# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral TTS acoustic flow-matching transformer for N150.

3-layer bidirectional transformer that runs the flow-matching ODE solver.
Shares architecture with the text decoder but adds:
  - llm_projection: projects text decoder hidden state h → D
  - time_projection: projects sinusoidal timestep embedding → D
  - input_projection: projects 36-dim acoustic noise x_t → D
  - acoustic_codebook_output: projects D → 36 velocity output
  - semantic_codebook_output: projects D → 8320 semantic logits (padded from 8192)

Inference:
  8 Euler ODE steps × 2 forward passes (conditioned + unconditioned CFG).
  Each forward pass: combined = h_proj + t_proj + x_proj → 3-layer transformer.

Weight keys (acoustic_transformer.* namespace, stripped to relative keys):
  llm_projection.weight         [3072, 3072]
  time_projection.weight        [3072, 3072]
  input_projection.weight       [3072, 36]
  norm.weight                   [3072]
  acoustic_codebook_output.weight [36, 3072]
  semantic_codebook_output.weight [8320, 3072]
  layers.{0-2}.attention.wq/wk/wv/wo
  layers.{0-2}.attention_norm.weight
  layers.{0-2}.feed_forward.w1/w2/w3.weight
  layers.{0-2}.ffn_norm.weight
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _sinusoidal_time_embedding(t_tensor: torch.Tensor, dim: int = 3072) -> torch.Tensor:
    """Sinusoidal timestep embedding on CPU → cast to bfloat16."""
    half = dim // 2
    inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    emb = t_tensor.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cat([emb.cos(), emb.sin()], dim=-1).bfloat16()


class TtVoxtralAcousticTransformer(LightweightModule):
    """Flow-matching acoustic transformer for Voxtral-4B-TTS-2603.

    Single-device N150, bidirectional SDPA (no causal mask, no KV cache).
    ODE loop is orchestrated in Python; this class runs one forward step.
    """

    def __init__(
        self,
        device,
        state_dict,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.device = device
        self.n_layers = 3
        self.dim = configuration.dim  # 3072
        self.n_heads = configuration.n_heads  # 32
        self.n_kv_heads = configuration.n_kv_heads  # 8
        self.head_dim = configuration.head_dim  # 128
        self.scale = self.head_dim**-0.5
        self.model_config = configuration.get_model_config()
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN

        cache_name = (
            (lambda n: weight_cache_path / f"acoustic_transformer.{n}")
            if weight_cache_path and not configuration.dummy_weights
            else (lambda _: None)
        )

        def _up(w, name, dtype_=None):
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),
                dtype=dtype_ or dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name),
            )

        # ── Input projections ──────────────────────────────────────────────
        self.llm_proj = _up(state_dict["llm_projection.weight"], "llm_proj")  # [3072→3072]
        self.time_proj = _up(state_dict["time_projection.weight"], "time_proj")  # [3072→3072]
        self.x_proj = _up(state_dict["input_projection.weight"], "x_proj")  # [36→3072]

        # ── Output projections ─────────────────────────────────────────────
        self.acoustic_out = _up(state_dict["acoustic_codebook_output.weight"], "acoustic_out")  # [3072→36]
        self.semantic_out = _up(state_dict["semantic_codebook_output.weight"], "semantic_out")  # [3072→8320]

        # ── Final norm ─────────────────────────────────────────────────────
        norm_w = state_dict["norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, self.dim)
        self.norm_w = ttnn.as_tensor(
            norm_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ── 3-layer transformer (same structure as text decoder, bidirectional) ──
        self.attn_norm_weights = []
        self.ffn_norm_weights = []
        self.wqkv_layers = []
        self.wo_layers = []
        self.w1_layers = []
        self.w2_layers = []
        self.w3_layers = []

        for i in range(self.n_layers):
            pfx = f"layers.{i}"

            # Norms
            attn_nw = state_dict[f"{pfx}.attention_norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, self.dim)
            ffn_nw = state_dict[f"{pfx}.ffn_norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, self.dim)
            self.attn_norm_weights.append(
                ttnn.as_tensor(
                    attn_nw,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            self.ffn_norm_weights.append(
                ttnn.as_tensor(
                    ffn_nw,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )

            # Fused wqkv [3072, 6144]
            wq = state_dict[f"{pfx}.attention.wq.weight"].T  # [3072, 4096]
            wk = state_dict[f"{pfx}.attention.wk.weight"].T  # [3072, 1024]
            wv = state_dict[f"{pfx}.attention.wv.weight"].T  # [3072, 1024]
            wqkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)
            self.wqkv_layers.append(
                ttnn.as_tensor(
                    wqkv,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name(f"layer{i}_wqkv"),
                )
            )

            wo = state_dict[f"{pfx}.attention.wo.weight"].T.unsqueeze(0).unsqueeze(0)  # [4096, 3072]
            self.wo_layers.append(
                ttnn.as_tensor(
                    wo,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name(f"layer{i}_wo"),
                )
            )

            # MLP
            self.w1_layers.append(_up(state_dict[f"{pfx}.feed_forward.w1.weight"], f"layer{i}_w1"))
            self.w2_layers.append(_up(state_dict[f"{pfx}.feed_forward.w2.weight"], f"layer{i}_w2"))
            self.w3_layers.append(_up(state_dict[f"{pfx}.feed_forward.w3.weight"], f"layer{i}_w3"))

    def _attention_layer(self, hidden, layer_idx):
        """Bidirectional attention (no causal mask, no KV cache)."""
        B = hidden.shape[0]
        N = hidden.shape[2]

        # QKV
        xqkv = ttnn.linear(
            hidden,
            self.wqkv_layers[layer_idx],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_heads,
            num_kv_heads=self.n_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # Bidirectional SDPA (no causal mask)
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](N),
            compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_11SH = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        out = ttnn.linear(
            attn_11SH,
            self.wo_layers[layer_idx],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_11SH)
        return out

    def _mlp_layer(self, hidden, layer_idx):
        w1_out = ttnn.linear(
            hidden,
            self.w1_layers[layer_idx],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        w3_out = ttnn.linear(
            hidden,
            self.w3_layers[layer_idx],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(hidden)
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        out = ttnn.linear(
            w2_in,
            self.w2_layers[layer_idx],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(w2_in)
        return out

    def forward(
        self,
        h_tt: ttnn.Tensor,  # [1, 1, N, 3072] text decoder hidden states
        x_t_tt: ttnn.Tensor,  # [1, 1, N, 36] current acoustic embedding
        t: float,  # ODE timestep ∈ [0, 1]
    ) -> tuple:
        """Returns (velocity_tt [1, 1, N, 36], semantic_logits_tt [1, 1, N, 8320])."""
        N = h_tt.shape[2]
        B = 1

        # Sinusoidal timestep embedding (on CPU)
        t_tensor = torch.tensor([t], dtype=torch.float32)
        t_emb = _sinusoidal_time_embedding(t_tensor, dim=self.dim)  # [1, 3072]
        t_emb_tt = ttnn.from_torch(
            t_emb.expand(N, -1).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, N, 3072]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Three projections: h, t, x_t → each [1, 1, N, 3072]
        h_proj = ttnn.linear(
            h_tt,
            self.llm_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        t_proj = ttnn.linear(
            t_emb_tt,
            self.time_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        x_proj = ttnn.linear(
            x_t_tt,
            self.x_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(t_emb_tt)

        # Sum all three projections
        combined = ttnn.add(h_proj, t_proj)
        ttnn.deallocate(h_proj)
        ttnn.deallocate(t_proj)
        combined = ttnn.add(combined, x_proj)
        ttnn.deallocate(x_proj)

        # 3-layer bidirectional transformer
        hidden = combined
        for i in range(self.n_layers):
            # Attention sub-layer
            normed = ttnn.rms_norm(hidden, weight=self.attn_norm_weights[i], epsilon=1e-5)
            attn_out = self._attention_layer(normed, i)
            hidden = ttnn.add(hidden, attn_out)
            ttnn.deallocate(attn_out)

            # MLP sub-layer
            normed2 = ttnn.rms_norm(hidden, weight=self.ffn_norm_weights[i], epsilon=1e-5)
            mlp_out = self._mlp_layer(normed2, i)
            hidden = ttnn.add(hidden, mlp_out)
            ttnn.deallocate(mlp_out)

        # Final norm
        hidden = ttnn.rms_norm(hidden, weight=self.norm_w, epsilon=1e-5)

        # Predict velocity and semantic logits — use HiFi4 for better precision on small-dim outputs
        velocity = ttnn.linear(
            hidden,
            self.acoustic_out,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        semantic_logits = ttnn.linear(
            hidden,
            self.semantic_out,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        ttnn.deallocate(hidden)

        return velocity, semantic_logits


def ode_solve_ttnn(
    h_tt: ttnn.Tensor,  # [1, 1, N, 3072]
    model: TtVoxtralAcousticTransformer,
    device,
    n_steps: int = 8,
    cfg_alpha: float = 1.2,
) -> tuple:
    """Euler ODE solve on device. Returns (acoustic_codes, x_continuous)."""
    N = h_tt.shape[2]
    dt = 1.0 / n_steps

    # Initial noise
    x_t = torch.randn(1, 1, N, 36, dtype=torch.bfloat16)
    x_t_tt = ttnn.from_torch(
        x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Null conditioning for CFG
    null_h_tt = ttnn.from_torch(
        torch.zeros(1, 1, N, 3072, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for step in range(n_steps):
        t = step * dt

        v_cond, _ = model.forward(h_tt, x_t_tt, t)
        v_uncond, _ = model.forward(null_h_tt, x_t_tt, t)

        # v_guided = cfg_alpha * v_cond + (1 - cfg_alpha) * v_uncond
        v_guided = ttnn.add(
            ttnn.mul(v_cond, cfg_alpha),
            ttnn.mul(v_uncond, 1.0 - cfg_alpha),
        )
        ttnn.deallocate(v_cond)
        ttnn.deallocate(v_uncond)

        # x_{t+1} = x_t + v_guided * dt
        x_t_tt = ttnn.add(x_t_tt, ttnn.mul(v_guided, dt))
        ttnn.deallocate(v_guided)

    ttnn.deallocate(null_h_tt)

    # Quantize: round x to FSQ levels [0, 20]
    x_continuous = ttnn.to_torch(x_t_tt).squeeze(0).squeeze(0)  # [N, 36]
    ttnn.deallocate(x_t_tt)
    acoustic_codes = x_continuous.round().long().clamp(0, 20)

    return acoustic_codes, x_continuous
