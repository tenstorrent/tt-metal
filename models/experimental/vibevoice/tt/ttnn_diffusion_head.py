# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice DiffusionHead — TTNN port.

Reference: VibeVoiceDiffusionHead in modular_vibevoice_diffusion_head.py
Components:
  - TimestepEmbedder: sin-cos freqs (precomputed on host) + 2-layer MLP (SiLU)
  - 4 x HeadLayer: adaLN (shift/scale/gate) + SwiGLU FFN
  - FinalLayer: adaLN (shift/scale) + linear projection → latent_size

No torch in forward().
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import ttnn


_COMPUTE_KERNEL_FP32 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Decode program configs (CFG batch-2 only) ──────────────────────────────────
# The head runs on the CFG-batched input [2,1,1,*] (B=2), so M_tiles=2 (each batch is
# padded to its own tile).  The stock auto config leaves the hot matmuls at ~20-40% DRAM
# BW; these swept 1D mcast_in0 configs (per_core_M=2) match the LM decode configs' shape
# family and recover 1.5-3.8x device time.  Precision-neutral (PCC vs auto ~1.0);
# validated in tests/perf/diffusion_progcfg_probe.py.  per_core_M=2 makes each config
# valid ONLY for the B=2 path — callers gate on input.shape[0]==2 (else auto).
def _mm1d(cx, cy, in0_block_w, per_core_n, out_subblock_w=None):
    # out_subblock_w must divide per_core_N (and out_block_w defaults to pcn).
    osw = out_subblock_w if out_subblock_w is not None else min(2, per_core_n)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=2,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


_MM_DOWN = _mm1d(8, 3, 8, 2)  # ffn down_proj   [.,4608]@[4608,1536]  146->38us (3.8x)
_MM_ADALN = _mm1d(8, 9, 4, 2)  # layer adaLN     [.,1536]@[1536,4608]  73->40us  (1.9x)
_MM_GATEUP = _mm1d(8, 9, 6, 2)  # ffn gate/up bf8 [.,1536]@[1536,4608]  61->35us  (1.7x)
_MM_HSQ = _mm1d(8, 3, 12, 2)  # cond_proj/t_mlp2[.,1536]@[1536,1536]  50->33us  (1.5x)
_MM_FADALN = _mm1d(8, 6, 12, 2)  # final adaLN     [.,1536]@[1536,3072]  70->34us  (2.0x)
# Untuned on auto (2 cores / ~36us SLOW in speech_frame_exp6). Isolation sweep
# (interleaved/WS/HS/BS/DRAMW): width/height/block/DRAM-sharded all lose on M=32;
# L1-interleaved 1D wins — final 1536→64 ~43us, noisy 64→1536 ~42us.
_MM_FINAL = _mm1d(2, 1, 12, 1)  # final_linear   [.,1536]@[1536,64]  (osw=1)
_MM_NOISY = _mm1d(8, 3, 2, 2)  # noisy_images   [.,64]@[64,1536]


def _is_cfg_batch2(x: ttnn.Tensor) -> bool:
    """CFG-batched path: B=2, each stream is one tile row (per_core_M=2 configs)."""
    return x.shape[0] == 2


def _pc(x: ttnn.Tensor, cfg):
    """Tuned config only on the B=2 CFG path (per_core_M=2); auto otherwise."""
    return cfg if _is_cfg_batch2(x) else None


def _act_mc(x: ttnn.Tensor):
    """Keep CFG-batch activations in L1 interleaved (tiny tensors; kills DRAM round-trips)."""
    return ttnn.L1_MEMORY_CONFIG if _is_cfg_batch2(x) else ttnn.DRAM_MEMORY_CONFIG


def _to_act_mc(x: ttnn.Tensor) -> ttnn.Tensor:
    """Move/ensure activation is on the preferred mem config for this path."""
    want = _act_mc(x)
    mc = x.memory_config()
    if mc.buffer_type == want.buffer_type and mc.memory_layout == want.memory_layout:
        return x
    return ttnn.to_memory_config(x, want)


@dataclass
class DiffusionHeadWeights:
    """All device tensors for VibeVoiceDiffusionHead."""

    # noisy_images_proj
    noisy_images_proj_w: ttnn.Tensor  # [latent, hidden]
    # cond_proj
    cond_proj_w: ttnn.Tensor  # [hidden, hidden]
    # timestep embedder MLP
    t_mlp0_w: ttnn.Tensor  # [hidden, freq_emb_size]
    t_mlp2_w: ttnn.Tensor  # [hidden, hidden]
    # precomputed frequency table for sin timestep embedding
    freq_table: ttnn.Tensor  # [1, 1, 1, freq_emb_size//2] — used with mul
    # per-layer weights
    layer_adaLN_w: List[ttnn.Tensor]  # each [3*hidden, hidden]
    layer_ffn_gate_w: List[ttnn.Tensor]  # [ffn_dim, hidden]
    layer_ffn_up_w: List[ttnn.Tensor]  # [ffn_dim, hidden]
    layer_ffn_down_w: List[ttnn.Tensor]  # [hidden, ffn_dim]
    layer_norm_w: List[ttnn.Tensor]  # [1,1,1,hidden]
    # final layer
    final_adaLN_w: ttnn.Tensor  # [2*hidden, hidden]
    final_linear_w: ttnn.Tensor  # [latent, hidden]
    # config
    hidden_size: int
    latent_size: int
    frequency_embedding_size: int = 256
    norm_eps: float = 1e-5


def _as_tile(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.as_tensor(
        t,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build_freq_table(frequency_embedding_size: int, max_period: int = 10000) -> torch.Tensor:
    """Precompute frequency table for sin timestep embeddings (host)."""
    half = frequency_embedding_size // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    # Shape [1, 1, 1, half] for broadcasting with timestep [1, 1, 1, 1]
    return freqs.view(1, 1, 1, half).to(torch.bfloat16)


def preprocess_diffusion_head_weights(
    hf_state: dict,
    device,
    hidden_size: int = 1536,
    latent_size: int = 64,
    head_ffn_ratio: float = 3.0,
    frequency_embedding_size: int = 256,
    norm_eps: float = 1e-5,
    num_layers: int = 4,
) -> DiffusionHeadWeights:
    """Convert host HF diffusion head state dict to device tensors.

    hf_state keys (prefix-stripped, e.g. from split_submodule_weights["diffusion_head"]):
      noisy_images_proj.weight, cond_proj.weight
      t_embedder.mlp.0.weight, t_embedder.mlp.2.weight
      layers.N.adaLN_modulation.1.weight, layers.N.norm.weight
      layers.N.ffn.gate_proj.weight, layers.N.ffn.up_proj.weight, layers.N.ffn.down_proj.weight
      final_layer.adaLN_modulation.1.weight, final_layer.linear.weight
    """

    def w(key) -> torch.Tensor:
        return hf_state[key].to(torch.bfloat16)

    def _w_tile(key: str, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        # ttnn.linear computes x @ W (no transpose), so store weights transposed [in, out]
        return _as_tile(w(key).t().unsqueeze(0).unsqueeze(0), device, dtype=dtype)

    def _norm_tile(w_1d: torch.Tensor) -> ttnn.Tensor:
        # ttnn.rms_norm requires gamma shape [1, 1, dim//32, 32] in ROW_MAJOR
        dim = w_1d.shape[0]
        return ttnn.as_tensor(
            w_1d.view(1, 1, dim // 32, 32),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    noisy_proj_w = _w_tile("noisy_images_proj.weight")
    cond_proj_w = _w_tile("cond_proj.weight")
    t_mlp0_w = _w_tile("t_embedder.mlp.0.weight")
    t_mlp2_w = _w_tile("t_embedder.mlp.2.weight")

    freq_table_torch = _build_freq_table(frequency_embedding_size)
    freq_table_tt = _as_tile(freq_table_torch, device, dtype=ttnn.bfloat16)

    layer_adaLN_w = []
    layer_ffn_gate_w = []
    layer_ffn_up_w = []
    layer_ffn_down_w = []
    layer_norm_w = []
    for i in range(num_layers):
        layer_adaLN_w.append(_w_tile(f"layers.{i}.adaLN_modulation.1.weight"))
        layer_norm_w.append(_norm_tile(w(f"layers.{i}.norm.weight")))
        # gate/up-proj weights bf8_b (weight-bandwidth-bound at B=2, M=1); down_proj stays
        # bf16 — it feeds the gated residual and this head runs iteratively (10 steps), so
        # down bf8_b drops single-forward PCC 0.9953->0.9940 (too tight without an e2e gate).
        layer_ffn_gate_w.append(_w_tile(f"layers.{i}.ffn.gate_proj.weight", dtype=ttnn.bfloat8_b))
        layer_ffn_up_w.append(_w_tile(f"layers.{i}.ffn.up_proj.weight", dtype=ttnn.bfloat8_b))
        layer_ffn_down_w.append(_w_tile(f"layers.{i}.ffn.down_proj.weight"))

    final_adaLN_w = _w_tile("final_layer.adaLN_modulation.1.weight")
    final_linear_w = _w_tile("final_layer.linear.weight")

    return DiffusionHeadWeights(
        noisy_images_proj_w=noisy_proj_w,
        cond_proj_w=cond_proj_w,
        t_mlp0_w=t_mlp0_w,
        t_mlp2_w=t_mlp2_w,
        freq_table=freq_table_tt,
        layer_adaLN_w=layer_adaLN_w,
        layer_ffn_gate_w=layer_ffn_gate_w,
        layer_ffn_up_w=layer_ffn_up_w,
        layer_ffn_down_w=layer_ffn_down_w,
        layer_norm_w=layer_norm_w,
        final_adaLN_w=final_adaLN_w,
        final_linear_w=final_linear_w,
        hidden_size=hidden_size,
        latent_size=latent_size,
        frequency_embedding_size=frequency_embedding_size,
        norm_eps=norm_eps,
    )


class TTDiffusionHead:
    """TTNN port of VibeVoiceDiffusionHead.

    forward(noisy_images, timesteps, condition) — no torch tensors allowed.
    """

    def __init__(self, weights: DiffusionHeadWeights):
        self.w = weights

    def _timestep_embedding(self, t_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Sinusoidal timestep embedding on device.

        t_tt: [B, 1, 1, 1] scalar timestep tensor (bfloat16)
        Returns: [B, 1, 1, freq_emb_size]
        """
        w = self.w
        # t_tt * freqs → [B, 1, 1, half]
        args = ttnn.mul(t_tt, w.freq_table, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos_half = ttnn.cos(args, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin_half = ttnn.sin(args, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # concat [B,1,1,half] and [B,1,1,half] → [B,1,1,freq_emb_size]
        embedding = ttnn.concat([cos_half, sin_half], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return embedding

    def _timestep_embedder(self, t_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Full timestep embedder: sin_emb → MLP (silu → linear) → cond_dim."""
        w = self.w
        t_freq = self._timestep_embedding(t_tt)  # [B, 1, 1, freq_emb_size]
        mc = _act_mc(t_freq)
        # MLP layer 0 + SiLU
        h = ttnn.linear(
            t_freq,
            w.t_mlp0_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=mc,
        )
        h = ttnn.silu(h, memory_config=mc)
        # MLP layer 2
        h = ttnn.linear(
            h,
            w.t_mlp2_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(h, _MM_HSQ),
            memory_config=mc,
        )
        return h  # [B, 1, 1, hidden_size]

    def _swiglu_ffn(self, x: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """SwiGLU FFN: gate * silu(gate) project → down."""
        w = self.w
        mc = _act_mc(x)
        # Fuse silu into the gate matmul (drops a separate Unary op per layer).
        # Keep gate/up/mul/down in L1 on the CFG path so the FFN stays DRAM-roundtrip-free.
        gate = ttnn.linear(
            x,
            w.layer_ffn_gate_w[layer_idx],
            activation="silu",
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(x, _MM_GATEUP),
            memory_config=mc,
        )
        up = ttnn.linear(
            x,
            w.layer_ffn_up_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(x, _MM_GATEUP),
            memory_config=mc,
        )
        hidden = ttnn.mul(gate, up, memory_config=mc)
        out = ttnn.linear(
            hidden,
            w.layer_ffn_down_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(hidden, _MM_DOWN),
            memory_config=mc,
        )
        return out

    def _head_layer(self, x: ttnn.Tensor, sc: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """Single HeadLayer: adaLN + SwiGLU residual.

        x:  [B, T, 1, hidden]  or [B, 1, 1, hidden] for latent
        sc: [B, 1, 1, hidden]  silu(conditioning), precomputed once per step (same for all layers)
        """
        w = self.w
        mc = _act_mc(x)
        # adaLN_modulation(silu(c)) → [B, 1, 1, 3*hidden]
        modulation = ttnn.linear(
            sc,
            w.layer_adaLN_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(sc, _MM_ADALN),
            memory_config=mc,
        )
        # chunk into 3 parts along last dim
        hidden_size = w.hidden_size
        shift = ttnn.slice(modulation, [0, 0, 0, 0], [modulation.shape[0], 1, 1, hidden_size])
        scale = ttnn.slice(modulation, [0, 0, 0, hidden_size], [modulation.shape[0], 1, 1, 2 * hidden_size])
        gate = ttnn.slice(modulation, [0, 0, 0, 2 * hidden_size], [modulation.shape[0], 1, 1, 3 * hidden_size])

        # RMSNorm(x)
        x_norm = ttnn.rms_norm(
            x,
            weight=w.layer_norm_w[layer_idx],
            epsilon=w.norm_eps,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=mc,
        )
        # modulate: x_norm * (1 + scale) + shift  (scalar +1.0 avoids a ones_like alloc op)
        x_mod = ttnn.add(
            ttnn.mul(
                x_norm,
                ttnn.add(scale, 1.0, memory_config=mc),
                memory_config=mc,
            ),
            shift,
            memory_config=mc,
        )
        # FFN + gated residual
        ffn_out = self._swiglu_ffn(x_mod, layer_idx)
        gated = ttnn.mul(gate, ffn_out, memory_config=mc)
        out = ttnn.add(x, gated, memory_config=mc)
        return out

    def _final_layer(self, x: ttnn.Tensor, sc: ttnn.Tensor) -> ttnn.Tensor:
        """FinalLayer: adaLN (shift/scale, no gate) + linear → latent_size.

        sc: silu(conditioning), precomputed once per step.
        """
        w = self.w
        mc = _act_mc(x)
        modulation = ttnn.linear(
            sc,
            w.final_adaLN_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(sc, _MM_FADALN),
            memory_config=mc,
        )
        hidden_size = w.hidden_size
        shift = ttnn.slice(modulation, [0, 0, 0, 0], [modulation.shape[0], 1, 1, hidden_size])
        scale = ttnn.slice(modulation, [0, 0, 0, hidden_size], [modulation.shape[0], 1, 1, 2 * hidden_size])

        # RMSNorm without learnable weight (elementwise_affine=False in reference)
        x_norm = ttnn.rms_norm(
            x,
            epsilon=w.norm_eps,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=mc,
        )
        # modulate: x_norm * (1 + scale) + shift  (scalar +1.0 avoids a ones_like alloc op)
        x_mod = ttnn.add(
            ttnn.mul(
                x_norm,
                ttnn.add(scale, 1.0, memory_config=mc),
                memory_config=mc,
            ),
            shift,
            memory_config=mc,
        )
        out = ttnn.linear(
            x_mod,
            w.final_linear_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(x_mod, _MM_FINAL),
            memory_config=mc,
        )
        return out

    def precompute_t_embs(self, t_tensors: list) -> list:
        """Precompute the timestep-embedder output for each denoising step's timestep.

        The DPM timesteps are fixed by set_timesteps(num_steps) and are identical on every
        frame, so t_emb (a pure function of the timestep) can be computed once and reused —
        removing the sin/cos/concat + two MLP matmuls from every step of every frame.
        Numerically exact.  Call once, outside any trace capture.
        """
        return [self._timestep_embedder(t) for t in t_tensors]

    def forward(
        self,
        noisy_images: ttnn.Tensor,
        timesteps: ttnn.Tensor,
        condition: ttnn.Tensor,
        t_emb: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            noisy_images: [B, 1, 1, latent_size] bfloat16 TILE
            timesteps:    [B, 1, 1, 1] bfloat16 scalar per batch (ignored if t_emb given)
            condition:    [B, 1, 1, hidden_size] bfloat16
            t_emb:        optional precomputed timestep embedding [B,1,1,hidden] (see
                          precompute_t_embs) — skips the per-step timestep embedder.

        Returns:
            [B, 1, 1, latent_size]
        """
        w = self.w
        # CFG batch-2: park activations in L1 for the whole head (tensors are tiny).
        noisy_images = _to_act_mc(noisy_images)
        condition = _to_act_mc(condition)
        if t_emb is not None:
            t_emb = _to_act_mc(t_emb)
        mc = _act_mc(noisy_images)

        # Project noisy latent to hidden_size
        x = ttnn.linear(
            noisy_images,
            w.noisy_images_proj_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(noisy_images, _MM_NOISY),
            memory_config=mc,
        )

        # Timestep embedding (precomputed once per step-index when supplied)
        if t_emb is None:
            t_emb = self._timestep_embedder(timesteps)  # [B, 1, 1, hidden]
            t_emb = _to_act_mc(t_emb)

        # Project condition
        cond_proj = ttnn.linear(
            condition,
            w.cond_proj_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_pc(condition, _MM_HSQ),
            memory_config=mc,
        )

        # Combine: c = cond_proj + t_emb
        c = ttnn.add(cond_proj, t_emb, memory_config=mc)
        # silu(c) is the input to every adaLN modulation and is identical across all
        # head layers + the final layer — compute it once per step instead of 5×.
        sc = ttnn.silu(c, memory_config=mc)

        # HeadLayers
        num_layers = len(w.layer_adaLN_w)
        for i in range(num_layers):
            x = self._head_layer(x, sc, i)

        # FinalLayer
        x = self._final_layer(x, sc)
        return x

    def __call__(
        self,
        noisy_images: ttnn.Tensor,
        timesteps: ttnn.Tensor,
        condition: ttnn.Tensor,
        t_emb: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        return self.forward(noisy_images, timesteps, condition, t_emb=t_emb)
