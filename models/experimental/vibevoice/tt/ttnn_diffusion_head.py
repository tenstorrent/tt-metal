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
from typing import List

import torch
import ttnn


_COMPUTE_KERNEL_FP32 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# Byte-identical B=2 program configs for the CFG diffusion head.  The head ALWAYS runs at B=2
# (sample_speech_latents concats [neg,pos] on dim0), on auto, so each head weight is read TWICE per
# step x 10 steps/frame — the same weight-read-twice waste the LM FFN had.  per_core_M=2 folds both
# CFG rows into M so the weights are read once.  in0_block_w=2 is auto's K-reduction block for these
# shapes (proven maxabsdiff==0 vs auto for both fp32 and bf16 inputs in
# tests/perf/diffusion_byteident_ibw_sweep.py) => same reduction order => long-form-safe (Tier-0),
# ~1.6-1.9x per matmul.  Applied only when B==2; a B=1 PCC-test call falls back to auto.
def _diff_b2_cfg(cx, cy, pn):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=2,
        per_core_N=pn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


_DIFF_N4608_B2 = _diff_b2_cfg(8, 9, 2)  # gate / up / head-layer modulation  (K=1536, N=4608)
_DIFF_N1536_B2 = _diff_b2_cfg(8, 3, 2)  # swiglu down                        (K=4608, N=1536)
_DIFF_N3072_B2 = _diff_b2_cfg(8, 6, 2)  # final-layer modulation             (K=1536, N=3072)


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

    def _w_tile(key: str) -> ttnn.Tensor:
        # ttnn.linear computes x @ W (no transpose), so store weights transposed [in, out]
        return _as_tile(w(key).t().unsqueeze(0).unsqueeze(0), device)

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
        layer_ffn_gate_w.append(_w_tile(f"layers.{i}.ffn.gate_proj.weight"))
        layer_ffn_up_w.append(_w_tile(f"layers.{i}.ffn.up_proj.weight"))
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
        # MLP layer 0 + SiLU
        h = ttnn.linear(
            t_freq,
            w.t_mlp0_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # MLP layer 2
        h = ttnn.linear(
            h,
            w.t_mlp2_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return h  # [B, 1, 1, hidden_size]

    def _swiglu_ffn(self, x: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """SwiGLU FFN: gate * silu(gate) project → down."""
        w = self.w
        b2 = x.shape[0] == 2  # CFG B=2 frame path → byte-identical weight-read-once progcfgs
        gate = ttnn.linear(
            x,
            w.layer_ffn_gate_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_DIFF_N4608_B2 if b2 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            w.layer_ffn_up_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_DIFF_N4608_B2 if b2 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.linear(
            hidden,
            w.layer_ffn_down_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_DIFF_N1536_B2 if b2 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def _head_layer(self, x: ttnn.Tensor, sc: ttnn.Tensor, layer_idx: int) -> ttnn.Tensor:
        """Single HeadLayer: adaLN + SwiGLU residual.

        x:  [B, T, 1, hidden]  or [B, 1, 1, hidden] for latent
        sc: [B, 1, 1, hidden]  = silu(conditioning), precomputed once per step (dedup) and shared
            across all HeadLayers + FinalLayer (byte-identical to computing silu(c) per layer).
        """
        w = self.w
        # adaLN_modulation(silu(c)) → [B, 1, 1, 3*hidden]
        modulation = ttnn.linear(
            sc,
            w.layer_adaLN_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_DIFF_N4608_B2 if sc.shape[0] == 2 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # modulate: x_norm * (1 + scale) + shift
        one = ttnn.ones_like(scale)
        x_mod = ttnn.add(
            ttnn.mul(
                x_norm,
                ttnn.add(one, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            shift,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # FFN + gated residual
        ffn_out = self._swiglu_ffn(x_mod, layer_idx)
        gated = ttnn.mul(gate, ffn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(x, gated, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    def _final_layer(self, x: ttnn.Tensor, sc: ttnn.Tensor) -> ttnn.Tensor:
        """FinalLayer: adaLN (shift/scale, no gate) + linear → latent_size.

        ``sc`` = silu(conditioning), shared with the HeadLayers (see _head_layer)."""
        w = self.w
        modulation = ttnn.linear(
            sc,
            w.final_adaLN_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_DIFF_N3072_B2 if sc.shape[0] == 2 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_size = w.hidden_size
        shift = ttnn.slice(modulation, [0, 0, 0, 0], [modulation.shape[0], 1, 1, hidden_size])
        scale = ttnn.slice(modulation, [0, 0, 0, hidden_size], [modulation.shape[0], 1, 1, 2 * hidden_size])

        # RMSNorm without learnable weight (elementwise_affine=False in reference)
        x_norm = ttnn.rms_norm(
            x,
            epsilon=w.norm_eps,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        one = ttnn.ones_like(scale)
        x_mod = ttnn.add(
            ttnn.mul(
                x_norm,
                ttnn.add(one, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            shift,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.linear(
            x_mod,
            w.final_linear_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def project_condition(self, condition: ttnn.Tensor) -> ttnn.Tensor:
        """cond_proj = Linear(condition).  Split out of forward() so the DPM loop can hoist this
        step-INVARIANT projection out of its per-step head calls (the condition is fixed for the
        whole frame; only the noisy latent + timestep change per step).  Byte-identical: same op,
        same input — computing it once vs per-step yields the identical tensor."""
        return ttnn.linear(
            condition,
            self.w.cond_proj_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_pre_cond(
        self,
        noisy_images: ttnn.Tensor,
        timesteps: ttnn.Tensor,
        cond_proj: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Head forward given the ALREADY-projected condition (cond_proj = project_condition(cond)).

        Args:
            noisy_images: [B, 1, 1, latent_size] bfloat16 TILE
            timesteps:    [B, 1, 1, 1] bfloat16 scalar per batch
            cond_proj:    [B, 1, 1, hidden_size]  = project_condition(condition)
        """
        w = self.w

        # Project noisy latent to hidden_size
        x = ttnn.linear(
            noisy_images,
            w.noisy_images_proj_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Timestep embedding
        t_emb = self._timestep_embedder(timesteps)  # [B, 1, 1, hidden]

        # Combine: c = cond_proj + t_emb
        c = ttnn.add(cond_proj, t_emb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # silu(c) is the adaLN input for every HeadLayer + FinalLayer — compute it ONCE per step and
        # share (byte-identical to the per-layer silu, saves 4 redundant silus/step).
        sc = ttnn.silu(c, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # HeadLayers
        num_layers = len(w.layer_adaLN_w)
        for i in range(num_layers):
            x = self._head_layer(x, sc, i)

        # FinalLayer
        x = self._final_layer(x, sc)
        return x

    def forward(
        self,
        noisy_images: ttnn.Tensor,
        timesteps: ttnn.Tensor,
        condition: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            noisy_images: [B, 1, 1, latent_size] bfloat16 TILE
            timesteps:    [B, 1, 1, 1] bfloat16 scalar per batch
            condition:    [B, 1, 1, hidden_size] bfloat16

        Returns:
            [B, 1, 1, latent_size]
        """
        return self.forward_pre_cond(noisy_images, timesteps, self.project_condition(condition))

    def __call__(
        self,
        noisy_images: ttnn.Tensor,
        timesteps: ttnn.Tensor,
        condition: ttnn.Tensor,
    ) -> ttnn.Tensor:
        return self.forward(noisy_images, timesteps, condition)
