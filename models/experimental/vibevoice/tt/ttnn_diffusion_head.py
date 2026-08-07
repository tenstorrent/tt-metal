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

from models.experimental.vibevoice.common.weight_cache import WeightCache

# bfloat8_b for the diffusion-head SwiGLU FFN gate/up weights (1536x4608) — DRAM-bound matmuls
# where bf8b halves the weight read. down_proj (4608x1536) is latency-bound, so it stays bf16.
# Not bit-exact vs bf16.
_DIFF_FFN_DTYPE = ttnn.bfloat8_b

_COMPUTE_KERNEL_FP32 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# Byte-identical B=2 program configs for the CFG diffusion head.  The head always runs at B=2
# (sample_speech_latents concats [neg, pos] on dim 0); on the auto config each head weight is
# therefore read twice per step, 10 steps per frame.  per_core_M=2 folds both CFG rows into M so
# each weight is read once, worth ~1.6-1.9x per matmul.  in0_block_w=2 is the K-reduction block
# auto picks for these shapes, so the reduction order — and hence the rounding — is unchanged
# (maxabsdiff==0 vs auto for both fp32 and bf16 inputs).
#
# Applied only when B==2; a B=1 PCC-test call falls back to auto.
def _diff_b2_cfg(cx, cy, pn, ibw=2):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=2,
        per_core_N=pn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


# gate / up / head-layer modulation (K=1536, N=4608).  in0_block_w=4 (12 K-blocks) vs the prior ib2:
# -11 us/op in the deployed frame (84 calls => -0.95 ms).  NOTE: isolation timing showed this flat —
# only the in-model device profiler surfaces it.  maxabsdiff==0 for both the bf8b gate/up and the
# fp32-act modulation (in0_block_w is K-stream granularity only, fp32 dest -> byte-identical).
_DIFF_N4608_B2 = _diff_b2_cfg(8, 9, 2, ibw=4)
# swiglu down (K=4608, N=1536): the 8x3=24-core / in0_block_w=2 config above ran at 74.4 us =
# 190 GB/s, less than half the 379 GB/s its same-weight-size sibling (_DIFF_N4608_B2) reaches.
# Widening the K-streaming granularity to in0_block_w=4 AND spreading N over 48 cores
# (per_core_N=1, 48*1 == Nt) gives 49.7 us = 285 GB/s (1.50x, 40 calls/frame => -1.0 ms).
# Neither change alone is enough: 48 cores at in0_block_w=2 is *slower* than the 24-core config
# (81.8 us).  Measured maxabsdiff==0 vs both auto and the previous config — in0_block_w only sets
# the K-tile streaming granularity and each core still reduces the full K into an fp32 dest, so
# the accumulation order (and the bf16 output) is unchanged.  Confirmed end-to-end: a full 93-min
# 4p_climate_100min render with this and the two LM config changes is BYTE-IDENTICAL to the
# pre-change render (same sha256 over all 134,163,200 samples, same 42,498 AR tokens).
_DIFF_N1536_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(6, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=2,
    per_core_N=1,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
_DIFF_N3072_B2 = _diff_b2_cfg(8, 6, 2)  # final-layer modulation             (K=1536, N=3072)
# cond_proj (B=2, K=1536, N=1536).  This was the ONLY matmul in the frame left on the auto
# program config, and auto lands it at 55 us / 19.6% of DRAM peak while its five progcfg'd fp32
# siblings (the adaLN modulations, same HiFi4 FP32 x BF16 dtypes) run at 63-79%.  So the fp32
# activation was never the problem — the missing config was.  The LM's wq/wo config is legal
# here (identical 1536x1536 shape, K=1536 -> 48 tiles, 24 cores x per_core_N=2 == Nt) and is what
# gets that shape to 16.4 us in the LM.  Measured -35 us/frame (-0.11% end-to-end; 4/5 wins on a
# paired head-to-head, pooled over 11 baseline and 8 candidate interleaved runs with a control
# arm); every run BYTE-IDENTICAL to the auto baseline, so this is blocking/placement only.
# Applied at B==2; B=1 PCC paths keep auto.
_COND_PROJ_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=2,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
# final_linear 1536→64: auto runs this on only 2 cores (~36 µs).  in0_block_w=2 matches auto's
# K-reduction (maxabsdiff==0 vs auto) and brings it to ~21 µs.  Any other in0_block_w changes the
# reduction order, so it is not byte-identical and must not be used on the long-form path.
_DIFF_N64_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(1, 2),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=2,
    per_core_N=1,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)


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
    # adaLN "+1" bias: 1.0 in the scale chunk, 0 elsewhere.  See _adaLN_plus_one_bias.
    adaLN_bias3: ttnn.Tensor  # [1,1,1,3*hidden]  for the HeadLayers
    adaLN_bias2: ttnn.Tensor  # [1,1,1,2*hidden]  for the FinalLayer
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


@dataclass
class ModSchedule:
    """All DPM steps' adaLN modulations, stacked on the batch axis.

    layer_mod[i]: [batch*steps, 1, 1, 3*hidden]   final_mod: [batch*steps, 1, 1, 2*hidden]
    Step ``s`` occupies rows ``[s*batch, (s+1)*batch)``.
    """

    layer_mod: List[ttnn.Tensor]
    final_mod: ttnn.Tensor
    steps: int
    batch: int

    def rows(self, step_idx: int) -> tuple:
        """Half-open row range holding step ``step_idx``."""
        return step_idx * self.batch, (step_idx + 1) * self.batch


def _mod_batch_cfg(cfg, per_core_m: int):
    """A per-step modulation progcfg re-issued for the step-batched M (per_core_M only)."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        in0_block_w=cfg.in0_block_w,
        out_subblock_h=cfg.out_subblock_h,
        out_subblock_w=cfg.out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=cfg.per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
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
    weight_cache: Optional[WeightCache] = None,
) -> DiffusionHeadWeights:
    """Convert host HF diffusion head state dict to device tensors.

    hf_state keys (prefix-stripped, e.g. from split_submodule_weights["diffusion_head"]):
      noisy_images_proj.weight, cond_proj.weight
      t_embedder.mlp.0.weight, t_embedder.mlp.2.weight
      layers.N.adaLN_modulation.1.weight, layers.N.norm.weight
      layers.N.ffn.gate_proj.weight, layers.N.ffn.up_proj.weight, layers.N.ffn.down_proj.weight
      final_layer.adaLN_modulation.1.weight, final_layer.linear.weight
    """

    wc = weight_cache if weight_cache is not None else WeightCache(None, enabled=False)

    def _w_tile(key: str, ckey: str, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        # ttnn.linear computes x @ W (no transpose), so store weights transposed [in, out].
        return wc.as_tensor(
            ckey,
            lambda: hf_state[key].to(torch.bfloat16).t().unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _norm_tile(key: str, ckey: str) -> ttnn.Tensor:
        # ttnn.rms_norm requires gamma shape [1, 1, dim//32, 32] in ROW_MAJOR
        def _mk():
            t = hf_state[key].to(torch.bfloat16)
            dim = t.shape[0]
            return t.view(1, 1, dim // 32, 32)

        return wc.as_tensor(
            ckey,
            _mk,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    noisy_proj_w = _w_tile("noisy_images_proj.weight", "noisy_images_proj")
    cond_proj_w = _w_tile("cond_proj.weight", "cond_proj")
    t_mlp0_w = _w_tile("t_embedder.mlp.0.weight", "t_mlp0")
    t_mlp2_w = _w_tile("t_embedder.mlp.2.weight", "t_mlp2")

    # freq_table is host-computed (not from the checkpoint) but cached for a uniform load path.
    freq_table_tt = wc.as_tensor(
        "freq_table",
        lambda: _build_freq_table(frequency_embedding_size),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    layer_adaLN_w = []
    layer_ffn_gate_w = []
    layer_ffn_up_w = []
    layer_ffn_down_w = []
    layer_norm_w = []
    for i in range(num_layers):
        layer_adaLN_w.append(_w_tile(f"layers.{i}.adaLN_modulation.1.weight", f"layers.{i}.adaLN"))
        layer_norm_w.append(_norm_tile(f"layers.{i}.norm.weight", f"layers.{i}.norm"))
        layer_ffn_gate_w.append(
            _w_tile(f"layers.{i}.ffn.gate_proj.weight", f"layers.{i}.ffn_gate", dtype=_DIFF_FFN_DTYPE)
        )
        layer_ffn_up_w.append(_w_tile(f"layers.{i}.ffn.up_proj.weight", f"layers.{i}.ffn_up", dtype=_DIFF_FFN_DTYPE))
        layer_ffn_down_w.append(_w_tile(f"layers.{i}.ffn.down_proj.weight", f"layers.{i}.ffn_down"))

    def _adaLN_plus_one_bias(n_chunks: int, ckey: str) -> ttnn.Tensor:
        """Fold adaLN's `1 + scale` into the modulation matmul as a bias.

        The reference adaLN_modulation Linear has NO bias (only `.weight` exists in the
        checkpoint, and the PCC test passes without one), so the bias slot is free.  Putting 1.0
        in the `scale` chunk makes the linear emit `1 + scale` directly, removing the standalone
        `ttnn.add(scale, 1.0)` that otherwise runs once per layer per DPM step — 50 ops/frame at
        ~5 us each.  Chunk order is (shift, scale[, gate]) in both the 3-chunk HeadLayer and the
        2-chunk FinalLayer, so the 1.0 always lands in [hidden, 2*hidden).

        NOT byte-identical, and not for the reason you would guess.  The bias VALUE is exact
        (1.0/0.0 round-trip in both bf16 and fp32, and a bf16 and an fp32 bias produce
        byte-for-byte the same wav), so the difference comes from the fused-bias epilogue itself
        taking a different path than a standalone `add` on the packed output.  Measured
        -0.27 ms/tok (-0.84%) on 2p_goat; validated on a full 4p_climate_100min render.
        """

        def _mk():
            b = torch.zeros(1, 1, 1, n_chunks * hidden_size, dtype=torch.float32)
            b[..., hidden_size : 2 * hidden_size] = 1.0
            return b

        return wc.as_tensor(
            ckey,
            _mk,
            device=device,
            dtype=ttnn.bfloat16,  # 0.0/1.0 exact in bf16; fp32 bias measured byte-for-byte identical
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    adaLN_bias3 = _adaLN_plus_one_bias(3, "adaLN_bias3")
    adaLN_bias2 = _adaLN_plus_one_bias(2, "adaLN_bias2")

    final_adaLN_w = _w_tile("final_layer.adaLN_modulation.1.weight", "final_adaLN")
    final_linear_w = _w_tile("final_layer.linear.weight", "final_linear")

    return DiffusionHeadWeights(
        noisy_images_proj_w=noisy_proj_w,
        cond_proj_w=cond_proj_w,
        t_mlp0_w=t_mlp0_w,
        t_mlp2_w=t_mlp2_w,
        freq_table=freq_table_tt,
        layer_adaLN_w=layer_adaLN_w,
        adaLN_bias3=adaLN_bias3,
        adaLN_bias2=adaLN_bias2,
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

    def embed_timestep(self, t_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Public alias of ``_timestep_embedder``.

        DPM inference timesteps are fixed for a schedule, so callers can precompute
        ``embed_timestep(t)`` once per step-index and reuse every frame (byte-identical
        to recomputing inside each head forward).
        """
        return self._timestep_embedder(t_tt)

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
        # Place the SwiGLU product (down_proj's in0) in L1: down_proj is the slowest head matmul
        # (K=4608, 24 cores, ~37% DRAM BW) and reads in0 ~9 µs faster from L1 (79.4 -> 70.7 µs).
        # Placement changes where the tensor lives, not its bits, so this stays byte-identical
        # (maxabsdiff==0).  Only down_proj benefits — L1 in0 regresses final_adaLN (41 -> 108 µs),
        # which therefore stays in DRAM.
        #
        # silu folded into the product as an in0 activation, dropping the standalone unary op
        # (40 calls/frame — 10 DPM steps x 4 head layers).  Measured maxabsdiff==0 vs
        # `mul(silu(gate), up)`: the fused form still rounds the activation to the operand dtype
        # before multiplying.  9.70 -> 7.99 us on the pair (1.21x).
        hidden = ttnn.mul(
            gate, up, memory_config=ttnn.L1_MEMORY_CONFIG, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        )
        out = ttnn.linear(
            hidden,
            w.layer_ffn_down_w[layer_idx],
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_DIFF_N1536_B2 if b2 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def precompute_modulations(self, cond_proj: ttnn.Tensor, t_embs: List[ttnn.Tensor]) -> "ModSchedule":
        """Compute EVERY DPM step's adaLN modulation up front — one matmul per layer, not per step.

        The adaLN input is ``sc_s = silu(cond_proj + t_embs[s])``.  ``cond_proj`` is frame-constant
        (already hoisted out of the DPM loop) and ``t_embs`` is schedule-constant, so all
        ``num_steps`` of ``sc`` are known *before* the loop runs.  One matmul per layer then
        produces every step's modulation, so each adaLN weight is read once per frame instead of
        once per step (40 -> 4 layer modulation matmuls, 10 -> 1 final; measured 400.5 -> 45.6 µs
        for a layer's ten calls).

        The steps stack on the BATCH axis, giving ``[2*S, 1, 1, hidden]``, so every per-step chunk
        slice below stays tile-aligned.  Two other layouts were measured end-to-end and lost:
        stacking on the tile HEIGHT axis makes the matmul far cheaper (Mt stays 2: 45.6 vs 400.5 µs
        for a layer's ten calls, against ~262 µs for this layout's Mt=2*S) but every chunk read
        becomes an unaligned intra-tile row slice (115 vs 65 µs) — with 3 chunks x S steps x 5
        modulations that was 3.6% SLOWER than not batching at all (42.02 vs 40.56 ms/tok), and
        extracting each step's row once before chunking still lost (36.91 vs 36.41).  The frame is
        op-count-bound at this scale: 50 extra slice ops outweigh the cheaper matmul.

        Byte-identical: matmul output rows are independent and ``in0_block_w`` (hence the K
        reduction order) is unchanged.  Verified ``maxabsdiff == 0`` against the per-step path.
        """
        w = self.w
        steps = len(t_embs)
        b = cond_proj.shape[0]
        # Broadcast the frame-constant condition across the step axis, then add the per-step
        # timestep embeddings in one shot.  Same values, same ops as the per-step add+silu.
        cond_all = ttnn.concat([cond_proj] * steps, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t_all = ttnn.concat(list(t_embs), dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sc_all = ttnn.silu(
            ttnn.add(cond_all, t_all, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # PACK the b*steps rows into tile ROWS before the matmul.  On the batch axis each of the 20
        # (b=2, S=10) rows owns a whole 32-row tile, so M is 20 tiles and the matmul computes 640
        # rows to use 20 — it is the one COMPUTE-bound matmul in the frame (29.2 TFLOPs, 29.3% of
        # peak, at only 96 GB/s: the DRAM column reads low because compute, not bandwidth, is the
        # wall).  As tile rows the same work is Mt=1 and the matmul becomes DRAM-bound like its
        # siblings: 315 -> 45 µs for N=4608, 291 -> 45 µs for the final N=3072.
        #
        # One reshape per modulation restores the batch axis, so every per-step chunk slice below
        # stays tile-aligned exactly as before — this is what makes the packing pay, unlike the two
        # height-axis layouts described above which pushed unaligned slices into the DPM loop.  The
        # reshapes cost ~60/44 µs and the one input-side reshape ~43 µs, against 1.27 ms of matmul
        # saved: measured 1.552 -> 0.561 ms per frame for the five modulations.
        rows = b * steps
        mt = (rows + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        sc_packed = ttnn.reshape(sc_all, [1, 1, rows, w.hidden_size])
        layer_cfg = _mod_batch_cfg(_DIFF_N4608_B2, mt) if b == 2 else None
        final_cfg = _mod_batch_cfg(_DIFF_N3072_B2, mt) if b == 2 else None

        def _mod(weight, cfg, n_out, bias):
            out = ttnn.linear(
                sc_packed,
                weight,
                bias=bias,
                compute_kernel_config=_COMPUTE_KERNEL_FP32,
                program_config=cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.reshape(out, [rows, 1, 1, n_out])

        layer_mod = [
            _mod(w.layer_adaLN_w[i], layer_cfg, 3 * w.hidden_size, w.adaLN_bias3) for i in range(len(w.layer_adaLN_w))
        ]
        final_mod = _mod(w.final_adaLN_w, final_cfg, 2 * w.hidden_size, w.adaLN_bias2)
        return ModSchedule(layer_mod=layer_mod, final_mod=final_mod, steps=steps, batch=b)

    def _head_layer(
        self,
        x: ttnn.Tensor,
        sc: ttnn.Tensor,
        layer_idx: int,
        mod: Optional["ModSchedule"] = None,
        step_idx: int = 0,
    ) -> ttnn.Tensor:
        """Single HeadLayer: adaLN + SwiGLU residual.

        x:  [B, T, 1, hidden]  or [B, 1, 1, hidden] for latent
        sc: [B, 1, 1, hidden]  = silu(conditioning), precomputed once per step (dedup) and shared
            across all HeadLayers + FinalLayer (byte-identical to computing silu(c) per layer).
            Unused (may be None) when ``mod`` carries a precomputed modulation schedule.
        mod/step_idx: optional ``precompute_modulations`` output + which step's row to read.
        """
        w = self.w
        if mod is not None:
            # Step ``step_idx``'s rows of the batch-stacked modulation.  The three chunk slices
            # below then carry the same [B, 1, 1, *] shapes as the per-step path.
            modulation = mod.layer_mod[layer_idx]
            r0, r1 = mod.rows(step_idx)
        else:
            # adaLN_modulation(silu(c)) → [B, 1, 1, 3*hidden]
            modulation = ttnn.linear(
                sc,
                w.layer_adaLN_w[layer_idx],
                bias=w.adaLN_bias3,
                compute_kernel_config=_COMPUTE_KERNEL_FP32,
                program_config=_DIFF_N4608_B2 if sc.shape[0] == 2 else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            r0, r1 = 0, modulation.shape[0]
        # chunk into 3 parts along last dim
        hidden_size = w.hidden_size
        shift = ttnn.slice(modulation, [r0, 0, 0, 0], [r1, 1, 1, hidden_size])
        scale = ttnn.slice(modulation, [r0, 0, 0, hidden_size], [r1, 1, 1, 2 * hidden_size])
        gate = ttnn.slice(modulation, [r0, 0, 0, 2 * hidden_size], [r1, 1, 1, 3 * hidden_size])

        # RMSNorm(x)
        x_norm = ttnn.rms_norm(
            x,
            weight=w.layer_norm_w[layer_idx],
            epsilon=w.norm_eps,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # modulate: x_norm * (1 + scale) + shift.  The +1 is folded into the modulation matmul's
        # bias (see _adaLN_plus_one_bias), so `scale` already IS (1 + scale) and the standalone
        # add is gone — 50 ops/frame removed, byte-identical.
        x_mod = ttnn.add(
            ttnn.mul(
                x_norm,
                scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            shift,
            # L1: x_mod is the in0 of _swiglu_ffn's gate/up pair (1536x4608, 120 calls/frame — the
            # largest DRAM-in0 matmul bucket left).  Placement only, maxabsdiff==0.
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # FFN + gated residual
        ffn_out = self._swiglu_ffn(x_mod, layer_idx)
        gated = ttnn.mul(gate, ffn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.add(x, gated, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    def _final_layer(
        self, x: ttnn.Tensor, sc: ttnn.Tensor, mod: Optional["ModSchedule"] = None, step_idx: int = 0
    ) -> ttnn.Tensor:
        """FinalLayer: adaLN (shift/scale, no gate) + linear → latent_size.

        ``sc`` = silu(conditioning), shared with the HeadLayers (see _head_layer)."""
        w = self.w
        if mod is not None:
            modulation = mod.final_mod
            r0, r1 = mod.rows(step_idx)
        else:
            modulation = ttnn.linear(
                sc,
                w.final_adaLN_w,
                bias=w.adaLN_bias2,
                compute_kernel_config=_COMPUTE_KERNEL_FP32,
                program_config=_DIFF_N3072_B2 if sc.shape[0] == 2 else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            r0, r1 = 0, modulation.shape[0]
        hidden_size = w.hidden_size
        shift = ttnn.slice(modulation, [r0, 0, 0, 0], [r1, 1, 1, hidden_size])
        scale = ttnn.slice(modulation, [r0, 0, 0, hidden_size], [r1, 1, 1, 2 * hidden_size])

        # RMSNorm without learnable weight (elementwise_affine=False in reference)
        x_norm = ttnn.rms_norm(
            x,
            epsilon=w.norm_eps,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # +1 folded into the modulation bias — see _head_layer.
        x_mod = ttnn.add(
            ttnn.mul(
                x_norm,
                scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            shift,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.linear(
            x_mod,
            w.final_linear_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            program_config=_DIFF_N64_B2 if x_mod.shape[0] == 2 else None,
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
            program_config=_COND_PROJ_B2 if condition.shape[0] == 2 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_pre_cond(
        self,
        noisy_images: ttnn.Tensor,
        timesteps: ttnn.Tensor,
        cond_proj: ttnn.Tensor,
        t_emb: ttnn.Tensor = None,
        mod: Optional["ModSchedule"] = None,
        step_idx: int = 0,
    ) -> ttnn.Tensor:
        """Head forward given the ALREADY-projected condition (cond_proj = project_condition(cond)).

        Args:
            noisy_images: [B, 1, 1, latent_size] bfloat16 TILE
            timesteps:    [B, 1, 1, 1] bfloat16 scalar per batch (ignored when ``t_emb`` set)
            cond_proj:    [B, 1, 1, hidden_size]  = project_condition(condition)
            t_emb:        optional precomputed ``embed_timestep(timesteps)``; when provided the
                          embedder is skipped (byte-identical if ``t_emb`` came from the same op)
        """
        w = self.w

        # Project noisy latent to hidden_size
        x = ttnn.linear(
            noisy_images,
            w.noisy_images_proj_w,
            compute_kernel_config=_COMPUTE_KERNEL_FP32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if mod is None:
            # Timestep embedding (or reuse a schedule-constant precompute)
            if t_emb is None:
                t_emb = self._timestep_embedder(timesteps)  # [B, 1, 1, hidden]

            # Combine: c = cond_proj + t_emb
            c = ttnn.add(cond_proj, t_emb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # silu(c) is the adaLN input for every HeadLayer + FinalLayer — compute it ONCE per step
            # and share (byte-identical to the per-layer silu, saves 4 redundant silus/step).
            sc = ttnn.silu(c, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Every step's adaLN modulation was batched into one matmul per layer before the loop
            # (see precompute_modulations); this step only reads row ``step_idx``.
            sc = None

        # HeadLayers
        num_layers = len(w.layer_adaLN_w)
        for i in range(num_layers):
            x = self._head_layer(x, sc, i, mod=mod, step_idx=step_idx)

        # FinalLayer
        x = self._final_layer(x, sc, mod=mod, step_idx=step_idx)
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
