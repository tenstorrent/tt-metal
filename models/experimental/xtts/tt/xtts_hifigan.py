# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 HiFi-GAN generator (``waveform_decoder``).

Mirrors ``reference/xtts_hifigan.py`` exactly:

    o = conv_pre(x) + cond_layer(g)
    for each of 4 upsample stages i:
        o = leaky_relu(o, 0.1); o = ups[i](o); o = o + conds[i](g)
        o = mean_j resblocks[i*3 + j](o)         # multi-receptive-field fusion
    o = leaky_relu(o, 0.01); o = conv_post(o); o = tanh(o)

Everything stays channels-last ``[N, L, C]`` ROW_MAJOR — the conv primitives and
all needed eltwise ops (leaky_relu, broadcast add, scalar mul, tanh) run in that
layout, so no per-op relayout is required. Weights come from the folded (weight-
norm removed) reference ``state_dict``.
"""

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_hifigan import (
    LRELU_SLOPE,
    RESBLOCK_DILATION_SIZES,
    RESBLOCK_KERNEL_SIZES,
    UPSAMPLE_RATES,
    get_padding,
)
from models.experimental.xtts.tt.xtts_conv import (
    TtConv1d,
    TtConvTranspose1d,
    height_shard_l1,
    sharded_chain_fits_l1,
)

FINAL_LRELU_SLOPE = 0.01  # coqui's pre-conv_post activation uses F.leaky_relu default

# Keep each resblock's residual chain L1-sharded (collapse the per-conv Interleaved<->Sharded
# round-trips). Global off-switch for A/B and trace bring-up.
_SHARD_RESBLOCKS = True

# Conv math fidelity. The bf16 stages run HiFi2 (2x math throughput vs HiFi4): validated on REAL
# GPT latents (a full sampled utterance) to be perceptually identical to HiFi4 — spectrogram-mag
# PCC 0.9995 vs the torch reference (same as HiFi4), worst-window 0.9889, and Whisper CER 0.000
# (identical transcription); HiFi2-vs-HiFi4 directly is spec-PCC 0.9998. An earlier caution against
# HiFi2 came from random N(0, 2.3) latents (a pessimistic proxy — the bf16 stages are far more
# accurate on real, structured latents), which don't reflect inference. The fp32 stage-0 conv stays
# HiFi4 (widest channels, most accumulation depth). Kept split (bf16/fp32) to tune separately.
_CONV_FIDELITY_BF16 = ttnn.MathFidelity.HiFi2
_CONV_FIDELITY_FP32 = ttnn.MathFidelity.HiFi4

# Full double-buffering (activations + weights) for the INTERLEAVED convs — the ones NOT kept in an
# L1-resident resblock chain (stage-0 resblocks, conv_pre/post, ups, conds). A per-stage config sweep
# measured -27% device time on stage 0 from this alone, bit-exact (PCC 1.0). The interleaved path has
# L1 room for the extra circular buffers; the sharded chains do NOT (they're deliberately L1-tight, so
# _shard_plan drops the double buffer there — enabling it would clash), so this is scoped to non-sharded
# convs only. Bit-exact: double-buffering changes how the conv streams data, not the math.
_INTERLEAVED_CONV_DB = {"enable_act_double_buffer": True, "enable_weights_double_buffer": True}

TILE = 32

# Best (grid_x, grid_y, per_core_N, out_subblock_w, fp32_dest_acc, fidelity) per speaker-conditioning
# output width N, from test_hifi_decoder_matmul_sweep.py (Blackhole, grid 11x10; all M=32, K=512),
# which re-tuned these across the full DRAM / L1-interleaved / L1-sharded cross-product for in0, in1
# and out. The dominant levers are HiFi2 + the tuned grid/out_subblock geometry (~2.2x vs the conv1d
# auto matmul the profiler flagged SLOW), then reading in0 (g) from L1 rather than DRAM: per shape
# 4.06 / 3.01 / 2.54 / 2.40 / 2.16us vs 4.89 / 3.92 / 3.64 / 3.30 / 2.43us, i.e. 18.2 -> 14.2us over
# the five. g is L1 height-sharded and all five run BEFORE the conv chain (see forward) so that copy
# is freed before the first conv — keeping g in L1 *across* the chain is what clashed the interleaved
# resblock convs' circular buffers on long sequences. The outputs stay in DRAM: the sweep's further
# ~2.6us from an L1 block-sharded out is given straight back by the ShardedToInterleaved its
# consumers (a broadcast add, a conv cond_bias) would then need.
# PCC held >=0.9999 per shape in the sweep (full-decoder PCC re-validated in test_hifi_decoder).
_COND_MM_CFG = {
    512: (8, 1, 2, 2, False, "HiFi2"),  # cond_layer
    256: (8, 1, 1, 1, False, "HiFi2"),  # conds[0]
    128: (4, 1, 1, 1, False, "HiFi2"),  # conds[1]
    64: (2, 1, 1, 1, False, "HiFi2"),  # conds[2]
    32: (1, 1, 1, 1, False, "HiFi2"),  # conds[3]
}


class TtCondProj(LightweightModule):
    """A 1x1 speaker-conditioning projection (``cond_layer`` / ``conds[i]``) as an explicit,
    per-shape-tuned ``ttnn.linear`` (``g @ Wᵀ + b``) instead of the ``ttnn.conv1d`` dispatch.

    A 1x1 conv over the length-1 speaker embedding IS this matmul (``g[1,512] @ [512,N]``); conv1d
    ran it on an auto config the profiler flagged SLOW. The tuned program-config + HiFi2 + an L1
    in0 (from test_hifi_decoder_matmul_sweep.py) is ~2.9x faster on these shapes at PCC >=0.9999.
    Pure device matmul with a fused bias epilogue — trace-safe (no host transfer)."""

    def __init__(self, device, weight, bias):
        super().__init__()
        self.device = device
        out_ch, in_ch, k = weight.shape
        assert k == 1, f"cond proj expects a 1x1 conv weight, got k={k}"
        assert out_ch in _COND_MM_CFG, f"no tuned cond-matmul config for N={out_ch}"
        assert in_ch % TILE == 0, f"K={in_ch} must be tile-aligned"
        self.n = out_ch
        self.k = in_ch
        # conv weight [out, in, 1] -> matmul in1 [in, out] = [K, N] (applied as g @ Wᵀ), bf16 in DRAM.
        w = weight.squeeze(-1).transpose(0, 1).contiguous()  # [in, out]
        self.tt_weight = ttnn.from_torch(
            w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.tt_bias = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(
                bias.reshape(1, -1).float(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        gx, gy, per_core_N, osw, fp32_acc, fid = _COND_MM_CFG[out_ch]
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=in_ch // TILE,  # Kt (single K-block)
            out_subblock_h=1,
            out_subblock_w=osw,
            out_block_h=1,
            out_block_w=per_core_N,
            per_core_M=1,  # M = 32 (g tile-padded from length 1) -> Mt = 1
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,  # required for a sharded in0; Mt is 1 either way for this rank-2 g
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=True,
        )

    def forward(self, g_mm):
        # g_mm: [1, K] TILE, L1 height-sharded by the caller and shared across all five projections
        # (one [TILE, K] shard — every shape has Mt = 1). Returns [1, N] TILE (fp32) in DRAM; see
        # _COND_MM_CFG for why the output does not follow in0 into L1. The caller reshapes it for the
        # downstream add / cond-bias fold.
        return ttnn.linear(
            g_mm,
            self.tt_weight,
            bias=self.tt_bias,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
        )


def _shard_plan(stage_i, kernel_size):
    """(sharded, act_double_buffer) for a resblock at upsample ``stage_i`` with ``kernel_size``.

    Tuned on Blackhole for the profiled decode length (latent_len 32): keeping the residual
    chain resident in L1 must leave room for the conv's circular buffers, which grow with
    channel width (early stages) and kernel size (halo). Only (stage 0, k7/k11) — the widest
    channels (256) with the largest halos — actually clash L1; every other block was verified to
    fit (k11 shards from stage 1 on, ch<=128). Enabling all three blocks of a stage is what lets
    ``_mrf`` keep the whole fusion L1-resident; stage 0 stays partly interleaved (k3 only), so its
    MRF still sums in DRAM. The double buffer is dropped on the sharding-capable k7/k11 blocks to
    fit their circular buffers. Anything not enabled here keeps the (correct, slower) interleaved
    round-trip, so this is safe to widen further once a shape is verified to fit."""
    if not _SHARD_RESBLOCKS:
        return False, True
    if kernel_size <= 3:
        return True, True
    return stage_i >= 1, False  # k7 and k11 shard from stage 1 on (ch<=128 fits); stage-0 k7/k11 clash


class TtResBlock1(LightweightModule):
    """HiFi-GAN ResBlock "1": 3 (dilated conv -> plain conv) residual pairs."""

    def __init__(
        self,
        device,
        state_dict,
        prefix,
        kernel_size,
        dilation,
        activations_dtype=ttnn.float32,
        sharded=False,
        act_double_buffer=True,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        conv_config_overrides=None,
    ):
        super().__init__()
        self.device = device
        # ``sharded``: this block is *capable* of keeping its whole residual chain L1-sharded —
        # shard the input once on entry, run all 6 convs + eltwise in L1 (no per-conv
        # Interleaved<->Sharded), gather once on exit. Every op is the same [1, L, C] shape, so
        # the convs share one shard spec. Capability (from _shard_plan) is a channel-width x
        # kernel property; whether it actually fits L1 also depends on sequence length, so the
        # final decision is made per-forward via sharded_chain_fits_l1 (+ a clash fallback).
        # ``_blocked_lengths`` memoizes any length that clashed so it is not retried.
        self.sharded = sharded
        self._blocked_lengths = set()
        # act_double_buffer is only meaningful for the sharding-capable convs (dropped on the
        # tighter k7/k11 blocks to fit their circular buffers); leave it at the ttnn default
        # (None) otherwise so non-sharded blocks are untouched.
        adb = act_double_buffer if sharded else None
        # The leaky_relu that sits between convs1 and convs2 is fused onto the
        # convs1 output (post-bias) — convs1's output feeds only that activation, so
        # ``leaky_relu(convs1(a))`` folds into the conv exactly. The pre-activation
        # before convs1 cannot fuse (its input is reused raw by the residual add).
        mid_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
        self.convs1 = [
            TtConv1d(
                device,
                state_dict[f"{prefix}convs1.{j}.weight"],
                state_dict[f"{prefix}convs1.{j}.bias"],
                padding=get_padding(kernel_size, d),
                dilation=d,
                activation=mid_act,
                activations_dtype=activations_dtype,
                act_double_buffer=adb,
                math_fidelity=math_fidelity,
                conv_config_overrides=conv_config_overrides,
            )
            for j, d in enumerate(dilation)
        ]
        self.convs2 = [
            TtConv1d(
                device,
                state_dict[f"{prefix}convs2.{j}.weight"],
                state_dict[f"{prefix}convs2.{j}.bias"],
                padding=get_padding(kernel_size, 1),
                dilation=1,
                activations_dtype=activations_dtype,
                act_double_buffer=adb,
                math_fidelity=math_fidelity,
                conv_config_overrides=conv_config_overrides,
            )
            for j in range(len(dilation))
        ]

    def will_shard(self, length, channels):
        """Whether forward WILL take the L1-sharded path for this shape (same gate as ``forward``).
        Lets the MRF caller keep the whole fusion L1-resident only when all its resblocks shard."""
        return (
            self.sharded
            and length not in self._blocked_lengths
            and sharded_chain_fits_l1(self.device, length, channels)
        )

    def forward(self, x):
        # Shard only when this block is capable AND the activation at *this* sequence length
        # fits L1 (short decodes shard, long ones fall back). The try/except is a safety net:
        # a circular-buffer clash is thrown at program-compile time (before enqueue), so the
        # device is unharmed — we memoize the length and use the interleaved path. This keeps
        # the demo (long sequences) working where the static length gate is too optimistic.
        length = x.shape[1]
        if (
            self.sharded
            and length not in self._blocked_lengths
            and sharded_chain_fits_l1(self.device, length, x.shape[2])
        ):
            try:
                return self._forward_sharded(x)
            except RuntimeError as e:
                if "circular buffer" not in str(e).lower() and "clash" not in str(e).lower():
                    raise
                self._blocked_lengths.add(length)
        return self._forward_interleaved(x)

    def _forward_interleaved(self, x):
        # Free each conv/activation temporary as soon as it is consumed. The block's
        # input ``x`` is preserved on the first iteration (the caller reuses it for the
        # other MRF resblocks); later residuals are internal and freed.
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            a = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE)
            b = c1(a)  # leaky_relu(0.1) is fused onto this conv's output
            ttnn.deallocate(a)
            d = c2(b)
            ttnn.deallocate(b)
            nxt = ttnn.add(d, x)
            ttnn.deallocate(d)
            if idx > 0:
                ttnn.deallocate(x)
            x = nxt
        return x

    def _forward_sharded(self, x, return_sharded=False):
        # ``return_sharded``: skip the exit gather and hand back the L1-sharded result, so the MRF
        # caller can sum the stage's resblocks in L1 (cheap adds) and gather once instead of per
        # block. The caller then owns the sharded tensor (must free it). On an L1 clash the partials
        # are freed and the exception propagates, so the caller can fall back to the interleaved MRF.
        # ``xs`` is our own L1-sharded copy of the block input; the caller's ``x`` is left
        # untouched (reused by the other MRF resblocks). Every intermediate stays sharded
        # with the same spec (same-shape convs), so leaky_relu / residual add run in L1 and
        # no Interleaved<->Sharded reshard happens between ops — only the entry shard and the
        # exit gather. Matches the interleaved path bit-for-bit (verified PCC ~1.0). On an L1
        # clash the partial temporaries are freed so the caller can retry interleaved cleanly.
        _, length, channels = x.shape
        a = b = d = nxt = None
        xs = height_shard_l1(self.device, x, channels)
        try:
            for c1, c2 in zip(self.convs1, self.convs2):
                a = ttnn.leaky_relu(xs, negative_slope=LRELU_SLOPE)
                b = c1(a, keep_sharded=True)  # leaky_relu(0.1) fused; L1-sharded in -> L1-sharded out
                ttnn.deallocate(a)
                a = None
                d = c2(b, keep_sharded=True)
                ttnn.deallocate(b)
                b = None
                nxt = ttnn.add(d, xs)
                ttnn.deallocate(d)
                d = None
                ttnn.deallocate(xs)  # always our internal copy
                xs = nxt
                nxt = None
        except Exception:
            for t in (xs, a, b, d, nxt):
                if isinstance(t, ttnn.Tensor) and t.is_allocated():
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            raise
        if return_sharded:
            return xs  # caller sums in L1 and gathers once; caller owns xs
        out = ttnn.to_memory_config(xs, ttnn.DRAM_MEMORY_CONFIG)  # gather once
        ttnn.deallocate(xs)
        return out


class TtHifiganGenerator(LightweightModule):
    """XTTS-v2 ``waveform_decoder``: GPT latent ``x`` (+ speaker embedding ``g``) -> waveform.

    Inputs are channels-last ROW_MAJOR: ``x`` is ``[N, T, 1024]``, ``g`` is
    ``[N, 1, 512]``. Output is ``[N, T*256, 1]``.
    """

    def __init__(self, device, state_dict, bf16_stages=None):
        super().__init__()
        self.device = device
        self.num_kernels = len(RESBLOCK_KERNEL_SIZES)  # 3
        self.num_upsamples = len(UPSAMPLE_RATES)  # 4
        self.inv_num_kernels = 1.0 / self.num_kernels

        # Mixed precision: stages listed in ``bf16_stages`` run their ups/conds/resblocks
        # in bf16, the rest in fp32. The late stages carry the largest activations (most
        # DRAM eltwise traffic) and the least remaining accumulation depth, so bf16 there
        # buys the most device time for the least PCC drift. Default = the last three
        # stages: on *real* GPT latents (std ~2.3, large outliers) this holds PCC ~0.995
        # up to ~100 mel frames (-24.7% device time vs stage-3-only), where random
        # latents ~N(0, 0.5) misleadingly suggested it fails — see tests. Stage 0 stays
        # fp32 (its wide 256-ch conv is where bf16 costs the most PCC for the least time).
        #
        # Widening this to all four stages was tried and REJECTED on listening: it measured
        # -10.9% decoder device time and looked fine on aggregate metrics (spectrogram-mag PCC
        # 0.99707 vs 0.99747, and it even improved end-to-end spec PCC), but it added an audible
        # robotic/metallic edge. Reverting only conv_pre/cond_layer/conv_post to fp32 (keeping
        # stage 0 bf16) did NOT clear it either, so stage 0 is implicated, not just the output
        # convs. Corroborating spectral evidence: broadband spectral flatness rose 0.30739 ->
        # 0.30958 and >6kHz energy share 0.19720 -> 0.19873. Aggregate PCC — waveform OR
        # spectrogram — did not predict this; only listening did.
        if bf16_stages is None:
            bf16_stages = {i for i in range(1, self.num_upsamples)}

        def act_dtype(i):
            return ttnn.bfloat16 if i in bf16_stages else ttnn.float32

        def conv_fid(i):
            # fidelity follows the stage's activation dtype (see _CONV_FIDELITY_* above)
            return _CONV_FIDELITY_BF16 if i in bf16_stages else _CONV_FIDELITY_FP32

        self.conv_pre = TtConv1d(
            device,
            state_dict["conv_pre.weight"],
            state_dict["conv_pre.bias"],
            padding=3,
            math_fidelity=_CONV_FIDELITY_FP32,
            conv_config_overrides=_INTERLEAVED_CONV_DB,
        )
        # Speaker-conditioning projections (1x1 convs over the length-1 embedding g) run as
        # per-shape-tuned explicit matmuls — see TtCondProj / _COND_MM_CFG.
        self.cond_layer = TtCondProj(device, state_dict["cond_layer.weight"], state_dict["cond_layer.bias"])

        # ups[i] consumes the *previous* stage's MRF mean (stages >=1) via leaky_relu; fold that
        # mean's 1/num_kernels scale into ups[i>=1]'s weights so the per-stage ttnn.mul is dropped
        # (see forward). ups[0] consumes conv_pre+cond (not a mean), so it is left unscaled.
        self.ups = [
            TtConvTranspose1d(
                device,
                state_dict[f"ups.{i}.weight"],
                state_dict[f"ups.{i}.bias"],
                stride=UPSAMPLE_RATES[i],
                activations_dtype=act_dtype(i),
                math_fidelity=conv_fid(i),
                weight_scale=self.inv_num_kernels if i >= 1 else 1.0,
                conv_config_overrides=_INTERLEAVED_CONV_DB,
            )
            for i in range(self.num_upsamples)
        ]
        # conds[i](g) is a length-1 per-channel constant folded into ups[i]'s bias (see forward);
        # its fp32 output feeds that fold directly. Tuned explicit matmul (TtCondProj), like cond_layer.
        self.conds = [
            TtCondProj(device, state_dict[f"conds.{i}.weight"], state_dict[f"conds.{i}.bias"])
            for i in range(self.num_upsamples)
        ]
        # in0 placement shared by all five projections: a single [TILE, K] L1 shard on one core
        # (every shape has Mt = 1, so the height-shard grid is 1x1). Built once here; the per-forward
        # copy of g is made and freed inside forward.
        self._g_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [TILE, self.cond_layer.k],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        self.resblocks = []
        for i in range(self.num_upsamples):
            for j, (k, d) in enumerate(zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES)):
                sharded, act_double_buffer = _shard_plan(i, k)
                # Non-sharded resblocks run interleaved with L1 room -> full double-buffering (the
                # measured stage-0 win). Sharded-chain blocks keep their L1-tight tuning (act_double_buffer
                # from _shard_plan; no weights-db, which would clash the resident chain).
                db_ov = None if sharded else _INTERLEAVED_CONV_DB
                self.resblocks.append(
                    TtResBlock1(
                        device,
                        state_dict,
                        f"resblocks.{i * self.num_kernels + j}.",
                        k,
                        d,
                        activations_dtype=act_dtype(i),
                        sharded=sharded,
                        act_double_buffer=act_double_buffer,
                        math_fidelity=conv_fid(i),
                        conv_config_overrides=db_ov,
                    )
                )

        # conv_post has no bias in XTTS. It consumes the final stage's MRF mean via leaky_relu, so
        # (as with ups[i>=1]) the mean's 1/num_kernels scale folds into its weights — no bias to keep
        # unscaled here, so the fold is exact and clean.
        self.conv_post = TtConv1d(
            device,
            state_dict["conv_post.weight"],
            None,
            padding=3,
            math_fidelity=_CONV_FIDELITY_FP32,
            weight_scale=self.inv_num_kernels,
            conv_config_overrides=_INTERLEAVED_CONV_DB,
        )

    def _mrf(self, i, o):
        """Multi-receptive-field fusion for upsample stage ``i``: sum the stage's ``num_kernels``
        resblocks over ``o`` (the mean's 1/num_kernels is folded into downstream weights, so this
        is a plain sum). Frees ``o``; returns the interleaved-DRAM stage output.

        Fast path keeps the whole fusion L1-resident — each resblock returns its output sharded, the
        sum runs in L1, and the result is gathered ONCE — replacing the per-block exit gather + the
        DRAM-priced sum adds. Taken only when every resblock in the stage will shard this shape; an
        L1 clash mid-sum falls back to the per-block interleaved path (and memoizes the length)."""
        nk = self.num_kernels
        rbs = self.resblocks[i * nk : (i + 1) * nk]
        length, channels = o.shape[1], o.shape[2]
        if all(rb.will_shard(length, channels) for rb in rbs):
            try:
                out = self._mrf_sharded(rbs, o)
                ttnn.deallocate(o)
                return out
            except RuntimeError as e:
                if "circular buffer" not in str(e).lower() and "clash" not in str(e).lower():
                    raise
                for rb in rbs:
                    rb._blocked_lengths.add(length)  # don't retry sharding this length
        out = self._mrf_interleaved(rbs, o)
        ttnn.deallocate(o)
        return out

    def _mrf_sharded(self, rbs, o):
        """L1-resident MRF sum: each resblock runs in L1 and returns sharded; sum incrementally in
        L1 (holding at most sum + one block output) and gather once. Does not free ``o``. On a clash
        the partial sum is freed and the exception propagates for the caller's interleaved fallback."""
        z_sum = None
        for rb in rbs:
            try:
                res = rb._forward_sharded(o, return_sharded=True)
            except Exception:
                if isinstance(z_sum, ttnn.Tensor) and z_sum.is_allocated():
                    ttnn.deallocate(z_sum)
                raise
            if z_sum is None:
                z_sum = res
            else:
                z_new = ttnn.add(z_sum, res)  # sharded + sharded (same spec) -> L1 add
                ttnn.deallocate(z_sum)
                ttnn.deallocate(res)
                z_sum = z_new
        out = ttnn.to_memory_config(z_sum, ttnn.DRAM_MEMORY_CONFIG)  # gather once for the whole stage
        ttnn.deallocate(z_sum)
        return out

    def _mrf_interleaved(self, rbs, o):
        """Per-block interleaved MRF sum (each resblock gathers to DRAM on exit). Does not free ``o``."""
        z_sum = None
        for rb in rbs:
            res = rb(o)  # does not free o
            if z_sum is None:
                z_sum = res
            else:
                z_new = ttnn.add(z_sum, res)
                ttnn.deallocate(z_sum)
                ttnn.deallocate(res)
                z_sum = z_new
        return z_sum

    def forward(self, x, g):
        # Deep conv chain — the vocoder's memory-dominant path, whose activation footprint grows
        # with output length. Each temporary is freed the moment the next op consumes it.
        # cond_layer/conds are 1x1 projections of the length-1 speaker embedding g, run as tuned
        # matmuls (TtCondProj). Reshape g to [1, 512] and tile it ONCE, shared by all five, and read
        # it from L1 (18.2 -> 14.2us over the five; see _COND_MM_CFG). All five run HERE, ahead of the
        # conv chain, so the L1 copy of g is freed before the first conv rather than competing with
        # the interleaved resblock convs' circular buffers for the whole chain. conds[i] depends only
        # on g, so hoisting it out of the loop is value-identical. Trace-safe (no host transfer).
        g_mm = ttnn.to_layout(ttnn.reshape(g, [1, g.shape[-1]]), ttnn.TILE_LAYOUT)
        g_l1 = ttnn.to_memory_config(g_mm, self._g_mem_config)
        ttnn.deallocate(g_mm)
        cond_global = ttnn.reshape(self.cond_layer(g_l1), [1, 1, self.cond_layer.n])  # [1,1,512], broadcasts over T
        # conds[i](g) is a length-1 per-channel constant, folded into ups[i]'s bias epilogue below.
        cond_biases = [ttnn.reshape(c(g_l1), [1, 1, 1, c.n]) for c in self.conds]
        ttnn.deallocate(g_l1)
        pre = self.conv_pre(x)
        ttnn.deallocate(x)  # upsampler output, not reused after conv_pre
        o = ttnn.add(pre, cond_global)
        ttnn.deallocate(pre)
        ttnn.deallocate(cond_global)

        for i in range(self.num_upsamples):
            a = ttnn.leaky_relu(o, negative_slope=LRELU_SLOPE)
            ttnn.deallocate(o)
            # ``conds[i](g)`` is a length-1, per-channel constant, so ``ups[i](a) + conds[i](g)`` is
            # just a per-channel bias add — fold it into the ups conv's fused bias epilogue.
            o = self.ups[i](a, cond_bias=cond_biases[i])
            ttnn.deallocate(a)
            ttnn.deallocate(cond_biases[i])
            o = self._mrf(i, o)  # sum the 3 resblocks (mean folded into weights); frees o
        a = ttnn.leaky_relu(o, negative_slope=FINAL_LRELU_SLOPE)
        ttnn.deallocate(o)
        p = self.conv_post(a)
        ttnn.deallocate(a)
        out = ttnn.tanh(p)
        ttnn.deallocate(p)
        return out
