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
    block_chain_fits_l1,
    block_shard_l1,
    height_shard_l1,
    sharded_chain_fits_l1,
)

FINAL_LRELU_SLOPE = 0.01  # coqui's pre-conv_post activation uses F.leaky_relu default

# Keep each resblock's residual chain L1-sharded (collapse the per-conv Interleaved<->Sharded
# round-trips). Global off-switch for A/B and trace bring-up.
_SHARD_RESBLOCKS = True

# Upsample stages whose resident chain uses a BLOCK rather than HEIGHT shard (see block_shard_grid).
# Stage 0 only -- it is the one stage wide enough (256 ch = 8 channel tiles) for block to score MORE
# cores than height (80 vs 35; height is capped by tile alignment at a 32-row shard), and its smaller
# per-core tile (16 vs 32 KB) is what lets the k7/k11 convs build resident AT ALL. Stages 1-3 are
# narrower (128/64/32 ch) so block would cap at 40/20/10 cores against height's 93-110.
#
# Worth -285us of device time (2512 -> 2227us, -11.3%): stage-0 convs 489 -> 433us and the whole
# stage-0 MRF goes L1-resident, deleting 12 I2S, 12 S2I, 12 Move and 6 DRAM residual adds. Needs the
# double buffering that _shard_plan/db_ov grant block stages -- without it k7/k11 run 33/48us instead
# of 21/29us and the win collapses to -98us.
#
# NOT bit-exact: PCC moves 0.993110 -> 0.993252 (len 32) because the MRF sum accumulates in a
# different order. Given this model's history of audible artifacts that aggregate PCC did not predict
# (see the bf16_stages note above), listen before shipping.
#
# Empty set reverts every stage to height sharding.
_BLOCK_SHARD_STAGES = {0}

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

# Shard layout forced on an upsample conv, overriding ttnn's auto pick. ttnn's auto-sharder is an
# L1-footprint minimiser applied one op at a time, and on ups[0] it gets the geometry wrong: that
# conv's input is 139 x 512 -- short and wide -- so it cuts channels only (WIDTH, 66 cores) and runs
# at 15.3% of peak FLOPs / 43.3us, while ups[1] does 1.8x the FLOPs in 12.7us at 47.1% on a BLOCK
# placement. Forcing BLOCK here recovers most of that gap. The other three ups are left on auto:
# they are long-and-narrow, where the auto pick is already the good one.
#
# Do NOT force HEIGHT on ups[0]: HEIGHT_SHARDED fails conv1d's DRAM slicer on exactly this
# wide-channel/short-length shape (see TtConv1d's "No forced shard_layout" note in xtts_conv.py).
_UPS_SHARD_OVERRIDE = {0: ttnn.TensorMemoryLayout.BLOCK_SHARDED}


def _ups_conv_overrides(i):
    ov = dict(_INTERLEAVED_CONV_DB)
    if i in _UPS_SHARD_OVERRIDE:
        ov["shard_layout"] = _UPS_SHARD_OVERRIDE[i]
    return ov


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


# Lowest upsample stage that emits its residual pre-activation as a second fused add instead of a
# standalone unary (see TtResBlock1._forward_sharded). Stage 0's shard geometry inverts the win —
# A/B'd end-to-end at 2520us with stage 0 included vs 2512us without — so it stays on the unary.
# Set to 0 to re-run that A/B, or to num_upsamples to disable the trick entirely.
_FUSED_PRE_ACT_MIN_STAGE = 1

# Hand each stage's MRF output to the next conv (ups[i+1], or conv_post for the last stage) still
# L1-sharded, instead of gathering it to DRAM for the conv to immediately scatter back.
#
# The MRF already finishes with its sum L1-resident, and ttnn.conv1d takes its L1 path whenever the
# input it is handed is already sharded -- so the gather and the scatter are a matched pair of pure
# waste. Measured around ups[2]: exit gather 8.5us + conv entry scatter 7.1us + conv exit gather
# 8.4us + next-MRF scatter 6.7us = 30.7us of movement wrapped around a 9.0us conv. Keeping the chain
# sharded collapses those into ~0.4us Reshards, and (because the conv now returns a sharded output)
# it is also what lets the stride<=2 sub-pixel shuffle reshape in L1 -- see TtConvTranspose1d.
#
# Best-effort: _mrf_interleaved still returns DRAM, and conv1d consumes either, so nothing depends
# on the hand-off succeeding. Holding the stage activation resident across the ups conv does raise
# peak L1, so forward() carries a gather-and-retry for a circular-buffer clash (same pattern as
# TtResBlock1.forward). Set False to revert every stage to the DRAM hand-off.
_SHARDED_STAGE_HANDOFF = True


def _is_l1_clash(exc):
    msg = str(exc).lower()
    return "circular buffer" in msg or "clash" in msg


def _fused_pre_act_plan(stage_i, sharded):
    """Whether resblocks at upsample ``stage_i`` should use the second-add pre-activation.
    Only meaningful on the L1-sharded path — an interleaved block's adds are DRAM-priced
    (~10us), so a second one there would cost far more than the unary it replaces."""
    return sharded and stage_i >= _FUSED_PRE_ACT_MIN_STAGE


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
    if stage_i in _BLOCK_SHARD_STAGES:
        # The BLOCK placement fits every kernel at this stage (see block_shard_grid), including the
        # k7/k11 blocks that clash L1 height-sharded, so the whole stage can go resident -- and at
        # half the per-core footprint of the height placement (16 vs 32 KB), which leaves room to keep
        # full double buffering. That is not optional: without it the k7/k11 convs measured 33/48us
        # against 21/29us with it.
        return True, True
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
        fused_pre_act=False,
        block_shard=False,
    ):
        super().__init__()
        self.device = device
        # ``block_shard``: this stage's resident chain is BLOCK- rather than HEIGHT-sharded. Both are
        # "one spec for the whole chain"; they differ only in how the activation is cut. See
        # block_shard_grid in xtts_conv.py for why stage 0 wants block and the rest want height.
        self.block_shard = block_shard
        # ``fused_pre_act`` (sharded path only): produce the next residual iteration's
        # ``leaky_relu(x)`` as a SECOND fused add over the same operands rather than as a
        # standalone unary — see ``_forward_sharded``. Tuned per stage by _fused_pre_act_plan.
        self.fused_pre_act = fused_pre_act
        self._pre_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
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

    def shard(self, x, channels):
        """Bring ``x`` to this block's L1 chain placement (block or height)."""
        return (block_shard_l1 if self.block_shard else height_shard_l1)(self.device, x, channels)

    def chain_fits_l1(self, length, channels):
        return (block_chain_fits_l1 if self.block_shard else sharded_chain_fits_l1)(self.device, length, channels)

    def will_shard(self, length, channels):
        """Whether forward WILL take the L1-sharded path for this shape (same gate as ``forward``).
        Lets the MRF caller keep the whole fusion L1-resident only when all its resblocks shard."""
        return self.sharded and length not in self._blocked_lengths and self.chain_fits_l1(length, channels)

    def forward(self, x, pre_act=None):
        # Shard only when this block is capable AND the activation at *this* sequence length
        # fits L1 (short decodes shard, long ones fall back). The try/except is a safety net:
        # a circular-buffer clash is thrown at program-compile time (before enqueue), so the
        # device is unharmed — we memoize the length and use the interleaved path. This keeps
        # the demo (long sequences) working where the static length gate is too optimistic.
        length = x.shape[1]
        if self.sharded and length not in self._blocked_lengths and self.chain_fits_l1(length, x.shape[2]):
            try:
                return self._forward_sharded(x, pre_act=pre_act)
            except RuntimeError as e:
                if "circular buffer" not in str(e).lower() and "clash" not in str(e).lower():
                    raise
                self._blocked_lengths.add(length)
        return self._forward_interleaved(x, pre_act=pre_act)

    # ``pre_act`` (both paths): ``leaky_relu(x)`` precomputed by the MRF caller. Every resblock
    # in a stage starts by activating the SAME stage activation with the same slope, so the caller
    # computes it once and lends it to all of them instead of each recomputing it (saves 2 of the
    # 3 per stage). Borrowed — never freed here; the caller frees it once every sibling is done.

    def _forward_interleaved(self, x, pre_act=None):
        # Free each conv/activation temporary as soon as it is consumed. The block's
        # input ``x`` is preserved on the first iteration (the caller reuses it for the
        # other MRF resblocks); later residuals are internal and freed.
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            if idx == 0 and pre_act is not None:
                b = c1(pre_act)  # borrowed activation; leaky_relu(0.1) fused onto this conv's output
            else:
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

    def _forward_sharded(self, x, return_sharded=False, pre_sharded=False, pre_act=None):
        # ``return_sharded``: skip the exit gather and hand back the L1-sharded result, so the MRF
        # caller can sum the stage's resblocks in L1 (cheap adds) and gather once instead of per
        # block. The caller then owns the sharded tensor (must free it). On an L1 clash the partials
        # are freed and the exception propagates, so the caller can fall back to the interleaved MRF.
        # ``pre_sharded``: ``x`` is already an L1-sharded copy the caller owns (the MRF sums several
        # resblocks over the same activation, so it shards ``o`` once and lends it to each) -- skip
        # the entry reshard and never free ``x`` itself; the caller frees it once every sibling
        # resblock is done with it.
        # ``xs`` is our own L1-sharded copy of the block input; the caller's ``x`` is left
        # untouched (reused by the other MRF resblocks). Every intermediate stays sharded
        # with the same spec (same-shape convs), so leaky_relu / residual add run in L1 and
        # no Interleaved<->Sharded reshard happens between ops — only the entry shard and the
        # exit gather. Matches the interleaved path bit-for-bit (verified PCC ~1.0). On an L1
        # clash the partial temporaries are freed so the caller can retry interleaved cleanly.
        #
        # ``fused_pre_act``: the next iteration needs BOTH the raw residual sum ``x`` (as its own
        # addend) and ``leaky_relu(x)`` (to feed convs1), so the activation cannot simply be fused
        # onto the sum. But a BinaryNg over two tensors with matching shard specs hits a fast path
        # a UnaryDeviceOperation does not: measured on the stage-1..3 shard (1.14M elems, bf16),
        # add = 0.9us, add + fused leaky_relu = 2.8us, standalone leaky_relu = 5.1us. So emitting a
        # SECOND add over the same (d, xs) pair costs 3.7us where add + unary costs 6.0us. On the
        # stage-0 shard (fp32, 256ch over only 35 cores) the numbers invert (3.0 / 4.4 / 2.6), which
        # is why _fused_pre_act_plan enables this from stage 1 on only. Peak L1 rises by one tensor,
        # but only across the two adds -- the convs still see three, exactly as before.
        _, length, channels = x.shape
        b = d = nxt = None
        xs = x if pre_sharded else self.shard(x, channels)
        entry = xs if pre_sharded else None  # borrowed from the caller -- never ours to free
        act, own_act = pre_act, False  # iteration 0's activation, borrowed from the MRF caller
        n_iters = len(self.convs1)
        try:
            for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
                if act is None:
                    act = ttnn.leaky_relu(xs, negative_slope=LRELU_SLOPE)
                    own_act = True
                b = c1(act, keep_sharded=True)  # leaky_relu(0.1) fused; L1-sharded in -> L1-sharded out
                if own_act:
                    ttnn.deallocate(act)
                act, own_act = None, False
                d = c2(b, keep_sharded=True)
                ttnn.deallocate(b)
                b = None
                nxt = ttnn.add(d, xs)
                if self.fused_pre_act and idx + 1 < n_iters:
                    act = ttnn.add(d, xs, activations=[self._pre_act])  # == leaky_relu(nxt)
                    own_act = True
                ttnn.deallocate(d)
                d = None
                if xs is not entry:
                    ttnn.deallocate(xs)  # our own internal copy; the borrowed entry is left alone
                xs = nxt
                nxt = None
        except Exception:
            for t in (b, d, nxt, act if own_act else None):
                if isinstance(t, ttnn.Tensor) and t.is_allocated():
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            if xs is not entry and isinstance(xs, ttnn.Tensor) and xs.is_allocated():
                try:
                    ttnn.deallocate(xs)
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
                conv_config_overrides=_ups_conv_overrides(i),
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
        self._cond = {}  # id(g) -> (g, cond_global, cond_biases) — see _conditioning

        self.resblocks = []
        for i in range(self.num_upsamples):
            for j, (k, d) in enumerate(zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES)):
                sharded, act_double_buffer = _shard_plan(i, k)
                # Non-sharded resblocks run interleaved with L1 room -> full double-buffering (the
                # measured stage-0 win), and so do BLOCK-sharded chains, whose per-core tile is half a
                # height shard's. Only HEIGHT-sharded chains stay L1-tight (act_double_buffer from
                # _shard_plan; no weights-db, which would clash the resident chain).
                db_ov = None if (sharded and i not in _BLOCK_SHARD_STAGES) else _INTERLEAVED_CONV_DB
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
                        fused_pre_act=_fused_pre_act_plan(i, sharded),
                        block_shard=i in _BLOCK_SHARD_STAGES,
                    )
                )

        # conv_post has no bias in XTTS. It consumes the final stage's MRF mean via leaky_relu, so
        # (as with ups[i>=1]) the mean's 1/num_kernels scale folds into its weights — no bias to keep
        # unscaled here, so the fold is exact and clean. The generator's output ``tanh`` is fused onto
        # this conv (its output feeds nothing else), removing the standalone unary — the largest
        # single eltwise op in the profile, 28us on the full-rate waveform.
        self.conv_post = TtConv1d(
            device,
            state_dict["conv_post.weight"],
            None,
            padding=3,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH),
            math_fidelity=_CONV_FIDELITY_FP32,
            weight_scale=self.inv_num_kernels,
            conv_config_overrides=_INTERLEAVED_CONV_DB,
        )
        # The stage pre-activations, fused onto the producing add rather than run as their own op:
        # ``_pre_act`` on the conditioning add (stage 0) and on stages 0-2's MRF sum, ``_final_act``
        # (coqui's default-slope leaky_relu) on stage 3's, whose consumer is conv_post.
        self._pre_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
        self._final_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, FINAL_LRELU_SLOPE)

    def _conditioning(self, g):
        """``(cond_global, cond_biases)`` for speaker embedding ``g``, memoised on ``g``.

        Every product here is a function of the speaker embedding ALONE, but ``g`` is fixed for a
        whole utterance while ``forward`` runs once per decoded chunk — so rebuilding them per call
        is pure waste. Measured on the profiled shape it is 54.7us of a 2227us pass, and 39.2us of
        that is a single ``TilizeWithValPadding`` padding the 512-float ``g`` up to one tile row on
        ONE core. Downstream, ``TtConvTranspose1d._inner_cond`` and ``TtConv1d``'s folded-bias cache
        key off the tensors returned here, so a hit here makes those hit too — together that is the
        whole ~91us conditioning path, and it removes the last host transfer from the fold.

        The cache holds a reference to ``g`` itself, which is what makes the ``is`` check sound:
        while we hold it, that buffer cannot be freed and re-issued to a different tensor at the
        same address. It does assume ``g`` is not mutated in place; no caller does (it comes
        straight out of the speaker encoder and is read-only from here on).

        Entries are **never evicted automatically**, and that is deliberate rather than lazy. These
        tensors are no longer recomputed inside a trace capture, so a captured trace holds recorded
        reads of these exact device addresses. Freeing an entry to make room for the next one lets
        those addresses be reissued, and a still-live trace then replays against whatever landed
        there — measured: correlation with the expected waveform drops to ~0.0, silently. Retaining
        them keeps every captured trace valid for as long as it lives, which is the behaviour
        callers had before this was memoised.

        The cost is that growth is per distinct ``g`` OBJECT, which is NOT the same as per speaker:
        ``inference_fully_traced`` rebuilds ``g`` every call (it is the setup trace's output), so
        re-decoding the same voice still adds an entry each time. Callers that loop must therefore
        call :meth:`release_conditioning` once the traces that used them are released — see there.

        The returned tensors are owned by this module — callers must not deallocate them."""
        hit = self._cond.get(id(g))
        if hit is not None and hit[0] is g:
            return hit[1], hit[2]
        # Reshape g to [1, 512] and tile it ONCE, shared by all five projections, and read it from
        # L1 (18.2 -> 14.2us over the five; see _COND_MM_CFG).
        g_mm = ttnn.to_layout(ttnn.reshape(g, [1, g.shape[-1]]), ttnn.TILE_LAYOUT)
        g_l1 = ttnn.to_memory_config(g_mm, self._g_mem_config)
        ttnn.deallocate(g_mm)
        cond_global = ttnn.reshape(self.cond_layer(g_l1), [1, 1, self.cond_layer.n])  # [1,1,512], broadcasts over T
        # conds[i](g) is a length-1 per-channel constant, folded into ups[i]'s bias epilogue.
        cond_biases = [ttnn.reshape(c(g_l1), [1, 1, 1, c.n]) for c in self.conds]
        ttnn.deallocate(g_l1)
        self._cond[id(g)] = (g, cond_global, cond_biases)
        return cond_global, cond_biases

    def release_conditioning(self):
        """Drop every memoised ``g``-derived product — this module's and the ups convs'.

        Safe ONLY once no captured trace that ran through them is still live: a trace records reads
        of these exact device addresses, so releasing them under a live trace makes it replay
        garbage (see :meth:`_conditioning`). The natural call site is right after the vocoder's
        ``release_trace``. Cheap and idempotent; the next forward rebuilds what it needs."""
        for entry in self._cond.values():
            for t in (entry[1], *entry[2]):
                if t.is_allocated():
                    ttnn.deallocate(t)
        self._cond.clear()
        for u in self.ups:
            u.release_cond_cache()

    def _mrf(self, i, o, post_act):
        """Multi-receptive-field fusion for upsample stage ``i``: sum the stage's ``num_kernels``
        resblocks over ``o`` (the mean's 1/num_kernels is folded into downstream weights, so this
        is a plain sum). Frees ``o``; returns the interleaved-DRAM stage output.

        ``post_act`` is the stage output's consumer-side activation (``leaky_relu``), fused onto the
        FINAL sum add as a BinaryNg post-activation instead of running as its own op — the sum is
        the activation's only consumer, so this is exact (verified bit-exact, DRAM and sharded).

        Fast path keeps the whole fusion L1-resident — each resblock returns its output sharded, the
        sum runs in L1, and the result is gathered ONCE — replacing the per-block exit gather + the
        DRAM-priced sum adds. Taken only when every resblock in the stage will shard this shape; an
        L1 clash mid-sum falls back to the per-block interleaved path (and memoizes the length)."""
        nk = self.num_kernels
        rbs = self.resblocks[i * nk : (i + 1) * nk]
        length, channels = o.shape[1], o.shape[2]
        if all(rb.will_shard(length, channels) for rb in rbs):
            try:
                out = self._mrf_sharded(rbs, o, post_act, keep_sharded=_SHARDED_STAGE_HANDOFF)
                ttnn.deallocate(o)
                return out
            except RuntimeError as e:
                if "circular buffer" not in str(e).lower() and "clash" not in str(e).lower():
                    raise
                for rb in rbs:
                    rb._blocked_lengths.add(length)  # don't retry sharding this length
        if o.memory_config().is_sharded():
            # ups[2]/ups[3] hand us their L1-sharded shuffle output (see TtConvTranspose1d). The
            # sharded MRF above consumes that directly, but the interleaved fallback's convs expect
            # the DRAM form, so gather before falling back.
            gathered = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            o = gathered
        out = self._mrf_interleaved(rbs, o, post_act)
        ttnn.deallocate(o)
        return out

    def _mrf_sharded(self, rbs, o, post_act, keep_sharded=False):
        """L1-resident MRF sum: every resblock in the stage reads the same activation, so shard
        ``o`` ONCE here and lend that single shard to each of them (rather than each resblock
        re-deriving its own copy of the identical shard) -- each still runs its own conv chain in
        L1 and returns sharded; sum incrementally in L1 (holding at most sum + one block output)
        and gather once. Does not free ``o``. On a clash the partial sum and the shared shard are
        freed and the exception propagates for the caller's interleaved fallback."""
        channels = o.shape[2]
        o_shard = rbs[0].shard(o, channels)  # every block in a stage shares one placement
        pre_act = z_sum = None
        try:
            # Every resblock's first op is leaky_relu over this same shard — compute it ONCE in L1
            # and lend it to all of them (see TtResBlock1's ``pre_act``).
            pre_act = ttnn.leaky_relu(o_shard, negative_slope=LRELU_SLOPE)
            for n, rb in enumerate(rbs):
                try:
                    res = rb._forward_sharded(o_shard, return_sharded=True, pre_sharded=True, pre_act=pre_act)
                except Exception:
                    if isinstance(z_sum, ttnn.Tensor) and z_sum.is_allocated():
                        ttnn.deallocate(z_sum)
                    raise
                if z_sum is None:
                    z_sum = res
                else:
                    # sharded + sharded (same spec) -> L1 add; the last one carries post_act
                    z_new = ttnn.add(z_sum, res, activations=[post_act] if n == len(rbs) - 1 else [])
                    ttnn.deallocate(z_sum)
                    ttnn.deallocate(res)
                    z_sum = z_new
        finally:
            ttnn.deallocate(o_shard)
            if isinstance(pre_act, ttnn.Tensor) and pre_act.is_allocated():
                ttnn.deallocate(pre_act)
        if keep_sharded:
            return z_sum  # next conv takes conv1d's L1 path straight off this — see _SHARDED_STAGE_HANDOFF
        out = ttnn.to_memory_config(z_sum, ttnn.DRAM_MEMORY_CONFIG)  # gather once for the whole stage
        ttnn.deallocate(z_sum)
        return out

    def _mrf_interleaved(self, rbs, o, post_act):
        """Per-block interleaved MRF sum (each resblock gathers to DRAM on exit). Does not free ``o``."""
        length, channels = o.shape[1], o.shape[2]
        pre_act = z_sum = None
        for n, rb in enumerate(rbs):
            if rb.will_shard(length, channels):
                res = rb(o)  # re-derives leaky_relu in L1, cheaper than resharding this DRAM copy
            else:
                if pre_act is None:  # shared by every interleaved block in this stage
                    pre_act = ttnn.leaky_relu(o, negative_slope=LRELU_SLOPE)
                res = rb(o, pre_act=pre_act)  # does not free o
            if z_sum is None:
                z_sum = res
            else:
                z_new = ttnn.add(z_sum, res, activations=[post_act] if n == len(rbs) - 1 else [])
                ttnn.deallocate(z_sum)
                ttnn.deallocate(res)
                z_sum = z_new
        if pre_act is not None:
            ttnn.deallocate(pre_act)
        return z_sum

    def forward(self, x, g):
        # Deep conv chain — the vocoder's memory-dominant path, whose activation footprint grows
        # with output length. Each temporary is freed the moment the next op consumes it.
        # cond_layer/conds are 1x1 projections of the length-1 speaker embedding g, run as tuned
        # matmuls (TtCondProj). All five run HERE, ahead of the conv chain, so the L1 copy of g is
        # freed before the first conv rather than competing with the interleaved resblock convs'
        # circular buffers for the whole chain. conds[i] depends only on g, so hoisting it out of the
        # loop is value-identical — and out of the CALL too, hence the memo (see _conditioning).
        # Owned by _conditioning; not freed here. Trace-safe (no host transfer).
        cond_global, cond_biases = self._conditioning(g)
        pre = self.conv_pre(x)
        ttnn.deallocate(x)  # upsampler output, not reused after conv_pre
        # Stage 0's pre-activation is the conditioning add's only consumer, so it rides along as
        # that add's fused post-activation rather than as its own op.
        a = ttnn.add(pre, cond_global, activations=[self._pre_act])
        ttnn.deallocate(pre)

        for i in range(self.num_upsamples):
            # ``conds[i](g)`` is a length-1, per-channel constant, so ``ups[i](a) + conds[i](g)`` is
            # just a per-channel bias add — fold it into the ups conv's fused bias epilogue.
            try:
                o = self.ups[i](a, cond_bias=cond_biases[i])
            except RuntimeError as e:
                # ``a`` may be the previous stage's still-resident L1 sum (_SHARDED_STAGE_HANDOFF).
                # That raises peak L1, and a clash is thrown at program-compile time (before
                # enqueue, so the device is unharmed) — gather and retry the DRAM way.
                if not (_is_l1_clash(e) and a.memory_config().is_sharded()):
                    raise
                a_dram = ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(a)
                a = a_dram
                o = self.ups[i](a, cond_bias=cond_biases[i])
            ttnn.deallocate(a)
            # Sum the 3 resblocks (mean folded into weights) and fuse the NEXT consumer's
            # pre-activation onto the sum's final add; frees o.
            last = i == self.num_upsamples - 1
            a = self._mrf(i, o, self._final_act if last else self._pre_act)
        out = self.conv_post(a)  # tanh is fused onto conv_post's output
        ttnn.deallocate(a)
        return out
