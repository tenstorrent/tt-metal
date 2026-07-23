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

# Conv math fidelity. Held at HiFi4 for audio quality: although HiFi3 matches HiFi4 on aggregate
# PCC (0.9907 @ latent_len=256), a localized per-window comparison showed HiFi3 slightly worsens
# the worst-window fit at high-energy transients (0.9827 -> 0.9799), which is audible as a mild
# robotic artifact on 1-2 phonemes. The 4th phase costs ~193us device time (only relevant under
# trace), so HiFi4 is the safe default. (HiFi2 was worse still — sinks aggregate PCC to ~0.982
# by latent_len=256.) Kept split (bf16/fp32) to tune separately if the trade is revisited.
_CONV_FIDELITY_BF16 = ttnn.MathFidelity.HiFi4
_CONV_FIDELITY_FP32 = ttnn.MathFidelity.HiFi4


def _shard_plan(stage_i, kernel_size):
    """(sharded, act_double_buffer) for a resblock at upsample ``stage_i`` with ``kernel_size``.

    Tuned on Blackhole for the profiled decode length (latent_len 32): keeping the residual
    chain resident in L1 must leave room for the conv's circular buffers, which grow with
    channel width (early stages) and kernel size (halo). The widest-channel x largest-kernel
    blocks — (stage 0, k7/k11) and (stage 1, k11) — clash L1 and stay on the interleaved path;
    the double buffer is dropped on the tighter feasible blocks. Anything not enabled here just
    keeps the original (correct, slower) interleaved round-trip, so this is safe to widen once
    a shape is verified to fit."""
    if not _SHARD_RESBLOCKS:
        return False, True
    if kernel_size <= 3:
        return True, True
    if kernel_size <= 7:
        return stage_i >= 1, False
    return stage_i >= 2, False


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
            )
            for j in range(len(dilation))
        ]

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

    def _forward_sharded(self, x):
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
        )
        self.cond_layer = TtConv1d(
            device,
            state_dict["cond_layer.weight"],
            state_dict["cond_layer.bias"],
            math_fidelity=_CONV_FIDELITY_FP32,
        )

        self.ups = [
            TtConvTranspose1d(
                device,
                state_dict[f"ups.{i}.weight"],
                state_dict[f"ups.{i}.bias"],
                stride=UPSAMPLE_RATES[i],
                activations_dtype=act_dtype(i),
                math_fidelity=conv_fid(i),
            )
            for i in range(self.num_upsamples)
        ]
        self.conds = [
            TtConv1d(
                device,
                state_dict[f"conds.{i}.weight"],
                state_dict[f"conds.{i}.bias"],
                activations_dtype=act_dtype(i),
                math_fidelity=conv_fid(i),
            )
            for i in range(self.num_upsamples)
        ]

        self.resblocks = []
        for i in range(self.num_upsamples):
            for j, (k, d) in enumerate(zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES)):
                sharded, act_double_buffer = _shard_plan(i, k)
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
                    )
                )

        # conv_post has no bias in XTTS.
        self.conv_post = TtConv1d(
            device,
            state_dict["conv_post.weight"],
            None,
            padding=3,
            math_fidelity=_CONV_FIDELITY_FP32,
        )

    def forward(self, x, g):
        # Deep conv chain — the vocoder's memory-dominant path, whose activation
        # footprint grows with output length. Each temporary is freed the moment the
        # next op consumes it; ``g``/``g_t`` are never freed (reused by every cond layer).
        # cond_layer/conds are 1x1 convs (matmuls); each would otherwise tilize ``g`` from
        # ROW_MAJOR internally, so tilize it ONCE here and hand the TILE copy to all 5 — turns
        # 5 TilizeWithValPadding ops into 1 (verified same result). Pure device op: trace-safe.
        g_t = ttnn.to_layout(g, ttnn.TILE_LAYOUT)
        cond_global = self.cond_layer(g_t)  # [N, 1, 512], broadcasts over T
        pre = self.conv_pre(x)
        ttnn.deallocate(x)  # upsampler output, not reused after conv_pre
        o = ttnn.add(pre, cond_global)
        ttnn.deallocate(pre)
        ttnn.deallocate(cond_global)

        for i in range(self.num_upsamples):
            a = ttnn.leaky_relu(o, negative_slope=LRELU_SLOPE)
            ttnn.deallocate(o)
            # ``conds[i](g)`` is a length-1, per-channel constant, so ``ups[i](a) +
            # conds[i](g)`` is just a per-channel bias add — fold it into the ups conv's
            # fused bias epilogue instead of a separate full-length broadcast add.
            cg = self.conds[i](g_t)  # [1, 1, C_i]
            cg = ttnn.reshape(cg, [1, 1, 1, cg.shape[-1]])
            if cg.dtype != ttnn.float32:
                cg = ttnn.typecast(cg, ttnn.float32)
            o = self.ups[i](a, cond_bias=cg)
            ttnn.deallocate(a)
            ttnn.deallocate(cg)
            z_sum = None
            for j in range(self.num_kernels):
                res = self.resblocks[i * self.num_kernels + j](o)  # does not free o
                if z_sum is None:
                    z_sum = res
                else:
                    z_new = ttnn.add(z_sum, res)
                    ttnn.deallocate(z_sum)
                    ttnn.deallocate(res)
                    z_sum = z_new
            o_new = ttnn.mul(z_sum, self.inv_num_kernels)
            ttnn.deallocate(z_sum)
            ttnn.deallocate(o)  # free the resblock input once the MRF sum is done
            o = o_new
        ttnn.deallocate(g_t)  # our tiled copy of g; last used by conds[-1] above
        a = ttnn.leaky_relu(o, negative_slope=FINAL_LRELU_SLOPE)
        ttnn.deallocate(o)
        p = self.conv_post(a)
        ttnn.deallocate(a)
        out = ttnn.tanh(p)
        ttnn.deallocate(p)
        return out
