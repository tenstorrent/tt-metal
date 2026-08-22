# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ConditionalDecoder: the UNet-1D that the flow-matching ODE calls at every step.

This is the model's hot spot. `ConditionalCFM` runs 10 Euler steps and each step
evaluates this network on a **batch of 2** -- the conditioned and unconditioned
rows of classifier-free guidance -- so the RTF of the whole flow stage is
essentially 20 forward passes of what is written here.

Shape, for the captured 608-frame utterance:

    in 320 = [x 80 | mu 80 | spks 80 | cond 80]
    down 0:  resnet 320->256, 4 transformers,  Conv1d(3, stride 2)   608 -> 304
    down 1:  resnet 256->256, 4 transformers,  Conv1d(3, stride 1)   304 -> 304
    mid:     12 x [resnet 256->256, 4 transformers]                  304
    up   0:  cat(skip) 512->256, 4 transformers, ConvTranspose1d(4,2,1)  304 -> 608
    up   1:  cat(skip) 512->256, 4 transformers, Conv1d(3, stride 1)     608 -> 608
    final:   Block1D 256->256, Conv1d(1) 256->80

16 ResnetBlock1D and **64 BasicTransformerBlock** per call.

Two things make the TTNN version structurally cheaper than the reference rather
than merely equivalent:

**No transposes.** The reference lives in `[B, C, T]` for its convolutions and
`rearrange`s to `[B, T, C]` and back around every transformer stack -- 32
contiguous copies per forward pass. `ttnn.conv1d` is channels-last and so are
linear, layer_norm and attention, so the entire UNet stays in `[B, T, C]` and
every one of those transposes disappears.

**GroupNorm as a LayerNorm.** See `TtGroupNorm` -- it is not an approximation,
it is the same statistic reached by a cheaper route, and on silicon it is more
accurate than the native kernel.

Masking: every captured call passes an all-ones mask, because CosyVoice-300M
feeds the flow one utterance at a time (chunked in streaming, but each chunk is
still dense). `mask=None` means that. A padded batch would need the mask threaded
through the `x * mask` products and the `[:, :, ::2]` stride in the down path;
that path is deliberately absent rather than present-and-unverified.
"""
from __future__ import annotations

import math
import os

import torch

import ttnn

from ..hifigan.conv import TtConv1d, accurate_compute_config
from ..hifigan.upsample import TtConvTranspose1d
from .encoder import _linear_fused  # same [out, in] -> [in, out] convention as `_linear` below

# Flash attention for the estimator's self-attention blocks. On by default: measured
# faster *and* more accurate on every gate (flow stage 0.707 -> 0.600 s, and all four
# PCCs improved, components included). `COSYVOICE_SDPA=0` restores the explicit
# matmul/softmax/matmul chain for A/B. See `TtAttention.__call__`.
COSY_SDPA = os.environ.get("COSYVOICE_SDPA", "1") == "1"

# `bfloat8_b` measured 1.00x on the AR decoder, twice, because that stage is bound by
# per-op latency rather than weight traffic. The flow decoder is a different regime --
# real tensors, batch 2, 64 blocks x 10 Euler steps -- so it is worth its own
# measurement rather than inheriting the decoder's verdict. `COSYVOICE_FLOW_BF8=1`.
FLOW_WEIGHTS_BF8 = os.environ.get("COSYVOICE_FLOW_BF8", "0") == "1"


# --------------------------------------------------------------------------
# small pieces
# --------------------------------------------------------------------------
def _linear(device, bag, name, dtype):
    """torch stores Linear weights as [out, in]; ttnn.linear wants [in, out].

    The weight may be stored narrower than the activations that flow through it; the
    bias stays at `dtype`, since it is one row against a full matrix and narrowing it
    buys no bandwidth while costing accuracy.
    """
    sub = bag.sub(name)
    wdt = ttnn.bfloat8_b if FLOW_WEIGHTS_BF8 else dtype
    w = ttnn.from_torch(sub.tensor("weight").t().contiguous(), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
    b = None
    if sub.has("bias"):
        b = ttnn.from_torch(sub.tensor("bias").reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


def _norm_affine(device, bag, name, dtype):
    """Per-channel gamma/beta as [1, 1, C], applied with multiply+add.

    Deliberately *not* the `[1, C]` TILE form `ttnn.layer_norm` wants for its own
    gamma: this affine is applied by hand after the normalisation, so it has to
    broadcast against a rank-3 `[B, T, C]` activation.
    """
    sub = bag.sub(name)
    g = ttnn.from_torch(sub.tensor("weight").reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(sub.tensor("bias").reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return g, b


def _ln_weights(device, bag, name, dtype):
    """LayerNorm gamma/beta in the [1, C] TILE form `ttnn.layer_norm` requires."""
    sub = bag.sub(name)
    g = ttnn.from_torch(sub.tensor("weight").reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(sub.tensor("bias").reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return g, b


class TtGroupNorm:
    """`torch.nn.GroupNorm(G, C)` on a channels-last `[B, T, C]` activation.

    GroupNorm pools each group's statistic over **channels-in-group and time
    jointly**. Reshaping `[B, T, C]` to `[B, G, T*(C/G)]` puts exactly that set on
    the last axis, so the normalisation is a plain LayerNorm and the affine is a
    per-channel multiply-add afterwards. No approximation is involved.

    TTNN does ship a native `group_norm`, and it is the wrong choice here on all
    three counts measured on Blackhole:

      native DRAM group_norm              PCC 0.9999231835
      native, use_welford=True            PCC 0.9998651553
      this, permute + layer_norm          PCC 0.9999931119

    It is also the only one of the three that needs no `core_grid` negotiation and
    does not have to be rebuilt when T changes -- and T changes between the down,
    mid and up stages of this very UNet. (The native op is not merely less accurate
    here, it is unavailable: it rejects `[2, 141, 256]` at group 8 on both parts.)

    **The reshape-permute route to that statistic is the estimator's single largest
    cost**, and it took a traced measurement to see it. Per call, on Blackhole:

        conv1d k3 256->256 @141    0.0320 ms
        this, permute + layer_norm 0.2190 ms      <- 6.8x the convolution it follows
        mish @141                  0.0076 ms

    33 of these run per Euler step, ~36% of the whole estimator. Untraced it looks
    ordinary (0.44 ms against the conv's 0.43) because dispatch dominates both -- the
    same trap PERF.md records for the decode step.

    The cost is the two `permute`s. Under `TILE_LAYOUT` they swap the tiled row axis,
    which is a genuine re-tiling shuffle rather than a view, and the intermediate's
    tiled face is `G x C/G` = `8 x 32` -- one tile carrying 8 useful rows out of 32.

    So take the same statistic without changing shape at all. Each group's sum over
    channels is a **matmul against a `[C, G]` indicator**; what remains is a reduction
    over T, an axis that needs no re-tiling. The statistics come back as `[B, 1, G]`,
    return to `[B, 1, C]` through the same indicator transposed, and normalise-plus-affine
    folds into one multiply and one add:

        out = x * (inv*w) + (b - mean*inv*w)

    Measured traced, against the permute form and a torch reference:

        [2, 141, 256]   0.2190 -> 0.0940 ms  (2.33x)   PCC 0.999988854
        [2, 282, 256]   0.3975 -> 0.0978 ms  (4.06x)   PCC 0.999992251

    Wormhole gives 2.07x and 3.33x. Note the matmul form is nearly **independent of T**
    where the permute form doubles with it, which is the re-tiling cost showing itself.

    `COSYVOICE_GN_PERMUTE=1` restores the permute form for A/B.
    """

    PERMUTE = os.environ.get("COSYVOICE_GN_PERMUTE") == "1"

    def __init__(self, device, bag, name, num_groups: int = 8, eps: float = 1e-5, dtype=ttnn.bfloat16):
        self.device, self.g, self.eps = device, num_groups, eps
        self.w, self.b = _norm_affine(device, bag, name, dtype)
        self._ind = self._indt = None
        self._dtype = dtype

    def _indicators(self, c: int):
        """`[1, C, G]` and `[1, G, C]` group-membership matrices, built once.

        Built lazily rather than in `__init__` because C is only known from the affine
        weight, and reading a shape off a device tensor at construction time is a host
        round trip this class otherwise never makes.
        """
        if self._ind is None:
            m = torch.zeros(c, self.g)
            m[torch.arange(c), torch.arange(c) // (c // self.g)] = 1.0
            self._ind = ttnn.from_torch(
                m.reshape(1, c, self.g), dtype=self._dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            self._indt = ttnn.from_torch(
                m.t().contiguous().reshape(1, self.g, c), dtype=self._dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._ind, self._indt

    def __call__(self, x):
        b, t, c = x.shape
        if self.PERMUTE:
            return self._permute_form(x, b, t, c)

        ind, indt = self._indicators(c)
        inv_n = 1.0 / float(t * (c // self.g))

        # Every intermediate is named and freed. Inside a captured trace an allocation is
        # part of the graph and lives for the trace's lifetime, so a leak here is
        # multiplied by 33 GroupNorms and never collected -- and the largest of these is
        # the full [B, T, C] product, not one of the [B, 1, G] statistics.
        sq = ttnn.multiply(x, x)
        s1 = ttnn.matmul(x, ind)  # [B, T, G] -- per-group channel sums
        s2 = ttnn.matmul(sq, ind)
        ttnn.deallocate(sq)

        # Reducing over T finishes the statistic. T is the tiled row axis, but a
        # reduction along it is a reduction, not a re-layout -- which is the whole point.
        t1 = ttnn.sum(s1, dim=1, keepdim=True)  # [B, 1, G]
        t2 = ttnn.sum(s2, dim=1, keepdim=True)
        ttnn.deallocate(s1)
        ttnn.deallocate(s2)
        mean = ttnn.multiply(t1, inv_n)
        ex2 = ttnn.multiply(t2, inv_n)
        ttnn.deallocate(t1)
        ttnn.deallocate(t2)

        m2 = ttnn.multiply(mean, mean)
        var = ttnn.subtract(ex2, m2)
        ttnn.deallocate(ex2)
        ttnn.deallocate(m2)
        veps = ttnn.add(var, self.eps)
        inv = ttnn.rsqrt(veps)
        ttnn.deallocate(var)
        ttnn.deallocate(veps)

        mean_c = ttnn.matmul(mean, indt)  # [B, 1, C]
        inv_c = ttnn.matmul(inv, indt)
        ttnn.deallocate(mean)
        ttnn.deallocate(inv)
        scale = ttnn.multiply(inv_c, self.w)
        ms = ttnn.multiply(mean_c, scale)
        shift = ttnn.subtract(self.b, ms)
        ttnn.deallocate(ms)
        ttnn.deallocate(mean_c)
        ttnn.deallocate(inv_c)
        xs = ttnn.multiply(x, scale)
        out = ttnn.add(xs, shift)
        ttnn.deallocate(xs)
        ttnn.deallocate(scale)
        ttnn.deallocate(shift)
        return out

    def _permute_form(self, x, b, t, c):
        """The original route, kept for A/B under `COSYVOICE_GN_PERMUTE=1`."""
        g = self.g
        cg = c // g
        h = ttnn.reshape(x, (b, t, g, cg))
        h = ttnn.permute(h, (0, 2, 1, 3))  # [B, G, T, C/G]
        h = ttnn.reshape(h, (b, g, t * cg))
        n = ttnn.layer_norm(h, epsilon=self.eps)
        ttnn.deallocate(h)
        n = ttnn.reshape(n, (b, g, t, cg))
        n = ttnn.permute(n, (0, 2, 1, 3))  # back to [B, T, G, C/G]
        n = ttnn.reshape(n, (b, t, c))
        scaled = ttnn.multiply(n, self.w)
        ttnn.deallocate(n)
        out = ttnn.add(scaled, self.b)
        ttnn.deallocate(scaled)
        return out


class TtBlock1D:
    """Conv1d(k=3, pad=1) -> GroupNorm(8) -> Mish."""

    def __init__(self, device, bag, dtype=ttnn.bfloat16):
        conv = bag.sub("block.0")
        self.conv = TtConv1d(device, conv.tensor("weight"), conv.optional("bias"), padding=1, dtype=dtype)
        self.norm = TtGroupNorm(device, bag, "block.1", num_groups=8, dtype=dtype)

    def __call__(self, x, length: int, batch: int = 1):
        h, _ = self.conv(x, length, batch)
        n = self.norm(h)
        ttnn.deallocate(h)
        out = ttnn.mish(n)
        ttnn.deallocate(n)
        return out


class TtResnetBlock1D:
    """block1 -> (+ time embedding) -> block2 -> (+ 1x1 conv of the input).

    `mlp` is `Sequential(Mish, Linear(time_embed_dim, dim_out))`, so only index 1
    carries weights. Its output is `[B, 1, dim_out]` and broadcasts over time.
    """

    def __init__(self, device, bag, cc=None, dtype=ttnn.bfloat16):
        self.cc = cc
        self.block1 = TtBlock1D(device, bag.sub("block1"), dtype)
        self.block2 = TtBlock1D(device, bag.sub("block2"), dtype)
        self.wm, self.bm = _linear(device, bag, "mlp.1", dtype)
        res = bag.sub("res_conv")
        self.res = TtConv1d(device, res.tensor("weight"), res.optional("bias"), padding=0, dtype=dtype)

    def __call__(self, x, t_emb, length: int, batch: int = 1):
        h = self.block1(x, length, batch)
        act = ttnn.mish(t_emb)
        proj = ttnn.linear(act, self.wm, bias=self.bm, compute_kernel_config=self.cc)  # [B, 1, C]
        ttnn.deallocate(act)
        h2 = ttnn.add(h, proj)
        ttnn.deallocate(h)
        ttnn.deallocate(proj)
        h3 = self.block2(h2, length, batch)
        ttnn.deallocate(h2)
        r, _ = self.res(x, length, batch)
        out = ttnn.add(h3, r)
        ttnn.deallocate(h3)
        ttnn.deallocate(r)
        return out


class TtAttention:
    """diffusers `Attention`, self-attention only, `attention_bias=False`.

    q/k/v project 256 -> 512 (8 heads x 64) with no bias; `to_out.0` projects back
    to 256 with one. `scale_qk` is on, so the scale is `dim_head ** -0.5` -- note
    that is the *head* dim, not the inner dim.
    """

    def __init__(self, device, bag, heads: int, dim_head: int, cc=None, dtype=ttnn.bfloat16):
        self.device, self.h, self.d, self.cc = device, heads, dim_head, cc
        self.scale = dim_head**-0.5
        # q, k and v project the same activation, so they are one matmul over a
        # concatenated weight. This block runs 64 times per Euler step and there are
        # ten steps at batch 2, so four ops saved here are 5 120 ops off the stage.
        self.wqkv, self.bqkv = _linear_fused(
            device,
            bag,
            ("to_q", "to_k", "to_v"),
            dtype,
            weights_dtype=ttnn.bfloat8_b if FLOW_WEIGHTS_BF8 else None,
            scales=(self.scale, 1.0, 1.0),
        )
        self.wo, self.bo = _linear(device, bag, "to_out.0", dtype)

    def _heads(self, x, b, t):
        x = ttnn.reshape(x, (b, t, self.h, self.d))
        return ttnn.permute(x, (0, 2, 1, 3))

    def __call__(self, x):
        b, t, _ = x.shape
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=self.cc)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=self.h, transpose_key=False)
        ttnn.deallocate(qkv)

        if COSY_SDPA:
            # Flash attention. This block is *plain* self-attention -- no mask, no
            # relative-position term -- so it is the one place in the model where SDPA
            # is a drop-in. The score matrix it avoids materialising is
            # [2, 8, T, T]; at T = 282 that is 2.5 MB written and read back per block,
            # 64 blocks x 10 Euler steps.
            #
            # `scale=1.0` because `1/sqrt(d)` is already folded into the q half of the
            # fused weight; letting SDPA apply its default would scale twice.
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=1.0, compute_kernel_config=self.cc
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        else:
            # `transpose_b` folds the [B, h, T, d] -> [B, h, d, T] permute into the matmul.
            # `1/sqrt(d)` is already baked into the q half of the fused weight, so the
            # separate `multiply(scores, scale)` is gone -- see `_linear_fused(scales=)`.
            scores = ttnn.matmul(q, k, transpose_b=True, compute_kernel_config=self.cc)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            attn = ttnn.softmax(scores, dim=-1)
            ttnn.deallocate(scores)

            ctx = ttnn.matmul(attn, v, compute_kernel_config=self.cc)
            ttnn.deallocate(attn)
            ttnn.deallocate(v)
        ctx = ttnn.transformer.concatenate_heads(ctx)  # [B, h, T, d] -> [B, T, h*d], one op
        out = ttnn.linear(ctx, self.wo, bias=self.bo, compute_kernel_config=self.cc)
        ttnn.deallocate(ctx)
        return out


class TtBasicTransformerBlock:
    """norm1 -> self-attention -> residual -> norm3 -> feed-forward -> residual.

    `act_fn='gelu'` in the checkpoint selects diffusers' `GELU`, which is a plain
    `Linear(256, 1024)` followed by the **erf** GELU -- not GEGLU and not the tanh
    approximation. `ttnn.gelu(fast_and_approximate_mode=False)` is the matching
    one; the approximate mode measured 0.99999 vs 0.99999_75 on Blackhole, a real
    if small difference over 64 blocks.
    """

    def __init__(self, device, bag, heads: int, dim_head: int, eps: float = 1e-5, cc=None, dtype=ttnn.bfloat16):
        self.eps, self.cc = eps, cc
        self.g1, self.b1 = _ln_weights(device, bag, "norm1", dtype)
        self.g3, self.b3 = _ln_weights(device, bag, "norm3", dtype)
        self.attn = TtAttention(device, bag.sub("attn1"), heads, dim_head, cc, dtype)
        self.wf1, self.bf1 = _linear(device, bag, "ff.net.0.proj", dtype)
        self.wf2, self.bf2 = _linear(device, bag, "ff.net.2", dtype)

    def __call__(self, x):
        # `x` belongs to the caller and is NOT freed here. Three ResBlocks sharing
        # one input is what broke the vocoder in P2; the rule that came out of it
        # is that a module frees only what it allocated.
        n = ttnn.layer_norm(x, weight=self.g1, bias=self.b1, epsilon=self.eps)
        a = self.attn(n)
        ttnn.deallocate(n)
        x1 = ttnn.add(a, x)
        ttnn.deallocate(a)

        n3 = ttnn.layer_norm(x1, weight=self.g3, bias=self.b3, epsilon=self.eps)
        f = ttnn.linear(n3, self.wf1, bias=self.bf1, compute_kernel_config=self.cc)
        ttnn.deallocate(n3)
        f = ttnn.gelu(f, fast_and_approximate_mode=False)
        f2 = ttnn.linear(f, self.wf2, bias=self.bf2, compute_kernel_config=self.cc)
        ttnn.deallocate(f)
        out = ttnn.add(f2, x1)
        ttnn.deallocate(f2)
        ttnn.deallocate(x1)
        return out


# --------------------------------------------------------------------------
# the UNet
# --------------------------------------------------------------------------
def sinusoidal_frequencies(dim: int, scale: float = 1000.0) -> torch.Tensor:
    """The constant half of `SinusoidalPosEmb`: `scale * exp(-i*log(10000)/(d/2-1))`.

    Folding `scale` in here means the device only ever does `t * freqs`, one
    broadcast multiply, instead of a multiply and a scale.
    """
    half = dim // 2
    step = math.log(10000.0) / (half - 1)
    return scale * torch.exp(torch.arange(half, dtype=torch.float32) * -step)


class TtConditionalDecoder:
    """The flow-matching estimator. Activations are channels-last `[B, T, C]`."""

    def __init__(self, device, bag, *, in_channels=320, num_heads=8, attention_head_dim=64, dtype=ttnn.bfloat16):
        self.device, self.dtype = device, dtype
        self.in_channels = in_channels
        self.heads, self.dim_head = num_heads, attention_head_dim
        # HiFi4 + fp32 accumulation on the matmuls too, not only the convolutions:
        # 64 transformer blocks compound bfloat16 drift the same way 40 convolutions
        # did in the vocoder, and the later Euler steps are where it shows.
        self.cc = accurate_compute_config(device)

        self.freqs = ttnn.from_torch(
            sinusoidal_frequencies(in_channels).reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.wt1, self.bt1 = _linear(device, bag, "time_mlp.linear_1", dtype)
        self.wt2, self.bt2 = _linear(device, bag, "time_mlp.linear_2", dtype)

        def transformers(sub):
            """`sub` is the ModuleList of BasicTransformerBlocks; its length is read
            off the exported names rather than hardcoded to n_blocks=4."""
            return [
                TtBasicTransformerBlock(device, sub.sub(str(j)), num_heads, attention_head_dim, cc=self.cc, dtype=dtype)
                for j in range(sub.children())
            ]

        self.down = []
        n_down = bag.sub("down_blocks").children()
        for i in range(n_down):
            sub = bag.sub(f"down_blocks.{i}")
            resnet = TtResnetBlock1D(device, sub.sub("0"), self.cc, dtype)
            blocks = transformers(sub.sub("1"))
            # Downsample1D wraps its conv in `.conv`; the last stage is a bare Conv1d.
            is_last = i == n_down - 1
            cs = sub.sub("2") if is_last else sub.sub("2.conv")
            conv = TtConv1d(
                device, cs.tensor("weight"), cs.optional("bias"), stride=1 if is_last else 2, padding=1, dtype=dtype
            )
            self.down.append((resnet, blocks, conv, 1 if is_last else 2))

        self.mid = []
        for i in range(bag.sub("mid_blocks").children()):
            sub = bag.sub(f"mid_blocks.{i}")
            self.mid.append((TtResnetBlock1D(device, sub.sub("0"), self.cc, dtype), transformers(sub.sub("1"))))

        self.up = []
        n_up = bag.sub("up_blocks").children()
        for i in range(n_up):
            sub = bag.sub(f"up_blocks.{i}")
            resnet = TtResnetBlock1D(device, sub.sub("0"), self.cc, dtype)
            blocks = transformers(sub.sub("1"))
            is_last = i == n_up - 1
            if is_last:
                cs = sub.sub("2")
                conv = TtConv1d(device, cs.tensor("weight"), cs.optional("bias"), padding=1, dtype=dtype)
                self.up.append((resnet, blocks, conv, False))
            else:
                cs = sub.sub("2.conv")
                conv = TtConvTranspose1d(
                    device, cs.tensor("weight"), cs.optional("bias"), stride=2, padding=1, dtype=dtype
                )
                self.up.append((resnet, blocks, conv, True))

        self.final_block = TtBlock1D(device, bag.sub("final_block"), dtype)
        fp = bag.sub("final_proj")
        self.final_proj = TtConv1d(device, fp.tensor("weight"), fp.optional("bias"), padding=0, dtype=dtype)

    # ----------------------------------------------------------------------
    def time_embedding(self, t):
        """t: `[B, 1, 1]` -> `[B, 1, time_embed_dim]`.

        `SinusoidalPosEmb` then `TimestepEmbedding(act='silu')`. The sinusoid is
        an outer product of t with a constant frequency vector, so on device it is
        one broadcast multiply and a sin/cos pair -- no host round-trip, which
        matters because this runs once per Euler step.
        """
        ang = ttnn.multiply(t, self.freqs)  # [B, 1, dim/2]
        s, c = ttnn.sin(ang), ttnn.cos(ang)
        ttnn.deallocate(ang)
        emb = ttnn.concat([s, c], dim=-1)
        ttnn.deallocate(s)
        ttnn.deallocate(c)
        h = ttnn.linear(emb, self.wt1, bias=self.bt1, compute_kernel_config=self.cc)
        ttnn.deallocate(emb)
        a = ttnn.silu(h)
        ttnn.deallocate(h)
        out = ttnn.linear(a, self.wt2, bias=self.bt2, compute_kernel_config=self.cc)
        ttnn.deallocate(a)
        return out

    def pack_input(self, x, mu, spks, cond, length: int, batch: int):
        """[x | mu | spks broadcast over T | cond] on the channel axis.

        The pieces are 80 channels each -- 2.5 tiles -- and `ttnn.concat` handles
        that on the last axis in TILE layout regardless (measured 0.9999986 on
        Blackhole), so no padding or split-convolution workaround is needed.
        """
        parts = [x, mu]
        if spks is not None:
            parts.append(ttnn.repeat(spks, ttnn.Shape([1, length, 1])))
        if cond is not None:
            parts.append(cond)
        out = ttnn.concat(parts, dim=-1)
        if spks is not None:
            ttnn.deallocate(parts[2])
        return out

    def pack_const(self, mu, spks, cond, length: int):
        """The `[mu | spks broadcast | cond]` tail of `pack_input`, built once.

        Only `x` changes across the solver's Euler steps -- `mu`, `spks` and `cond` are
        fixed for the whole utterance -- so the broadcast of `spks` over time and the
        concatenation of the three constant blocks are loop-invariant. Doing them once
        before trace capture leaves the traced body a single two-way concat with `x`.

        Worth the extra entry point rather than a cache inside `pack_input`, because
        the tensor is read by a captured trace for the life of that trace: an eviction
        would be a use-after-free, and the caller here already owns the lifetime.
        """
        parts = [mu]
        if spks is not None:
            parts.append(ttnn.repeat(spks, ttnn.Shape([1, length, 1])))
        if cond is not None:
            parts.append(cond)
        out = ttnn.concat(parts, dim=-1) if len(parts) > 1 else mu
        if spks is not None:
            ttnn.deallocate(parts[1])
        return out

    def __call__(self, x, mu, t, spks=None, cond=None, mask=None, batch: int = 2, packed_const=None):
        """x/mu/cond: `[B, T, 80]`; spks: `[B, 1, 80]`; t: `[B, 1, 1]`.

        Returns `[B, T, 80]`, the estimated dphi/dt.

        `packed_const` is the output of `pack_const` -- the loop-invariant
        `[mu | spks | cond]` block. Passing it turns the per-step input assembly from a
        broadcast plus a four-way concat into a single two-way concat; the solver builds
        it once per utterance. `mu`/`spks`/`cond` are then unused and may be `None`.
        """
        if mask is not None:
            raise NotImplementedError(
                "padded batches are not supported: every captured call passes an all-ones mask. "
                "Adding one means threading it through the x*mask products and the [:, :, ::2] "
                "stride in the down path -- see the module docstring."
            )
        length = x.shape[1]
        t_emb = self.time_embedding(t)
        if packed_const is not None:
            h = ttnn.concat([x, packed_const], dim=-1)
        else:
            h = self.pack_input(x, mu, spks, cond, length, batch)

        # Every hand-off below is explicit about ownership: a sub-module frees only
        # what it allocated, so the caller frees the tensor it passed in once the
        # result no longer aliases it. The one exception is the skip tensors, which
        # stay live across the whole mid stack.
        def run_blocks(h, blocks):
            for blk in blocks:
                nxt = blk(h)
                ttnn.deallocate(h)
                h = nxt
            return h

        skips, lengths = [], []
        cur = length
        for resnet, blocks, conv, _stride in self.down:
            nxt = resnet(h, t_emb, cur, batch)
            ttnn.deallocate(h)
            h = run_blocks(nxt, blocks)
            skips.append(h)  # kept for the up path -- deliberately not freed
            lengths.append(cur)
            h, cur = conv(h, cur, batch)

        for resnet, blocks in self.mid:
            nxt = resnet(h, t_emb, cur, batch)
            ttnn.deallocate(h)
            h = run_blocks(nxt, blocks)

        for resnet, blocks, conv, _is_transpose in self.up:
            skip = skips.pop()
            skip_len = lengths.pop()
            if cur != skip_len:
                # `x[:, :, :skip.shape[-1]]` upstream -- a transposed convolution can
                # overshoot the skip it has to meet.
                trimmed = ttnn.slice(h, [0, 0, 0], [batch, skip_len, h.shape[-1]])
                ttnn.deallocate(h)
                h, cur = trimmed, skip_len
            cat = ttnn.concat([h, skip], dim=-1)
            ttnn.deallocate(h)
            ttnn.deallocate(skip)
            nxt = resnet(cat, t_emb, cur, batch)
            ttnn.deallocate(cat)
            h = run_blocks(nxt, blocks)
            out, cur = conv(h, cur, batch)
            ttnn.deallocate(h)
            h = out

        ttnn.deallocate(t_emb)
        f = self.final_block(h, cur, batch)
        ttnn.deallocate(h)
        out, _ = self.final_proj(f, cur, batch)
        ttnn.deallocate(f)
        return out
