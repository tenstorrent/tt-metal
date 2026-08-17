# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttml autograd Functions for convolutional (UNet) training — Phase 1.

Everything here is a "pattern 2" ttml.autograd.Function (see
ttml/autograd/function.py): forward() takes ttml tensors, computes with raw
ttnn ops on .get_value(), returns raw ttnn tensors (auto-wrapped); backward()
receives raw ttnn grads and returns one raw ttnn grad per tensor input.

CANONICAL ACTIVATION SHAPE: [B, 1, H*W, C] channels-last tokens, TILE
layout, bf16. H and W travel alongside as python ints (static per call
site). This keeps every broadcast (bias add, per-block embedding add) in the
already-validated [B,1,T,D] (+) [B,1,1,D] row-broadcast form and lets 1x1
convs / attention reuse plain LinearLayers.

LAYOUT POLICY: pad/slice/concat/permute and the pool/upsample composites
run in ROW_MAJOR NHWC (TILE front-padding is restricted); matmul, elementwise
adds, and moreh group_norm run in TILE. Conversions are explicit via
_rm/_tile. Phase 1 optimizes for correctness, not layout-churn count.
Pool/upsample avoid the ttnn sliding-window kernels entirely — see the note
above _sum_pool2x2_rm (ttml opens the mesh with l1_small_size=0, which the
halo path needs).

CONV WEIGHT ROW_ORDER (must match reference_unet.py exactly):
    3x3 conv weight is a TILE matrix [1, 1, 9*C_in, C_out] with row index
        r = (kh*3 + kw)*C_in + c_in,   kh, kw in {0,1,2}
    i.e. flatten order (kh, kw, c_in), kernel window scanned row-major.
    _im2col concatenates its 9 shifted views in the same (kh, kw) order, so
    patches @ W_flat == conv2d(x, W_flat.view(3,3,Ci,Co).permute(3,2,0,1),
    padding=1). Weight-gradient rows land in the same order.

MEMORY POLICY: conv backward recomputes im2col from the saved input rather
than saving the 9x-wider patch tensor.
"""

from __future__ import annotations

import ttnn
from ttnn.operations import moreh as ttnn_moreh  # not attached to the ttnn namespace in all builds

import ttml


def _rm(v):
    return ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT)


def _tile(v):
    return ttnn.to_layout(v, ttnn.TILE_LAYOUT)


def _tokens_to_nhwc_rm(v, b, h, w, c):
    """[B,1,HW,C] (any layout) -> ROW_MAJOR [B,H,W,C]."""
    return ttnn.reshape(_rm(v), (b, h, w, c))


def _nhwc_rm_to_tokens(v, b, h, w, c):
    """ROW_MAJOR [B,H,W,C] -> TILE [B,1,HW,C]."""
    return _tile(ttnn.reshape(v, (b, 1, h * w, c)))


def _im2col(v, b, h, w, c):
    """[B,1,HW,C] -> TILE [1,1,B*H*W, 9C] patches (ROW_ORDER = (kh, kw, c_in))."""
    x = _tokens_to_nhwc_rm(v, b, h, w, c)
    p = ttnn.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], 0.0)
    views = []
    for kh in range(3):
        for kw in range(3):
            views.append(ttnn.slice(p, [0, kh, kw, 0], [b, kh + h, kw + w, c]))
    cols = ttnn.concat(views, dim=-1)  # RM [B,H,W,9C]
    return _tile(ttnn.reshape(cols, (1, 1, b * h * w, 9 * c)))


class Permute(ttml.autograd.Function):
    """Autograd permute: ttnn.permute fwd, inverse permute bwd."""

    @staticmethod
    def forward(ctx, x, dims):
        dims = tuple(dims)
        ctx.inverse = tuple(sorted(range(len(dims)), key=dims.__getitem__))
        return ttnn.permute(x.get_value(), dims)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.permute(grad_output, ctx.inverse)


class Scale(ttml.autograd.Function):
    """Multiply by a python scalar (ttml.ops.binary.mul is tensor-tensor only)."""

    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return ttnn.multiply(x.get_value(), factor)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(grad_output, ctx.factor)


class ConcatChannels(ttml.autograd.Function):
    """Skip concat along the channel (last) dim of [B,1,HW,C] tokens.

    Backward slices the grad back apart. TILE slice/concat require the
    boundary channel count to be tile-aligned; all UNet channel counts are
    multiples of 32 wherever concat is used.
    """

    @staticmethod
    def forward(ctx, a, b):
        va, vb = a.get_value(), b.get_value()
        ctx.ca, ctx.cb = va.shape[-1], vb.shape[-1]
        return ttnn.concat([va, vb], dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        shape = list(grad_output.shape)
        begins = [0] * len(shape)
        ends_a = shape[:-1] + [ctx.ca]
        begins_b = begins[:-1] + [ctx.ca]
        ends_b = shape[:-1] + [ctx.ca + ctx.cb]
        da = ttnn.slice(grad_output, begins, ends_a)
        db = ttnn.slice(grad_output, begins_b, ends_b)
        return da, db


class Conv3x3Im2col(ttml.autograd.Function):
    """3x3 same-pad conv as im2col + matmul; weight flat [1,1,9Cin,Cout].

    apply(x [B,1,HW,Cin], weight, bias [1,1,1,Cout], H, W) -> [B,1,HW,Cout].
    Backward: dW = im2col(x)^T @ dOut (im2col recomputed), db = sum_BHW dOut,
    dX = col2im(dOut @ W^T) via 9 pad-shift adds.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, h, w):
        v, wv, bv = x.get_value(), weight.get_value(), bias.get_value()
        b = v.shape[0]
        c = v.shape[-1]
        cout = wv.shape[-1]
        cols = _im2col(v, b, h, w, c)  # [1,1,BHW,9C]
        out = ttnn.matmul(cols, wv)  # [1,1,BHW,Cout]
        out = ttnn.add(out, bv)
        ctx.save_for_backward(x, weight)
        ctx.dims = (b, h, w, c, cout)
        return _tile(ttnn.reshape(_rm(out), (b, 1, h * w, cout)))

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        b, h, w, c, cout = ctx.dims
        wv = weight.get_value()
        g = _tile(ttnn.reshape(_rm(grad_output), (1, 1, b * h * w, cout)))
        # dW / db
        cols = _im2col(x.get_value(), b, h, w, c)  # recompute (memory policy)
        dw = ttnn.matmul(cols, g, transpose_a=True)  # [1,1,9C,Cout]
        db = ttnn.sum(g, dim=2, keepdim=True)  # [1,1,1,Cout]
        # dX: col2im of dOut @ W^T
        dcols = ttnn.matmul(g, wv, transpose_b=True)  # [1,1,BHW,9C]
        d = ttnn.reshape(_rm(dcols), (b, h, w, 9 * c))
        acc = None
        for kh in range(3):
            for kw in range(3):
                k = kh * 3 + kw
                part = ttnn.slice(d, [0, 0, 0, k * c], [b, h, w, (k + 1) * c])
                shifted = ttnn.pad(part, [(0, 0), (kh, 2 - kh), (kw, 2 - kw), (0, 0)], 0.0)
                shifted = _tile(shifted)  # elementwise add on TILE
                acc = shifted if acc is None else ttnn.add(acc, shifted)
        dp = _rm(acc)  # [B,H+2,W+2,C]
        dx = ttnn.slice(dp, [0, 1, 1, 0], [b, h + 1, w + 1, c])
        dx = _nhwc_rm_to_tokens(dx, b, h, w, c)
        return dx, dw, db


# NOTE: pool/upsample are deliberately NOT ttnn.avg_pool2d / ttnn.upsample.
# Those go through the sliding-window halo path, which allocates from the
# L1_SMALL region — and ttml opens its mesh with DEFAULT_L1_SMALL_SIZE = 0
# (tt-train/sources/ttml/core/mesh_device.cpp), so they die with an
# out-of-memory TT_FATAL in bank_manager.cpp. For fixed 2x2/stride-2 kernels
# the same math is a handful of RM reshape/slice/concat + TILE adds — all
# primitives already exercised by the conv im2col/col2im paths.


def _sum_pool2x2_rm(v_rm, b, h, w, c):
    """RM [B,H,W,C] -> RM [B,H/2,W/2,C], each output = SUM of its 2x2 window.

    Pure data movement + adds, no halo: pair adjacent W pixels via a
    [B,H,W/2,2C] view (last-dim slices = even/odd columns), then pair
    adjacent rows via a [B,H/2,W,C] view (dim-2 slices = even/odd rows).
    """
    r = ttnn.reshape(v_rm, (b, h, w // 2, 2 * c))
    even_w = ttnn.slice(r, [0, 0, 0, 0], [b, h, w // 2, c])
    odd_w = ttnn.slice(r, [0, 0, 0, c], [b, h, w // 2, 2 * c])
    s = _rm(ttnn.add(_tile(even_w), _tile(odd_w)))  # [B,H,W/2,C]
    s = ttnn.reshape(s, (b, h // 2, w, c))  # rows (2i, 2i+1) side by side in dim 2
    even_h = ttnn.slice(s, [0, 0, 0, 0], [b, h // 2, w // 2, c])
    odd_h = ttnn.slice(s, [0, 0, w // 2, 0], [b, h // 2, w, c])
    return _rm(ttnn.add(_tile(even_h), _tile(odd_h)))  # [B,H/2,W/2,C]


def _nearest_up2_rm(v_rm, b, h, w, c):
    """RM [B,H,W,C] -> RM [B,2H,2W,C] nearest-neighbor: concat-duplicate each
    pixel along W (channel-concat + reshape), then each row along H."""
    r = ttnn.concat([v_rm, v_rm], dim=-1)  # [B,H,W,2C]: pixel duplicated adjacently
    r = ttnn.reshape(r, (b, h, 1, 2 * w * c))
    r = ttnn.concat([r, r], dim=-1)  # [B,H,1,4WC]: row duplicated adjacently
    return ttnn.reshape(r, (b, 2 * h, 2 * w, c))


class AvgPool2x2(ttml.autograd.Function):
    """2x2/stride-2 average pool on [B,1,HW,C] tokens (halo-free composite).

    Backward is the exact adjoint: nearest-2x upsample of dOut times 0.25.
    """

    @staticmethod
    def forward(ctx, x, h, w):
        v = x.get_value()
        b, c = v.shape[0], v.shape[-1]
        ctx.dims = (b, h, w, c)
        s = _sum_pool2x2_rm(_tokens_to_nhwc_rm(v, b, h, w, c), b, h, w, c)
        return ttnn.multiply(_nhwc_rm_to_tokens(s, b, h // 2, w // 2, c), 0.25)

    @staticmethod
    def backward(ctx, grad_output):
        b, h, w, c = ctx.dims
        g = _tokens_to_nhwc_rm(grad_output, b, h // 2, w // 2, c)
        up = _nearest_up2_rm(g, b, h // 2, w // 2, c)  # RM [B,H,W,C]
        return ttnn.multiply(_nhwc_rm_to_tokens(up, b, h, w, c), 0.25)


class UpsampleNearest2(ttml.autograd.Function):
    """Nearest-neighbor 2x upsample on [B,1,HW,C] tokens (halo-free composite).

    Backward is the exact adjoint: 2x2 SUM pool of dOut. AvgPool2x2 and
    UpsampleNearest2 form an adjoint pair; the device gate checks both
    directions numerically.
    """

    @staticmethod
    def forward(ctx, x, h, w):
        v = x.get_value()
        b, c = v.shape[0], v.shape[-1]
        ctx.dims = (b, h, w, c)
        up = _nearest_up2_rm(_tokens_to_nhwc_rm(v, b, h, w, c), b, h, w, c)
        return _nhwc_rm_to_tokens(up, b, 2 * h, 2 * w, c)

    @staticmethod
    def backward(ctx, grad_output):
        b, h, w, c = ctx.dims
        g = _tokens_to_nhwc_rm(grad_output, b, 2 * h, 2 * w, c)
        down = _sum_pool2x2_rm(g, b, 2 * h, 2 * w, c)  # RM [B,H,W,C]
        return _nhwc_rm_to_tokens(down, b, h, w, c)


class GroupNormMoreh(ttml.autograd.Function):
    """GroupNorm on [B,1,HW,C] tokens via the moreh group_norm kernels (NCHW TILE).

    apply(x, gamma [1,1,1,C], beta [1,1,1,C], num_groups, H, W).
    Wraps the moreh op in NHWC<->NCHW permutes (done in ROW_MAJOR — plain
    data movement); saves the NCHW input plus (mean, rstd) for backward.
    """

    EPS = 1e-6

    @staticmethod
    def forward(ctx, x, gamma, beta, num_groups, h, w):
        v = x.get_value()
        b, c = v.shape[0], v.shape[-1]
        nhwc = _tokens_to_nhwc_rm(v, b, h, w, c)
        nchw = _tile(ttnn.permute(nhwc, (0, 3, 1, 2)))  # [B,C,H,W]
        out, mean, rstd = ttnn_moreh.group_norm(
            nchw,
            num_groups,
            eps=GroupNormMoreh.EPS,
            gamma=gamma.get_value(),
            beta=beta.get_value(),
            are_required_outputs=[True, True, True],
        )
        ctx.save_for_backward(gamma)
        ctx.nchw, ctx.mean, ctx.rstd = nchw, mean, rstd
        ctx.dims = (b, h, w, c)
        ctx.num_groups = num_groups
        out_nhwc = ttnn.permute(_rm(out), (0, 2, 3, 1))
        return _nhwc_rm_to_tokens(out_nhwc, b, h, w, c)

    @staticmethod
    def backward(ctx, grad_output):
        (gamma,) = ctx.saved_tensors
        b, h, w, c = ctx.dims
        g_nhwc = _tokens_to_nhwc_rm(grad_output, b, h, w, c)
        g_nchw = _tile(ttnn.permute(g_nhwc, (0, 3, 1, 2)))
        dx_nchw, dgamma, dbeta = ttnn_moreh.group_norm_backward(
            g_nchw,
            ctx.nchw,
            ctx.mean,
            ctx.rstd,
            ctx.num_groups,
            are_required_outputs=[True, True, True],
            gamma=gamma.get_value(),
        )
        dx_nhwc = ttnn.permute(_rm(dx_nchw), (0, 2, 3, 1))
        dx = _nhwc_rm_to_tokens(dx_nhwc, b, h, w, c)
        return dx, dgamma, dbeta
