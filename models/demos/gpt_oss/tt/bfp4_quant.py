# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Accurate MXFP4 -> BFP4 (ttnn.bfloat4_b) rounding of the gpt-oss expert weights of the fused decode path.

ttnn.bfloat4_b stores, per group of 16 consecutive values of one row of a 16x16 face, one shared 8-bit exponent E and
per value a sign and a 3-bit magnitude m: value = +-m * 2^(E - 127 - 2), m = 0..7. For a [K, N] weight in TILE layout a
group is W[k, 16 j : 16 j + 16]: 16 consecutive *output* columns at one K. ttnn's host packer
(tt_metal/impl/data_format/blockfloat_common.cpp) always takes E = the group's largest exponent and rounds to nearest
even, which wastes the 3 bits whenever one value is much larger than the rest: the gpt-oss experts are MXFP4
(E2M1 values {0, .5, 1, 1.5, 2, 3, 4, 6} times a power-of-two scale per 32 values along K), so a group mixes 16 output
channels with up to ~4 different scale exponents (down_proj), and even within one scale 0.5 / 1.5 cannot sit next to
4 / 6.

mxfp4_to_bfp4() instead picks, per group, the shared exponent (the largest exponent or one below it, the large
values then saturating at 7) with the least squared reconstruction error, and returns the weight rounded onto that
grid. Values already on a BFP4 grid are packed losslessly by ttnn's packer (the group's largest rounded value fixes the
same exponent, or a smaller one whose finer grid holds them), so the returned float tensor goes through ttnn.as_tensor
/ ttnn.from_torch(dtype=ttnn.bfloat4_b) unchanged: no custom packer and no raw tile bytes are needed.

K-grouped copies (mxfp4_to_bfp4_k): the decode-only expert copies of gpt-oss-20b on 1x4 store every 32x32 tile
transposed, so a BFP4 group is 16 consecutive K values of one output column, inside one MXFP4 block (one scale); the
stream kernels transpose the tiles back on unpack (custom_mm transpose, no decode-time cost; fused_decode/stream.py).

GPT_OSS_EXPERT_BFP4 (read when the model is built) selects the rounding of the fused decode path's BFP4 expert tensors
(expert_bfp4_rounding); the original decode path always keeps ttnn's packer:
  kmse  the default with decode-only expert copies (gpt-oss-20b on 1x4): the copies K-grouped, the prefill copies
        N-grouped, both least-squared-error.
  mse   the default for the ND experts (fused_decode/nd.py, which prefill and decode share): N-grouped,
        least-squared-error.
  max   ttnn's packer.
gpt-oss-20b on 1x4, teacher-forced top-1 / mean KL against the fp32 HF model (4 prompts x 256 decode positions): BFP4
max 89.2% / 0.068, mse 90.8% / 0.048, kmse 92.9% / 0.026; BFP8 decode copies 95.0% / 0.015.
"""

import os

import torch

GROUP = 16  # values per shared exponent in ttnn.bfloat4_b (one row of a 16x16 face)
ENV = "GPT_OSS_EXPERT_BFP4"
# Exponent offsets tried per group: "max" = ttnn's packer, "mse" / "kmse" = the largest exponent or one below it (two
# or three below never win on the gpt-oss experts).
ROUNDINGS = {"max": None, "mse": (0, 1), "kmse": (0, 1)}


def bfp4_round(w, offsets=(0,)):
    """w [..., N] (N % 16 == 0, groups = 16 consecutive values along the last dim) -> float32 values on the bfloat4_b
    grid. Per group the shared exponent is (largest exponent - off) for the off in offsets with the least squared error
    (ties: the earlier offset). offsets=(0,) reproduces ttnn's packer bit for bit."""
    assert w.shape[-1] % GROUP == 0, w.shape
    g = w.float().reshape(*w.shape[:-1], w.shape[-1] // GROUP, GROUP)
    a = g.abs()
    # step0 = 2^(floor(log2 amax) - 2), the grid spacing at the largest exponent; s = |w| in units of step0
    amax = a.amax(-1, keepdim=True)
    step0 = torch.ldexp(torch.ones_like(amax), torch.frexp(amax)[1] - 3)
    s = a.div_(step0)
    best = best_err = None
    for off in offsets:
        t = s * float(2**off) if off else s
        m = torch.round(t).clamp_(max=7)  # torch.round: half to even, as the packer
        if len(offsets) == 1:
            best = m
            break
        d = (t - m).double().square_()  # float64: exact for bf16 / MXFP4 inputs (ties resolve as the numba path)
        err = d.sum(-1, keepdim=True).mul_(4.0**-off)
        if off:
            m.mul_(2.0**-off)  # back to units of step0 (exact: powers of two)
        if best is None:
            best, best_err = m, err
        else:
            take = err < best_err
            best = torch.where(take, m, best)
            best_err = torch.where(take, err, best_err)
    return best.mul_(step0).copysign_(g).masked_fill_(g == 0, 0.0).reshape(w.shape)


def _numba_kernel():
    """numba kernel of bfp4_round for float32 [groups, 16] (same arithmetic, float32 throughout); None without numba."""
    try:
        import numba
        import math

        import numpy as np
    except ImportError:
        return None

    @numba.njit(parallel=True, cache=False, fastmath=False)
    def kernel(x, out, offsets):
        seven = np.float32(7.0)
        for gi in numba.prange(x.shape[0]):
            amax = np.float32(0.0)
            for j in range(16):
                v = abs(x[gi, j])
                if v > amax:
                    amax = v
            if amax == 0:
                for j in range(16):
                    out[gi, j] = 0.0
                continue
            e = math.frexp(amax)[1]
            step0 = np.float32(math.ldexp(1.0, e - 3))
            best_err = np.inf
            best_off = 0
            for oi in range(offsets.shape[0]):
                off = offsets[oi]
                scale = np.float32(math.ldexp(1.0, off))
                err = 0.0  # float64: exact for bf16 / MXFP4 inputs, so ties resolve as in the torch path
                for j in range(16):
                    t = (abs(x[gi, j]) / step0) * scale
                    q = min(np.float32(np.rint(t)), seven)
                    d = np.float64(t - q)
                    err += d * d
                err *= math.ldexp(1.0, -2 * off)
                if oi == 0 or err < best_err:
                    best_err = err
                    best_off = off
            scale = np.float32(math.ldexp(1.0, best_off))
            for j in range(16):
                v = x[gi, j]
                t = (abs(v) / step0) * scale
                q = min(np.float32(np.rint(t)), seven) / scale * step0
                out[gi, j] = 0.0 if (v == 0 or q == 0) else (q if v > 0 else -q)

    return kernel


_KERNEL = []


def _kernel():
    if not _KERNEL:
        _KERNEL.append(_numba_kernel())
    return _KERNEL[0]


def mxfp4_to_bfp4(w, offsets=ROUNDINGS["mse"], chunk=1 << 24):
    """The least-squared-error bfloat4_b rounding of a [..., K, N] weight (the layout handed to ttnn.as_tensor in TILE
    layout; BFP4 groups = 16 consecutive values along the last dim), in w's dtype (the BFP4 values are exact in BF16).
    Uses a numba kernel when numba is importable (~10x faster; same result as bfp4_round for BF16 / MXFP4 inputs),
    else bfp4_round on about `chunk` values at a time (bounded host memory)."""
    import numpy as np

    flat = w.reshape(-1, w.shape[-1])
    out = torch.empty(flat.shape, dtype=w.dtype)
    kernel = _kernel()
    rows = max(1, chunk // w.shape[-1])
    offs = np.asarray(offsets, dtype=np.int64)
    for r in range(0, flat.shape[0], rows):
        part = flat[r : r + rows]
        if kernel is not None:
            x = part.float().contiguous().numpy().reshape(-1, GROUP)
            q = np.empty_like(x)
            kernel(x, q, offs)
            out[r : r + rows] = torch.from_numpy(q).reshape(part.shape)
        else:
            out[r : r + rows] = bfp4_round(part, offsets)
    return out.reshape(w.shape)


def mxfp4_to_bfp4_k(w, offsets=ROUNDINGS["mse"]):
    """As mxfp4_to_bfp4, with the BFP4 groups along K (dim -2) of a [..., K, N] weight: the rounding of the transposed
    tiles of the K-grouped decode copies (fused_decode/stream.py transposed=True)."""
    return mxfp4_to_bfp4(w.transpose(-1, -2), offsets).transpose(-1, -2)


def expert_bfp4_rounding(nd):
    """The rounding of the fused decode path's BFP4 expert tensors: GPT_OSS_EXPERT_BFP4, else "kmse" with decode-only
    expert copies and "mse" for the ND experts (nd: fused_decode/config.py nd_experts_on), which prefill reads too, so
    they cannot be K-grouped ("kmse" means "mse" there)."""
    mode = os.environ.get(ENV, "").strip().lower() or ("mse" if nd else "kmse")
    if mode not in ROUNDINGS:
        raise ValueError(f"{ENV}={mode!r}: expected one of {sorted(ROUNDINGS)}")
    return "mse" if nd and mode == "kmse" else mode


def n_grouped(rounding):
    """The rounding of the N-grouped copies under `rounding` (K-grouping applies to the decode-only copies only)."""
    return "mse" if rounding == "kmse" else rounding


def _is_bfp4(dtype):
    import ttnn

    return dtype == ttnn.bfloat4_b


def prepare_bfp4_expert_weight(w, dtype, rounding="max"):
    """The float weight to hand ttnn.as_tensor(dtype=dtype, layout=TILE) for an N-grouped expert weight w [..., K, N]:
    unchanged unless dtype is bfloat4_b and the rounding is not ttnn's ("max")."""
    if w is None or not _is_bfp4(dtype):
        return w
    offsets = ROUNDINGS[n_grouped(rounding)]
    return w if offsets is None else mxfp4_to_bfp4(w, offsets)


def bfp4_cache_tag(dtype, rounding="max"):
    """Suffix of an expert tensor's cache name: '' for ttnn's rounding (and every non-BFP4 dtype), else '_bfp4<rounding>'
    (rounding: of this tensor, "mse" or "kmse")."""
    if not _is_bfp4(dtype) or ROUNDINGS[rounding] is None:
        return ""
    return f"_bfp4{rounding}"


LAYER_DTYPES_ENV = "GPT_OSS_ND_LAYER_DTYPES"


def nd_layer_dtypes(layer_idx, default):
    """Per-layer override of the ND experts' (gate / up, down) dtypes: GPT_OSS_ND_LAYER_DTYPES="<layers>:<gu>/<dn>[;...]"
    with <layers> a comma list of indices or ranges (e.g. "4,13,22,31:bfp4/bfp4" or "30-35:bfp8/bfp4") and <gu> / <dn>
    in bfp4 | bfp8. Lets BFP8 experts fill the DRAM a QB2 chip has: each gpt-oss-120b layer whose down moves to BFP4
    frees ~0.14 GB per chip at TP=4 (gate / up as well: ~0.41 GB). The fused decode builds one op set per dtype pair
    (ccl.get_decode_expert_nd keys on it); every dtype's tensors keep their own cache names. default: the model-wide
    (gate / up, down) ttnn dtypes."""
    import ttnn

    spec = os.environ.get(LAYER_DTYPES_ENV, "").strip()
    if not spec or layer_idx is None:
        return default
    names = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
    for item in filter(None, spec.split(";")):
        layers, _, pair = item.partition(":")
        gu, _, dn = pair.partition("/")
        idx = set()
        for part in layers.split(","):
            lo, _, hi = part.partition("-")
            idx.update(range(int(lo), int(hi or lo) + 1))
        if layer_idx in idx:
            return names[gu.strip()], names[(dn or gu).strip()]
    return default
