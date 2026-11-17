# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stride-2 subsample variants for the PDL res3.0 / res4.0 1x1 shortcuts on genify_matmul (#65150).

The input is the block input as the model holds it: [1, H, W, C] bfp8 TILE in DRAM. Each variant must
return exactly x[:, ::2, ::2, :] as a TILE tensor; run under `--profile` to compare device times
(signposts SUBSAMPLE_<variant>_<stage>).

  rows   : the even image rows as a tile-aligned slice of a [1, H/2, 2W, C] view (bfp8, TILE)
  v1     : rows -> untilize -> strided row-major slice (even columns) -> tilize (bf16)
  v2     : rows -> untilize -> row-major reshape [.., W/2, 2C] -> slice [..., :C] -> tilize (bf16)
  v3     : rows -> selection matmul on genify_matmul: S [1, H/2, W/2, W] @ rows [1, H/2, W, C] (batched B,
           S[j, 2j] = 1, HiFi4 + fp32 DEST, so every output is one exact input value; bfp8 out)
  v4     : rows -> TILE reshape [.., W/2, 2C] -> tile-aligned slice [..., :C] (bfp8)
  v3h2   : v3 at HiFi2 (S = 1.0 sits in SrcB, whose top bits LoFi/HiFi2 already use whole; x in SrcA needs 2 phases)
  v3g    : v3h2 on groups of 64 pixels: S' [1, H/2*W/64, 32, 64] @ rows viewed [1, H/2*W/64, 64, C] (K 64, not W)
  v3gl   : v3g at LoFi (expected inexact: LoFi drops SrcA's low mantissa bits)

    scripts/run_safe_pytest.sh --profile models/experimental/op_genify/integration/pdl_test_subsample.py -s
"""

import os

import pytest
import torch
import ttnn

from models.experimental.op_genify import genify_matmul_adapter as genify

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_args, **_kwargs):
        pass


L1 = ttnn.L1_MEMORY_CONFIG


def rows(x, n, h, w, c):
    xv = ttnn.reshape(x, (n, h // 2, 2 * w, c))
    return ttnn.slice(xv, [0, 0, 0, 0], [n, h // 2, w, c], memory_config=L1)


def v1(xr, n, h, w, c):
    x_rm = ttnn.to_layout(xr, ttnn.ROW_MAJOR_LAYOUT, memory_config=L1)
    xs = ttnn.slice(x_rm, [0, 0, 0, 0], [n, h // 2, w, c], [1, 1, 2, 1], memory_config=L1)
    return ttnn.to_layout(xs, ttnn.TILE_LAYOUT, memory_config=L1)


def v2(xr, n, h, w, c):
    x_rm = ttnn.to_layout(xr, ttnn.ROW_MAJOR_LAYOUT, memory_config=L1)
    x_pairs = ttnn.reshape(x_rm, (n, h // 2, w // 2, 2 * c))
    xs = ttnn.slice(x_pairs, [0, 0, 0, 0], [n, h // 2, w // 2, c], memory_config=L1)
    return ttnn.to_layout(xs, ttnn.TILE_LAYOUT, memory_config=L1)


def selection(device, n, h, w):
    s = torch.zeros(w // 2, w)
    s[torch.arange(w // 2), 2 * torch.arange(w // 2)] = 1.0
    s = s.expand(n, h // 2, w // 2, w).contiguous()
    return ttnn.from_torch(
        s, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def cfg(fidelity):
    return ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False)


EXACT = cfg(ttnn.MathFidelity.HiFi4)


def v3(xr, s, compute=EXACT):
    return genify.matmul(s, xr, dtype=ttnn.bfloat8_b, memory_config=L1, compute_kernel_config=compute)


def grouped_selection(device, n, h, w, group=64):
    s = torch.zeros(group // 2, group)
    s[torch.arange(group // 2), 2 * torch.arange(group // 2)] = 1.0
    s = s.expand(n, (h // 2) * (w // group), group // 2, group).contiguous()
    return ttnn.from_torch(
        s, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def v3g(xr, sg, n, h, w, c, fidelity, group=64):
    xg = ttnn.reshape(xr, (n, (h // 2) * (w // group), group, c))
    out = genify.matmul(sg, xg, dtype=ttnn.bfloat8_b, memory_config=L1, compute_kernel_config=cfg(fidelity))
    return ttnn.reshape(out, (n, h // 2, w // 2, c))


def v4(xr, n, h, w, c):
    x_pairs = ttnn.reshape(xr, (n, h // 2, w // 2, 2 * c))
    return ttnn.slice(x_pairs, [0, 0, 0, 0], [n, h // 2, w // 2, c], memory_config=L1)


@pytest.mark.parametrize("h, w, c", [(128, 256, 256), (64, 128, 512)], ids=["res3", "res4"])
def test_subsample_variants(device, h, w, c):
    torch.manual_seed(0)
    n = 1
    xt = torch.randn(n, h, w, c)
    x = ttnn.from_torch(
        xt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ref = ttnn.to_torch(x)[:, ::2, ::2, :]  # the bfp8 values themselves
    s = selection(device, n, h, w)
    sg = grouped_selection(device, n, h, w)
    only = os.environ.get("PDL_SUBSAMPLE_VARIANTS")  # e.g. "v3,v3h2,v3g,v3gl"
    variants = {
        "v1": lambda xr: v1(xr, n, h, w, c),
        "v2": lambda xr: v2(xr, n, h, w, c),
        "v3": lambda xr: v3(xr, s),
        "v4": lambda xr: v4(xr, n, h, w, c),
        "v3h2": lambda xr: v3(xr, s, cfg(ttnn.MathFidelity.HiFi2)),
        "v3g": lambda xr: v3g(xr, sg, n, h, w, c, ttnn.MathFidelity.HiFi2),
        "v3gl": lambda xr: v3g(xr, sg, n, h, w, c, ttnn.MathFidelity.LoFi),
    }
    if only:
        variants = {k: v for k, v in variants.items() if k in only.split(",")}
    results = {}
    for name, fn in variants.items():
        try:
            for it in range(2):  # 2nd iteration: compiled, cached
                signpost(f"SUBSAMPLE_{name}_{it}_START")
                xr = rows(x, n, h, w, c)
                out = fn(xr)
                ttnn.synchronize_device(device)
                signpost(f"SUBSAMPLE_{name}_{it}_END")
            got = ttnn.to_torch(out).reshape(ref.shape).float()
            results[name] = (got == ref.float()).all().item(), (got - ref.float()).abs().max().item(), out.dtype
        except Exception as e:  # noqa: BLE001 - a variant the ops reject is a result too
            results[name] = (False, f"error: {str(e).splitlines()[0][:200]}", None)
    for name, r in results.items():
        print(f"SUBSAMPLE {h}x{w}x{c} {name}: exact={r[0]} maxdiff={r[1]} dtype={r[2]}")
    assert results[next(iter(variants))][0], results
