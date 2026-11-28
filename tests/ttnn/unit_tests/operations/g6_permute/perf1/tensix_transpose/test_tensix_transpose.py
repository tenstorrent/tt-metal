# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the tensix_transpose experiment (perf_experiments/tensix_transpose): bit-exact vs
torch.permute for every bf16 bit pattern, per variant and CB format, plus the regime's other shapes."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[7]
_spec = importlib.util.spec_from_file_location(
    "tensix_transpose_exp",
    ROOT / "ttnn/ttnn/operations/g6_permute/perf_experiments/tensix_transpose/tensix_transpose.py",
)
tt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tt
_spec.loader.exec_module(tt)

NHWC = (0, 3, 1, 2)
MEM = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}


def _bits(t):
    return t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _classify(b):
    exp = (b >> 7) & 0xFF
    man = b & 0x7F
    if bool(exp == 0xFF) and bool(man != 0):
        return "nan"
    if bool(exp == 0xFF):
        return "inf"
    if bool(exp == 0) and bool(man != 0):
        return "denormal"
    if bool(b == 0x8000):
        return "-0"
    if bool(b == 0):
        return "+0"
    return "normal"


def _all_patterns(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    numel = 1
    for d in shape:
        numel *= d
    reps = (numel + 65535) // 65536
    pats = torch.cat([torch.randperm(65536, generator=g) for _ in range(reps)])[:numel]
    pats = torch.where(pats >= 0x8000, pats - 0x10000, pats).to(torch.int16)
    return pats.view(torch.bfloat16).reshape(shape)


def _run(x_torch, device, buffer, cfg):
    x = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=MEM[buffer]
    )
    y = tt.permute(x, cfg)
    return ttnn.to_torch(y)


def _check(actual, x_torch, label=""):
    expected = x_torch.permute(*NHWC).contiguous()
    assert list(actual.shape) == list(expected.shape)
    a, e = _bits(actual), _bits(expected)
    bad = a != e
    n_bad = int(bad.sum())
    if n_bad:
        classes = {}
        for ev, av in zip(e[bad][:20000].tolist(), a[bad][:20000].tolist()):
            k = _classify(torch.tensor(ev))
            classes.setdefault(k, []).append((hex(ev), hex(av)))
        summary = {k: (len(v), v[:4]) for k, v in classes.items()}
        print(f"\nBITCHECK {label} mismatches={n_bad}/{bad.numel()} by class: {summary}")
    else:
        print(f"\nBITCHECK {label} bit-exact ({bad.numel()} values)")
    assert n_bad == 0


@pytest.mark.parametrize("fmt", ["uint16", "bf16"])
@pytest.mark.parametrize("variant", ["two_pass", "fused", "fused_raw", "fused_d2b"])
def test_all_bit_patterns(variant, fmt, device):
    tt._CACHE.clear()
    x = _all_patterns((1, 2, 1024, 32))
    cfg = tt.Cfg(variant=variant, fmt=fmt)
    _check(_run(x, device, "l1", cfg), x, f"{variant}/{fmt}")


CELL_SHAPES = [
    ("p44", (1, 128, 256, 32), "l1"),
    ("p44_dram", (1, 128, 256, 32), "dram"),
    ("odd_c_w_tail", (1, 3, 100, 3), "dram"),
    ("c5_l1", (1, 4, 40, 5), "l1"),
    ("c1", (1, 2, 64, 1), "dram"),
    ("w5_whole_rows", (2, 7, 5, 32), "l1"),
    ("c64_long_page", (1, 8, 48, 64), "dram"),
    ("batch2_tile_aligned", (2, 32, 64, 32), "l1"),
    ("h_non_aligned_c64", (1, 5, 40, 64), "dram"),
    ("w_non_aligned", (1, 6, 33, 70), "l1"),
    ("one_long_row", (1, 1, 1000, 16), "dram"),
    ("many_rows", (1, 200, 96, 24), "l1"),
    ("w_tails_1", (1, 4, 65, 32), "l1"),
    ("w_tails_4", (1, 4, 68, 32), "l1"),
    ("w_tails_8", (1, 4, 72, 32), "dram"),
    ("w_tails_16", (1, 4, 80, 32), "l1"),
    ("c128_wide", (1, 16, 128, 128), "l1"),
    ("special_values", (1, 6, 40, 32), "l1"),
]


@pytest.mark.parametrize(
    "variant,fmt,trid,reader",
    [
        ("two_pass", "uint16", False, "own"),
        ("two_pass", "uint16", True, "own"),
        ("fused_d2b", "uint16", True, "own"),
        ("fused_d2b", "uint16", False, "op"),
        ("fused_d2b", "uint16", True, "auto"),
        ("two_pass", "uint16", True, "auto"),
    ],
)
@pytest.mark.parametrize("name,shape,buffer", CELL_SHAPES, ids=[c[0] for c in CELL_SHAPES])
def test_cell_shapes(name, shape, buffer, variant, fmt, trid, reader, device):
    tt._CACHE.clear()
    x = _all_patterns(shape, seed=hash(name) & 0xFFFF)
    cfg = tt.Cfg(variant=variant, fmt=fmt, use_trid=trid, reader=reader)
    _check(_run(x, device, buffer, cfg), x, f"{name}/{variant}/{fmt}/trid={trid}/reader={reader}")
    print(f"PLAN {name} block_w={tt._CACHE[next(iter(tt._CACHE))][0].block_w}")


# Forced unit widths whose row tail is narrower in w-tiles than the unit (kw_tail < kw): the compute
# switches pack_untilize width per unit and the reader pads the tail unit's CB pages.
FORCED = [
    ("kw2_tail1", (1, 4, 80, 32), "l1", 2),  # block_w 64, tail 16 (kw_tail 1)
    ("kw4_tail3_dram", (1, 3, 200, 32), "dram", 4),  # block_w 128, tail 72 (kw_tail 3)
    ("kw4_tail1_c70", (1, 2, 130, 70), "l1", 4),  # block_w 128, tail 2 (kw_tail 1), Ct 3
    ("kw2_tail1_dram_c64", (2, 3, 97, 64), "dram", 2),  # block_w 64, tail 33 (kw_tail 2 = kw) + N=2
]


@pytest.mark.parametrize(
    "variant,trid,reader",
    [("fused_d2b", True, "own"), ("fused_d2b", False, "op"), ("two_pass", True, "own"), ("fused_d2b", True, "auto")],
)
@pytest.mark.parametrize("name,shape,buffer,kw", FORCED, ids=[c[0] for c in FORCED])
def test_forced_tails(name, shape, buffer, kw, variant, trid, reader, device):
    tt._CACHE.clear()
    x = _all_patterns(shape, seed=7)
    cfg = tt.Cfg(variant=variant, fmt="uint16", use_trid=trid, reader=reader, kw=kw)
    _check(_run(x, device, buffer, cfg), x, f"{name}/{variant}/trid={trid}/reader={reader}")
