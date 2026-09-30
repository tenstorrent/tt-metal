# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off for perf_experiments/column_stream (column-granular reader -> compute -> writer handoff).

test_cs_correct: baseline vs candidate(s) on the same inputs, bit-exact, plus torch allclose.
test_cs_perf:    one op per test (profiler CSV row order == test order; CS_LABELS=1 logs the labels).
Variant sets are picked with CS_VARIANTS (comma list of names in VARIANTS), shapes with CS_SHAPES (FOCUS/SWEEP/ALL).
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.mhc_post.perf_experiments.column_stream import bench

BF, FP = ttnn.bfloat16, ttnn.float32

VARIANTS = {
    "baseline": "baseline",
    "g1": {"G": 1},  # G=1, in flight B columns, caps 2B
    "g2": {"G": 2},
    "g4": {"G": 4},
    "g1_if2b": {"G": 1, "INFLIGHT_COLS": 16},  # in flight up to 2B / 16 groups
    "g2_if2b": {"G": 2, "INFLIGHT_COLS": 16},
    "g1_bar": {"G": 1, "WRITER_FLUSH": False},
    "g8": {"G": 8},  # == B at C7168: handoff not refined, isolates writer flush + trid depth
    "g1_if12": {"G": 1, "INFLIGHT_COLS": 12},
    "g1_if6": {"G": 1, "INFLIGHT_COLS": 6},
    "g2_if12": {"G": 2, "INFLIGHT_COLS": 12},
    "g1_cap3b": {"G": 1, "INFLIGHT_COLS": 16, "IN_CAP_COLS": 24},
    "g1_if4": {"G": 1, "INFLIGHT_COLS": 4},
    "g1_if2": {"G": 1, "INFLIGHT_COLS": 2},
    "g1_w8": {"G": 1, "WG": 8},  # writer window 8 columns (= B at C7168)
    "g1_w8_bar": {"G": 1, "WG": 8, "WRITER_FLUSH": False},  # writer behaves like the op's
    "g1_w4": {"G": 1, "WG": 4},
    "g2_w8": {"G": 2, "WG": 8},
    "g4_w8": {"G": 4, "WG": 8},
    "g1_nopoll": {"G": 1, "READER_DEFINES": ("CS_RESERVE_NOPOLL",)},
    "g1_rb8": {"G": 1, "RB": 8},  # op's request order + burst timing, per-column trid push
    "g1_rb8_if15": {"G": 1, "RB": 8, "INFLIGHT_COLS": 15},
    "g1_rb4": {"G": 1, "RB": 4},  # two 4-column batches in flight (B = 8)
    "g2_rb8": {"G": 2, "RB": 8},
    "g1_rb8_w8": {"G": 1, "RB": 8, "WG": 8},
    "cs_b": {"G": 1, "RB": "B", "WG": "B"},  # DM keeps the op's B-column NoC batches; compute handoff per column
    "cs_b_t4": {"G": 1, "RB": "B", "WG": "B", "TAIL_COLS": 4},
    "cs_b_t8": {"G": 1, "RB": "B", "WG": "B", "TAIL_COLS": 8},
    "cs_b_t16": {"G": 1, "RB": "B", "WG": "B", "TAIL_COLS": 16},
    "cs_b_if15": {"G": 1, "RB": "B", "WG": "B", "INFLIGHT_COLS": 15},
    "cs_b2": {"G": 2, "RB": "B", "WG": "B"},
    "cs_b4": {"G": 4, "RB": "B", "WG": "B"},
    # ablations (output garbage: perf test skips the check for these)
    "g1_nodm": {"G": 1, "STUB_DM": True},
    "g8_nodm": {"G": 8, "STUB_DM": True},
    "g1_nocmp": {"G": 1, "STUB_COMPUTE": True},
    "g8_nocmp": {"G": 8, "STUB_COMPUTE": True},
    "g2_nodm": {"G": 2, "STUB_DM": True},
    "g4_nodm": {"G": 4, "STUB_DM": True},
}

# (T, C, n, x_dtype, f_dtype)
FOCUS = [
    (640, 7168, 4, BF, BF),
    (640, 1792, 4, BF, BF),
    (1280, 4096, 4, BF, BF),
]
SWEEP = [
    (640, 7168, 4, FP, FP),
    (640, 1792, 4, FP, FP),
    (640, 7168, 4, FP, BF),
    (640, 1792, 4, FP, BF),
    (1000, 1792, 4, BF, BF),
    (1000, 7168, 4, BF, BF),
    (32, 32, 4, BF, BF),
    (17, 128, 4, FP, FP),
    (640, 1792, 1, BF, BF),
    (640, 1792, 2, BF, BF),
    (640, 1792, 3, FP, FP),
    (640, 1792, 5, BF, BF),
    (320, 7168, 2, BF, BF),
    (96, 800, 4, BF, BF),  # 3 rows x 25 cols over 75 cores... ragged, row-straddling
    (130, 1344, 5, FP, BF),  # ragged T, 42 cols, n=5 grouped, row-straddling cores
]
PROBE = [
    (640, 7200, 4, BF, BF),  # Ct = 225 (odd): the n X tiles + F tile of one column land in different DRAM banks
    (640, 1824, 4, BF, BF),  # Ct = 57
]
FP32_PROBE = [
    (640, 1792, 4, FP, FP),
    (640, 1792, 4, FP, BF),
    (640, 1792, 3, FP, FP),
    (1000, 1792, 4, FP, FP),
    (640, 4096, 4, FP, FP),
]
_SHAPES = {"FP32_PROBE": FP32_PROBE, "FOCUS": FOCUS, "SWEEP": SWEEP, "ALL": FOCUS + SWEEP, "PROBE": PROBE}[
    os.environ.get("CS_SHAPES", "ALL")
]
_VARS = os.environ.get("CS_VARIANTS", "baseline,g1,g2,g4").split(",")


def _sid(s):
    T, C, n, xd, fd = s
    return f"T{T}_C{C}_n{n}_X{'bf16' if xd == BF else 'fp32'}_F{'bf16' if fd == BF else 'fp32'}"


@pytest.mark.parametrize("shape", _SHAPES, ids=[_sid(s) for s in _SHAPES])
def test_cs_correct(device, shape):
    T, C, n, xd, fd = shape
    host = bench.make_inputs(T, C, n, xd, fd)
    tensors = bench.to_device(device, *host, xd, fd)
    ref = bench.reference(*host, n, xd, fd)
    outs, build_failed = {}, {}
    for name in _VARS:
        try:
            outs[name] = ttnn.to_torch(bench.run(device, VARIANTS[name], tensors)).float()
        except RuntimeError as e:  # kernel build failure (host side); a hang would not return here
            if "TIMEOUT" in str(e):
                raise
            build_failed[name] = str(e).splitlines()[0][:160]
    print(f"\nCS_CORRECT {_sid(shape)} ran={list(outs)} build_failed={build_failed}")
    tol = 1e-4 if xd == FP else 2e-2
    for name, got in outs.items():
        assert torch.allclose(got, ref, rtol=tol, atol=tol), name
    ref_name = next(iter(outs))
    for name, got in outs.items():
        assert torch.equal(
            got, outs[ref_name]
        ), f"{name} not bit-exact vs {ref_name}: max diff {(got - outs[ref_name]).abs().max()}"
    # A candidate may not fail where the baseline builds.
    if "baseline" in outs:
        assert not build_failed, f"candidate build failures where the baseline builds: {build_failed}"


@pytest.mark.parametrize("variant", _VARS)
@pytest.mark.parametrize("shape", _SHAPES, ids=[_sid(s) for s in _SHAPES])
def test_cs_perf(device, shape, variant):
    T, C, n, xd, fd = shape
    host = bench.make_inputs(T, C, n, xd, fd)
    tensors = bench.to_device(device, *host, xd, fd)
    got = ttnn.to_torch(bench.run(device, VARIANTS[variant], tensors, label=f"{_sid(shape)}|{variant}")).float()
    if isinstance(VARIANTS[variant], dict) and (
        VARIANTS[variant].get("STUB_DM") or VARIANTS[variant].get("STUB_COMPUTE")
    ):
        return
    ref = bench.reference(*host, n, xd, fd)
    tol = 1e-4 if xd == FP else 2e-2
    assert torch.allclose(got, ref, rtol=tol, atol=tol)
