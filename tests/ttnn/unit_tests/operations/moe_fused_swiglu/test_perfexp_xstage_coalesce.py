# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin pytest entry point for the xstage_coalesce isolated bake-off (blocking-perf-part-optimizer).

ALL logic (program descriptors, kernels, variants) lives under the experiment dir:
    ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/xstage_coalesce/

This file only exists here (rather than inside that dir) because pytest's
--import-mode=importlib cannot collect a test_*.py placed directly inside the `ttnn/ttnn/`
package tree: it derives a dotted module path starting with "ttnn" and ends up re-executing
ttnn/ttnn/__init__.py under a second qualified name ("ttnn.ttnn"), which crashes on duplicate C++
op registration ("bernoulli" already registered). Every real test in this repo lives under
`tests/...` for the same reason. This file is named uniquely for this idea (xstage_coalesce) so it
cannot collide with a sibling part-optimizer's own escape-hatch entry point.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_xstage_coalesce.py::test_xstage_correctness
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_xstage_coalesce.py::test_xstage_perf_focus
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_xstage_coalesce.py::test_xstage_perf_sweep
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_xstage_coalesce.py::test_rateprobe_perf
"""

import torch
import pytest
from loguru import logger

import ttnn
from ttnn.operations.moe_fused_swiglu.perf_experiments.xstage_coalesce import xstage_bench as xb

from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Correctness — every graduatable variant must match ttnn's own tilize of the same slice, AND
# match VARIANT 0 bit-for-bit (same bytes, same tilize, only the read path differs).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", xb.CORRECTNESS_GATED)
def test_xstage_correctness(device, variant):
    emb, kr_pad, kstart_tiles = 7168, 23, 0
    torch_x, result = xb.run_variant(device, variant, emb=emb, kr_pad=kr_pad, kstart_tiles=kstart_tiles)
    got = ttnn.to_torch(result).to(torch.float32)
    expected = xb.reference_tiles(torch_x, kstart_tiles, kr_pad)
    assert list(got.shape) == list(expected.shape), f"{got.shape} != {expected.shape}"
    assert_with_pcc(expected, got, 0.999)


def test_xstage_correctness_nonzero_kstart(device):
    """Addressing sanity check: kstart_tiles > 0 (a mid-row column offset) on the baseline path."""
    emb, kr_pad, kstart_tiles = 7168, 10, 5
    torch_x, result = xb.run_variant(device, 0, emb=emb, kr_pad=kr_pad, kstart_tiles=kstart_tiles)
    got = ttnn.to_torch(result).to(torch.float32)
    expected = xb.reference_tiles(torch_x, kstart_tiles, kr_pad)
    assert_with_pcc(expected, got, 0.999)


def test_xstage_variants_agree_bit_exact(device):
    """The sharper check: every RM variant reads the SAME bytes through the SAME tilize, so their
    outputs must be bit-identical to VARIANT 0 (mirrors this op's own 'scheduling-only change ->
    bit-identical output' test philosophy, e.g. test_moe_fused_swiglu_r3_residency.py)."""
    emb, kr_pad, kstart_tiles = 7168, 23, 0
    _, ref = xb.run_variant(device, 0, emb=emb, kr_pad=kr_pad, kstart_tiles=kstart_tiles, seed=1)
    ref_t = ttnn.to_torch(ref)
    for variant in (1, 2, 3):
        _, out = xb.run_variant(device, variant, emb=emb, kr_pad=kr_pad, kstart_tiles=kstart_tiles, seed=1)
        out_t = ttnn.to_torch(out)
        assert torch.equal(ref_t, out_t), f"variant {xb.VARIANTS[variant]} diverged from baseline bytes"


# ---------------------------------------------------------------------------
# Perf — one fresh-cache run per variant, focus shape first, then the predicate sweep.
# ---------------------------------------------------------------------------
def test_xstage_perf_focus(device):
    """FOCUS SHAPE: emb=7168, KR_PAD=23 (the count-256 injector's real shape)."""
    emb, kr_pad = 7168, 23
    ns = {}
    for variant in sorted(xb.VARIANTS):
        run_fn = lambda v=variant: xb.run_variant(device, v, emb=emb, kr_pad=kr_pad)
        ns[variant] = xb.measure_once(device, run_fn)
    lines = ["", "=== xstage_coalesce focus (emb=7168, KR_PAD=23) ==="]
    for variant in sorted(xb.VARIANTS):
        lines.append(f"  {xb.VARIANTS[variant]:<24} {ns[variant]:>10.1f} ns")
    logger.info("\n".join(lines))


@pytest.mark.parametrize(
    "emb,kr_pad",
    [(7168, 23), (7168, 22), (6144, 20), (6144, 19)],
)
def test_xstage_perf_sweep(device, emb, kr_pad):
    """PREDICATE SWEEP over KR_PAD (emb 7168: 23/22; emb 6144: 20/19), both input formats."""
    ns = {}
    for variant in sorted(xb.VARIANTS):
        run_fn = lambda v=variant: xb.run_variant(device, v, emb=emb, kr_pad=kr_pad)
        ns[variant] = xb.measure_once(device, run_fn)
    lines = [f"", f"=== xstage_coalesce sweep emb={emb} KR_PAD={kr_pad} ==="]
    for variant in sorted(xb.VARIANTS):
        lines.append(f"  {xb.VARIANTS[variant]:<24} {ns[variant]:>10.1f} ns")
    logger.info("\n".join(lines))


def test_rateprobe_perf(device):
    """Sweep transaction COUNT at FIXED 1472B (KR_PAD=23) transaction size: fit ns ~= a + b*N and
    report b (measured ns/transaction) against the ~110-125 ns single-core floor."""
    emb, kr_pad = 7168, 23
    _, tt_x = xb.make_x_bf16_rm(device, emb)
    counts = (1, 2, 4, 8, 16, 24, 32)
    ns = {}
    for n in counts:
        run_fn = lambda nn=n: xb.run_rateprobe(device, tt_x, kr_pad=kr_pad, num_reads=nn)
        ns[n] = xb.measure_once(device, run_fn)

    lines = ["", "=== xstage_coalesce rate probe (1472 B sub-page reads, fixed size, count sweep) ==="]
    prev = None
    prev_n = None
    for n in counts:
        delta = "" if prev is None else f"  (+{(ns[n] - prev):.1f} ns for +{n - prev_n} reads)"
        lines.append(f"  N={n:>3}  {ns[n]:>9.1f} ns{delta}")
        prev, prev_n = ns[n], n
    slope = (ns[counts[-1]] - ns[counts[0]]) / (counts[-1] - counts[0])
    lines.append(f"  slope (N={counts[0]}->{counts[-1]}): {slope:.1f} ns/transaction")
    logger.info("\n".join(lines))
