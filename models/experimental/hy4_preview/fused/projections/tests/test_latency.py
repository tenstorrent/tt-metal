# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced latency and DRAM read rate of one chip's projection groups for 8 token rows (fused/projections/program.py):
each program captured as a trace, 20 replays back to back, one synchronize (fused/common/tracing.py), next to the
ttnn form of the same stages (PERF.md 3.2, traced, per user, every chip working):

    pre     latent projections 0.311 ms (q_a, kv_a, norms, RoPE), indexer projections 0.425 ms on full layers, and the
            gate's share of "query and gate projections" 0.119 ms
    query   the q_b share of "query and gate projections" 0.119 ms (the absorption is part of the attention core there)
    post    output projection 0.093 ms (the value side of kv_b is part of the attention core there)

Variants (HY4_LATENCY_VARIANTS, comma separated) show what bounds a program: "noread" (the streamers issue no DRAM
reads), "nomath" (the matmuls are skipped), "timing" (every unit reports its cycles from the release to its last
hand-over), "empty" (every kernel returns at once: the launch floor of the program's shape). Weights are random (the time does not depend on them). On the mock Galaxy the programs are compiled only. Log lines start with HY4_MK7."""

import os

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.common.tracing import traced_ms
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.projections.prepare import placeholder_share
from models.experimental.hy4_preview.fused.projections.tests.harness import FID_KW, FID_TAG
from models.experimental.hy4_preview.fused.projections.tests.test_pre import FORMATS

TRACE_REGION = 64 << 20
CYCLE_NS = 1e3 / 1350  # the wall clock counts AICLK cycles (1350 MHz)
VARIANTS = {
    "noread": ("HY4_NO_READ", "1"),
    "nomath": ("HY4_NO_MATH", "1"),
    "timing": ("HY4_TIMING", "1"),
    "empty": ("HY4_EMPTY", "1"),
}
TTNN_US = {
    "pre": "latent 311 + gate's part of 119 (+ indexer 425 on full layers), per user",
    "query": "q_b's part of 119, per user",
    "post": "output projection 93, per user",
}
# name: (group, chip, full layer, builder kwargs)
CASES = {
    "pre": ("pre", 0, False, {}),
    "pre_full": ("pre", 0, True, {}),
    "pre_gate1": ("pre", 0, False, {"gate_ct": 1}),
    "pre_gate4": ("pre", 0, False, {"gate_ct": 4}),
    "query": ("query", 0, False, {}),
    "query_n1a2": ("query", 0, False, {"nope_ct": 1, "absorb_ct": 2}),
    "query_n6a8": ("query", 0, False, {"nope_ct": 6, "absorb_ct": 8}),
    "post": ("post", 0, False, {}),
    "post_o2": ("post", 0, False, {"o_ct": 2}),
    "post_o8": ("post", 0, False, {"o_ct": 8}),
}
BUILD = {"pre": P.pre_program, "query": P.query_program, "post": P.post_program}
_cases = [c for c in os.environ.get("HY4_LATENCY_CASES", "pre,pre_full,query,post").split(",") if c]
_variants = [v for v in os.environ.get("HY4_LATENCY_VARIANTS", "").split(",") if v]
_formats = [f for f in os.environ.get("HY4_LATENCY_FORMATS", "bfp8,bfp4x2,bfp4").split(",") if f]
# calls of the program per trace: 1 = the stage as a trace of its own (every replay sends the kernel binaries again,
# what PERF.md's per-stage numbers are); more = what it costs inside a long trace
_calls = int(os.environ.get("HY4_LATENCY_CALLS", "1"))


def random_share(chip, full, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda t: torch.randn(t.shape, generator=g) * t.shape[0] ** -0.5
    w = placeholder_share(chip, full)
    return {k: [rnd(m) for m in v] if isinstance(v, list) else rnd(v) for k, v in w.items()}


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("fmt", _formats)
@pytest.mark.parametrize("case", _cases)
def test_latency(device, case, fmt):
    group, chip, full, kwargs = CASES[case]
    f, terms = FORMATS[fmt]
    w = random_share(chip, full)
    g = torch.Generator().manual_seed(1)
    # the variants on the default shapes with BFP8 only
    variants = _variants if fmt == "bfp8" and case in ("pre", "pre_full", "query", "post") else []
    for variant in [()] + [(v,) for v in variants]:
        op = BUILD[group](device, w, fmt=f, terms=terms, defines=[VARIANTS[v] for v in variant], **kwargs, **FID_KW)
        op.set_x(torch.randn(L.ROWS, L.TILE * op.x_tiles, generator=g))
        op.set_params(lanes=torch.full((L.ROWS, 8), 256.0))
        if P.MOCK:  # compile only (this also fills the kernel cache the board run uses)
            op()
            continue
        t = traced_ms(device, op, calls=_calls)
        _, _, status = op.read()
        us = t["replay_ms"] * 1e3
        tag = f"{case} {fmt}{FID_TAG}" + "".join(f" {v}" for v in variant)
        logger.info(
            f"HY4_MK7 latency {tag} ({_calls} per trace): {us:.1f} us per call (rounds {[round(v * 1e3, 1) for v in t['rounds_ms']]}), "
            f"{len(op.units)} units, {op.weight_bytes / 1e6:.2f} MB of weights, "
            f"{op.weight_bytes / (us * 1e-6) / 1e9:.0f} GB/s over the whole program; ttnn form: {TTNN_US[group]} us"
        )
        op.check_status(status)
        if "timing" in variant:
            cyc = op.unit_cycles(status)
            by_kind = {}
            for u, c in zip(op.units, cyc):
                by_kind.setdefault(u.name.split(".")[-1].rstrip("0123456789"), []).append(c * CYCLE_NS / 1e3)
            logger.info(
                f"HY4_MK7 timing {tag}: hub release sent after {status[8] * CYCLE_NS / 1e3:.1f} us; units, us from "
                f"their release to their last hand-over (min / max): "
                f"{ {k: (round(min(v), 1), round(max(v), 1)) for k, v in by_kind.items()} }"
            )
