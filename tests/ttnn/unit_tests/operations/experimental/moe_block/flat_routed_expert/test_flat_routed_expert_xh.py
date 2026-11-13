# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flat routed expert's x / h data formats (FlatLayoutOptions.x_bf16 / h_bf16: x after the tilize, h after
gate/up, bfp8 by default) on one Blackhole chip, each (x, h) combination on the same inputs and weights:

* test_flat_expert_xh_accuracy: every routed row against a torch reference of the bfp4-quantized weights - the worst
  expert's PCC and the largest absolute difference (XH_ACCURACY lines).
* test_flat_expert_xh_perf: the op's device time (real-time profiler), WARMUP calls first, then the median of RUNS
  calls (XH_PERF lines). No baseline: the lines compare the combinations.

Cases and inputs: test_flat_routed_expert_checks.py (the same routings / weights)."""

import importlib.util
import statistics
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chk = _load("test_flat_routed_expert_checks", HERE / "test_flat_routed_expert_checks.py")
cases, weights_py = chk.cases, chk.weights_py

WARMUP = 25  # (at least 20 warm-up calls before any measured one)
RUNS = 10
XH = [(False, False), (True, False), (False, True), (True, True)]
CASES = [
    chk.FlatCase("glm53", "8x4", 5120, "uniform"),
    chk.FlatCase("k3", "8x4", 5120, "uniform"),
    chk.FlatCase("dsv41_flash", "8x4", 5120, "uniform"),
]


def _xh_id(x, h):
    return f"x{'bf16' if x else 'bfp8'}-h{'bf16' if h else 'bfp8'}"


PARAMS = pytest.mark.parametrize(
    "case, xh", [pytest.param(c, xh, id=f"{c.id}-{_xh_id(*xh)}") for c in CASES for xh in XH]
)


def _setup(case, device, x_bf16, h_bf16):
    md = cases.MODELS[case.model]
    x, c_, r_, gids, spans = chk._inputs(case)
    wsets = cases.weight_sets(md["H"], md["I"])
    op = weights_py.FlatRoutedExpert(
        device,
        [[wsets[e % cases.N_W] for e in range(len(gids))]],
        m=case.chunk,
        H=md["H"],
        I=md["I"],
        gids=[gids],
        n_global=md["E"],
        act=md["act"],
        pin=1,
        layout={"x_bf16": x_bf16, "h_bf16": h_bf16},
    )
    args = (chk._rm(x, ttnn.bfloat16, device), chk._rm(c_, ttnn.uint32, device), chk._rm(r_, ttnn.uint32, device))
    return op, args, (x, spans), wsets


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", chk.DEVICE_PARAMS, indirect=True)
@PARAMS
def test_flat_expert_xh_accuracy(device, case, xh):
    chk._skip_unless_supported(device)
    md = cases.MODELS[case.model]
    op, args, (x, spans), wsets = _setup(case, device, *xh)
    logger.info(f"{case.id} {_xh_id(*xh)}: plan {op.layout}")
    y = ttnn.to_torch(op(*args)).float()
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa: E731
    wq = [tuple(q(w) for w in ws) for ws in wsets]
    refs, gots, worst = [], [], (1.0, None)
    for e, (o, c) in enumerate(spans):
        if not c:
            continue
        xe = x[o : o + c].to(torch.bfloat16).float()
        wg, wu, wd = wq[e % cases.N_W]
        ref = weights_py.act_ref(xe @ wg, xe @ wu, md["act"]) @ wd
        got = y[o : o + c]
        assert torch.isfinite(got).all(), f"{case.id} {_xh_id(*xh)}: expert {e} non-finite"
        if c > 1:
            _, pcc = comp_pcc(ref, got, chk.PCC)
            worst = min(worst, (float(pcc), e))
        refs.append(ref)
        gots.append(got)
    ref, got = torch.cat(refs), torch.cat(gots)
    _, pcc_all = comp_pcc(ref, got, chk.PCC)
    diff = (ref - got).abs()
    logger.info(
        f"XH_ACCURACY {case.id} {_xh_id(*xh)}: pcc {float(pcc_all):.6f} (worst expert {worst[1]} {worst[0]:.6f}), "
        f"max abs diff {diff.max().item():.5f}, mean abs diff {diff.mean().item():.6f} over {got.shape[0]} rows"
    )
    assert worst[0] >= chk.PCC, f"{case.id} {_xh_id(*xh)}: expert {worst[1]} pcc {worst[0]:.5f} < {chk.PCC}"
    op.close()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", chk.DEVICE_PARAMS, indirect=True)
@PARAMS
def test_flat_expert_xh_perf(device, case, xh):
    chk._skip_unless_supported(device)
    op, args, _, _ = _setup(case, device, *xh)
    for _ in range(WARMUP):  # (warm-up: program build, then steady state)
        ttnn.deallocate(op(*args))
    ttnn.synchronize_device(device)

    def run():
        for _ in range(RUNS):
            ttnn.deallocate(op(*args))

    t = chk.op_device_times(device, run, chk.FLAT_OPS)["flat_routed_expert"][-RUNS:]
    assert len(t) == RUNS, f"{case.id}: {len(t)} flat_routed_expert programs profiled, expected {RUNS}"
    us = statistics.median(t) / 1e3
    logger.info(
        f"XH_PERF {case.id} {_xh_id(*xh)}: flat_routed_expert median {us:.1f} us (min {min(t) / 1e3:.1f}, "
        f"max {max(t) / 1e3:.1f}) over {RUNS} calls after {WARMUP} warm-up"
    )
    op.close()
