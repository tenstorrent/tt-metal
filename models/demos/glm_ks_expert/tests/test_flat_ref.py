# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The real flat routed expert at GLM-5.3's 4x4 shape (16 experts a chip), one chip: the reference the generic_op model
is compared with (device time, median of PERF_RUNS calls)."""

import statistics
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

T = Path(__file__).resolve().parents[4] / "tests/ttnn/unit_tests/operations/experimental/moe_block"
sys.path.insert(0, str(T / "flat_routed_expert"))
sys.path.insert(0, str(T))
import test_flat_routed_expert_checks as chk  # noqa: E402
from op_checks import PERF_RUNS, op_device_times  # noqa: E402

W = chk.weights_py
CAP = 5120


def _inputs(device, md, gids, M):
    NG, H = md["E"], md["H"]
    pad = -(-M // 32) * 32
    counts = torch.zeros(1, NG, dtype=torch.int32)
    regions = torch.zeros(1, NG, dtype=torch.int32)
    for e, gid in enumerate(gids):
        counts[0, gid], regions[0, gid] = M, e * pad
    x = torch.randn(len(gids) * pad + 64, H, generator=torch.Generator().manual_seed(M)) * 0.3
    return chk._rm(x, ttnn.bfloat16, device), chk._rm(counts, ttnn.uint32, device), chk._rm(regions, ttnn.uint32, device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("M", [320, 384, 512])
def test_flat_ref(device, M):
    chk._skip_unless_supported(device)
    md = chk.cases.MODELS["glm53"]
    epc = 16
    gids = [(md["E"] // epc) * e + 1 for e in range(epc)]
    op = W.FlatRoutedExpert(
        device, "uninitialized", m=max(CAP, epc * M), H=md["H"], I=md["I"], gids=[gids], n_global=md["E"],
        act=md["act"], pin=1,
    )  # fmt: skip
    args = _inputs(device, md, gids, M)
    ttnn.deallocate(op(*args))
    t = op_device_times(device, lambda: [ttnn.deallocate(op(*args)) for _ in range(PERF_RUNS)], chk.FLAT_OPS)
    us = statistics.median(t["flat_routed_expert"][-PERF_RUNS:]) / 1e3
    sb = epc * -(-M // 128)
    logger.info(f"FLATREF glm53 e{epc} M{M}: {us:.1f} us, {sb} sub-blocks a core ({us / sb:.2f} us a sub-block)")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("M", [384])
def test_flat_ref_profile(device, M):
    """3 calls under the device profiler (run with TT_METAL_DEVICE_PROFILER=1); the CSV is copied by the caller."""
    import json

    chk._skip_unless_supported(device)
    md = chk.cases.MODELS["glm53"]
    epc = 16
    gids = [(md["E"] // epc) * e + 1 for e in range(epc)]
    op = W.FlatRoutedExpert(
        device, "uninitialized", m=max(CAP, epc * M), H=md["H"], I=md["I"], gids=[gids], n_global=md["E"],
        act=md["act"], pin=1,
    )  # fmt: skip
    lay = op.lay
    meta = {
        "H": md["H"], "I": md["I"], "epc": epc, "M": M, "rows_sb": lay["row_tiles_per_sub_block"] * 32,
        "n_gu": len(lay["gate_up_cores"]), "n_dn": lay["down_core_count"], "dn_cols": lay["down_columns_per_core"],
        "sub_blocks_per_expert": -(-M // (lay["row_tiles_per_sub_block"] * 32)),
    }  # fmt: skip
    out = Path("generated/zz_balprof")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"best-ref-M{M}.meta.json").write_text(json.dumps(meta))
    args = _inputs(device, md, gids, M)
    for _ in range(3):
        ttnn.deallocate(op(*args))
    ttnn.ReadDeviceProfiler(device)
