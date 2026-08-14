# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""L1-headroom half of the P5 domain: does a wider block ladder BREAK anything?

The block ladder's budget is what bounds `block_rows`, and `block_rows` scales
FOUR CBs (`cb_sq_partials`, `cb_gathered_partials`/`cb_slice_stat`,
`cb_rms_bcast`, `cb_rms_recip`).  Widening the budget therefore trades L1
headroom for fewer combine round trips, and the changelog records one sharded
resilience cell (`13x777x1023` WIDTH_SHARDED) that ALREADY fails a CB-vs-L1
clash at the conservative budget.  This file measures whether the wider ladder
changes any of that: it runs the ugliest sharded resilience geometries under the
shipped plan and under the candidate and compares the OUTCOME SET (ok / raised /
pcc), not the ns.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/block_count/test_l1_headroom.py -s
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_harness import PLAN_GLOBALS, guard_no_ablation, make_hook, make_split_budget_hook, target_compute_config  # noqa

pytestmark = pytest.mark.use_module_device

_ST = ttnn.ShardStrategy

# The sharded-resilience corpus, restricted to the cells whose per-core shard is
# big enough for the CB ladder to matter (verbatim shapes from
# eval/golden_tests/rms_norm/feature_spec.py's _RESILIENCE_SHAPES).
SHAPES = [
    (13, 777, 1023),  # the recorded CB-vs-L1 clash
    (3, 1, 736, 5119),
    (1, 224, 11008),
    (7136, 736),
    (3104, 4064),
    (2047, 2047),
    (100, 5120),
    (99991, 64),
    (5, 3, 928, 544),
    (1, 1, 992, 3000),
]
STRATEGIES = [("W", _ST.WIDTH), ("H", _ST.HEIGHT), ("B", _ST.BLOCK)]

VARIANTS = {
    "baseline": None,
    "split_1.43mb": (1.0, 1464 / 1024.0),  # the arch-safe real L1 (min of WH/BH)
    "split_1.46mb": (1.0, 1.46),
}


def _run(device, x, gamma, label, split):
    from ttnn.operations.rms_norm.rms_norm import rms_norm

    saved = PLAN_GLOBALS["_plan"]
    try:
        if split is None:
            PLAN_GLOBALS["_plan"] = make_hook(label)
        else:
            PLAN_GLOBALS["_plan"] = make_split_budget_hook(label, search_mb=split[0], ladder_mb=split[1])
        out = ttnn.to_torch(
            rms_norm(x, gamma=gamma, compute_kernel_config=target_compute_config(), memory_config=x.memory_config())
        ).to(torch.float32)
        return out, None
    except Exception as exc:  # a clash raises; that IS the datum
        msg = " | ".join(l.strip() for l in str(exc).splitlines() if l.strip())[:160]
        return None, f"{type(exc).__name__}: {msg}"
    finally:
        PLAN_GLOBALS["_plan"] = saved


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("sname,strategy", STRATEGIES, ids=[s[0] for s in STRATEGIES])
def test_l1_headroom(device, shape, sname, strategy):
    guard_no_ablation()
    from eval.sharding import auto_shard_config

    torch.manual_seed(0)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn(shape[-1], dtype=torch.float32).to(torch.bfloat16)
    try:
        mc = auto_shard_config(shape, strategy, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    except Exception as exc:
        pytest.skip(f"no legal shard config: {exc}")

    x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32)

    results = {}
    try:
        for vname, split in VARIANTS.items():
            out, err = _run(device, x, gamma, f"{shape}/{sname}/{vname}", split)
            if err is not None:
                results[vname] = ("RAISED", err)
                continue
            a, b = out.flatten(), expected.flatten()
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            results[vname] = ("OK", f"pcc={pcc:.6f}")
    finally:
        x.deallocate()
        gamma.deallocate()

    for vname, (status, note) in results.items():
        logger.info(f"L1H {str(shape):20s} {sname} {vname:14s} {status:7s} {note}")

    base = results["baseline"][0]
    for vname, (status, note) in results.items():
        # The candidate must never turn an OK into a RAISED.  A RAISED->OK is a
        # bonus, not a failure.
        assert not (base == "OK" and status == "RAISED"), f"{shape}/{sname}: {vname} regressed to {note}"
