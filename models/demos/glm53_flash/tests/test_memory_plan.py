# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the serving envelope of the 8x4 from the config (tt/memory_plan.py), pinned to the observed outcomes:
one user per row at 1 M fits without the stacked expert copy and not with it; the partitioned cache at 1 M fits
only without the SP-prefill scratch slot (sparse_1m.md section 7)."""

from __future__ import annotations

from loguru import logger

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.memory_plan import GIB, MemoryPlan


def _cfg():
    return GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")


def test_memory_plan_8x4_table():
    cfg = _cfg()
    for name, kw in (
        ("replicated cache (today's long runs)", dict(partition=False)),
        ("replicated + keep_stacked", dict(partition=False, keep_stacked=True)),
        ("partitioned with the scratch slot (today)", dict(partition=True, scratch_slot=True)),
        ("partitioned, no scratch slot (heads->users prefill)", dict(partition=True, scratch_slot=False)),
    ):
        plan = MemoryPlan.estimate(cfg, 8, 4, **kw)
        logger.info(
            f"\n### {name}: weights {plan.weights_bytes / GIB:.1f} GiB, cache {plan.cache_per_token / 1024:.2f} KiB/token/user, "
            f"fixed {plan.fixed_per_user / 2**20:.0f} MiB/user, headroom at 1 M {plan.headroom(1048576) / GIB:.1f} GiB\n"
            + plan.table()
        )
    rep = MemoryPlan.estimate(cfg, 8, 4, partition=False)
    assert rep.fits(1, 1048576), "the 1 M single-slot run fits (observed: 4.0 of 4.2 GB per bank in use)"
    assert not MemoryPlan.estimate(cfg, 8, 4, partition=False, keep_stacked=True).fits(1, 1048576)
    assert not MemoryPlan.estimate(cfg, 8, 4, partition=True, scratch_slot=True).fits(4, 1048576)
    assert MemoryPlan.estimate(cfg, 8, 4, partition=True, scratch_slot=False).fits(4, 1048576)
    assert rep.fits(4, 65536), "v22: 4 users per row at 64 K with both expert forms ran"
    assert MemoryPlan.estimate(cfg, 8, 4, partition=False, keep_stacked=True).fits(4, 65536)
