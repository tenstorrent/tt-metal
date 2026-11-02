# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the serving envelope of the 8x4 from the config (tt/memory_plan.py), pinned to the observed outcomes:
one user per row at 1 M fits without the stacked expert copy and not with it; the partitioned cache at 1 M fits
only without the SP-prefill scratch slot (sparse_1m.md section 7)."""

from __future__ import annotations

from loguru import logger

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.memory_plan import GIB, MemoryPlan, default_prefill_qblock, reserve_bytes


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
    part = MemoryPlan.estimate(cfg, 8, 4, partition=True, scratch_slot=False)
    # 2026-10-27: the 128 K x 32-users served boot ran out of DRAM in the 32 K warm-up prompt (fragmented 0.6 GB left);
    # the chunk working-set term keeps >= 2 GiB after the caches: 28 users per row (7 slots per chip) is the 128 K profile
    assert not part.fits(32, 131072) and part.fits(28, 131072) and part.max_users_per_row(131072) == 28
    assert rep.fits(4, 65536), "4 users per row at 64 K (one expert copy: the regions-sourced streaming decode) runs"
    # waves (row_ownership.md step 8): W = 2048 rows per row is memory-neutral on the 8x4 (8 x 2048 = the 16 K SP chunk
    # the working-set term already models); a wider wave costs 4 x (rows x W - 16384) x 4384 x 4 B of chunk working set,
    # so the full 1 M x 4 cell admits no wave above 2048 and the planner's bound says so
    from dataclasses import replace

    assert replace(part, wave_rows=2048).headroom(1048576) == part.headroom(1048576)
    wide = reserve_bytes(1048576, part.trace_mb, chunk_tokens=8 * 4096) - reserve_bytes(1048576, part.trace_mb)
    assert wide > 0 and part.headroom(1048576) - replace(part, wave_rows=4096).headroom(1048576) == wide
    w_1m, w_128k = part.max_wave_rows(4, 1048576), part.max_wave_rows(28, 131072)
    logger.info(f"max wave rows: 1 M x 4 per row {w_1m}, 128 K x 28 per row {w_128k}")
    assert 2048 <= w_1m < 4096 and part.max_wave_rows(5, 1048576) == 0
    assert w_128k >= 2048 and part.max_wave_rows(4, 131072) >= 4096, "the 128 K cell has room for wider waves"
    # (the 2026-10-2x "both expert forms at 64 K x 4" pin is retired: GLM53_REGIONS_DECODE=stream never builds the
    # stacked copy, and the chunk working-set floor sizes the reserve for the one-copy configurations only)


def test_default_prefill_qblock_follows_the_slot_length():
    """The scorer's [rows, pools] bf16 block stays within 256 MB: 2048 rows up to 256 K, 512 at 1 M (the served 1 M
    profile ran 2048 on 2026-10-27 and failed a 201 MB score-block allocation at ~196 K positions)."""
    assert default_prefill_qblock(None) == 2048
    assert default_prefill_qblock(131072) == 2048 and default_prefill_qblock(262144) == 2048
    assert default_prefill_qblock(524288) == 512 and default_prefill_qblock(1048576) == 256
    assert reserve_bytes(1048576) == reserve_bytes(1048576, qblock=256)


def test_memory_plan_layer_subset_on_a_2_row_block():
    """The served entry point plans the depth it builds: layers 0-3 on a (2, 4) block (the image / APC / spec block
    tests: 4 users per row x 1536 positions) fit, while the full 45-layer estimate on 2 rows does not even hold one
    user (its 2-row expert shards exceed a chip) -- the refusal served_image_block hit on 2026-10-28."""
    cfg = _cfg()
    full = MemoryPlan.estimate(cfg, 2, 4, partition=True)
    assert full.max_users_per_row(1536) == 0
    sub = MemoryPlan.estimate(cfg, 2, 4, partition=True, layers=[0, 1, 2, 3])
    assert sub.fits(4, 1536), sub.headroom(1536) / GIB
    assert sub.weights_bytes < 4 * GIB  # one MoE layer (layer 3) of 2-row shards + the scaled static remainder
    # the per-layer terms follow the subset: no DSA layer among 0-3 -> no latent cache per token; KDA states of 0-3 only
    types = cfg.layer_types[:4]
    assert sub.cache_per_token == 0.0 if all(t == "linear_attention" for t in types) else sub.cache_per_token > 0
    assert sub.fixed_per_user < MemoryPlan.estimate(cfg, 2, 4, partition=True).fixed_per_user
    assert sub.fixed_per_user > 0
