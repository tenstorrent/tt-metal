# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""G1 (device-free): the Blackhole prefetcher core layout.

qwen3.6's prefetcher path (prefetcher_common.TtLlamaPrefetcherSetup) calls the
llama70b ``get_core_ranges(12, 2, ...)`` which HARDCODES the Wormhole layout:
``all_dram_cores = [CoreCoord(idx,0) for idx in range(12)]`` (12 banks, cols 0-11
on row 0) + senders on cols 0/4. Blackhole GLX has only **8** DRAM banks at
X=[1,3,2,0,5,7,6,4], senders on cols **0/7**, receivers in cols **1-7 / 8-11**.
``range(12)`` therefore indexes a nonexistent bank -> the "bank x=8" failure that
forced ``use_prefetcher=False``.

The BH-correct layout already exists in
``models/tt_transformers/tt/prefetcher.py`` (``PrefetcherCoreConfig`` +
``prefetcher_config.yaml``). This test pins the contract of a BH adapter
``get_bh_prefetcher_core_ranges`` that returns the SAME 8-tuple shape that
prefetcher_common.py unpacks, but built from the BH config.

CPU-ONLY (no device): PrefetcherCoreConfig.__post_init__ uses only the yaml cfg.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_bh_core_ranges.py -v
"""
from __future__ import annotations

import ttnn

# Expected BH 8-bank coords (from prefetcher_config.yaml blackhole.dram_banks), row 0.
_BH_DRAM_BANK_X = [1, 3, 2, 0, 5, 7, 6, 4]
# qwen3.6 uses a 24-core ring; ring_size = num_senders(8) * num_receivers.
# So BH uses 3 receivers/sender (8*3=24), matching the decode MLP ring config.
_NUM_RECEIVERS = 3


def _crs_size(crs: ttnn.CoreRangeSet) -> int:
    """Number of cores covered by a CoreRangeSet."""
    return sum(cr.grid_size().x * cr.grid_size().y for cr in crs.ranges())


def test_bh_prefetcher_core_ranges_contract():
    from models.demos.qwen3_6_galaxy_v2.tt.prefetcher_common import get_bh_prefetcher_core_ranges

    (
        active_sender_cores,
        dram_cores,
        all_sender_cores,
        active_receiver_cores_list,
        all_receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    ) = get_bh_prefetcher_core_ranges(num_global_cb_receivers=_NUM_RECEIVERS)

    # --- DRAM banks: exactly the 8 BH banks, at row 0, no phantom x>=8 col-0 bank ---
    assert len(dram_cores) == 8, f"BH has 8 DRAM banks, got {len(dram_cores)}"
    got_banks = {(c.x, c.y) for c in dram_cores}
    assert got_banks == {(x, 0) for x in _BH_DRAM_BANK_X}, f"BH bank coords wrong: {sorted(got_banks)}"

    # --- senders: 8 (one per bank), only on cols 0 (left) and 7 (right) ---
    assert len(active_sender_cores) == 8, f"expected 8 active senders, got {len(active_sender_cores)}"
    sx = {c.x for c in active_sender_cores}
    assert sx == {0, 7}, f"BH senders must be on cols 0/7, got {sorted(sx)}"
    assert sum(c.x == 0 for c in active_sender_cores) == 4
    assert sum(c.x == 7 for c in active_sender_cores) == 4

    # --- receivers: one CoreRangeSet per sender, each with num_global_cb_receivers cores ---
    assert len(all_receiver_cores) == len(all_sender_cores), "every sender needs a receiver set (global_cb pairing)"
    sizes = {_crs_size(crs) for crs in all_receiver_cores}
    assert sizes == {
        _NUM_RECEIVERS
    }, f"every receiver set must have {_NUM_RECEIVERS} cores (global_cb requires uniform), got {sizes}"

    # receiver cols: left senders -> cols 1-7, right senders -> cols 8-11
    rx = set()
    for crs in all_receiver_cores:
        for cr in crs.ranges():
            rx.add(cr.start.x)
            rx.add(cr.end.x)
    assert rx, "no receiver cores"
    assert all(1 <= x <= 11 for x in rx), f"receiver cols out of BH range 1-11: {sorted(rx)}"

    # --- senders and receivers are disjoint ---
    sender_set = {(c.x, c.y) for c in all_sender_cores}
    receiver_set = set()
    for crs in all_receiver_cores:
        for cr in crs.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    receiver_set.add((x, y))
    assert sender_set.isdisjoint(receiver_set), f"senders overlap receivers: {sender_set & receiver_set}"

    # --- active_receiver_cores_list: flat (x,y) list, len == senders * receivers ---
    assert len(active_receiver_cores_list) == len(all_sender_cores) * _NUM_RECEIVERS, (
        f"flat receiver list len {len(active_receiver_cores_list)} != " f"{len(all_sender_cores)}*{_NUM_RECEIVERS}"
    )
    assert all(isinstance(t, tuple) and len(t) == 2 for t in active_receiver_cores_list)

    # --- worker_cores_range_set is a valid CoreRangeSet (sub-device target) ---
    assert isinstance(worker_cores_range_set, ttnn.CoreRangeSet)
    assert _crs_size(worker_cores_range_set) > 0

    # The worker sub-device MUST be disjoint from the prefetcher (sender)
    # sub-device, else create_sub_device_manager raises "SubDevices ... intersect".
    # BH senders sit on cols 0 and 7 -> the worker grid must exclude both.
    worker_set = set()
    for cr in worker_cores_range_set.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                worker_set.add((x, y))
    assert worker_set.isdisjoint(sender_set), f"worker grid overlaps senders: {sorted(worker_set & sender_set)}"
    worker_cols = {x for (x, _) in worker_set}
    assert not (
        {0, 7} & worker_cols
    ), f"sender cols 0/7 must not be in the worker grid, found {sorted({0, 7} & worker_cols)}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
