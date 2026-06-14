# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Task 1 (device-free): ring-40 prefetcher SETUP layout 8-tuple.

Extends get_bh_prefetcher_core_ranges with a ``ring40=True`` mode that returns the
4-bank x 10-receiver ring-40 layout (instead of the 8-bank x 3 ring-24 layout), so
TtLlamaPrefetcherSetup can build a ring-40 global_cb without a padding regression.

Pins the 8-tuple contract for ring40 mode:
  - 4 senders (the left DRAM banks, col 0), 40 receiver cores total
  - receiver sets uniform (10 each), all in the 12x10 grid, disjoint from senders
  - dram_cores = 4 left banks
  - worker grid disjoint from the (col-0) senders, contains all receivers

CPU-only. Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_bh_ring40_setup.py -v
"""
from __future__ import annotations

import ttnn

_GX, _GY = 12, 10
_RING = 40


def _crs_cores(crs):
    s = set()
    for cr in crs.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                s.add((x, y))
    return s


def test_bh_ring40_setup_core_ranges():
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
    ) = get_bh_prefetcher_core_ranges(ring40=True)

    # 4 senders (left banks, col 0)
    assert len(all_sender_cores) == 4, f"ring-40 uses 4 senders, got {len(all_sender_cores)}"
    assert all(
        c.x == 0 for c in all_sender_cores
    ), f"left senders must be col 0, got {[(c.x, c.y) for c in all_sender_cores]}"

    # 4 DRAM banks
    assert len(dram_cores) == 4, f"ring-40 reads 4 banks, got {len(dram_cores)}"

    # 40 receivers, uniform 10/sender
    assert len(all_receiver_cores) == 4, "one receiver set per sender"
    per = {len(_crs_cores(crs)) for crs in all_receiver_cores}
    assert per == {10}, f"each sender must feed 10 receivers, got {per}"
    all_recv = set()
    for crs in all_receiver_cores:
        all_recv |= _crs_cores(crs)
    assert len(all_recv) == _RING, f"expected {_RING} distinct receivers, got {len(all_recv)}"
    assert all(0 <= x < _GX and 0 <= y < _GY for (x, y) in all_recv), "receiver off grid"

    # flat list len == 40
    assert len(active_receiver_cores_list) == _RING

    # senders disjoint from receivers
    sset = {(c.x, c.y) for c in all_sender_cores}
    assert sset.isdisjoint(all_recv), "senders overlap receivers"

    # worker grid: valid, disjoint from senders, contains all receivers
    assert isinstance(worker_cores_range_set, ttnn.CoreRangeSet)
    wset = _crs_cores(worker_cores_range_set)
    assert wset.isdisjoint(sset), f"worker overlaps senders: {sorted(wset & sset)}"
    assert all_recv <= wset, f"worker grid must contain all receivers; missing {sorted(all_recv - wset)}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
