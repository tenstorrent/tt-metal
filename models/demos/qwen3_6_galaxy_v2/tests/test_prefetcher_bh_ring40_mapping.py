# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ring-40 prefetcher feasibility on Blackhole (device-free geometry).

qwen3.6's decode MLP uses a 40-core ring (least padding: K=1280=40 tiles, N
2176->2560). The BH prefetcher receiver ring = num_banks * num_receivers; ring-40
as 8 banks * 5 receivers is INFEASIBLE (5 not in legal_receiver_cores [1,2,3,8,10];
receiver cols spill to x=12, off the 12-wide grid). It IS reachable as
**4 banks * 10 receivers** (10 is legal). This test pins that the
``get_bh_ring40_prefetcher_mapping`` override is geometrically valid:
  - exactly 40 distinct receiver cores, all inside the 12x10 BH grid
  - disjoint from the (col-0) senders
  - uniform 10 receivers per sender (a global_circular_buffer requirement)

This is the structural gate that says "ring-40 + prefetcher is possible" without
forcing the decode MLP onto ring-24's heavier padding. The on-device step (build
the global_cb with this override + run the ring-40 matmul through it) follows.

CPU-only. Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_bh_ring40_mapping.py -v
"""
from __future__ import annotations

_GX, _GY = 12, 10  # BH compute grid: cols 0-11, rows 0-9
_RING = 40


def test_bh_ring40_mapping_geometry():
    from models.demos.qwen3_6_galaxy_v2.tt.prefetcher_common import get_bh_ring40_prefetcher_mapping

    mapping = get_bh_ring40_prefetcher_mapping()

    senders = list(mapping.keys())
    all_recv = [r for recvs in mapping.values() for r in recvs]

    # 4 senders, each with 10 receivers -> ring 40
    assert len(senders) == 4, f"expected 4 (left-bank) senders, got {len(senders)}"
    assert len(all_recv) == _RING, f"expected {_RING} receivers, got {len(all_recv)}"

    # uniform receivers-per-sender (global_cb requires uniform CoreRangeSet sizes)
    sizes = {len(v) for v in mapping.values()}
    assert sizes == {10}, f"receivers/sender must be uniform 10, got {sizes}"

    # 40 DISTINCT receiver cores
    assert len(set(all_recv)) == _RING, f"receivers not distinct: {len(set(all_recv))} unique"

    # all in the BH 12x10 grid
    assert all(0 <= x < _GX and 0 <= y < _GY for (x, y) in all_recv), "a receiver is off the 12x10 grid"

    # senders on col 0 (left), disjoint from receivers
    assert all(sx == 0 for (sx, _) in senders), f"left senders must be col 0, got {senders}"
    assert set(senders).isdisjoint(set(all_recv)), "senders overlap receivers"

    # receivers must avoid BOTH potential sender columns (0 and 7) is NOT required here:
    # only the col-0 left senders are active in this 4-left-bank config, so col 7 is free.
    recv_cols = {x for (x, _) in all_recv}
    assert 0 not in recv_cols, "receivers must not sit on the col-0 sender column"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
