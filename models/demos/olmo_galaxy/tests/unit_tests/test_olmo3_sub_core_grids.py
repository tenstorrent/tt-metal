# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Invariants for OLMo3's worker-core layout after freeing col 4 from prefetcher
reservation.

Rationale: OLMo3 has use_prefetcher=False, so col 4 (the Wormhole right-sender
column per prefetcher_config.yaml) is not reserved and can be used by workers.
Widening self.sub_core_grids to a contiguous cols 1-6 × rows 0-9 rectangle
gives the SDPA decode kernel a contiguous compute region (the previous
non-contiguous 50-core layout was implicated in batch-row corruption — see
ISSUE_1_paged_sdpa_batch32_collapse.md).

CREATE_HEAD_OUTPUT_MEMCFG remains pinned to the old 50-core narrow set
because its sharded shape encodes the cardinality of the set in the tensor's
logical height.
"""

import pytest

import ttnn
from models.demos.olmo_galaxy.tt.olmo_model_config import TtOlmoModelArgs


def _enumerate_cores(crs: ttnn.CoreRangeSet) -> set:
    out = set()
    for rng in crs.ranges():
        for x in range(rng.start.x, rng.end.x + 1):
            for y in range(rng.start.y, rng.end.y + 1):
                out.add((x, y))
    return out


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="galaxy_8x4")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_olmo3_sub_core_grids_includes_col4(mesh_device):
    """After the col-4-free change, self.sub_core_grids must be a single
    contiguous rectangle cols 1-6 × rows 0-9 = 60 cores, including col 4."""
    args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=128 * 1024)

    cores = _enumerate_cores(args.sub_core_grids)

    # Col 4 must be present, all 10 rows.
    for y in range(10):
        assert (4, y) in cores, f"col 4 row {y} missing from sub_core_grids"

    # Col 0 must still be excluded — flagged for long-prefill NOC hangs
    # (separate follow-up to widen).
    for y in range(10):
        assert (0, y) not in cores, f"col 0 row {y} unexpectedly in sub_core_grids"

    # Col 7 must still be excluded — dispatch core when dispatch_core_axis=COL.
    for y in range(10):
        assert (7, y) not in cores, f"col 7 row {y} unexpectedly in sub_core_grids (dispatch col)"

    # Expected exactly cols 1-6 × rows 0-9 = 60 cores.
    expected = {(x, y) for x in range(1, 7) for y in range(10)}
    assert cores == expected, f"sub_core_grids = {sorted(cores)} vs expected {sorted(expected)}"


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="galaxy_8x4")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_olmo3_create_head_output_memcfg_stays_50_cores(mesh_device):
    """CREATE_HEAD_OUTPUT_MEMCFG's shard core set must remain the original
    50-core non-contiguous layout regardless of how self.sub_core_grids is
    widened — its shard count encodes the tensor's logical height."""
    args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=128 * 1024)

    memcfg = args.model_config["CREATE_HEAD_OUTPUT_MEMCFG"]
    cores = _enumerate_cores(memcfg.shard_spec.grid)

    assert len(cores) == 50, (
        f"CREATE_HEAD_OUTPUT_MEMCFG shard set size {len(cores)} != 50 — "
        f"shard layout has changed and downstream tensor shape assumptions "
        f"will break."
    )

    # Col 4 must NOT be in the create-head shard set (the narrow set must
    # preserve the 50-core layout — cols 1-3 ∪ cols 5-6).
    for y in range(10):
        assert (4, y) not in cores, f"col 4 row {y} unexpectedly in CREATE_HEAD_OUTPUT_MEMCFG shard set"
