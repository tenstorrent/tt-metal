# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""On-device validation of the GlobalCB ring configs for Flash's real weight shapes.

Runs Flash's MLA weight shapes through the canonical `run_prefetcher_mm` harness
(tests/ttnn/unit_tests/operations/prefetcher_common.py) -- the same harness that
backs test_prefetcher_TG.py, which passes on this Galaxy. Each case does
dram_prefetcher into a GlobalCB, consumes it from a gather_in0 ring matmul, and
PCC-checks the result against a reference matmul.

WHAT THIS DOES AND DOES NOT COVER

Covers: that a given (K, N, ring size) actually streams and computes correctly on
silicon. This is the part that deadlocks when the ring does not divide both
dimensions, and a deadlock costs a Galaxy reset -- so it is worth isolating here,
before the shapes are wired into a 47-layer model where a hang is far more
expensive to attribute.

Does NOT cover: Flash's own core layout. The canonical harness uses llama's
get_core_ranges, which is hardcoded for the 7x10 COL-dispatch grid (it places cores
at y=9). Flash runs ETH dispatch and gets an 8x9 grid, so it needs its own layout --
that is what prefetcher_setup.get_glm_core_ranges is for, ported from REAP's
device-proven 8x9 layout. Ring arithmetic is grid-independent, which is what makes
this split useful: validate the shapes here, trust REAP's layout there.

The host-side arithmetic that decides which rings are legal is pinned separately in
test_prefetcher_config.py (no device required).
"""

from __future__ import annotations

import os

import pytest

import ttnn

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_HW_TESTS") != "1",
        reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
    ),
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_MULTI_DEVICE_TESTS") != "1",
        reason="Enable with TT_ENABLE_MULTI_DEVICE_TESTS=1 (opens a multi-device mesh).",
    ),
]


# The canonical harness hardcodes its GlobalCB at 600 tiles
# (prefetcher_common.py: `global_cb_size = 600 * max_tile_size`). Cases needing more
# than this cannot run here; they skip with the measured requirement rather than
# failing, since the constraint is the harness's, not Flash's.
HARNESS_GLOBAL_CB_TILES = 600

# (id, K, N, num_reader_cores, num_layers)
#
# num_reader_cores x 2 receivers = the ring width.
#
# w_o is the first-prototype target: Flash's largest 2D decode weight (11.1 MB/layer
# in bf8) and, at 160 x 64 tiles, dimensionally identical to REAP's QKV, which REAP
# runs at a 16-core ring. Full size needs a 640-tile CB, over the harness cap -- so it
# is listed to document the requirement (the skip message carries the number) and a
# reduced-K variant carries the actual ring-16 validation. 144 x 64 tiles keeps the
# ring-16 divisibility and the N extent identical, at a 576-tile payload that fits.
#
# w_q_b is the planned next increment and only admits an 8-core ring (24 x 160 tiles,
# gcd 8), exercised at full size to confirm the narrower ring is sound before
# committing to a mixed-ring-width GlobalCB contract.
FLASH_RING_CASES = [
    ("w_o_ring16_full", 5120, 2048, 8, 5),
    ("w_o_ring16_reduced_k", 4608, 2048, 8, 5),
    ("w_q_b_ring8", 768, 5120, 4, 5),
]


@pytest.mark.parametrize(
    "case_id,K,N,num_reader_cores,num_layers",
    FLASH_RING_CASES,
    ids=[c[0] for c in FLASH_RING_CASES],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_flash_weight_shape_streams_through_global_cb(
    mesh_device,
    case_id: str,
    K: int,
    N: int,
    num_reader_cores: int,
    num_layers: int,
    function_level_defaults,
) -> None:
    """One Flash weight shape, prefetched and consumed by a ring matmul."""
    # Imported here so collection does not require the llama demo tree.
    from tests.ttnn.unit_tests.operations.prefetcher_common import run_prefetcher_mm

    from models.experimental.glm4_moe_lite.tt.prefetcher_setup import global_cb_tiles_for, ring_feasibility

    ring = num_reader_cores * 2
    feasible = ring_feasibility(K, N, max_cores=ring)
    assert ring in feasible, (
        f"{case_id}: a {ring}-core ring does not divide K={K} N={N} (feasible: {feasible}). "
        "Running it would deadlock rather than fail -- fix the config, do not run this."
    )

    required_tiles = global_cb_tiles_for(K, N, ring)
    if required_tiles > HARNESS_GLOBAL_CB_TILES:
        pytest.skip(
            f"{case_id} needs a {required_tiles}-tile GlobalCB "
            f"({required_tiles * 1088} B); the canonical harness hardcodes "
            f"{HARNESS_GLOBAL_CB_TILES} tiles ({HARNESS_GLOBAL_CB_TILES * 1088} B). "
            "Verified on device: dram_prefetcher asserts "
            "'largest tensor 696320 must fit in global cb 652800' for full-size w_o, "
            "which confirms global_cb_tiles_for(). prefetcher_setup sizes its own CB "
            "correctly, so this bounds the harness, not the model."
        )

    device_grid = (
        mesh_device.compute_with_storage_grid_size().x,
        mesh_device.compute_with_storage_grid_size().y,
    )
    if device_grid != (7, 10):
        pytest.skip(
            f"canonical harness core layout requires a 7x10 grid, got {device_grid}. "
            "Flash's own 8x9 layout lives in prefetcher_setup.get_glm_core_ranges."
        )

    run_prefetcher_mm(
        mesh_device,
        1,  # num_tensors per layer
        [(K, N)],
        num_layers,
        num_reader_cores,
        [ttnn.bfloat8_b],
    )
