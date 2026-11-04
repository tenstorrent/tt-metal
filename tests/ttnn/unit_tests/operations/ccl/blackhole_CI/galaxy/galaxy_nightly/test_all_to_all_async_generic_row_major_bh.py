# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_to_all_async_generic with a row-major input, which the op tilizes on the way: exact against torch on the 8x4
Blackhole torus, along both axes.

The cases cover the schedule shapes the tilize path has to handle:
- one destination range per sender (4-chip axis, large message);
- several ranges per sender (8-chip axis: more destinations than workers per direction; small messages on the compact
  schedule);
- tile rows that don't split evenly across links;
- tile rows that aren't a whole number of packets, so ranges end in a short packet;
- input rows wider than a fabric packet, which only carries tiles.

A float32 row-major input is rejected: the tilize would round it.
"""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0

NUM_LINKS = 2


def _expected(chip_inputs, num_devices, in_dim, out_dim):
    """Chip d's output: every chip's slice d of out_dim, concatenated along in_dim in chip order."""
    split = chip_inputs[0].shape[out_dim] // num_devices
    return [
        torch.cat([x.narrow(out_dim, d * split, split) for x in chip_inputs], dim=in_dim) for d in range(num_devices)
    ]


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1, 0], ids=["axis1_4chip", "axis0_8chip"])
@pytest.mark.parametrize(
    "heads, rows, width",
    [
        (16, 640, 576),  # large: 20 tile rows of 18 tiles, not a whole number of packets
        (16, 96, 512),  # 3 tile rows over 2 links
        (8, 32, 96),  # small: 1 tile row of 3 tiles, compact schedule
        (8, 32, 2304),  # wide: a 4608 B input row, more than a 4352 B packet
    ],
    ids=["large", "odd_tile_rows", "small", "wide_rows"],
)
def test_all_to_all_async_generic_row_major_input(mesh_device, cluster_axis, heads, rows, width):
    torch.manual_seed(0)
    num_devices = mesh_device.shape[cluster_axis]
    in_dim, out_dim = 2, 1  # concatenate along the rows, split the heads
    chips = [torch.randn(1, heads, rows, width).to(torch.bfloat16) for _ in range(mesh_device.get_num_devices())]
    x = ttnn.from_host_shards(
        [ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for c in chips], mesh_device.shape
    ).to(mesh_device, ttnn.DRAM_MEMORY_CONFIG)

    out = ttnn.experimental.all_to_all_async_generic(
        x,
        in_dim=in_dim,
        out_dim=out_dim,
        num_links=NUM_LINKS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=cluster_axis,
    )
    assert out.layout == ttnn.TILE_LAYOUT
    got = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]

    # Chips are row-major over the mesh; the all-to-all runs within each line along cluster_axis.
    mesh_rows, mesh_cols = mesh_device.shape
    for line in range(mesh_cols if cluster_axis == 0 else mesh_rows):
        if cluster_axis == 0:
            members = [r * mesh_cols + line for r in range(mesh_rows)]
        else:
            members = [line * mesh_cols + c for c in range(mesh_cols)]
        expected = _expected([chips[m] for m in members], num_devices, in_dim, out_dim)
        for position, member in enumerate(members):
            assert torch.equal(got[member], expected[position]), f"chip {member} (position {position}) differs"


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_to_all_async_generic_row_major_input_rejects_float32(mesh_device, expect_error):
    x = ttnn.from_torch(
        torch.randn(1, 8, 32, 96),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with expect_error(RuntimeError, "must be bfloat16"):
        ttnn.experimental.all_to_all_async_generic(
            x, in_dim=2, out_dim=1, num_links=NUM_LINKS, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
        )
