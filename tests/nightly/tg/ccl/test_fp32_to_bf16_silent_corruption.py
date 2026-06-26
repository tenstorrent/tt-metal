# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Silent data-corruption detector for the fp32 -> bf16 conversion path.

On a faulty Galaxy this reproduces silent, intermittent data corruption when
``ttnn.from_torch`` converts a large fp32 host tensor to bf16 on device.

The host tensor is filled with 240.0, which is *exactly* representable in bf16,
so a correct device must read back exactly 240.0 for every element; any element
that differs is silent hardware corruption.

Scope of the fault (device-confirmed on the affected chip): the corruption is
chip-wide -- it is spread across most compute cores and most pages of the shard,
not localized to one core, so this test reports the affected chip and the spread
rather than a single core coordinate. It appears only in ``from_torch``'s
internal tilize+convert path: writing the same buffer with a pure fp32 DMA (no
conversion), or converting it afterwards with a standalone ``ttnn.typecast``,
reads back clean. That rules out a software bug and points at the on-device
conversion compute. A healthy device reads back clean and the test passes.
"""

import pytest
import torch
import ttnn

# 240.0 is exact in bf16; any deviation after a fp32 -> bf16 convert is corruption.
FILL_VALUE = 240.0
# Large, with a non-tile-multiple last dim, so the conversion path runs on every
# chip with enough work to expose the marginal fault.
TENSOR_SHAPE = (256, 256, 2, 7000)


def _deallocate(tensor):
    if tensor is not None:
        ttnn.deallocate(tensor, force=True)


@pytest.mark.parametrize(
    "device_params",
    # Default dispatch-core axis (kept identical to the tilize detector). The
    # conversion fault is chip-wide, so the dispatch axis does not affect detection.
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    ids=["fabric_2d"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((8, 4), (8, 4), id="8x4_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("num_iters", [10])
def test_fp32_to_bf16_silent_corruption(mesh_device, mesh_shape, num_iters):
    assert mesh_shape == (8, 4)

    host = torch.full(TENSOR_SHAPE, FILL_VALUE, dtype=torch.float32)

    total_corrupt = 0
    for i in range(num_iters):
        tt = None
        try:
            tt = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )

            for chip_idx, chip_tensor in enumerate(ttnn.get_device_tensors(tt)):
                actual = ttnn.to_torch(chip_tensor).float()
                diff = actual != FILL_VALUE
                corrupt_count = int(diff.sum().item())
                if corrupt_count == 0:
                    continue

                total_corrupt += corrupt_count
                # Show the corruption is chip-wide, not localized: how many distinct
                # outer pages (dim0, dim1) of this shard contain at least one bad value.
                bad = diff.nonzero(as_tuple=False)
                page_ids = bad[:, 0] * actual.shape[1] + bad[:, 1]
                pages_hit = int(torch.unique(page_ids).numel())
                total_pages = actual.shape[0] * actual.shape[1]
                pct = 100.0 * corrupt_count / actual.numel()
                print(
                    f"CORRUPT chip={chip_idx} iter={i} corrupt_count={corrupt_count} "
                    f"pct={pct:.3f} pages_hit={pages_hit}/{total_pages} (chip-wide)"
                )
        finally:
            _deallocate(tt)

    assert total_corrupt == 0
