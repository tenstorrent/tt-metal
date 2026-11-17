# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Vision outputs must survive later device allocations.

DropInVisionTransformer slices each output to ``out_hidden_size`` and mesh-partitions it before
returning it. A full-width slice is (today) a view of its source, and mesh_partition returns its input on a
1-device mesh, so freeing either source frees the returned tensor's memory. Nothing raises at that
point; the next allocation reuses the buffer and overwrites the output. Each case below therefore
allocates and fills another tensor before checking the returned values.
"""

import pytest
import torch

import ttnn
from models.demos.qwen3_vl.tt.model import _free_unless_aliased, _mesh_partition_and_free

SEQ_LEN = 256
HIDDEN = 2560


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "source_width",
    [HIDDEN, HIDDEN + 64],
    ids=["full_width", "padded_width"],
)
def test_vision_output_survives_reallocation(mesh_device, source_width):
    torch.manual_seed(0)
    source_torch = torch.randn(1, 1, SEQ_LEN, source_width, dtype=torch.bfloat16)
    expected = source_torch[0, 0, :, :HIDDEN]

    source = ttnn.from_torch(source_torch, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Same steps as DropInVisionTransformer.forward_single_user.
    output = ttnn.reshape(source[:, 0:1, :, :HIDDEN], (-1, HIDDEN))
    aliased = output.buffer_address() == source.buffer_address()
    _free_unless_aliased(source, output)
    output = _mesh_partition_and_free(output, 1)

    # Reuse device memory: with the old unconditional deallocate, this lands on the output's buffer.
    filler = ttnn.from_torch(
        torch.full((1, 1, SEQ_LEN, source_width), 7.0, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    assert output.is_allocated(), "returned output was freed"
    actual = ttnn.to_torch(output).reshape(SEQ_LEN, HIDDEN)
    assert torch.equal(actual, expected), f"output overwritten: {(actual == 7.0).float().mean():.1%} of values are 7"

    # The source must be freed exactly when the output does not share its buffer: kept alive for a view
    # (today the full-width slice), freed for a copy so it does not leak.
    assert source.is_allocated() == aliased

    ttnn.deallocate(filler)
