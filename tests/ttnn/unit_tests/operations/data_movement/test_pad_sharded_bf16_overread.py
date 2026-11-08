# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #64834: PadRmShardedWidthOnlyProgramFactory BFLOAT16 2-byte stack buffer over-read.

The bug: pad_rm_sharded_width_only_program_factory.cpp:108-109 used
  uint16_t bfloat_pad_value_bits = std::bit_cast<uint16_t>(bfloat16(pad_value));
  padding_value_as_u32 = *reinterpret_cast<uint32_t*>(&bfloat_pad_value_bits);
which reads 4 bytes from a 2-byte stack variable (buffer over-read).

The fix: use pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)})
matching the pattern in all 6 other pad factories.

These tests exercise the PadRmShardedWidthOnlyProgramFactory path with BFLOAT16:
  - ROW_MAJOR layout, HEIGHT_SHARDED, L1
  - input_w != output_w (width padding only)
  - input_tot_h == output_tot_h (height unchanged)
"""

import pytest
import torch

import ttnn


torch.manual_seed(0)


def _run_sharded_width_only_pad(device, n, c, h, w, pad_w_after, pad_value, dtype):
    """Pad width only on a HEIGHT_SHARDED L1 ROW_MAJOR tensor and compare to torch."""
    torch_input = torch.full((n, c, h, w), pad_value * 0.5, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_input = torch_input.bfloat16()
    torch_ref = torch.nn.functional.pad(
        torch_input.float(), (0, pad_w_after), mode="constant", value=pad_value
    )
    if dtype == ttnn.bfloat16:
        torch_ref = torch_ref.bfloat16()

    compute_grid = device.compute_with_storage_grid_size()
    num_cores_x = min(8, compute_grid.x)
    num_cores_y = min(8, compute_grid.y)
    num_cores = num_cores_x * num_cores_y

    shard_h = ttnn.core.divup(n * c * h, num_cores)
    grid_coord = ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})

    shard_spec_in = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_in
    )

    out_h = n * c * (h)
    shard_h_out = ttnn.core.divup(out_h, num_cores)
    shard_spec_out = ttnn.ShardSpec(
        shard_grid, (shard_h_out, w + pad_w_after), ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_out
    )

    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem_config
    )
    padding = ((0, 0), (0, 0), (0, pad_w_after))
    tt_output = ttnn.pad(tt_input, padding=padding, value=pad_value, memory_config=output_mem_config)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert tt_output.shape == torch_ref.shape, f"Shape mismatch: {tt_output.shape} vs {torch_ref.shape}"
    if dtype == ttnn.bfloat16:
        assert torch.allclose(tt_output.float(), torch_ref.float(), atol=1e-2, rtol=1e-2), (
            f"BFLOAT16 pad value mismatch: max diff = {(tt_output.float() - torch_ref.float()).abs().max()}"
        )
    else:
        assert torch.equal(tt_output, torch_ref)


@pytest.mark.parametrize("pad_value", [0.0, 1.5, -3.0, 7.25])
@pytest.mark.parametrize("w", [4, 8, 16])
def test_pad_rm_sharded_width_only_bf16(device, w, pad_value):
    """BFLOAT16 width-only pad through PadRmShardedWidthOnlyProgramFactory.

    Exercises the fixed code path where pack_two_bfloat16_into_uint32 replaces
    the unsafe reinterpret_cast<uint32_t*>(&uint16_var).
    """
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x < 1 or core_grid.y < 1:
        pytest.skip("Device does not have enough cores for sharded test")
    _run_sharded_width_only_pad(device, n=1, c=1, h=32, w=w, pad_w_after=12, pad_value=pad_value, dtype=ttnn.bfloat16)


@pytest.mark.parametrize("pad_value", [0.0, 2.5, -1.0])
def test_pad_rm_sharded_width_only_f32(device, pad_value):
    """FLOAT32 width-only pad — verifies the fix doesn't break the float32 path."""
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x < 1 or core_grid.y < 1:
        pytest.skip("Device does not have enough cores for sharded test")
    _run_sharded_width_only_pad(device, n=1, c=1, h=32, w=8, pad_w_after=16, pad_value=pad_value, dtype=ttnn.float32)


def test_pad_rm_sharded_width_only_bf16_large_grid(device):
    """BFLOAT16 width-only pad on a larger core grid — stress test for the packing fix."""
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x < 4 or core_grid.y < 4:
        pytest.skip("Device does not have enough cores for large grid test")
    _run_sharded_width_only_pad(
        device, n=1, c=1, h=3 * 64, w=4, pad_w_after=12, pad_value=5.5, dtype=ttnn.bfloat16
    )
