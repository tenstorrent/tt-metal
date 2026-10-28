# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_negative_dim(device, shape, dim, dtype):
    """
    Regression test for negative dimension support in ttnn.gather.
    Verifies that negative dimension values work correctly and match PyTorch behavior.
    """
    torch_input = torch.rand(shape, dtype=dtype)

    # Create index tensor for gathering
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)  # Gather subset
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)

    # PyTorch reference with negative dim
    torch_output_neg = torch.gather(torch_input, dim, torch_index.long())

    # Convert to ttnn using the matching test dtype
    torch_to_ttnn_dtype = {
        torch.bfloat16: ttnn.bfloat16,
        torch.float32: ttnn.float32,
    }
    ttnn_dtype = torch_to_ttnn_dtype[dtype]

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    # Test with negative dim
    ttnn_output = ttnn.gather(ttnn_input, dim, ttnn_index)
    output = ttnn.to_torch(ttnn_output)

    assert (
        output.shape == torch_output_neg.shape
    ), f"Output shape {output.shape} does not match expected {torch_output_neg.shape}"
    # Use tighter tolerance for float32
    pcc_threshold = 0.9999 if dtype == torch.float32 else 0.999
    assert_with_pcc(torch_output_neg, output, pcc_threshold)


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((4, 128), -1),  # Last dimension
        ((4, 128), -2),  # First dimension
        ((2, 3, 4), -1),  # 3D tensor, last dim
        ((2, 3, 4), -2),  # 3D tensor, middle dim
        ((2, 3, 4), -3),  # 3D tensor, first dim
    ],
)
def test_gather_negative_dim_equals_positive(device, shape, dim):
    """
    Verify that negative and positive dim produce identical results.
    """
    positive_dim = len(shape) + dim

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)

    # Get results for both negative and positive dims
    torch_output_neg = torch.gather(torch_input, dim, torch_index.long())
    torch_output_pos = torch.gather(torch_input, positive_dim, torch_index.long())

    # Verify PyTorch results match
    assert torch.allclose(torch_output_neg, torch_output_pos)

    # Test ttnn
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    ttnn_output_neg = ttnn.to_torch(ttnn.gather(ttnn_input, dim, ttnn_index))
    ttnn_output_pos = ttnn.to_torch(ttnn.gather(ttnn_input, positive_dim, ttnn_index))

    assert_with_pcc(ttnn_output_neg, ttnn_output_pos, 0.9999)
    assert_with_pcc(torch_output_neg, ttnn_output_neg, 0.999)


@pytest.mark.parametrize("N", [151936, 4800, 2048, 4096, 3520, 2200, 1980])
def test_gather_rm_wide_index_multicore(device, N):
    """Regression for #55450: ROW_MAJOR ttnn.gather with an index last-dim > GATHER_WT_THRESHOLD*32 (=1920)
    uses the multi-core factory, which reads/writes each core's column slice via a sharded NoC transfer at byte
    offset w_start*elem_size. When that offset is not a multiple of the ARCHITECTURE's DRAM alignment the tail of
    each core's slice is silently wrong.

    The alignment is architectural, not a constant -- Wormhole 16B, Blackhole 64B -- so the failing widths differ
    per board and a width that is clean on one silently corrupts on the other. Widths below cover both:
      151936  vocab width; per-core offset off a 16B boundary on Wormhole (192 wrong of 151936)
      4096,   2048  clean on Wormhole, both corrupt on a 110-core Blackhole before the fix
      3520    32 elements = 64B per core on Blackhole: the aligned control, clean before and after
      2200,   1980  20 and 18 elements per core (40B, 36B): divide evenly into 110 cores yet corrupt,
                    which is why the per-core ELEMENT count is not the property that matters
    Data movement => require EXACT equality (a PCC check would mask a small corrupted tail)."""
    torch.manual_seed(0)
    x = torch.rand([1, N], dtype=torch.bfloat16)
    idx = torch.randint(0, N, [1, N], dtype=torch.int32)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    it = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.gather(xt, dim=-1, index=it)).contiguous().flatten()
    ref = torch.gather(x, -1, idx.long()).contiguous().flatten()
    mismatches = int((out != ref).sum().item())
    assert mismatches == 0, f"gather returned {mismatches}/{N} wrong elements for ROW_MAJOR width N={N}"
