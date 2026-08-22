# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_allclose
from models.common.utility_functions import is_blackhole

TILE_HEIGHT = 32
TILE_WIDTH = 32


def _residual_ht_shapes(device):
    """
    Build Ht values where Ht % grid_x != 0 and grid_x <= Ht < grid_x * grid_y.
    This is the exact condition that enters the additional CoreRange branch
    in SingleRowSingleCore / SingleRowMultiCore sort program factories.
    """
    grid = device.compute_with_storage_grid_size()
    total = grid.x * grid.y
    shapes = []
    for r in range(1, grid.x):
        ht = grid.x + r
        if ht < total:
            shapes.append(ht)
    if grid.x * 2 + 1 < total:
        shapes.append(grid.x * 2 + 1)
    return shapes


def test_sort_residual_core_range(device):
    """
    Regression test for an off-by-one in the additional CoreRange end
    coordinate that allocated one extra core, causing OOB DRAM writes.
    Shapes are derived from the device grid so the path is always hit.
    """
    ht_values = _residual_ht_shapes(device)
    for ht in ht_values:
        for descending in (False, True):
            torch.manual_seed(0)
            shape = [ht * TILE_HEIGHT, TILE_WIDTH]
            input_tensor = torch.randn(shape, dtype=torch.bfloat16)

            ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
            torch_values, _ = torch.sort(input_tensor, dim=-1, descending=descending)
            ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)

            assert list(ttnn_values.shape) == shape
            assert_equal(torch_values, ttnn.to_torch(ttnn_values))
            ttnn_gathered = torch.gather(input_tensor, -1, ttnn.to_torch(ttnn_indices).to(torch.int64))
            assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, False),
        ([32, 128], -1, False),
        ([1, 1, 32, 64], -1, True),
        ([32, 128], 1, True),
        ([1], 0, True),
        ([], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
        ([1, 1, 512, 64], -1, False),
        ([1, 1, 2112, 64], -1, False),
        ([1, 64, 64], 0, False),
        ([1, 64, 64], 1, True),
        ([1, 64, 64], 2, False),
        ([1, 64], 0, False),
        ([1, 64], 1, True),
        ([237], 0, False),
    ],
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_sort_standard(shape, dim, descending, device, torch_dtype, ttnn_dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn_dtype, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
        assert torch_sort_indices == ttnn.to_torch(ttnn_sort_indices).to(torch.int64)
    else:
        # Validate sorted values
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values, dtype=torch_dtype))

        # Validate that the indices correctly index into the original tensor
        ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn.to_torch(ttnn_sort_indices).to(torch.int64))
        assert_equal(torch_sort_values, ttnn_torch_gather_from_indices)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, False),
        ([32, 128], -1, False),
        ([1, 1, 32, 64], -1, True),
        ([32, 128], 1, True),
        ([1], 0, True),
        ([], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
        ([1, 1, 512, 64], -1, False),
        ([1, 1, 2112, 64], -1, False),
    ],
)
def test_sort_prealocated_output(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    ttnn_sort_values = ttnn.zeros_like(ttnn_input)
    ttnn_sort_indices = ttnn.zeros_like(ttnn_input, dtype=ttnn.uint16)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(ttnn_sort_values, ttnn_sort_indices))

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([1, 1, 1, 2 * TILE_WIDTH], -1, False),
        ([1, 1, 1, 8192 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 96 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
        ([1, 151936], -1, False),
        ([1, 128256], -1, False),
        ([1, 16384 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_long_tensor(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


def _sort_fp32_wide(descending, device):
    # A few finite logits in a sea of -inf, as in masked vocab-size logits.
    torch.manual_seed(0)
    n = 151936
    input = torch.full((1, n), float("-inf"), dtype=torch.float32)
    input[..., torch.randperm(n)[:328]] = torch.randn(328) * 8.0

    ttnn_input = ttnn.from_torch(input, ttnn.float32, layout=ttnn.Layout.TILE, device=device)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)
    return input, ttnn_sort_values, ttnn_sort_indices


@pytest.mark.parametrize("descending", [False, True])
def test_sort_fp32_wide_values(descending, device):
    input, ttnn_sort_values, ttnn_sort_indices = _sort_fp32_wide(descending, device)

    torch_sort_values, _ = torch.sort(input, dim=-1, descending=descending)

    assert ttnn_sort_indices.dtype == ttnn.uint32
    assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "descending",
    [
        False,
        # Descending order pads the row with -inf, which ties with real -inf
        # entries and can emit padding indices (>= n) into the output.
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/tenstorrent/tt-metal/issues/53326: padding indices leak",
            ),
        ),
    ],
)
def test_sort_fp32_wide_index_correctness(descending, device):
    # Ties make exact torch index parity undefined, so check invariants instead:
    # indices form a valid permutation and gather back the sorted values.
    input, ttnn_sort_values, ttnn_sort_indices = _sort_fp32_wide(descending, device)

    n = input.shape[-1]
    values = ttnn.to_torch(ttnn_sort_values)
    indices = ttnn.to_torch(ttnn_sort_indices).reshape(-1).to(torch.int64)

    assert (
        indices.min() >= 0 and indices.max() < n
    ), f"indices out of range [0, {n}): min={int(indices.min())}, max={int(indices.max())}"
    unique_count = indices.unique().numel()
    assert unique_count == n, f"{n - unique_count} duplicated indices"
    assert_equal(input.reshape(-1)[indices], values.reshape(-1))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([1, 96], -1, True),
        ([1, 1, 32, 96 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_l1_memory_tensor(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        input,
        ttnn.bfloat16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([64, 64], -1, True),
        ([1, 1, 32, 64], -1, False),
        ([32, 128], -1, True),
        ([1, 1, 32, 128 * TILE_WIDTH], -1, False),
        ([1, 1, 32, 256 * TILE_WIDTH], -1, False),
    ],
)
def test_sort_program_cache(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    test_iterations = 3
    for _ in range(test_iterations):
        # Run the sort operation multiple times to fill the program cache
        with device.cache_entries_counter.measure():
            ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)
        ttnn_sort_values_torch = ttnn.to_torch(ttnn_sort_values)

        assert torch_sort_values.shape == ttnn_sort_values.shape
        assert torch_sort_indices.shape == ttnn_sort_indices.shape

        assert list(ttnn_sort_values.shape) == shape
        assert list(ttnn_sort_indices.shape) == shape

        assert_equal(torch_sort_values, ttnn_sort_values_torch)
        ttnn.synchronize_device(device)
    device.disable_and_clear_program_cache()
    assert (
        device.cache_entries_counter.total == 1
    ), "Expected only one program cache entry for sort operation, but found {}".format(
        device.cache_entries_counter.total
    )


@pytest.mark.parametrize(
    "shape, dim, descending, torch_value_dtype, ttnn_value_dtype, ttnn_index_dtype",
    [
        ([32, 64], -1, False, torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        ([32, 64], -1, False, torch.bfloat16, ttnn.bfloat16, ttnn.uint32),
        # UINT16 input always produces UINT32 indices (fp32_dest_acc_en is forced
        # so the SFPU uses 32-bit index writes). UINT16-index output is not valid.
        ([32, 64], -1, False, torch.uint8, ttnn.uint16, ttnn.uint32),
        # ([1, 8], -1, False, torch.uint8, ttnn.uint16, ttnn.uint16), # GH issue: #33473
    ],
)
def test_sort_datatypes(shape, dim, descending, torch_value_dtype, ttnn_value_dtype, ttnn_index_dtype, device):
    torch.manual_seed(0)

    if torch_value_dtype == torch.uint8 or torch_value_dtype == torch.int16:
        input = torch.randint(100, shape, dtype=torch_value_dtype)
    else:
        input = torch.randn(shape, dtype=torch_value_dtype)
    ttnn_input = ttnn.from_torch(input, ttnn_value_dtype, layout=ttnn.Layout.TILE, device=device)

    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)

    ttnn_sort_values = ttnn.zeros_like(ttnn_input, dtype=ttnn_value_dtype)
    ttnn_sort_indices = ttnn.zeros_like(ttnn_input, dtype=ttnn_index_dtype)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(ttnn_sort_values, ttnn_sort_indices))

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    if len(shape) == 0 or len(shape) == 1:
        assert torch_sort_values == ttnn.to_torch(ttnn_sort_values)
    else:
        assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values, dtype=torch_value_dtype))


@pytest.mark.parametrize(
    "n, hi, descending",
    [
        # Values ≤ 255: these pass even without the fix (bf16 is exact for 0..255)
        (64, 63, False),
        (256, 255, False),
        # Values > 256: these fail without fp32_dest_acc_en (GH issue #46331)
        (512, 511, False),
        (512, 511, True),
        (551, 550, False),  # Qwen window_index size
        (551, 550, True),
        (128, 512, False),  # values well above the bf16 threshold
    ],
)
def test_sort_uint16_index_correctness(n, hi, descending, device):
    torch.manual_seed(0)
    shape = [1, n]

    # Build n UNIQUE integers so argsort has a single correct answer.
    # Using float scaling loses uniqueness due to truncation, so we sample via randperm.
    # All test cases satisfy hi + 1 >= n.
    assert hi + 1 >= n, f"hi+1={hi+1} must be >= n={n} to guarantee uniqueness"
    if descending:
        # For descending sort, the sort pads with 0 (the smallest UINT16 sentinel).
        # Shift values to [1, hi+1] so that 0 is never an actual value; the
        # padding zeros then unambiguously sort past all real elements.
        vals = torch.randperm(hi + 1)[:n].to(torch.int32) + 1
    else:
        # For ascending sort the padding sentinel is 65535.  Values sampled from
        # [0, hi] (≤ 550 in all test cases) never reach 65535, so no conflict.
        vals = torch.randperm(hi + 1)[:n].to(torch.int32)
    x_torch = vals.reshape(shape)

    ttnn_input = ttnn.from_torch(x_torch, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)

    _sort_vals, ttnn_idx = ttnn.sort(ttnn_input, dim=-1, descending=descending)

    # After the fix UINT16 input always yields UINT32 indices.
    assert ttnn_idx.dtype == ttnn.uint32, f"Expected UINT32 indices for UINT16 input, got {ttnn_idx.dtype}"

    dev_idx = ttnn.to_torch(ttnn_idx).to(torch.int64)
    golden = torch.argsort(x_torch.to(torch.int64), dim=-1, descending=descending)

    assert_equal(golden, dev_idx)


@pytest.mark.parametrize(
    "n, hi, descending",
    [
        # pre_sort_transform_tensor pads the last dim to the next power of two
        # (>= 2*TILE_WIDTH = 64), so Wt = next_pow2(n_tile_aligned) / TILE_WIDTH.
        # Anything with Wt > SORT_WT_THRESHOLD (64) routes to the MultiCore
        # factory: n=2080 -> pad 4096 -> Wt=128; n=4096 -> Wt=128; n=8192 -> Wt=256.
        # Use hi <= 65535 so all values fit in UINT16.
        (2080, 2079, False),
        (2080, 2079, True),
        (4096, 4095, False),
        (8192, 8191, False),
    ],
)
def test_sort_uint16_index_correctness_multicore(n, hi, descending, device):
    """
    Regression test for UINT16 sort keys with Wt > SORT_WT_THRESHOLD.
    Selects the MultiCore factory (SortProgramFactorySingleRowMultiCore) whose
    reader/writer kernels now perform an element-wise UInt16↔Float32 conversion
    so the SFPU compares keys in fp32_dest_acc_en mode (exact for 0..65535).
    """
    torch.manual_seed(0)
    shape = [1, n]

    assert hi + 1 >= n, f"hi+1={hi+1} must be >= n={n} to guarantee uniqueness"
    if descending:
        vals = torch.randperm(hi + 1)[:n].to(torch.int32) + 1
    else:
        vals = torch.randperm(hi + 1)[:n].to(torch.int32)
    x_torch = vals.reshape(shape)

    ttnn_input = ttnn.from_torch(x_torch, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)

    _sort_vals, ttnn_idx = ttnn.sort(ttnn_input, dim=-1, descending=descending)

    assert ttnn_idx.dtype == ttnn.uint32, f"Expected UINT32 indices for UINT16 input, got {ttnn_idx.dtype}"

    dev_idx = ttnn.to_torch(ttnn_idx).to(torch.int64)
    golden = torch.argsort(x_torch.to(torch.int64), dim=-1, descending=descending)

    assert_equal(golden, dev_idx)


@pytest.mark.parametrize(
    "n, hi, descending",
    [
        # ROW_MAJOR UINT16 sort — pre_sort_transform_tensor pads the last dim
        # to the next power of two (>= 64), so Wt = next_pow2(n) / TILE_WIDTH.
        # UINT16 + ROW_MAJOR uses SORT_WT_THRESHOLD_UINT16_ROW_MAJOR = 32 so
        # Wt <= 32 (n <= 1024) routes to SingleCore RM and everything above
        # (including the Wt = 64 boundary that used to OOM in SingleCore) routes
        # to MultiCore RM, whose reader/writer now also do UInt16<->Float32.
        (64, 63, False),  # pad 64  -> Wt=2, SingleCore RM
        (551, 550, False),  # pad 1024 -> Wt=32, SingleCore RM (upper SingleCore bound)
        (551, 550, True),
        (2048, 2047, False),  # pad 2048 -> Wt=64, MultiCore RM (first width the new threshold reroutes)
        (2080, 2079, False),  # pad 4096 -> Wt=128, MultiCore RM
        (4096, 4095, False),  # pad 4096 -> Wt=128, MultiCore RM
        (4096, 4095, True),
    ],
)
def test_sort_uint16_row_major_correctness(n, hi, descending, device):
    """
    Regression test for UINT16 sort in ROW_MAJOR layout across both SingleCore
    (Wt <= 64) and MultiCore (Wt > 64) factories.  All values in [0, 65535] must
    round-trip through the UInt16↔Float32 conversion loops with exact indices.
    """
    torch.manual_seed(0)
    shape = [32, n]  # combined_h must be a multiple of TILE_HEIGHT (32) for RM

    assert hi + 1 >= n, f"hi+1={hi+1} must be >= n={n} to guarantee uniqueness"
    if descending:
        vals = torch.stack([torch.randperm(hi + 1)[:n].to(torch.int32) + 1 for _ in range(shape[0])])
    else:
        vals = torch.stack([torch.randperm(hi + 1)[:n].to(torch.int32) for _ in range(shape[0])])
    x_torch = vals

    ttnn_input = ttnn.from_torch(x_torch, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_vals, ttnn_idx = ttnn.sort(ttnn_input, dim=-1, descending=descending)

    assert ttnn_idx.dtype == ttnn.uint32, f"Expected UINT32 indices for UINT16 input, got {ttnn_idx.dtype}"

    dev_vals = ttnn.to_torch(ttnn_vals).to(torch.int64)
    dev_idx = ttnn.to_torch(ttnn_idx).to(torch.int64)
    golden_vals, golden_idx = torch.sort(x_torch.to(torch.int64), dim=-1, descending=descending)

    assert_equal(golden_vals, dev_vals)
    assert_equal(golden_idx, dev_idx)


def create_descending_tensor(shape, dim, dtype=torch.bfloat16):
    size_along_dim = shape[dim]

    # Step 1: Create descending range [size-1, size-2, ..., 0]
    descending_values = torch.arange(size_along_dim - 1, -1, -1, dtype=dtype)

    # Step 2: Reshape to fit into the target dimension with unsqueeze
    view_shape = [1] * len(shape)
    view_shape[dim] = size_along_dim
    descending_values = descending_values.view(*view_shape)

    # Step 3: Broadcast to full shape
    descending_tensor = descending_values.expand(*shape)

    return descending_tensor


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([32, 64], -1, False),
        ([1, 2048, 1, 64], -1, False),
        ([1, 55, 43], -1, True),
        ([11, 29, 14, 1], -1, True),
    ],
)
def test_sort_indices(shape, dim, descending, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = create_descending_tensor(shape, dim, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    torch_converted_indices = ttnn.to_torch(ttnn_sort_indices).to(torch.int64)

    assert_equal(torch_sort_values, ttnn.to_torch(ttnn_sort_values))
    assert_allclose(torch_sort_indices.to(torch.int64), torch_converted_indices)


@pytest.mark.parametrize(
    "shape, dim, descending, torch_dtype, ttnn_dtype",
    [
        ([64, 64], -1, False, torch.bfloat16, ttnn.bfloat16),
        ([32, 128], -1, True, torch.bfloat16, ttnn.bfloat16),
        ([1, 1, 32, 64], -1, False, torch.float32, ttnn.float32),
        ([1, 55, 43], -1, True, torch.bfloat16, ttnn.bfloat16),
        ([32, 128], 1, False, torch.bfloat16, ttnn.bfloat16),
        ([1, 64, 64], 0, True, torch.bfloat16, ttnn.bfloat16),
        ([1, 64, 64], 1, False, torch.float32, ttnn.float32),
        ([1, 1, 64, 64], 0, False, torch.float32, ttnn.float32),
        ([237], 0, False, torch.bfloat16, ttnn.bfloat16),
    ],
)
def test_sort_row_major_layout(shape, dim, descending, torch_dtype, ttnn_dtype, device):
    torch.manual_seed(0)

    input_t = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(input_t, ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_values, torch_indices = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert ttnn_values.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "Output layout must be ROW_MAJOR"
    assert ttnn_indices.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "Index layout must be ROW_MAJOR"

    assert list(ttnn_values.shape) == shape
    assert list(ttnn_indices.shape) == shape

    out_vals = ttnn.to_torch(ttnn_values, dtype=torch_dtype)
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    # For non-last-dim fp32, the composite layer wraps the device sort in a pair of
    # ttnn::transpose calls.  The RM transpose compute kernel routes data through DEST
    # (which holds ~10-bit mantissa on Wormhole even with fp32_dest_acc_en=true), so
    # fp32 values pick up ~1 TF32 ULP of error per hop and pairs of near-equal inputs
    # may swap places.  bf16's 7-bit mantissa is coarser than DEST, so bf16 stays
    # bit-exact and we keep the strict check there.
    is_dim_last_idx = (dim == -1) or (dim == len(shape) - 1)
    if torch_dtype == torch.float32 and not is_dim_last_idx:
        assert_allclose(torch_values, out_vals, rtol=1e-2, atol=1e-2)
        assert_allclose(torch_values, ttnn_gathered, rtol=1e-2, atol=1e-2)
    else:
        assert_equal(torch_values, out_vals)
        assert_equal(torch_values, ttnn_gathered)


def _make_sharded_cfg(memory_layout, grid_end_x, grid_end_y, shard_h, shard_w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_end_x, grid_end_y))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, spec)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
        ([8 * TILE_HEIGHT, TILE_WIDTH * 2], -1, False),
    ],
)
def test_sort_sharded_input(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    shard_height = TILE_HEIGHT
    shard_width = shape[-1]
    sharded_cfg = _make_sharded_cfg(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, shard_height, shard_width
    )

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        input_t,
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_cfg,
    )

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert list(ttnn_values.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
        ([8 * TILE_HEIGHT, TILE_WIDTH * 2], -1, False),
    ],
)
def test_sort_sharded_output(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    shard_height = TILE_HEIGHT
    shard_width = shape[-1]
    sharded_cfg = _make_sharded_cfg(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, shard_height, shard_width
    )

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        input_t,
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending, memory_config=sharded_cfg)

    assert ttnn_values.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert list(ttnn_values.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
    ],
)
def test_sort_row_major_sharded(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    shard_height = TILE_HEIGHT
    shard_width = shape[-1]
    sharded_cfg = _make_sharded_cfg(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, shard_height, shard_width
    )

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        input_t,
        ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_cfg,
    )

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert list(ttnn_values.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, True),
        ([1, 1, 32, 64], -1, False),
    ],
)
def test_sort_preallocated_row_major_outputs(shape, dim, descending, device):
    torch.manual_seed(0)

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_vals = ttnn.zeros(shape, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_idx = ttnn.zeros(shape, dtype=ttnn.uint16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(out_vals, out_idx))

    assert out_vals.get_layout() == ttnn.ROW_MAJOR_LAYOUT
    assert out_idx.get_layout() == ttnn.ROW_MAJOR_LAYOUT
    assert list(out_vals.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(out_vals))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(out_idx).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([4 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([8 * TILE_HEIGHT, TILE_WIDTH * 2], -1, True),
    ],
)
def test_sort_preallocated_sharded_outputs(shape, dim, descending, device):
    torch.manual_seed(0)

    num_shards = shape[0] // TILE_HEIGHT
    sharded_cfg = _make_sharded_cfg(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 0, num_shards - 1, TILE_HEIGHT, shape[-1])

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_vals = ttnn.zeros(shape, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)
    out_idx = ttnn.zeros(shape, dtype=ttnn.uint16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=sharded_cfg)

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn.sort(ttnn_input, dim=dim, descending=descending, out=(out_vals, out_idx))

    assert out_vals.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert list(out_vals.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(out_vals))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(out_idx).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim, descending",
    [
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], -1, False),
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], 0, False),
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], 2, True),
        ([2, 2, 2, 2 * TILE_HEIGHT, TILE_WIDTH], 3, False),
    ],
)
def test_sort_rank5_all_dims(shape, dim, descending, device):
    """Rank > 4 with sort dim ranging over all logical positions.

    The composite layer permutes the sort dim to the last position, squeezes
    leading dims into 4D, runs the kernel, then restores the original rank
    and order.  The rank-restoration reshape targets the *transposed* shape,
    keeping the last dim unchanged, so it routes through ttnn::reshape's
    `this_is_view` fast path (metadata-only `view`).  This avoids the device
    reshape kernel entirely — important because that kernel rejects UINT16.
    """
    torch.manual_seed(0)

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_values, _ = torch.sort(input_t, dim=dim, descending=descending)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending)

    assert list(ttnn_values.shape) == shape
    assert list(ttnn_indices.shape) == shape
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    ttnn_gathered = torch.gather(input_t, dim, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ([4 * TILE_HEIGHT, 2 * TILE_WIDTH], 0),
        ([2, 3 * TILE_HEIGHT, 2 * TILE_WIDTH], 1),
    ],
)
def test_fp32_non_last_dim_index_validation(shape, dim, device):
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    ref_vals, _ = torch.sort(t, dim=dim)

    x = ttnn.from_torch(
        t, dtype=ttnn.float32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    v, i = ttnn.sort(x, dim=dim)

    assert i.dtype == ttnn.uint32, f"FP32 input must produce UINT32 indices, got {i.dtype}"

    out_vals = ttnn.to_torch(v).float()
    out_idx = ttnn.to_torch(i).to(torch.int64)

    # The kernel pipeline routes fp32 through DEST (which holds ~10-bit mantissa even
    # with fp32_dest_acc_en=true on Wormhole) during the bitonic merge stage. This
    # causes two kinds of small deviations vs torch.sort:
    #   * sorted-values output: per-element ATOL up to ~1 TF32 ULP at the value's
    #     magnitude (~4e-3 for values around 4).
    #   * returned indices: pairs of inputs that differ by less than the kernel's
    #     precision can swap their relative sort order, so a returned index may point
    #     to a neighbour of the position torch.sort picked — which is still "correct"
    #     up to the same ULP tolerance when we gather from the original fp32 input.
    assert_allclose(out_vals, ref_vals, rtol=1e-2, atol=1e-2)

    gathered = torch.gather(t, dim, out_idx)
    assert_allclose(gathered.float(), ref_vals.float(), rtol=1e-2, atol=1e-2)


def test_fp32_input_uint16_preallocated_index_rejected(device, expect_error):
    shape = [TILE_HEIGHT, 2 * TILE_WIDTH]
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(
        t, dtype=ttnn.float32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )
    out_v = ttnn.zeros(
        shape, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_i = ttnn.zeros(
        shape, dtype=ttnn.uint16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with expect_error(RuntimeError, "must be UINT32 when input dtype is FLOAT32"):
        ttnn.sort(x, dim=-1, out=(out_v, out_i))


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_width_sharded(layout, device):
    torch.manual_seed(0)
    shape = [2 * TILE_HEIGHT, 4 * TILE_WIDTH]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref_vals, _ = torch.sort(t, dim=-1)

    cfg = _make_sharded_cfg(ttnn.TensorMemoryLayout.WIDTH_SHARDED, 3, 0, 2 * TILE_HEIGHT, TILE_WIDTH)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, memory_config=cfg, layout=layout)
    v, i = ttnn.sort(x, dim=-1)

    out = ttnn.to_torch(v).float()
    assert list(v.shape) == shape
    assert torch.allclose(
        out, ref_vals.float(), rtol=1e-2, atol=1e-2
    ), f"values mismatch max_diff={(out - ref_vals.float()).abs().max():.4f}"
    gathered = torch.gather(t, -1, ttnn.to_torch(i).to(torch.int64))
    assert torch.allclose(gathered.float(), ref_vals.float(), rtol=1e-2, atol=1e-2), "index gather mismatch"


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@pytest.mark.timeout(600, method="thread")
@pytest.mark.parametrize("descending", [False, True])
def test_sort_multi_row_multi_core_no_deadlock(descending, device):
    """
    Guard for the DRAM multi-core sort path (SortProgramFactorySingleRowMultiCore).

    The coordinator core collects two logically distinct worker signals -- the reader's
    per-row "ready" and the writer's per-pair "done" -- on two separate cores->coordinator
    semaphores, one producer signal each.  They are kept separate so each coordinator wait
    has an exact, monotonic per-producer target: were both folded onto one shared counter,
    at a tile-row boundary (Ht >= 2) a fast reader's next-row "ready" increment could push
    the counter past the "done" target an exact-match wait is looking for, stranding the
    wait and deadlocking the op.

    This exercises the multi-core Ht >= 2 path -- otherwise only covered at Ht == 1 -- and
    checks it runs to completion with correct output.  It is not a deterministic deadlock
    reproducer: the mismatch a shared counter would cause is timing-dependent (the exact-match
    poll normally out-races the NoC atomics), so it needs timing pressure to surface.

    The worker-thread watchdog + pytest-timeout are a best-effort regression guard,
    not a clean recovery mechanism: a genuine deadlock wedges the device (which needs
    a reset regardless), so the join times out and the assertion below fires, but the
    function-scoped `device` fixture teardown (close_device) may then block until the
    process-level pytest-timeout terminates the run.  The guarantee is only that a
    regression surfaces as a CI *failure* (never a silent pass); cleanly isolating a
    hang from teardown would require running the op in a killable subprocess.

    The multi-core factory is only selected when the (power-of-two padded) tile width
    exceeds total_cores * 128, so the width is sized from the device grid.
    """
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    wt = _next_pow2(total_cores * 128 + 1)  # smallest pow2 Wt on the DRAM multi-core path
    shape = [1, 1, 2 * TILE_HEIGHT, wt * TILE_WIDTH]  # Ht = 2

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    result = {}

    def _run():
        try:
            values, indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)
            ttnn.synchronize_device(device)
            result["values"] = values
            result["indices"] = indices
        except Exception as exc:  # surface device/compile errors to the main thread
            result["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=300.0)

    assert not worker.is_alive(), (
        "ttnn.sort did not complete on the DRAM multi-core path (Ht=2): the coordinator's "
        "cores->coordinator wait was starved -- likely a regression of the ready/done "
        "semaphore split."
    )
    if "error" in result:
        raise result["error"]

    torch_values, _ = torch.sort(input_t, dim=-1, descending=descending)
    assert list(result["values"].shape) == shape
    assert_equal(torch_values, ttnn.to_torch(result["values"]))
    ttnn_gathered = torch.gather(input_t, -1, ttnn.to_torch(result["indices"]).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_block_sharded(layout, device):
    torch.manual_seed(0)
    shape = [2 * TILE_HEIGHT, 4 * TILE_WIDTH]
    t = torch.randn(shape, dtype=torch.bfloat16)
    ref_vals, _ = torch.sort(t, dim=-1)

    cfg = _make_sharded_cfg(ttnn.TensorMemoryLayout.BLOCK_SHARDED, 1, 1, TILE_HEIGHT, 2 * TILE_WIDTH)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, memory_config=cfg, layout=layout)
    v, i = ttnn.sort(x, dim=-1)

    out = ttnn.to_torch(v).float()
    assert list(v.shape) == shape
    assert torch.allclose(
        out, ref_vals.float(), rtol=1e-2, atol=1e-2
    ), f"values mismatch max_diff={(out - ref_vals.float()).abs().max():.4f}"
    gathered = torch.gather(t, -1, ttnn.to_torch(i).to(torch.int64))
    assert torch.allclose(gathered.float(), ref_vals.float(), rtol=1e-2, atol=1e-2), "index gather mismatch"


@pytest.mark.parametrize("descending", [False, True])
def test_sort_row_major_multi_core_correctness(descending, device):
    torch.manual_seed(42)

    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    wt = _next_pow2(total_cores * 128 + 1)  # smallest pow2 Wt on the DRAM multi-core path
    shape = [1, 1, TILE_HEIGHT, wt * TILE_WIDTH]

    input_t = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(input_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)

    assert ttnn_values.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "output layout must be ROW_MAJOR"
    assert ttnn_indices.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "index layout must be ROW_MAJOR"
    assert list(ttnn_values.shape) == shape, f"values shape mismatch: got {list(ttnn_values.shape)}, expected {shape}"
    assert (
        list(ttnn_indices.shape) == shape
    ), f"indices shape mismatch: got {list(ttnn_indices.shape)}, expected {shape}"

    torch_values, _ = torch.sort(input_t, dim=-1, descending=descending)

    out_vals = ttnn.to_torch(ttnn_values)
    assert_equal(torch_values, out_vals)

    ttnn_gathered = torch.gather(input_t, -1, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn_gathered)


# ---------------------------------------------------------------------------
# stable=True (issue #33492)
#
# The stable contract is torch.sort(..., stable=True): equal values keep their
# original (ascending-index) order in the output, in BOTH sort directions.
# Assertions are EXACT (assert_equal on indices against the torch-stable
# reference), never PCC: a stable sort that is "almost right" is wrong.
#
# Device-side value canonicalization folds -0.0 into +0.0 (torch compares
# -0.0 == +0.0, so the tie CLASSES and therefore the index reference are
# unaffected); value comparisons below normalize zero signs on both sides.
# ---------------------------------------------------------------------------


def _fold_zero_sign(t):
    return torch.where(t == 0, torch.zeros_like(t), t)


def _run_stable_sort_case(device, input_tensor, dim, descending, ttnn_dtype, layout=ttnn.Layout.TILE):
    torch_values, torch_indices = torch.sort(input_tensor, dim=dim, descending=descending, stable=True)

    ttnn_input = ttnn.from_torch(input_tensor, ttnn_dtype, layout=layout, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending, stable=True)

    assert list(ttnn_values.shape) == list(input_tensor.shape)
    assert list(ttnn_indices.shape) == list(input_tensor.shape)

    dev_indices = ttnn.to_torch(ttnn_indices).to(torch.int64)
    dev_values = ttnn.to_torch(ttnn_values)

    # Indices must exactly match the torch-stable reference (this subsumes the
    # permutation property and the tie ordering in one check).
    assert_equal(torch_indices, dev_indices)
    # Values must match bit-exactly up to the -0.0 -> +0.0 fold.
    if dev_values.dtype.is_floating_point:
        assert_equal(_fold_zero_sign(torch_values), _fold_zero_sign(dev_values))
    else:
        assert_equal(torch_values, dev_values)


def _tie_heavy_tensor(shape, levels, seed, dtype=torch.bfloat16):
    """Random draw from a small set of exactly-representable levels: guarantees
    massive tie groups spanning tile / 64-column / per-core partition boundaries."""
    g = torch.Generator().manual_seed(seed)
    choice = torch.randint(0, len(levels), shape, generator=g)
    return torch.tensor(levels, dtype=dtype)[choice]


@pytest.mark.parametrize(
    "shape, dim, descending, stable",
    [
        ([32, 64], -1, False, True),
        ([32, 64], -1, True, True),
    ],
)
def test_sort_stable_issue33492_repro(shape, dim, descending, device, stable):
    """Literal reproduction from issue #33492."""
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    torch_sort_values, torch_sort_indices = torch.sort(input, dim=dim, descending=descending, stable=stable)
    ttnn_sort_values, ttnn_sort_indices = ttnn.sort(ttnn_input, dim=dim, descending=descending, stable=stable)

    assert torch_sort_values.shape == ttnn_sort_values.shape
    assert torch_sort_indices.shape == ttnn_sort_indices.shape

    assert list(ttnn_sort_values.shape) == shape
    assert list(ttnn_sort_indices.shape) == shape

    assert_allclose(torch_sort_values, ttnn.to_torch(ttnn_sort_values))
    if stable:
        assert_allclose(
            torch_sort_indices.to(torch.int64),
            ttnn.to_torch(ttnn_sort_indices).to(torch.int64),
        )


# Widths route to all three program factories (BH p150a 13x10 grid = 130 cores;
# CrossCore capacity = cores * min(128, max(Wt//cores, 2)) tiles):
#   W=64    -> Wt=2    SingleRowSingleCore
#   W=2048  -> Wt=64   SingleRowSingleCore (threshold boundary)
#   W=4096  -> Wt=128  CrossCoreDataExchange
#   W=8192  -> Wt=256  CrossCoreDataExchange (capacity boundary)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("width", [64, 2048, 4096, 8192])
@pytest.mark.parametrize(
    "levels",
    [
        # Negative ties: the sign-magnitude SFPSWAP asymmetry killer.
        [-3.5, -1.25, -1.25, 2.0],
        # Mixed +-0: one tie class, folded on device, ordered by index.
        [-1.0, -0.0, 0.0, 1.5],
        # +-Inf ties (also collide with the composite's +-inf padding sentinels).
        [float("-inf"), -2.0, 2.0, float("inf")],
    ],
    ids=["neg_ties", "signed_zero", "inf_ties"],
)
def test_sort_stable_tie_heavy(width, descending, levels, device):
    input_tensor = _tie_heavy_tensor([32, width], levels, seed=width + int(descending))
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("width", [64, 4096])
def test_sort_stable_all_equal(width, descending, device):
    """Every element ties: output indices must be the identity permutation."""
    input_tensor = torch.full([32, width], 1.25, dtype=torch.bfloat16)
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_padded_width(descending, device):
    """W=96 pads to 128 with +-inf sentinels; tie levels include +-inf so real
    infs tie with the padding and must still come out in index order."""
    levels = [float("-inf"), -1.5, -1.5, 0.0, 1.5, float("inf")]
    input_tensor = _tie_heavy_tensor([32, 96], levels, seed=7)
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_non_last_dim(descending, device):
    """Composite transpose path: stable along dim=-2."""
    levels = [-2.5, -0.5, -0.5, 0.5, 2.5]
    input_tensor = _tie_heavy_tensor([1, 1, 64, 64], levels, seed=11)
    _run_stable_sort_case(device, input_tensor, -2, descending, ttnn.bfloat16)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_row_major(descending, device):
    levels = [-4.0, -1.0, -1.0, 0.0, 1.0]
    input_tensor = _tie_heavy_tensor([1, 1, 32, 128], levels, seed=13)
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_single_core_fused_dtype(descending, device):
    """Stable bf16 sorts in the single-core width band (Wt <= 64) return UINT32 indices — the
    32-bit-DEST rule the fused true-index-tag engine requires (same rule fp32/u16 inputs follow).
    The dtype contract holds regardless of which factory the router picks: on Blackhole this
    Ht=1 W=2048 cell runs the mergesort row engine (W=512/1024 siblings reroute to the
    CrossCore comparator), elsewhere it runs the fused single-core engine like the large-Ht
    sibling test below — every engine must produce UINT32 indices and torch-stable exactness."""
    levels = [-1.5, -0.5, -0.5, 0.5, 1.5]
    input_tensor = _tie_heavy_tensor([32, 2048], levels, seed=31)

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True)

    assert ttnn_indices.dtype == ttnn.uint32
    assert_equal(torch_indices, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))


# ---------------------------------------------------------------------------
# Mergesort row engine (issue #33492 roadmap): on Blackhole, stable bfloat16
# sorts of padded width 2048/4096 run a full per-row sort on the TopK XL SFPU
# kernels (fused linearly-tagged keys, both-halves merge level) — one row per
# core. On other archs the same shapes keep the previous engines; every test
# below asserts torch-stable exactness either way, so they double as parity
# tests for whichever engine the router picks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("width", [2048, 4096])
def test_sort_stable_mergesort_row_engine(width, layout, descending, device):
    """Tie-heavy (negative ties, zeros, positive ties) exactness on the
    mergesort widths, both layouts. W=4096 exercises the both-halves merge
    level; W=2048 is leaf-only."""
    levels = [-1.5, -0.5, -0.5, 0.0, 0.5, 1.5]
    input_tensor = _tie_heavy_tensor([32, width], levels, seed=61 + width + int(descending))
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16, layout=layout)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_mergesort_multi_tile_row(descending, device):
    """H=96 (three tile-rows, one row per core) plus a non-multiple-of-32 H:
    the row-parallel work split must cover every real row exactly once."""
    levels = [-2.5, -0.5, -0.5, 0.5, 2.5]
    input_tensor = _tie_heavy_tensor([96, 2048], levels, seed=67)
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16)

    input_tensor = _tie_heavy_tensor([40, 4096], levels, seed=71)
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_mergesort_signed_zero(descending, device):
    """-0.0 and +0.0 are one tie class under the stable contract (torch
    semantics); the engine's copy path canonicalizes -0.0 on the way in."""
    levels = [-1.0, -0.0, 0.0, 0.0, 1.0]
    input_tensor = _tie_heavy_tensor([32, 2048], levels, seed=73)
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_mergesort_w4096_index_dtype(descending, device):
    """On Blackhole the W=4096 stable bf16 cell runs the mergesort engine whose
    fused tags live in 32-bit DEST, so its indices are UINT32 (previously this
    width returned UINT16 from the CrossCore comparator). Other archs keep the
    old dtype; both must be torch-stable exact."""
    levels = [-1.5, -0.5, -0.5, 0.5, 1.5]
    input_tensor = _tie_heavy_tensor([32, 4096], levels, seed=79)

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True)

    if is_blackhole():
        assert ttnn_indices.dtype == ttnn.uint32
    assert_equal(torch_indices, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_mergesort_prealloc_u16_opt_out(descending, device):
    """A caller who preallocates UINT16 index tensors opts out of the mergesort
    engine (its indices are inherently 32-bit); the previous routing must still
    produce a torch-stable-exact result."""
    levels = [-1.5, -0.5, -0.5, 0.5, 1.5]
    input_tensor = _tie_heavy_tensor([32, 2048], levels, seed=83)

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    out_values = ttnn.zeros_like(ttnn_input)
    out_indices = ttnn.zeros((32, 2048), dtype=ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True, out=(out_values, out_indices))

    assert out_indices.dtype == ttnn.uint16
    assert_equal(torch_indices, ttnn.to_torch(out_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(out_values))


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_single_core_fused_large_ht(descending, device):
    """Large-Ht guard for the small-Ht CrossCore reroute: Ht=9 exceeds the Wt/8=8 cutoff at
    Wt=64, so this shape stays on the single-core factory and runs the FUSED true-index-tag
    engine (per-core tile-row fanning is optimal there). UINT32 indices + stable exactness."""
    levels = [-1.5, -0.5, -0.5, 0.5, 1.5]
    input_tensor = _tie_heavy_tensor([9 * 32, 2048], levels, seed=37)

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True)

    assert ttnn_indices.dtype == ttnn.uint32
    assert_equal(torch_indices, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize("width", [512, 1024, 2048])
def test_sort_small_ht_cross_core_reroute(width, stable, descending, device):
    """Small-Ht STABLE cells in the single-core width band (Wt in [16, 64], Ht <= Wt/8, Blackhole)
    reroute to the CrossCore comparator (measured 5-12x faster there; the single-core factory
    would idle the rest of the grid). Unstable cells deliberately keep the single-core engine:
    whenever CrossCore serves an unstable sort it resolves ties positionally and emits
    duplicate indices inside tie groups (issue #54043) — measured on silicon at every width
    it serves, W=512-4096, on plain randn (bf16 quantization makes ties common), not just
    tie-heavy input at its native W=4096/8192 band. Exactness parity on tie-heavy input for
    both stability modes, and a full permutation check on the unstable arm."""
    levels = [-1.5, -0.5, -0.5, 0.5, 1.5]
    input_tensor = _tie_heavy_tensor([32, width], levels, seed=41 + width + int(descending))

    if stable:
        _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.bfloat16)
        return

    torch_values, _ = torch.sort(input_tensor, dim=-1, descending=descending)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)

    dev_indices = ttnn.to_torch(ttnn_indices).to(torch.int64)
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))
    # Unstable contract: exact values, the indices gather the sorted values, AND every row's
    # indices are a permutation of [0, W) — the single-core engine moves value+index
    # atomically per SFPSWAP, so ties never duplicate an index.
    gathered = torch.gather(input_tensor, -1, dev_indices)
    assert_equal(torch_values, gathered)
    expected_perm = torch.arange(width, dtype=torch.int64).expand(dev_indices.shape[0], -1)
    assert torch.equal(
        torch.sort(dev_indices, dim=-1).values, expected_perm
    ), "unstable sort indices are not a permutation per row"


@pytest.mark.parametrize("width", [512, 1024, 2048])
def test_sort_unstable_all_ones_permutation(width, device):
    """Regression (adversarial swarm minimal repro): [32, W] all-ones bf16 TILE stable=False
    must return a valid permutation per row. When the small-Ht reroute predicate did not
    exclude stable=False, these cells landed on the CrossCore factory whose unstable exchange
    resolves ties positionally — every row came back with duplicate indices (448/512 duplicate
    entries per row at W=512). They now run the single-core engine again and pass."""
    input_tensor = torch.full((32, width), 1.0).bfloat16()

    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, stable=False)

    assert_equal(input_tensor, ttnn.to_torch(ttnn_values))  # all-ones: values must be unchanged
    dev_indices = ttnn.to_torch(ttnn_indices).to(torch.int64)
    expected_perm = torch.arange(width, dtype=torch.int64).expand(32, -1)
    assert torch.equal(
        torch.sort(dev_indices, dim=-1).values, expected_perm
    ), "unstable sort indices are not a permutation per row"


@pytest.mark.parametrize("descending", [False, True])
def test_sort_small_ht_cross_core_reroute_fp32(descending, device):
    """fp32 unstable input at the small-Ht shape: stays on the single-core engine (the reroute
    is stable-only) with UINT32 indices (the fp32 dtype rule is routing-independent)."""
    torch.manual_seed(53)
    input_tensor = torch.randn([32, 2048], dtype=torch.float32)

    torch_values, _ = torch.sort(input_tensor, dim=-1, descending=descending)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.float32, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending)

    assert ttnn_indices.dtype == ttnn.uint32
    dev_indices = ttnn.to_torch(ttnn_indices).to(torch.int64)
    assert_equal(torch_values, ttnn.to_torch(ttnn_values, dtype=torch.float32))
    gathered = torch.gather(input_tensor, -1, dev_indices)
    assert_equal(torch_values, gathered)


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_multicore_dram_comparator(descending, device):
    """W=32768: Wt=1024 exceeds the CrossCore capacity, so this is the SingleRowMultiCore
    DRAM factory, which deliberately keeps the index-aware COMPARATOR stable engine with u16
    index transport: it is data-movement-bound (its comparator runs within ~0.3% of the
    unstable floor), so the fused engine's u32-index transport only adds cost there (+10%
    measured at this width). Tie groups straddle every per-substage pair boundary."""
    levels = [-2.0, -1.0, -1.0, 0.0, 1.0, 3.0]
    input_tensor = _tie_heavy_tensor([32, 32768], levels, seed=23)

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True)

    assert ttnn_indices.dtype == ttnn.uint16
    assert_equal(torch_indices, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_wide_u32_index(descending, device):
    """W=65536 >= the u16 index ceiling: UINT32 indices, 32-bit DEST, and (on a
    grid whose core count does not divide Wt=2048) the SingleRowMultiCore DRAM
    factory. Tie groups straddle every per-core partition boundary."""
    levels = [-2.0, -1.0, -1.0, 0.0, 1.0, 3.0]
    input_tensor = _tie_heavy_tensor([32, 65536], levels, seed=17)

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True)

    assert ttnn_indices.dtype == ttnn.uint32
    assert_equal(torch_indices, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))


@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_past_tag_ceiling_comparator(descending, device):
    """W=131072: stable sort keeps the index-aware comparator network — the width exceeds the
    fused engine's u16 tag ceiling (a 16-bit tag cannot address 131072 positions), and
    per-merge position-derived rank stamps are unsound at the bitonic network's inner
    substages (tiles there hold elements of MIXED half-origin, so no static per-pair rank
    range assignment orders cross-tile ties by true index). Tie-heavy parity must hold on
    the comparator across every per-substage boundary."""
    levels = [-2.0, -1.0, -1.0, 0.0, 1.0, 3.0]
    input_tensor = _tie_heavy_tensor([32, 131072], levels, seed=29)

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True)

    assert ttnn_indices.dtype == ttnn.uint32
    assert_equal(torch_indices, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(ttnn_values))


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("width", [64, 4096], ids=["single_core", "multi_core"])
def test_sort_stable_float32(width, descending, device):
    """FLOAT32 values (32-bit DEST + UINT32 indices), tie-heavy including a
    mixed +-0.0 class."""
    levels = [-3.25, -1.0, -0.0, 0.0, 0.0, 1.0]
    input_tensor = _tie_heavy_tensor([32, width], levels, seed=19, dtype=torch.float32)
    _run_stable_sort_case(device, input_tensor, -1, descending, ttnn.float32)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("width", [64, 4096], ids=["single_core", "multi_core"])
def test_sort_stable_uint16(width, descending, device):
    """UINT16 values (uint16-in-32-bit-DEST path; Wt>threshold routes to the
    SingleRowMultiCore factory). Values deliberately include the padding
    sentinels (0 and 65535): with stable sort the real elements' lower indices
    order them ahead of the padding, so the sentinel collision is benign."""
    g = torch.Generator().manual_seed(width + int(descending))
    tie_levels = torch.tensor([0, 1, 1, 7, 300, 65535], dtype=torch.int32)
    choice = torch.randint(0, len(tie_levels), [32, width], generator=g)
    input_tensor = tie_levels[choice]

    torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=descending, stable=True)
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=descending, stable=True)

    assert ttnn_indices.dtype == ttnn.uint32
    assert_equal(torch_indices, ttnn.to_torch(ttnn_indices).to(torch.int64))
    assert_equal(torch_values, ttnn.to_torch(ttnn_values).to(torch.int32))


@pytest.mark.parametrize(
    "shape",
    [
        [32, 64],  # SingleRowSingleCore
        [32, 4096],  # CrossCoreDataExchange (mergesort row engine on Blackhole)
    ],
    ids=["single_core", "cross_core"],
)
def test_sort_stable_program_cache(shape, device):
    """stable must enter the program hash: alternate stable/unstable on the
    same shape and re-run the stable program from cache on fresh data. A hash
    that ignores `stable` returns the unstable program on run 3."""
    levels = [-1.5, -0.5, -0.5, 0.5, 1.5]

    entries = []
    for iteration, stable in enumerate([True, False, True, False]):
        input_tensor = _tie_heavy_tensor(shape, levels, seed=100 + iteration)
        torch_values, torch_indices = torch.sort(input_tensor, dim=-1, descending=True, stable=True)

        ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
        with device.cache_entries_counter.measure():
            ttnn_values, ttnn_indices = ttnn.sort(ttnn_input, dim=-1, descending=True, stable=stable)

        dev_indices = ttnn.to_torch(ttnn_indices).to(torch.int64)
        if stable:
            assert_equal(torch_indices, dev_indices)
            assert_equal(torch_values, ttnn.to_torch(ttnn_values))
        else:
            # Unstable run: values still exact, indices a valid permutation of a correct sort.
            assert_equal(torch_values, ttnn.to_torch(ttnn_values))
            gathered = torch.gather(input_tensor, -1, dev_indices)
            assert_equal(torch_values, gathered)
        ttnn.synchronize_device(device)
        entries.append(device.cache_entries_counter.total)

    device.disable_and_clear_program_cache()
    # Two distinct programs total (stable and unstable), each compiled exactly once.
    assert entries[-1] == 2, f"Expected 2 program cache entries (stable + unstable), found {entries[-1]}"
