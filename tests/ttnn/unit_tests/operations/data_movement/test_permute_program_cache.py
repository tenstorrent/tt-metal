# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache regression tests for ttnn.permute (PermuteDeviceOperation).

These tests pin the program-cache keying granularity so it can be verified
BEFORE and AFTER removing PermuteDeviceOperation::compute_program_hash and
falling back to the framework default hash. The default hashes the full
operation_attributes (dims, output_mem_config, pad_value) plus each input
tensor's TensorSpec (logical_shape + tensor_layout, from which padded_shape
is derived), so it must produce exactly the same number of cache entries:

- Same config -> reuse (1 entry). Different data alone must NOT re-key.
- Different dims / shape / dtype -> distinct entries.
- Two permutations that select DIFFERENT program factories -> distinct entries,
  even though the default hash does NOT fold factory.index() (the custom hash
  did). This holds because select_program_factory is a pure function of dims +
  layout, which the default already hashes.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_permute(device, shape, dims, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, seed=0):
    """Run ttnn.permute inside the cache counter; return (torch_ref, tt_out)."""
    torch.manual_seed(seed)
    torch_input = torch.rand(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    torch_ref = torch.permute(torch_input, dims)

    tt_input = ttnn.from_torch(torch_input, layout=layout, dtype=dtype, device=device)
    with device.cache_entries_counter.measure():
        tt_out = ttnn.permute(tt_input, dims)
    tt_out = ttnn.to_torch(tt_out)
    return torch_ref, tt_out


# =============================================================================
# Cache reuse (fields correctly NOT keyed)
# =============================================================================


def test_permute_cache_reuse_same_config(device, isolate_program_cache):
    """Same shape/dims/dtype run twice with different data -> 1 entry, different outputs."""
    shape = (1, 1, 32, 64)
    dims = (0, 1, 3, 2)

    ref1, out1 = run_permute(device, shape, dims, seed=0)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_permute(device, shape, dims, seed=42)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 1
    assert not torch.equal(out1, out2)


# =============================================================================
# Cache miss (fields correctly keyed)
# =============================================================================


def test_permute_cache_miss_different_dims(device, isolate_program_cache):
    """Different permutations (different program factories) -> 2 entries.

    (0,1,3,2) selects MultiCoreTileInvariant; (0,2,1,3) selects
    MultiCoreTileRowInvariant. Distinct despite no factory.index() in the hash.
    """
    shape = (1, 1, 32, 64)

    ref1, out1 = run_permute(device, shape, (0, 1, 3, 2))
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_permute(device, shape, (0, 2, 1, 3))
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_permute_cache_miss_different_shape(device, isolate_program_cache):
    """Same dims, different input shape -> 2 entries."""
    dims = (0, 1, 3, 2)

    ref1, out1 = run_permute(device, (1, 1, 32, 64), dims)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_permute(device, (1, 1, 64, 96), dims)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_permute_cache_miss_different_dtype(device, isolate_program_cache):
    """Same dims/shape, different dtype -> 2 entries."""
    shape = (1, 1, 32, 64)
    dims = (0, 1, 3, 2)

    ref1, out1 = run_permute(device, shape, dims, dtype=ttnn.bfloat16)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_permute(device, shape, dims, dtype=ttnn.float32)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2


def test_permute_cache_miss_different_factory_row_major(device, isolate_program_cache):
    """Row-major permutations selecting different factories -> 2 entries.

    (1,0,2,3) keeps the last dim (MultiCoreRowInvariant); (0,1,3,2) moves it
    (MultiCoreBlockedGeneric). The default hash distinguishes them via dims.
    """
    shape = (2, 3, 32, 64)

    ref1, out1 = run_permute(device, shape, (1, 0, 2, 3), layout=ttnn.ROW_MAJOR_LAYOUT)
    assert_with_pcc(ref1, out1, 0.9999)
    ref2, out2 = run_permute(device, shape, (0, 1, 3, 2), layout=ttnn.ROW_MAJOR_LAYOUT)
    assert_with_pcc(ref2, out2, 0.9999)

    assert device.cache_entries_counter.total == 2
