# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("w", [1, 4, 8, 32])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.int32,
        ttnn.uint32,
    ],
)
def test_plus_one(device, w, dtype):
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, device=device)
    ttnn.plus_one(input_tensor)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
def test_plus_one_subdevice(device, w):
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(
        input_tensor, sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))])
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("input_shape", [(16, 32), (32, 32), (4, 16, 32), (4, 8, 16, 32)])
def test_plus_one_subdevice_nd(device, input_shape):
    torch_input_tensor = torch.randint(32000, input_shape)
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(
        input_tensor,
        sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))]),
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
@pytest.mark.parametrize("val", [-1, -100])
def test_plus_one_with_neg_entries(device, w, val):
    torch_input_tensor = torch.randint(32000, (w,))
    mask = torch.rand(w) < 0.3
    torch_input_tensor[mask] = val
    torch_output_tensor = torch.where(torch_input_tensor < 0, torch_input_tensor, torch_input_tensor + 1)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor, skip_negative_entries=True)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("input_shape", [(16, 32), (32, 32), (4, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("val", [-1, -100])
def test_plus_one_with_neg_entries_nd(device, input_shape, val):
    torch_input_tensor = torch.randint(32000, input_shape)
    mask = torch.rand(input_shape) < 0.3
    torch_input_tensor[mask] = val
    torch_output_tensor = torch.where(torch_input_tensor < 0, torch_input_tensor, torch_input_tensor + 1)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor, skip_negative_entries=True)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def test_plus_one_program_cache(device):
    """Base-case (opt-out) cache contract for the MetalV2 factory.

    plusone's only per-dispatch value is which tensor is bound: the buffer address is excluded from the
    cache key, and on a cache hit it is re-applied via UpdateTensorArgs (from the spec's run_args). This
    test pins both halves of that contract so neither can regress:

      * a second tensor of the SAME shape (different allocation/address) must reuse the cached program
        (no new entry) -> guards against the address leaking into the key (recompile-per-call).
      * that second tensor must be incremented correctly -> guards against a cache hit reusing the first
        call's stale baked-in address (the frozen-binding bug) instead of re-applying the current one.
      * a different shape must add a new cache entry, and the right program must be selected for each.
    """
    device.enable_program_cache()
    device.clear_program_cache()
    w = 32

    a = ttnn.from_torch(torch.full((w,), 5, dtype=torch.int32), dtype=ttnn.int32, device=device)
    ttnn.plus_one(a)
    assert torch.equal(ttnn.to_torch(a), torch.full((w,), 6, dtype=torch.int32))
    assert device.num_program_cache_entries() == 1, "first call should create exactly one cache entry"

    # Fresh allocation => different buffer address, same shape. Must cache-hit (no new entry) AND increment
    # the NEW tensor -- proving the address was re-applied (UpdateTensorArgs), not left stale at a's address.
    b = ttnn.from_torch(torch.full((w,), 10, dtype=torch.int32), dtype=ttnn.int32, device=device)
    ttnn.plus_one(b)
    assert (
        device.num_program_cache_entries() == 1
    ), "same shape, different address must reuse the program (address is not part of the key)"
    assert torch.equal(
        ttnn.to_torch(b), torch.full((w,), 11, dtype=torch.int32)
    ), "cache hit must increment the CURRENT tensor (address re-applied), not stale first-call data"

    # A different shape is a different key => a second, coexisting cache entry, selected correctly.
    c = ttnn.from_torch(torch.full((64,), 1, dtype=torch.int32), dtype=ttnn.int32, device=device)
    ttnn.plus_one(c)
    assert device.num_program_cache_entries() == 2, "a new shape must add a cache entry"
    assert torch.equal(ttnn.to_torch(c), torch.full((64,), 2, dtype=torch.int32))

    device.disable_and_clear_program_cache()
