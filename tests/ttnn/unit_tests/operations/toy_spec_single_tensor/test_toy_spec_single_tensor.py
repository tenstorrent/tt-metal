# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.generic_op with a ONE-element io_tensors: a write-only program and an in-place program.

Both were rejected by the old arity check (`io_tensors.size() >= 2`), which stated a semantic rule
("at least one input and one output") that the op never actually needed -- only that io_tensors.back()
names the output tensor.
"""

import pytest
import torch

import ttnn
from ttnn.operations.toy_spec_single_tensor import toy_spec_fill, toy_spec_square_
from ttnn.operations.toy_spec_single_tensor.toy_spec_single_tensor import TP_OUT, create_fill_spec


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _on_device(device, torch_tensor):
    return ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 128, 128), (1, 2, 256, 64)])
@pytest.mark.parametrize("value", [1.0, -2.5, 0.75])
def test_fill_writes_every_tile(device, shape, value):
    """Write-only program: the sentinel must be gone everywhere, so every tile was written."""
    sentinel = torch.full(shape, -7.0, dtype=torch.bfloat16)
    out = _on_device(device, sentinel)

    returned = toy_spec_fill(out, value)

    assert returned.buffer_address() == out.buffer_address(), "the single io tensor is the output"
    got = ttnn.to_torch(returned).float()
    assert torch.equal(got, torch.full(shape, value, dtype=torch.float32)), got


def test_fill_hits_the_program_cache(device):
    """A second fill of the same shape must reuse the entry, i.e. the spec path is stable."""
    out1 = _on_device(device, torch.zeros((1, 1, 64, 64), dtype=torch.bfloat16))
    toy_spec_fill(out1, 3.0)
    entries = device.num_program_cache_entries()

    out2 = _on_device(device, torch.zeros((1, 1, 64, 64), dtype=torch.bfloat16))
    toy_spec_fill(out2, -1.5)

    assert device.num_program_cache_entries() == entries, "expected a cache hit, got a new entry"
    assert torch.equal(ttnn.to_torch(out2).float(), torch.full((1, 1, 64, 64), -1.5))
    assert torch.equal(ttnn.to_torch(out1).float(), torch.full((1, 1, 64, 64), 3.0)), "first fill was clobbered"


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 96, 128), (1, 3, 64, 64)])
def test_square_in_place(device, shape):
    """In-place program: one tensor, bound as the reader's input and the writer's output."""
    torch.manual_seed(0)
    src = torch.randn(*shape, dtype=torch.bfloat16)
    t = _on_device(device, src)
    address = t.buffer_address()

    returned = toy_spec_square_(t)

    assert returned.buffer_address() == address, "in-place must not reallocate"
    got = ttnn.to_torch(t).float()
    expected = (src.float() * src.float()).bfloat16().float()
    assert torch.allclose(got, expected, atol=0.05, rtol=0.02), (got - expected).abs().max()


def test_square_in_place_is_repeatable(device):
    """Applying it twice must square twice, which a stale cached address would not do."""
    torch.manual_seed(1)
    src = torch.rand((1, 1, 64, 64), dtype=torch.bfloat16) + 0.5  # keep away from under/overflow
    t = _on_device(device, src)

    toy_spec_square_(t)
    toy_spec_square_(t)

    got = ttnn.to_torch(t).float()
    expected = (src.float() ** 4).bfloat16().float()
    assert torch.allclose(got, expected, atol=0.1, rtol=0.05), (got - expected).abs().max()


def test_empty_io_tensors_is_rejected(device, expect_error):
    """The relaxed check is still structural: something has to be the output tensor."""
    out = _on_device(device, torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16))
    spec, run_args = create_fill_spec(out, 1.0)

    with expect_error(RuntimeError, "must contain at least the output tensor"):
        ttnn.generic_op([], spec, run_args, {TP_OUT: 0})
