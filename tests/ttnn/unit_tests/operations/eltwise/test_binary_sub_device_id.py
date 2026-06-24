# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for sub_device_id parameter on binary ops (issue #43977)."""

import pytest
import torch
import ttnn
from models.common.utility_functions import skip_for_slow_dispatch


def setup_sub_device(device):
    """Create a sub-device manager with two sub-devices and load it."""
    tensix_cores0 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    tensix_cores1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 4))})
    sub_device_0 = ttnn.SubDevice([tensix_cores0])
    sub_device_1 = ttnn.SubDevice([tensix_cores1])
    sub_device_manager = device.create_sub_device_manager([sub_device_0, sub_device_1], 3200)
    device.load_sub_device_manager(sub_device_manager)
    return sub_device_manager


def teardown_sub_device(device, sub_device_manager):
    """Unload and remove the sub-device manager."""
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


# ---------------------------------------------------------------------------
# Tensor-Tensor binary ops with sub_device_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op_fn, op_name",
    [
        (ttnn.add, "add"),
        (ttnn.subtract, "subtract"),
        (ttnn.multiply, "multiply"),
    ],
)
@pytest.mark.parametrize("sub_device_idx", [0, 1])
@skip_for_slow_dispatch()
def test_binary_tensor_tensor_with_sub_device_id(device, op_fn, op_name, sub_device_idx):
    """Binary tensor-tensor op runs correctly on a specific sub-device."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        result = op_fn(tt_a, tt_b, sub_device_id=ttnn.SubDeviceId(sub_device_idx))
        result_torch = ttnn.to_torch(result)

        if op_name == "add":
            expected = torch_a + torch_b
        elif op_name == "subtract":
            expected = torch_a - torch_b
        elif op_name == "multiply":
            expected = torch_a * torch_b

        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, f"{op_name} with sub_device_id={sub_device_idx} failed"
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# Tensor-Scalar binary ops with sub_device_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op_fn, op_name",
    [
        (ttnn.add, "add"),
        (ttnn.multiply, "multiply"),
    ],
)
@skip_for_slow_dispatch()
def test_binary_tensor_scalar_with_sub_device_id(device, op_fn, op_name):
    """Binary tensor-scalar op runs correctly on a specific sub-device."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        scalar = 2.5

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        result = op_fn(tt_a, scalar, sub_device_id=ttnn.SubDeviceId(0))
        result_torch = ttnn.to_torch(result)

        if op_name == "add":
            expected = torch_a + scalar
        elif op_name == "multiply":
            expected = torch_a * scalar

        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, f"{op_name} scalar with sub_device_id failed"
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# Mutual exclusion: sub_core_grids + sub_device_id = TT_FATAL
# ---------------------------------------------------------------------------
@skip_for_slow_dispatch()
def test_binary_sub_device_id_and_sub_core_grids_mutual_exclusion(device):
    """TT_FATAL when both sub_core_grids and sub_device_id are provided."""
    torch.manual_seed(0)
    shape = [1, 1, 32, 32]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        some_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

        with pytest.raises(RuntimeError, match="Cannot specify both"):
            ttnn.add(
                tt_a,
                tt_b,
                sub_core_grids=some_cores,
                sub_device_id=ttnn.SubDeviceId(0),
            )
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# Inplace binary ops with sub_device_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op_fn", [ttnn.add_, ttnn.multiply_])
@skip_for_slow_dispatch()
def test_binary_inplace_with_sub_device_id(device, op_fn):
    """Inplace binary op runs correctly on a specific sub-device."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        result = op_fn(tt_a, tt_b, sub_device_id=ttnn.SubDeviceId(0))
        result_torch = ttnn.to_torch(result)

        if op_fn == ttnn.add_:
            expected = torch_a + torch_b
        elif op_fn == ttnn.multiply_:
            expected = torch_a * torch_b

        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, f"Inplace op with sub_device_id failed"
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# NEW: Different dtypes with sub_device_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@skip_for_slow_dispatch()
def test_binary_sub_device_id_with_dtype(device, dtype):
    """Binary add with sub_device_id works across dtypes."""
    torch.manual_seed(42)
    shape = [1, 1, 32, 64]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
        torch_b = torch.randn(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)

        tt_a = ttnn.from_torch(torch_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        result = ttnn.add(tt_a, tt_b, sub_device_id=ttnn.SubDeviceId(0))
        result_torch = ttnn.to_torch(result)

        expected = torch_a + torch_b
        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, f"add with dtype={dtype} and sub_device_id failed"
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# NEW: Larger shapes to verify multi-core dispatch on sub-device
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 128, 128],
        [1, 1, 256, 256],
        [2, 1, 64, 64],
    ],
)
@skip_for_slow_dispatch()
def test_binary_sub_device_id_various_shapes(device, shape):
    """Binary multiply with sub_device_id on various tensor shapes."""
    torch.manual_seed(0)

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        result = ttnn.multiply(tt_a, tt_b, sub_device_id=ttnn.SubDeviceId(0))
        result_torch = ttnn.to_torch(result)

        expected = torch_a * torch_b
        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, f"multiply with shape={shape} and sub_device_id failed"
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# NEW: sub_device_id with output_dtype
# ---------------------------------------------------------------------------
@skip_for_slow_dispatch()
def test_binary_sub_device_id_with_output_dtype(device):
    """Binary add with sub_device_id and explicit output_dtype."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        result = ttnn.add(tt_a, tt_b, dtype=ttnn.bfloat16, sub_device_id=ttnn.SubDeviceId(1))
        result_torch = ttnn.to_torch(result)

        expected = torch_a + torch_b
        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, "add with output_dtype and sub_device_id failed"
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# NEW: sub_device_id with memory_config
# ---------------------------------------------------------------------------
@skip_for_slow_dispatch()
def test_binary_sub_device_id_with_memory_config(device):
    """Binary add with sub_device_id and explicit L1 memory config."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        result = ttnn.add(
            tt_a,
            tt_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            sub_device_id=ttnn.SubDeviceId(0),
        )
        result_torch = ttnn.to_torch(result)

        expected = torch_a + torch_b
        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, "add with memory_config and sub_device_id failed"
        assert result.memory_config().buffer_type == ttnn.BufferType.L1
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# NEW: Verify sub_device_id=None has no effect (backward compat)
# ---------------------------------------------------------------------------
@skip_for_slow_dispatch()
def test_binary_sub_device_id_none_backward_compat(device):
    """Passing sub_device_id=None is equivalent to not passing it."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result_default = ttnn.add(tt_a, tt_b)
    result_none = ttnn.add(tt_a, tt_b, sub_device_id=None)

    r1 = ttnn.to_torch(result_default)
    r2 = ttnn.to_torch(result_none)
    assert torch.equal(r1, r2), "sub_device_id=None should be identical to default"


# ---------------------------------------------------------------------------
# NEW: Relational binary op with sub_device_id
# ---------------------------------------------------------------------------
@skip_for_slow_dispatch()
def test_binary_relational_with_sub_device_id(device):
    """Relational binary op (gt) with sub_device_id."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    sub_device_manager = setup_sub_device(device)
    try:
        torch_a = torch.randn(shape, dtype=torch.bfloat16)
        torch_b = torch.randn(shape, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        result = ttnn.gt(tt_a, tt_b, sub_device_id=ttnn.SubDeviceId(0))
        result_torch = ttnn.to_torch(result)

        expected = (torch_a > torch_b).to(torch.bfloat16)
        passing = torch.allclose(expected, result_torch, atol=0.1, rtol=0.01)
        assert passing, "gt with sub_device_id failed"
    finally:
        teardown_sub_device(device, sub_device_manager)
