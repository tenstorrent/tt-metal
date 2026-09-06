# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for sub_device_id parameter on binary ops (issue #43977)."""

import pytest
import torch
import ttnn
from models.common.utility_functions import skip_for_slow_dispatch


def setup_sub_device(device, local_l1_size=3200):
    """Create a sub-device manager with two sub-devices and load it."""
    tensix_cores0 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    tensix_cores1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 4))})
    sub_device_0 = ttnn.SubDevice([tensix_cores0])
    sub_device_1 = ttnn.SubDevice([tensix_cores1])
    sub_device_manager = device.create_sub_device_manager([sub_device_0, sub_device_1], local_l1_size)
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


@pytest.mark.parametrize("op_fn", [ttnn.remainder, ttnn.fmod])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("scalar", [1.5, 2.0])
@skip_for_slow_dispatch()
def test_binary_int32_float_scalar_promotion_with_sub_device_id(device, op_fn, layout, scalar):
    """INT32-to-FLOAT32 promotion and the scalar op both run on the requested sub-device."""
    torch_input = torch.tensor([-5, -1, 0, 1, 5], dtype=torch.int32)

    sub_device_manager = setup_sub_device(device)
    try:
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=layout, device=device)

        result = op_fn(tt_input, scalar, sub_device_id=ttnn.SubDeviceId(1))
        actual = ttnn.to_torch(result)
        assert result.dtype == ttnn.float32
        assert result.layout == layout
        expected = torch.remainder(torch_input, scalar) if op_fn == ttnn.remainder else torch.fmod(torch_input, scalar)

        assert torch.equal(expected, actual)
    finally:
        teardown_sub_device(device, sub_device_manager)


@pytest.mark.parametrize("op_fn", [ttnn.remainder, ttnn.fmod])
@pytest.mark.parametrize("interleaved_output", [False, True])
@skip_for_slow_dispatch()
def test_binary_int32_float_scalar_row_major_sharded_with_sub_device_id(device, op_fn, interleaved_output):
    """Tilize, promotion, arithmetic and untilize stay on the non-origin shard workers."""
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 3))})
    memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    torch_input = (torch.arange(4096, dtype=torch.int32) % 1024 - 512).reshape(1, 1, 32, 128)
    sub_device_manager = setup_sub_device(device, local_l1_size=128 * 1024)
    try:
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG if interleaved_output else memory_config
        # Repeat with different allocations and values to exercise cached conversion programs.
        for offset in [0, 1]:
            tt_input = ttnn.from_torch(
                torch_input + offset,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=memory_config,
                device=device,
            )
            result = op_fn(tt_input, 1.5, memory_config=output_memory_config, sub_device_id=ttnn.SubDeviceId(1))
            expected = ttnn.get_golden_function(op_fn)((torch_input + offset).float(), 1.5, device=device)
            assert result.dtype == ttnn.float32
            assert result.layout == ttnn.ROW_MAJOR_LAYOUT
            assert result.memory_config() == output_memory_config
            assert torch.equal(expected, ttnn.to_torch(result))
    finally:
        teardown_sub_device(device, sub_device_manager)


@pytest.mark.parametrize("scalar", [257.25, -257.25])
@skip_for_slow_dispatch()
def test_remainder_int32_row_major_bfloat16_output_with_sub_device_id(device, scalar):
    torch_input = (torch.arange(32 * 1056, dtype=torch.int32) % 1024 - 512).reshape(32, 1056)
    sub_device_manager = setup_sub_device(device)
    try:
        for offset in [0, 1]:
            input_tensor = ttnn.from_torch(
                torch_input + offset, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            output_tensor = ttnn.from_torch(
                torch.zeros_like(torch_input, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            result = ttnn.remainder(
                input_tensor, scalar, output_tensor=output_tensor, sub_device_id=ttnn.SubDeviceId(1)
            )
            expected = torch.remainder((torch_input + offset).float(), scalar).to(torch.bfloat16)
            assert result.buffer_address() == output_tensor.buffer_address()
            assert result.dtype == ttnn.bfloat16
            assert result.layout == ttnn.ROW_MAJOR_LAYOUT
            assert torch.equal(ttnn.to_torch(output_tensor), expected)
    finally:
        teardown_sub_device(device, sub_device_manager)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@skip_for_slow_dispatch()
def test_div_int32_float_scalar_promotion_with_sub_device_id(device, rounding_mode, layout):
    torch_input = torch.tensor([-(2**31), -7, -5, 0, 5, 7, 2**24 + 3], dtype=torch.int32)
    sub_device_manager = setup_sub_device(device)
    try:
        for offset in [0, 1]:
            input_tensor = ttnn.from_torch(torch_input + offset, dtype=ttnn.int32, layout=layout, device=device)
            result = ttnn.div(input_tensor, 2.0, rounding_mode=rounding_mode, sub_device_id=ttnn.SubDeviceId(1))
            expected = torch.div((torch_input + offset).float(), 2.0, rounding_mode=rounding_mode)
            assert result.dtype == ttnn.float32
            assert result.layout == layout
            assert torch.equal(ttnn.to_torch(result), expected)
    finally:
        teardown_sub_device(device, sub_device_manager)


@skip_for_slow_dispatch()
def test_binary_int32_float_scalar_sharded_output_with_sub_device_id(device, expect_error):
    """The unsupported output typecast must not be reported as an input-promotion failure."""
    torch_input = torch.arange(-512, 512, dtype=torch.int32).reshape(1, 1, 32, 32)
    sharded_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    # Allocate in global L1 before loading the manager with its small local L1 region.
    tt_output = ttnn.from_torch(
        torch.zeros_like(torch_input),
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_memory_config,
    )

    sub_device_manager = setup_sub_device(device)
    try:
        with expect_error(
            RuntimeError, "Remainder output typecast on a restricted grid requires a tiled interleaved tensor"
        ):
            ttnn.remainder(tt_input, 1.5, output_tensor=tt_output, sub_device_id=ttnn.SubDeviceId(1))
    finally:
        teardown_sub_device(device, sub_device_manager)


# ---------------------------------------------------------------------------
# Mutual exclusion: sub_core_grids + sub_device_id = TT_FATAL
# ---------------------------------------------------------------------------
@skip_for_slow_dispatch()
def test_binary_sub_device_id_and_sub_core_grids_mutual_exclusion(device, expect_error):
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

        with expect_error(RuntimeError, "Cannot specify both"):
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
