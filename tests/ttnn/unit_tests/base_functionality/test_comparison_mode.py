# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.operations.core import _typecast_golden_function
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import skip_for_wormhole_b0
from models.common.utility_functions import torch_random


def _compare_torch_tensors(golden, output, *, fail_on_bad_comparison=True):
    ttnn.decorators.set_tensor_id(golden, force=True)
    ttnn.decorators.set_tensor_id(output, force=True)
    return ttnn.decorators.compare_tensors_using_pcc(
        "ttnn.test_operation",
        golden,
        output,
        desired_pcc=0.99,
        level="locally",
        fail_on_bad_comparison=fail_on_bad_comparison,
    )


def test_ulp_comparison_policy_for_degenerate_output(expect_error):
    golden = torch.tensor([1.0], dtype=torch.bfloat16)
    one_ulp_away = torch.nextafter(golden, torch.tensor([2.0], dtype=torch.bfloat16))
    two_ulps_away = torch.nextafter(one_ulp_away, torch.tensor([2.0], dtype=torch.bfloat16))
    ttnn.decorators.set_golden_comparison_config(golden, method="ulp", scope="degenerate", ulp_threshold=1)

    comparison_records = _compare_torch_tensors(golden, one_ulp_away)

    assert comparison_records[0]["matches"]
    with expect_error(RuntimeError):
        _compare_torch_tensors(golden, two_ulps_away)


def test_allclose_comparison_policy_for_degenerate_output():
    golden = torch.tensor([1.0])
    output = torch.tensor([1.25])
    ttnn.decorators.set_golden_comparison_config(golden, method="allclose", scope="degenerate", rtol=0.3, atol=0.0)

    comparison_records = _compare_torch_tensors(golden, output)

    assert comparison_records[0]["matches"]


def test_degenerate_comparison_rejects_shape_mismatch():
    golden = torch.ones(1)
    output = torch.ones(1, 1)

    comparison_records = _compare_torch_tensors(golden, output, fail_on_bad_comparison=False)

    assert not comparison_records[0]["matches"]


def test_skip_comparison_policy_for_all_outputs():
    golden = torch.zeros(2)
    output = torch.ones(2)
    ttnn.decorators.set_golden_comparison_config(golden, method="skip", scope="all")

    assert _compare_torch_tensors(golden, output) == []


def test_comparison_policy_mask_excludes_unpopulated_values():
    golden = torch.tensor([1.0, 2.0])
    output = torch.tensor([1.0, 99.0])
    ttnn.decorators.set_golden_comparison_config(
        golden,
        method="allclose",
        scope="all",
        rtol=0.0,
        atol=0.0,
        mask=torch.tensor([True, False]),
    )

    comparison_records = _compare_torch_tensors(golden, output)

    assert comparison_records[0]["matches"]


def test_comparison_policy_masks_matching_nonfinite_positions(expect_error):
    golden = torch.tensor([float("nan"), 1.0])
    output = torch.tensor([float("inf"), 1.0])
    ttnn.decorators.set_golden_comparison_config(
        golden, method="allclose", scope="all", rtol=0.0, atol=0.0, nonfinite="mask"
    )

    comparison_records = _compare_torch_tensors(golden, output)

    assert comparison_records[0]["matches"]
    with expect_error(RuntimeError):
        _compare_torch_tensors(golden, torch.tensor([1.0, float("inf")]))


def test_prepare_backward_golden_inputs_clears_accumulated_gradients():
    input_tensor = torch.tensor([2.0], requires_grad=True)
    prepared_args, _ = ttnn.decorators.prepare_backward_golden_inputs(((input_tensor,), {}))
    (prepared_args[0] * 2).sum().backward()
    assert torch.equal(input_tensor.grad, torch.tensor([2.0]))

    prepared_args, _ = ttnn.decorators.prepare_backward_golden_inputs(((input_tensor,), {}))
    assert input_tensor.grad is None
    (prepared_args[0] * 3).sum().backward()

    assert torch.equal(input_tensor.grad, torch.tensor([3.0]))


def test_scalar_output_comparison(monkeypatch, expect_error):
    monkeypatch.setattr(ttnn.graph, "record_tensor_comparison_data", lambda **_: None)

    integer_records = ttnn.decorators.compare_scalar_outputs(
        "ttnn.test_operation",
        7,
        7,
        desired_pcc=0.99,
        level="locally",
        fail_on_bad_comparison=True,
    )
    float_records = ttnn.decorators.compare_scalar_outputs(
        "ttnn.test_operation",
        1.0,
        1.0 + 1e-6,
        desired_pcc=0.99,
        level="locally",
        fail_on_bad_comparison=True,
    )

    assert integer_records[0]["matches"]
    assert float_records[0]["matches"]
    with expect_error(RuntimeError):
        ttnn.decorators.compare_scalar_outputs(
            "ttnn.test_operation",
            7,
            8,
            desired_pcc=0.99,
            level="locally",
            fail_on_bad_comparison=True,
        )


def test_stored_global_golden_preserves_mesh_index():
    output = torch.tensor([0.0])
    golden = torch.tensor([1.0])
    ttnn.decorators.set_tensor_id(output, force=True)
    golden._ttnn_mesh_index = 2
    ttnn.decorators.set_golden_comparison_config(golden, method="skip", scope="all")

    try:
        ttnn.decorators.postprocess_global_golden_function_outputs(output, golden)
        stored_golden = ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[output.tensor_id]

        assert stored_golden._ttnn_mesh_index == 2
        assert stored_golden._ttnn_comparison_config == golden._ttnn_comparison_config
    finally:
        ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR.pop(output.tensor_id, None)


def test_mesh_index_selects_requested_device_shard(monkeypatch):
    class FakeTensor:
        def __init__(self, value=None):
            self.dtype = ttnn.bfloat16
            self.value = value

    runtime_output = FakeTensor()
    device_tensors = [FakeTensor(torch.tensor([0.0])), FakeTensor(torch.tensor([1.0]))]
    golden = torch.tensor([1.0])
    golden._ttnn_mesh_index = 1
    monkeypatch.setattr(ttnn, "Tensor", FakeTensor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda _: device_tensors)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **_: tensor.value)

    selected_output = ttnn.decorators.to_torch_for_comparison(runtime_output, golden)

    assert torch.equal(selected_output, golden)


def test_typecast_golden_prefers_explicit_bfloat16_metadata():
    input_tensor = torch.tensor([1.7], dtype=torch.bfloat16)

    captured_dtype_result = _typecast_golden_function(
        input_tensor,
        output_dtype=ttnn.uint16,
        input_dtype=ttnn.bfloat8_b,
        _ttnn_input_dtype=ttnn.bfloat16,
        _ttnn_arch_name="wormhole_b0",
    )
    explicit_dtype_result = _typecast_golden_function(
        input_tensor,
        ttnn.bfloat8_b,
        ttnn.uint16,
        input_dtype=ttnn.bfloat16,
        _ttnn_arch_name="wormhole_b0",
    )

    assert captured_dtype_result.item() == 2
    assert explicit_dtype_result.item() == 2


def test_global_typecast_inputs_receive_local_host_metadata():
    local_inputs = (
        (),
        {
            "_ttnn_input_dtype": ttnn.bfloat16,
            "_ttnn_is_host": True,
            "_ttnn_arch_name": "wormhole_b0",
        },
    )
    global_inputs = ([torch.tensor([-1.0], dtype=torch.bfloat16)], {"output_dtype": ttnn.uint8})

    global_args, global_kwargs = ttnn.decorators._merge_local_golden_metadata_into_global_inputs(
        local_inputs, global_inputs
    )
    result = _typecast_golden_function(*global_args, **global_kwargs)

    assert global_kwargs["_ttnn_input_dtype"] == ttnn.bfloat16
    assert global_kwargs["_ttnn_is_host"]
    assert result.item() == 0


def test_assign_golden_uses_nanobind_argument_names_and_casts_dtype():
    from ttnn.operations.binary import _golden_function_assign

    input_tensor = torch.tensor([1.234567], dtype=torch.float32)
    expected = input_tensor.to(torch.bfloat16)

    allocating_result = _golden_function_assign(
        input_tensor=input_tensor,
        memory_config=None,
        dtype=ttnn.bfloat16,
        output_tensor=None,
    )
    destination_result = _golden_function_assign(
        input_a=input_tensor,
        input_b=torch.zeros_like(expected),
    )

    assert torch.equal(allocating_result, expected)
    assert torch.equal(destination_result, expected)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [-1])
def test_softmax(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.softmax(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    with ttnn.manage_config("enable_comparison_mode", True), ttnn.manage_config("comparison_mode_pcc", 0.99):
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        input_tensor = ttnn.to_device(input_tensor, device)
        output_tensor = ttnn.softmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
def test_exp(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.exp(torch_input_tensor)

    with ttnn.manage_config("enable_comparison_mode", True):
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.exp(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1])
def test_failed_comparison(device, batch_size, h, w, dim, expect_error):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)

    ttnn.softmax.golden_function = lambda x, **_: x  # override the proper golden function implementation

    def run():
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        input_tensor = ttnn.to_device(input_tensor, device)
        ttnn.softmax(input_tensor, dim=dim)

    with ttnn.manage_config("enable_comparison_mode", True), ttnn.manage_config("comparison_mode_pcc", 0.99):
        with ttnn.manage_config("comparison_mode_should_raise_exception", False):
            run()

        with ttnn.manage_config("comparison_mode_should_raise_exception", True):
            with expect_error(RuntimeError):
                run()
