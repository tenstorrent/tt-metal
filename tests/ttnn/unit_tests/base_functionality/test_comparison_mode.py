# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random


def _compare_torch_tensors(golden, output, *, fail_on_bad_comparison=True):
    ttnn.decorators.set_tensor_id(ttnn.decorators.get_all_tensors(golden), force=True)
    ttnn.decorators.set_tensor_id(ttnn.decorators.get_all_tensors(output), force=True)
    return ttnn.decorators.compare_tensors_using_pcc(
        "ttnn.test_operation",
        golden,
        output,
        desired_pcc=0.99,
        level="locally",
        fail_on_bad_comparison=fail_on_bad_comparison,
    )


class _FakeDistributedTensor:
    def __init__(self, value=None, topology=None):
        self.dtype = ttnn.bfloat16
        self.value = value
        self.topology = topology
        self.tensor_id = None

    def tensor_topology(self):
        return self.topology

    def device(self):
        return None

    @property
    def shape(self):
        return self.value.shape


class _FakeTensorTopology:
    def __init__(self, *, mesh_coords, placements=(), distribution_shape=()):
        self._mesh_coords = mesh_coords
        self._placements = placements
        self._distribution_shape = distribution_shape

    def mesh_coords(self):
        return self._mesh_coords

    def placements(self):
        return self._placements

    def distribution_shape(self):
        return self._distribution_shape


def test_ulp_comparison_policy_for_degenerate_output(expect_error):
    golden = torch.tensor([1.0], dtype=torch.bfloat16)
    one_ulp_away = torch.nextafter(golden, torch.tensor([2.0], dtype=torch.bfloat16))
    two_ulps_away = torch.nextafter(one_ulp_away, torch.tensor([2.0], dtype=torch.bfloat16))
    ttnn.decorators.set_golden_comparison_config(golden, method="ulp", scope="degenerate", ulp_threshold=1)

    comparison_records = _compare_torch_tensors(golden, one_ulp_away)

    assert comparison_records[0]["matches"]
    with expect_error(RuntimeError, r"Comparing output tensor 0 against CPU locally failed"):
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
    with expect_error(RuntimeError, r"Comparing output tensor 0 against CPU locally failed"):
        _compare_torch_tensors(golden, torch.tensor([1.0, float("inf")]))


def test_reduction_goldens_match_ttnn_output_contracts():
    input_tensor = torch.tensor([[3, 1], [2, 4]], dtype=torch.int32)

    max_output = ttnn.get_golden_function(ttnn.max)(input_tensor, dim=1)
    std_output = ttnn.get_golden_function(ttnn.std)(input_tensor.float(), dim=0, correction=False)
    argmax_output = ttnn.get_golden_function(ttnn.argmax)(input_tensor.to(torch.uint32), dim=1)

    assert torch.equal(max_output, torch.tensor([3, 4], dtype=torch.int32))
    assert torch.equal(std_output, torch.std(input_tensor.float(), dim=0, correction=False))
    assert torch.equal(argmax_output, torch.tensor([0, 1]))


def test_batch_norm_golden_accepts_ttnn_channel_shapes_and_one_running_stat():
    input_tensor = torch.randn(2, 3, 4, 4)
    running_mean = torch.zeros(1, 3, 1, 1)

    output = ttnn.get_golden_function(ttnn.batch_norm)(
        input_tensor,
        running_mean=running_mean,
        running_var=None,
        training=True,
    )

    assert output.shape == input_tensor.shape


def test_data_movement_goldens_cover_unsigned_inplace_fold_and_tied_sort():
    plus_one_input = torch.tensor([0, 2**32 - 1], dtype=torch.int64).to(torch.uint32)
    plus_one_output = ttnn.get_golden_function(ttnn.plus_one)(plus_one_input.clone())
    assert torch.equal(plus_one_output, torch.tensor([1, 0], dtype=torch.int64).to(torch.uint32))

    nchw_input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    fold_output = ttnn.get_golden_function(ttnn.fold)(
        nchw_input, 2, 2, use_transpose_as_fold=True, padding=[0, 0, 0, 0, 0, 0]
    )
    assert torch.equal(fold_output, torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]]))

    _, indices = ttnn.get_golden_function(ttnn.sort)(torch.tensor([3.0, 1.0, 1.0, 2.0]), stable=False)
    comparison_config = indices._ttnn_comparison_config
    assert comparison_config.method == "allclose"
    assert comparison_config.scope == "all"
    assert comparison_config.rtol == 0.0
    assert comparison_config.atol == 0.0
    assert torch.equal(comparison_config.mask, torch.tensor([False, False, True, True]))


def test_addmm_golden_ignores_invalid_addend_shape_when_beta_is_zero():
    invalid_addend = torch.zeros(8, 8)
    mat1 = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    mat2 = torch.eye(4)

    output = ttnn.get_golden_function(ttnn.addmm)(invalid_addend, mat1, mat2, alpha=2.0, beta=0.0)

    assert torch.equal(output, 2.0 * mat1)


def test_unary_goldens_cover_unsigned_chain_integer_hardswish_and_gelu_policy():
    unsigned_input = torch.tensor([0, 2**32 - 1], dtype=torch.int64).to(torch.uint32)
    chain_output = ttnn.get_golden_function(ttnn.unary_chain)(
        unsigned_input, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
    )
    hardswish_output = ttnn.get_golden_function(ttnn.hardswish)(torch.zeros(2, dtype=torch.int32))
    gelu_output = ttnn.get_golden_function(ttnn.gelu)(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.bfloat16))

    assert torch.equal(chain_output, unsigned_input)
    assert torch.equal(hardswish_output, torch.full((2,), 0x3F000000, dtype=torch.int32))
    assert gelu_output._ttnn_comparison_config.method == "ulp"
    assert gelu_output._ttnn_comparison_config.ulp_threshold == 10


def test_moreh_dot_golden_uses_degenerate_allclose_policy():
    golden_function = ttnn.get_golden_function(ttnn.moreh_dot)
    golden = golden_function(torch.tensor([1.0]), torch.tensor([1.0]))
    comparison_config = golden._ttnn_comparison_config

    comparison_records = _compare_torch_tensors(golden, golden + 0.05)

    assert comparison_config.method == "allclose"
    assert comparison_config.scope == "degenerate"
    assert comparison_config.rtol == 0.1
    assert comparison_config.atol == 0.1
    assert comparison_records[0]["matches"]


def test_moreh_clip_grad_norm_golden_masks_matching_nonfinite_values():
    golden_function = ttnn.get_golden_function(ttnn.moreh_clip_grad_norm)
    golden = golden_function([torch.tensor([float("inf")])], max_norm=1.0)
    comparison_config = golden._ttnn_comparison_config

    comparison_records = _compare_torch_tensors(golden, torch.tensor([[float("nan")]]))

    assert golden.shape == (1, 1)
    assert comparison_config.method == "allclose"
    assert comparison_config.scope == "degenerate"
    assert comparison_config.rtol == 0.1
    assert comparison_config.atol == 0.1
    assert comparison_config.nonfinite == "mask"
    assert comparison_records[0]["matches"]


def test_moreh_layer_norm_golden_uses_allclose_policy_for_statistics():
    golden_function = ttnn.get_golden_function(ttnn.moreh_layer_norm)
    golden = golden_function(
        torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
        1,
        mean=torch.empty(2),
        rstd=torch.empty(2),
    )
    output = [
        golden[0].clone(),
        golden[1] + torch.tensor([0.05, -0.05]),
        golden[2] + torch.tensor([0.05, -0.05]),
    ]

    comparison_records = _compare_torch_tensors(golden, output)

    for statistic in golden[1:]:
        comparison_config = statistic._ttnn_comparison_config
        assert comparison_config.method == "allclose"
        assert comparison_config.scope == "all"
        assert comparison_config.rtol == 0.1
        assert comparison_config.atol == 0.1
    assert all(record["matches"] for record in comparison_records)


def test_moreh_adam_golden_skips_param_and_compares_optimizer_state():
    golden_function = ttnn.get_golden_function(ttnn.moreh_adam)
    golden = golden_function(
        torch.ones(2),
        torch.ones(2),
        torch.zeros(2),
        torch.zeros(2),
        step=1,
    )

    comparison_records = _compare_torch_tensors(
        golden,
        [torch.full((2,), 100.0), golden[1].clone(), golden[2].clone(), None],
    )

    comparison_config = golden[0]._ttnn_comparison_config
    assert comparison_config.method == "skip"
    assert comparison_config.scope == "all"
    assert len(comparison_records) == 2
    assert all(record["matches"] for record in comparison_records)


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
    with expect_error(RuntimeError, r"Comparing scalar output against CPU locally failed"):
        ttnn.decorators.compare_scalar_outputs(
            "ttnn.test_operation",
            7,
            8,
            desired_pcc=0.99,
            level="locally",
            fail_on_bad_comparison=True,
        )


# Verifies mixed tensor/scalar structured outputs are paired in order
# and that scalar-only outputs are routed to scalar comparison.
def test_structured_output_pairs_preserve_order_and_scalar_routing(monkeypatch):
    golden_tensor = torch.tensor([1.0])
    output_tensor = torch.tensor([1.0])

    output_pairs = ttnn.decorators._structured_output_pairs(
        (golden_tensor, 2, {"value": 3.0}),
        (output_tensor, 2, {"value": 3.0}),
    )

    assert output_pairs[0][0] is golden_tensor
    assert output_pairs[0][1] is output_tensor
    assert output_pairs[1:] == ((2, 2), (3.0, 3.0))
    assert ttnn.decorators.should_compare_scalar_outputs((1, {"value": 2.0}), (1, {"value": 2.0}))
    assert not ttnn.decorators.should_compare_scalar_outputs((1, 2), (1,))

    monkeypatch.setattr(ttnn.graph, "record_tensor_comparison_data", lambda **_: None)
    comparison_records = ttnn.decorators.compare_tensors_using_pcc(
        "ttnn.test_operation",
        (1, {"value": 2.0}),
        (1, {"value": 2.0}),
        desired_pcc=0.99,
        level="globally",
        fail_on_bad_comparison=True,
    )
    assert len(comparison_records) == 2
    assert all(record["matches"] for record in comparison_records)


# Checks that None outputs (op produced no tensor) are skipped, yielding no comparison records.
def test_none_outputs_produce_no_comparison_records():
    compare = ttnn.decorators.compare_tensors_using_pcc
    comparison_kwargs = {
        "python_fully_qualified_name": "ttnn.test_operation",
        "desired_pcc": 0.99,
        "level": "globally",
        "fail_on_bad_comparison": True,
    }

    assert compare(golden_outputs=None, outputs=None, **comparison_kwargs) == []
    assert compare(golden_outputs=(None,), outputs=(None,), **comparison_kwargs) == []


# Ensures a TypeError is raised when golden and runtime outputs disagree on which outputs are None.
@pytest.mark.parametrize(
    "golden_outputs, outputs",
    [
        (None, torch.tensor([1.0])),
        (torch.tensor([1.0]), None),
        ((None,), (torch.tensor([1.0]),)),
        ((torch.tensor([1.0]),), (None,)),
    ],
)
def test_none_output_structure_mismatch_raises(golden_outputs, outputs, expect_error):
    with expect_error(TypeError, "Output structure mismatch"):
        ttnn.decorators.compare_tensors_using_pcc(
            "ttnn.test_operation",
            golden_outputs,
            outputs,
            desired_pcc=0.99,
            level="globally",
            fail_on_bad_comparison=True,
        )


# Checks the stored global golden tensor preserves its mesh index and comparison config metadata.
def test_stored_global_golden_preserves_mesh_index():
    output = torch.tensor([0.0])
    golden = torch.tensor([1.0])
    golden._ttnn_mesh_index = 2
    ttnn.decorators.set_golden_comparison_config(golden, method="skip", scope="all")
    ttnn.decorators.set_tensor_id(output, force=True)

    try:
        ttnn.decorators.postprocess_global_golden_function_outputs(output, golden)
        stored_golden = ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[output.tensor_id]

        assert torch.equal(stored_golden, golden)
        assert stored_golden._ttnn_mesh_index == 2
        assert stored_golden._ttnn_comparison_config == golden._ttnn_comparison_config
    finally:
        ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR.pop(output.tensor_id, None)


# Verifies local comparison selects the device shard at the golden's mesh index
# and emits exactly one comparison record for it.
def test_mesh_index_selects_requested_device_shard_and_emits_one_record(monkeypatch):
    runtime_output = _FakeDistributedTensor()
    runtime_output.tensor_id = 17
    # Device tensors are stored in physical storage order; the golden selects index 1.
    device_tensors = [_FakeDistributedTensor(torch.tensor([0.0])), _FakeDistributedTensor(torch.tensor([1.0]))]
    golden = torch.tensor([1.0])
    golden._ttnn_mesh_index = 1
    ttnn.decorators.set_tensor_id(golden, force=True)
    monkeypatch.setattr(ttnn, "Tensor", _FakeDistributedTensor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda _: device_tensors)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **_: tensor.value)

    comparison_records = ttnn.decorators.compare_tensors_using_pcc(
        "ttnn.test_operation",
        golden,
        runtime_output,
        desired_pcc=0.99,
        level="locally",
        fail_on_bad_comparison=False,
    )

    assert len(comparison_records) == 1
    assert comparison_records[0]["tensor_id"] == runtime_output.tensor_id
    assert comparison_records[0]["matches"]
    assert "mesh_index" not in comparison_records[0]


# Ensures an error is raised when the runtime tensor has no shard at the golden's mesh index.
def test_mesh_index_requires_matching_runtime_shard(monkeypatch, expect_error):
    runtime_output = _FakeDistributedTensor()
    golden = torch.tensor([1.0])
    golden._ttnn_mesh_index = 1
    monkeypatch.setattr(ttnn, "Tensor", _FakeDistributedTensor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda _: [_FakeDistributedTensor(torch.tensor([1.0]))])
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **_: tensor.value)

    with expect_error(ValueError, r"no shard at mesh index 1"):
        ttnn.decorators.to_torch_for_comparison(runtime_output, golden)


# Verifies to_torch_for_comparison stitches per-device shards back into the full logical tensor.
def test_to_torch_for_comparison_composes_mesh_shards(monkeypatch):
    topology = _FakeTensorTopology(mesh_coords=(), placements=(), distribution_shape=())
    runtime_output = _FakeDistributedTensor(topology=topology)
    device_tensors = [
        _FakeDistributedTensor(torch.tensor([1.0])),
        _FakeDistributedTensor(torch.tensor([2.0])),
    ]
    golden = torch.tensor([1.0, 2.0])
    monkeypatch.setattr(ttnn, "Tensor", _FakeDistributedTensor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda _: device_tensors)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **_: tensor.value)

    output = ttnn.decorators.to_torch_for_comparison(runtime_output, golden)

    assert torch.equal(output, golden)


# Verifies global golden decomposition splits shards in physical row-major order.
def test_decompose_global_golden_splits_shards_in_physical_order(monkeypatch):
    topology = _FakeTensorTopology(
        mesh_coords=(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)),
        placements=(ttnn.PlacementShard(0),),
        distribution_shape=(2,),
    )
    input_tensor = _FakeDistributedTensor(topology=topology)
    device_tensors = [
        _FakeDistributedTensor(torch.tensor([1.0, 2.0])),
        _FakeDistributedTensor(torch.tensor([3.0, 4.0])),
    ]
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda _: device_tensors)

    golden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    shards = ttnn.decorators._decompose_global_golden_mesh_tensor(input_tensor, golden)

    assert torch.equal(shards[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(shards[1], torch.tensor([3.0, 4.0]))


# Verifies a mismatch in a nested structured output reports the failing leaf path.
def test_structured_output_comparison_reports_failing_leaf(expect_error):
    golden = [torch.tensor([1.0]), {"weight": torch.tensor([2.0])}]
    output = [torch.tensor([1.0]), {"weight": torch.tensor([3.0])}]
    for leaf in (golden[0], golden[1]["weight"], output[0], output[1]["weight"]):
        ttnn.decorators.set_tensor_id(leaf, force=True)

    with expect_error(RuntimeError, r"Comparing output tensor at output\[1\]\['weight'\] against CPU locally failed"):
        ttnn.decorators.compare_tensors_using_pcc(
            "ttnn.test_operation",
            golden,
            output,
            desired_pcc=0.99,
            level="locally",
            fail_on_bad_comparison=True,
        )


def test_typecast_golden_prefers_explicit_bfloat16_metadata():
    _typecast_golden_function = ttnn.get_golden_function(ttnn.typecast)

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
    _typecast_golden_function = ttnn.get_golden_function(ttnn.typecast)

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
    _golden_function_assign = ttnn.get_golden_function(ttnn.assign)
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


def test_copy_golden_propagates_positional_destination():
    golden_function = ttnn.get_golden_function(ttnn.copy)
    source = torch.tensor([1.25, -2.5], dtype=torch.float32)
    destination = torch.zeros(2, dtype=torch.bfloat16)

    output = golden_function(source, destination, _ttnn_global_golden=True)

    assert golden_function._ttnn_mutates_global_inputs
    assert output is destination
    assert torch.equal(destination, source.to(torch.bfloat16))


def test_copy_transfer_goldens_sync_positional_and_keyword_destinations():
    host_to_device = ttnn.get_golden_function(ttnn.copy_host_to_device_tensor)
    device_to_host = ttnn.get_golden_function(ttnn.copy_device_to_host_tensor)
    source = torch.tensor([1.25, -2.5])

    assert host_to_device._ttnn_mutates_global_inputs
    assert device_to_host._ttnn_mutates_global_inputs

    positional_destination = torch.zeros(2)
    assert host_to_device(source, positional_destination, _ttnn_global_golden=True) is None
    assert torch.equal(positional_destination, source)

    keyword_destination = torch.zeros(2)
    assert device_to_host(device_tensor=source, host_tensor=keyword_destination, _ttnn_global_golden=True) is None
    assert torch.equal(keyword_destination, source)

    # The local (non-global) path is a no-op and preserves the void return contract.
    untouched = torch.zeros(2)
    assert host_to_device(source, untouched) is None
    assert torch.equal(untouched, torch.zeros(2))


def test_ema_golden_matches_recurrence():
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = ttnn.get_golden_function(ttnn.ema)(input_tensor, 0.5)

    torch.testing.assert_close(output, torch.tensor([[1.0, 1.5, 2.25, 3.125]]))


def test_var_hw_and_std_hw_goldens_use_biased_correction():
    input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    var_output = ttnn.get_golden_function(ttnn.var_hw)(input_tensor)
    std_output = ttnn.get_golden_function(ttnn.std_hw)(input_tensor)

    torch.testing.assert_close(var_output, torch.tensor([[[[1.25]]]]))
    torch.testing.assert_close(std_output, torch.tensor([[[[1.25**0.5]]]]))


def test_prod_golden_reduces_each_axis_in_dims():
    input_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    output = ttnn.get_golden_function(ttnn.prod)(input_tensor, dims=(2, 0))

    expected = torch.prod(torch.prod(input_tensor, dim=0, keepdim=True), dim=2, keepdim=True)
    assert torch.equal(output, expected)


def test_quantize_goldens_support_scalar_and_per_channel_args():
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    quantized = ttnn.get_golden_function(ttnn.quantize)(input_tensor, 0.5, 2.0)
    assert quantized.dtype == torch.int32
    assert torch.equal(quantized, torch.tensor([[4, 6], [8, 10]], dtype=torch.int32))

    dequantized = ttnn.get_golden_function(ttnn.dequantize)(quantized.float(), 0.5, 2.0)
    torch.testing.assert_close(dequantized, (quantized.float() - 2.0) * 0.5)

    scale = torch.tensor([0.5, 2.0])
    zero_point = torch.tensor([1.0, 0.0])
    per_channel = ttnn.get_golden_function(ttnn.quantize)(input_tensor, scale, zero_point, axis=-1)
    assert torch.equal(per_channel, torch.round(input_tensor / scale + zero_point).to(torch.int32))

    requantized = ttnn.get_golden_function(ttnn.requantize)(input_tensor, 0.5, 2.0, 0.25, 1.0)
    expected = torch.round((input_tensor - 2.0) * (0.5 / 0.25) + 1.0).to(torch.int32)
    assert torch.equal(requantized, expected)


def test_nonzero_golden_packs_count_and_flat_indices_with_masks(expect_error):
    input_tensor = torch.tensor([[[[0.0, 5.0], [0.0, 7.0]]]])
    golden_function = ttnn.get_golden_function(ttnn.nonzero)

    count, indices = golden_function(input_tensor)

    assert count.shape == (1, 1, 1, 8)
    assert count[0, 0, 0, 0] == 2
    assert count._ttnn_comparison_config.mask.sum() == 1
    assert indices.shape == (1, 1, 1, 16)
    assert torch.equal(indices[0, 0, 0, :8], torch.tensor([0, 0, 0, 1, 0, 0, 1, 1]))
    assert indices._ttnn_comparison_config.mask.sum() == 8
    with expect_error(ValueError, "requires rank-4 input"):
        golden_function(torch.ones(2, 2))


def test_sparse_matmul_golden_zeros_out_inactive_sparse_groups():
    input_a = torch.randn(2, 3)
    input_b = torch.randn(2, 3, 4)
    sparsity = torch.tensor([1, 0])

    output = ttnn.get_golden_function(ttnn.sparse_matmul)(input_a, input_b, sparsity=sparsity)

    torch.testing.assert_close(output[0], input_a @ input_b[0])
    assert torch.equal(output[1], torch.zeros(2, 4))


def test_broadcast_golden_copies_sender_shard_to_cluster_group():
    sender_shard = torch.tensor([[1.0, 2.0]])
    other_shard = torch.tensor([[9.0, 9.0]])
    golden_function = ttnn.get_golden_function(ttnn.broadcast)

    output = golden_function(
        [other_shard, sender_shard],
        (0, 1),
        cluster_axis=1,
        _ttnn_golden_mesh_shape=(1, 2),
        _ttnn_golden_mesh_shard_dims=(None, None),
    )

    assert torch.equal(output, sender_shard)
    # Without mesh metadata the golden has no distributed context and declines to compare.
    assert golden_function([other_shard], (0,)) is None


def test_mesh_partition_golden_round_trips_replicated_input():
    full = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    golden_function = ttnn.get_golden_function(ttnn.mesh_partition)

    output = golden_function(
        [full, full.clone()],
        0,
        cluster_axis=0,
        _ttnn_golden_mesh_shape=(2, 1),
        _ttnn_golden_mesh_shard_dims=(None, None),
    )

    assert torch.equal(output, full)
    assert golden_function([full], 0) is None


def test_allocate_tensor_goldens_return_skip_marked_uninitialized_storage():
    shape_overload = ttnn.get_golden_function(ttnn.allocate_tensor_on_device)(
        (2, 3), ttnn.bfloat16, ttnn.TILE_LAYOUT, None, None
    )
    assert shape_overload.shape == (2, 3)
    assert shape_overload.dtype == torch.bfloat16
    assert shape_overload._ttnn_comparison_config.method == "skip"

    spec = ttnn.TensorSpec(ttnn.Shape([4, 5]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    spec_overload = ttnn.get_golden_function(ttnn.allocate_tensor_on_host)(spec, None)
    assert spec_overload.shape == (4, 5)
    assert spec_overload.dtype == torch.float32
    assert spec_overload._ttnn_comparison_config.method == "skip"


def test_load_tensor_golden_round_trips_dumped_tensor(tmp_path):
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    file_name = str(tmp_path / "tensor.bin")
    ttnn._ttnn.tensor.dump_tensor_flatbuffer(file_name, ttnn.from_torch(source, layout=ttnn.ROW_MAJOR_LAYOUT))

    golden = ttnn.get_golden_function(ttnn.load_tensor)(file_name)

    assert torch.equal(golden, source)


def test_unary_chain_golden_applies_param_ops_in_order():
    input_tensor = torch.tensor([0.5, -1.0, 2.0])
    ops_chain = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 2.0),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 3.0),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2.0),
    ]

    output = ttnn.get_golden_function(ttnn.unary_chain)(input_tensor, ops_chain)

    torch.testing.assert_close(output, ((input_tensor + 2.0) * 3.0) ** 2.0)


def test_unary_chain_golden_shares_unsigned_and_hardswish_handling():
    # The chain map and the standalone unary goldens share the canonical dtype-aware helpers.
    unsigned_input = torch.tensor([0, 2**32 - 1], dtype=torch.int64).to(torch.uint32)
    chain_output = ttnn.get_golden_function(ttnn.unary_chain)(
        unsigned_input, [ttnn.UnaryWithParam(ttnn.UnaryOpType.GTZ)]
    )
    standalone_output = ttnn.get_golden_function(ttnn.gtz)(unsigned_input)
    assert torch.equal(chain_output, torch.tensor([False, True]))
    assert torch.equal(chain_output, standalone_output)

    hardswish_input = torch.tensor([0.0, 1.0])
    chain_hardswish = ttnn.get_golden_function(ttnn.unary_chain)(
        hardswish_input, [ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDSWISH)]
    )
    torch.testing.assert_close(chain_hardswish, torch.nn.functional.hardswish(hardswish_input))

    # Name-bridged ops resolve to the same callable as the standalone golden.
    input_tensor = torch.tensor([0.5, -1.0])
    chain_exp = ttnn.get_golden_function(ttnn.unary_chain)(input_tensor, [ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP)])
    torch.testing.assert_close(chain_exp, ttnn.get_golden_function(ttnn.exp)(input_tensor))


def test_snake_beta_golden_matches_activation_formula():
    input_tensor = torch.tensor([0.5, -1.0, 2.0])

    output = ttnn.get_golden_function(ttnn.snake_beta)(input_tensor, 2.0, 4.0)

    torch.testing.assert_close(output, input_tensor + torch.sin(2.0 * input_tensor) ** 2 / 4.0)


def test_complex_tensor_golden_combines_real_and_imag():
    real = torch.tensor([1.0, -2.0])
    imag = torch.tensor([0.5, 3.0])

    output = ttnn.get_golden_function(ttnn.complex_tensor)(real, imag)

    assert output.dtype == torch.complex64
    assert torch.equal(output, torch.complex(real, imag))


def test_embedding_bw_golden_scatters_output_grad_into_indexed_weight_rows():
    input_tensor = torch.tensor([[0, 2], [0, 1]])
    weight_tensor = torch.randn(3, 4)
    output_gradient = torch.randn(2, 2, 4)

    golden = ttnn.get_golden_function(ttnn.embedding_bw)(input_tensor, weight_tensor, output_gradient)

    expected = torch.zeros(3, 4).index_add_(0, input_tensor.reshape(-1), output_gradient.reshape(-1, 4))
    torch.testing.assert_close(golden, expected)


def test_golden_function_output_tensor_kwargs_default_and_override():
    default_operation = SimpleNamespace()
    custom_operation = SimpleNamespace()

    ttnn.attach_golden_function(default_operation, golden_function=torch.clone)
    ttnn.attach_golden_function(
        custom_operation,
        golden_function=torch.clone,
        output_tensor_kwarg_names=("input_grad", "bias_grad"),
    )

    assert default_operation.output_tensor_kwarg_names == ttnn.decorators.DEFAULT_OUTPUT_TENSOR_KWARG_NAMES
    assert "bias_grad" not in default_operation.output_tensor_kwarg_names
    assert custom_operation.output_tensor_kwarg_names == ("input_grad", "bias_grad")


def test_custom_output_tensor_kwargs_preserve_golden_output_order(monkeypatch):
    input_grad = SimpleNamespace(tensor_id=900_001)
    bias_grad = SimpleNamespace(tensor_id=900_002)
    input_tensor = SimpleNamespace(tensor_id=900_003)
    golden_input_grad = torch.tensor([1.0, 2.0])
    golden_bias_grad = torch.tensor([3.0, 4.0])
    function_kwargs = {
        "bias_grad": bias_grad,
        "input": input_tensor,
        "input_grad": input_grad,
    }

    monkeypatch.setattr(ttnn.decorators, "get_ttnn_tensors", lambda value: [value])
    output_tensors = ttnn.decorators.get_inplace_output_tensors(
        function_kwargs,
        ("input_grad", "bias_grad"),
    )

    try:
        ttnn.decorators.refresh_or_invalidate_global_goldens(
            output_tensors,
            (golden_input_grad, golden_bias_grad),
        )

        assert output_tensors == [input_grad, bias_grad]
        assert torch.equal(
            ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[input_grad.tensor_id],
            golden_input_grad,
        )
        assert torch.equal(
            ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[bias_grad.tensor_id],
            golden_bias_grad,
        )
        assert input_tensor.tensor_id not in ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR
    finally:
        for tensor in (input_grad, bias_grad, input_tensor):
            ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR.pop(tensor.tensor_id, None)


@pytest.mark.parametrize(
    "alias, operation_name",
    [
        ("abs", "moreh_abs_pow"),
        ("adam", "moreh_adam"),
        ("adamw", "moreh_adamw"),
        ("arange", "moreh_arange"),
        ("bmm", "moreh_bmm"),
        ("bmm_backward", "moreh_bmm_backward"),
        ("clip_grad_norm", "moreh_clip_grad_norm"),
        ("cumsum", "moreh_cumsum"),
        ("cumsum_backward", "moreh_cumsum_backward"),
        ("dot", "moreh_dot"),
        ("dot_backward", "moreh_dot_backward"),
        ("fold", "moreh_fold"),
        ("getitem", "moreh_getitem"),
        ("group_norm", "moreh_group_norm"),
        ("group_norm_backward", "moreh_group_norm_backward"),
        ("layer_norm", "moreh_layer_norm"),
        ("layer_norm_backward", "moreh_layer_norm_backward"),
        ("linear", "moreh_linear"),
        ("linear_backward", "moreh_linear_backward"),
        ("logsoftmax", "moreh_logsoftmax"),
        ("logsoftmax_backward", "moreh_logsoftmax_backward"),
        ("matmul", "moreh_matmul"),
        ("matmul_backward", "moreh_matmul_backward"),
        ("mean", "moreh_mean"),
        ("mean_backward", "moreh_mean_backward"),
        ("nll_loss", "moreh_nll_loss"),
        ("nll_loss_backward", "moreh_nll_loss_backward"),
        ("nll_loss_unreduced_backward", "moreh_nll_loss_unreduced_backward"),
        ("norm", "moreh_norm"),
        ("norm_backward", "moreh_norm_backward"),
        ("sgd", "moreh_sgd"),
        ("softmax", "moreh_softmax"),
        ("softmax_backward", "moreh_softmax_backward"),
        ("softmin", "moreh_softmin"),
        ("softmin_backward", "moreh_softmin_backward"),
        ("sum", "moreh_sum"),
        ("sum_backward", "moreh_sum_backward"),
    ],
)
def test_moreh_alias_uses_registered_operation(alias, operation_name):
    operation = getattr(ttnn, operation_name)

    assert getattr(ttnn.operations.moreh, alias) is operation
    assert operation.golden_function is not None


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
            with expect_error(RuntimeError, r"ttnn\.softmax: Comparing output tensor 0 against CPU locally failed"):
                run()
