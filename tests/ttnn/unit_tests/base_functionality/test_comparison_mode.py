# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
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


def test_stored_global_golden_preserves_distributed_metadata():
    output = torch.tensor([0.0])
    mesh_coord = ttnn.MeshCoordinate(0, 1)
    topology = ttnn.TensorTopologySnapshot(
        distribution_shape=(1, 2),
        placements=(ttnn.PlacementReplicate(), ttnn.PlacementReplicate()),
        mesh_coords=(ttnn.MeshCoordinate(0, 0), mesh_coord),
    )
    golden_shard = torch.tensor([1.0])
    ttnn.decorators.set_golden_comparison_config(golden_shard, method="skip", scope="all")
    golden = ttnn.DistributedGolden(
        topology=topology,
        shards={mesh_coord: golden_shard},
        compare_coords=frozenset({mesh_coord}),
    )
    ttnn.decorators.set_tensor_id(output, force=True)

    try:
        ttnn.decorators.postprocess_global_golden_function_outputs(output, golden)
        stored_golden = ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[output.tensor_id]

        assert stored_golden.topology == topology
        assert stored_golden.compare_coords == frozenset({mesh_coord})
        assert torch.equal(stored_golden.shards[mesh_coord], golden_shard)
        assert stored_golden.shards[mesh_coord]._ttnn_comparison_config == golden_shard._ttnn_comparison_config
    finally:
        ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR.pop(output.tensor_id, None)


def test_distributed_comparison_selects_requested_device_shard(monkeypatch):
    mesh_coords = (ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1))
    topology = ttnn.TensorTopologySnapshot(
        distribution_shape=(1, 2),
        placements=(ttnn.PlacementReplicate(), ttnn.PlacementReplicate()),
        mesh_coords=mesh_coords,
    )
    runtime_output = _FakeDistributedTensor(topology=topology)
    runtime_output.tensor_id = 17
    device_tensors = [_FakeDistributedTensor(torch.tensor([0.0])), _FakeDistributedTensor(torch.tensor([1.0]))]
    golden_shard = torch.tensor([1.0])
    golden = ttnn.DistributedGolden(
        topology=topology,
        shards={mesh_coords[1]: golden_shard},
        compare_coords=frozenset({mesh_coords[1]}),
    )
    monkeypatch.setattr(ttnn, "Tensor", _FakeDistributedTensor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda _: device_tensors)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **_: tensor.value)

    comparison_pairs = ttnn.decorators._distributed_comparison_pairs(golden, runtime_output)

    assert len(comparison_pairs) == 1
    selected_golden, selected_output = comparison_pairs[0]
    assert selected_output._ttnn_mesh_coord == (0, 1)
    assert torch.equal(selected_golden, golden_shard)
    assert torch.equal(selected_output, device_tensors[1].value)


def test_distributed_comparison_records_mesh_coordinate(monkeypatch, expect_error):
    mesh_coords = (ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1))
    topology = ttnn.TensorTopologySnapshot(
        distribution_shape=(1, 2),
        placements=(ttnn.PlacementReplicate(), ttnn.PlacementReplicate()),
        mesh_coords=mesh_coords,
    )
    runtime_output = _FakeDistributedTensor(topology=topology)
    runtime_output.tensor_id = 17
    device_tensors = [_FakeDistributedTensor(torch.tensor([0.0])), _FakeDistributedTensor(torch.tensor([1.0]))]
    golden = ttnn.DistributedGolden(
        topology=topology,
        shards={
            mesh_coords[0]: torch.tensor([0.0]),
            mesh_coords[1]: torch.tensor([2.0]),
        },
        compare_coords=frozenset(mesh_coords),
    )
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

    assert len(comparison_records) == 2
    assert {record["tensor_id"] for record in comparison_records} == {runtime_output.tensor_id}
    assert [record["mesh_coord"] for record in comparison_records] == [(0, 0), (0, 1)]
    assert len({record["golden_tensor_id"] for record in comparison_records}) == 2
    with expect_error(RuntimeError, r"mesh coordinate \(0, 1\)"):
        ttnn.decorators.compare_tensors_using_pcc(
            "ttnn.test_operation",
            golden,
            runtime_output,
            desired_pcc=0.99,
            level="locally",
            fail_on_bad_comparison=True,
        )


def test_distributed_golden_conversion_does_not_mask_unexpected_errors(monkeypatch, expect_error):
    def raise_unexpected_error(_):
        raise AttributeError("unexpected discovery bug")

    monkeypatch.setattr(ttnn, "get_device_tensors", raise_unexpected_error)

    with expect_error(AttributeError, "unexpected discovery bug"):
        ttnn.decorators.distributed_golden_for_comparison(object())


def test_collective_golden_does_not_require_remote_shards():
    """A local collective group can be evaluated without shards owned by other
    processes in the global mesh."""
    mesh_coords = tuple(ttnn.MeshCoordinate(row, column) for row in range(2) for column in range(4))
    topology = ttnn.TensorTopologySnapshot(
        distribution_shape=(2, 4),
        placements=(ttnn.PlacementReplicate(), ttnn.PlacementReplicate()),
        mesh_coords=mesh_coords,
    )
    local_coords = mesh_coords[:4]
    remote_coords = mesh_coords[4:]
    input_golden = ttnn.DistributedGolden(
        topology=topology,
        shards={
            mesh_coord: torch.tensor([column + 1.0], dtype=torch.bfloat16)
            for column, mesh_coord in enumerate(local_coords)
        },
        compare_coords=frozenset(local_coords),
    )

    assert not set(remote_coords) & set(input_golden.shards)

    all_reduce_golden = ttnn.get_golden_function(ttnn.all_reduce)
    output_golden = all_reduce_golden(input_golden, cluster_axis=1)

    assert output_golden.global_value is None
    assert output_golden.shards is not None
    assert set(output_golden.shards) == set(local_coords)
    assert output_golden.compare_coords == frozenset(local_coords)

    expected_shard = torch.tensor([10.0], dtype=torch.bfloat16)
    for output_shard in output_golden.shards.values():
        assert torch.equal(output_shard, expected_shard)


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
