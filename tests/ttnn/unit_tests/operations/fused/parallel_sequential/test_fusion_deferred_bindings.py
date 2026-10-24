# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.device_utils import is_blackhole

from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from models.experimental.ops.descriptors.op_descriptor import LazyOutputList, OpDescriptor, _DeferredOutput
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.test_fusion_output_ownership import (
    _check_constant,
    _constant,
    _core,
    _identity,
    _tensor,
)
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.test_parallel_sequential_global_cb import (
    build_identity_op,
)


pytestmark = pytest.mark.skipif(not is_blackhole(), reason="not yet run on Wormhole")


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_build_cache()
    yield
    clear_build_cache()


@pytest.mark.parametrize("supplied_at_construction", [False, True])
def test_deferred_plain_list_tracks_original_bindings(device, supplied_at_construction):
    source = _constant(3, device)
    slots = [_constant(0, device) if supplied_at_construction else None]

    def factory():
        if slots[0] is None:
            slots[0] = _constant(0, device)
        return build_identity_op(source, slots[0], _core(0), 1).descriptor

    op = OpDescriptor(factory_fn=factory, input_tensors=[source], output_tensors=slots, program_cache_key=81723)
    fused = Sequential(op)
    [retained] = fused.run()
    _check_constant(retained, 3)
    op.update(_constant(7, device))
    [returned] = fused.run()
    _check_constant(returned, 7)
    _check_constant(retained, 3)
    assert returned.buffer_address() != retained.buffer_address()


@pytest.mark.parametrize("construction", ["cold", "warm", "persistent"])
def test_binding_before_factory_materialization_is_written(device, construction, monkeypatch):
    from models.experimental.ops.descriptors import op_descriptor as descriptor_impl

    retained = []
    for value in [3, 7]:
        source = _tensor(torch.full((1, 1, 32, 128), value, dtype=torch.bfloat16), device)
        supplied = [_tensor(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16), device) for _ in range(2)]
        ops = []
        for x in range(2):
            if construction == "persistent":
                op = _identity(x=x)
            elif construction == "warm":
                # Force the cached shell for this exact factory/core signature.
                with monkeypatch.context() as patch:
                    patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
                    _identity(source, x=x)
                with monkeypatch.context() as patch:
                    patch.setattr(
                        descriptor_impl, "_program_key_cache_get", lambda cache, key: next(reversed(cache.values()))
                    )
                    op = _identity(source, x=x)
                assert isinstance(op.output_tensors[0], _DeferredOutput)
            else:
                op = _identity(source, x=x)
            ops.append(op)
        assert len(ops) == len(supplied)
        for op, output in zip(ops, supplied):
            op.output_tensors[0] = output
            if construction == "persistent":
                op.update(source)
        outputs = Parallel(*ops).run()
        assert len(outputs) == len(supplied)
        for output, target in zip(outputs, supplied):
            _check_constant(target, value)
            assert output.buffer_address() == target.buffer_address()
        retained.append((outputs, value, ops, source, supplied))
        for previous, previous_value, *_ in retained:
            for tensor in previous:
                _check_constant(tensor, previous_value)


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_partial_mesh_binding_does_not_narrow_other_branch(mesh_device):
    if mesh_device.get_num_devices() != 2:
        pytest.skip("This regression requires a 1x2 mesh")

    def replicated(value):
        return ttnn.from_torch(
            torch.full((32, 32), value, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    first_input, second_input = replicated(1), replicated(2)
    first_output, second_output = replicated(0), replicated(0)
    first = build_identity_op(first_input, first_output, _core(0), 1)
    second = build_identity_op(second_input, second_output, _core(1), 1)
    fused = Parallel(first, second)
    fused.run()
    fresh_first, fresh_second = replicated(0), replicated(0)
    partial = ttnn.get_device_tensors(fresh_first)[0]
    first.output_tensors[0] = partial
    second.output_tensors[0] = fresh_second
    returned = fused.run()
    assert len(returned[0].device_coords()) == 2
    assert returned[0].buffer_address() != partial.buffer_address()
    for tensor, expected in [(returned[0], 1), (returned[1], 2)]:
        for shard in ttnn.get_device_tensors(tensor):
            _check_constant(shard, expected)
    for shard in ttnn.get_device_tensors(fresh_first):
        _check_constant(shard, 0)


@pytest.mark.parametrize("slice_assignment", [False, True])
def test_none_assignment_unbinds_lazy_slot(device, slice_assignment):
    source, supplied = _constant(3, device), _constant(0, device)
    op = build_identity_op(source, supplied, _core(0), 1)
    op.output_tensors = LazyOutputList([None])
    op.output_tensors[0] = supplied
    fused = Sequential(op)
    fused.run()
    if slice_assignment:
        op.output_tensors[:] = [None]
    else:
        op.output_tensors[0] = None
    assert not op.output_tensors._any_explicit
    assert op.output_tensors._explicit_bindings == [False]
    op.update(_constant(7, device))
    [returned] = fused.run()
    _check_constant(returned, 7)
    _check_constant(supplied, 3)
    assert returned.buffer_address() != supplied.buffer_address()


def test_unbound_none_cold_error_names_both_causes(device, expect_error):
    source, supplied = _constant(3, device), _constant(0, device)
    op = build_identity_op(source, supplied, _core(0), 1)
    op.output_tensors = LazyOutputList([None])
    with expect_error(RuntimeError, "cleanup or an unbound None assignment"):
        Sequential(op).run()


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_partial_mesh_binding_allocates_full_mesh(mesh_device):
    # Allocation-contract check: also usable with Metal's two-device mock cluster.
    # The numerical branch coverage is in the test above.
    def allocate():
        return ttnn.empty((32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    source, target = allocate(), allocate()
    op = build_identity_op(source, target, _core(0), 1)
    fused = Sequential(op)
    fused.run()
    partial = ttnn.get_device_tensors(target)[0]
    op.output_tensors[0] = partial
    [returned] = fused.run()
    assert len(partial.device_coords()) == 1
    assert len(returned.device_coords()) == mesh_device.get_num_devices() == 2
    assert returned.buffer_address() != partial.buffer_address()
