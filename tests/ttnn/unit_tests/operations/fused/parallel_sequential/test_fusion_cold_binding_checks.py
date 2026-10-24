# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
import torch
import ttnn

from models.common.device_utils import is_blackhole

from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from models.experimental.ops.descriptors.op_descriptor import LazyOutputList, OpDescriptor
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


def test_new_spec_cold_eager_binding_is_rejected(device, expect_error):
    source, original = _constant(3, device), _constant(0, device)
    op = build_identity_op(source, original, _core(0), 1)
    Sequential(op).run()
    supplied = _tensor(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), device)
    op.output_tensors[0] = supplied
    with expect_error(RuntimeError, "construct a new descriptor for a different output tensor"):
        Sequential(op).run()
    _check_constant(supplied, 0)
    _check_constant(original, 3)


@pytest.mark.parametrize("entry", ["run", "build", "launch"])
def test_cold_factory_rebinding_is_rejected(device, expect_error, entry):
    source = _tensor(torch.full((1, 1, 32, 128), 3, dtype=torch.bfloat16), device)
    supplied = _tensor(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16), device)
    first, second = _identity(x=0), _identity(x=1)
    first.update(source)
    second.update(source)
    original = first.output_tensors[0]  # Keep the original buffer alive during the comparison.
    first.output_tensors[0] = supplied
    with expect_error(RuntimeError, "construct a new descriptor for a different output tensor"):
        if entry == "launch":
            first.launch()
        elif entry == "build":
            Parallel(first, second).build()
        else:
            Parallel(first, second).run()
    _check_constant(supplied, 0)
    _check_constant(original, 0)


def test_same_spec_binding_reuses_factory_cache(device):
    source = _tensor(torch.full((1, 1, 32, 128), 3, dtype=torch.bfloat16), device)
    first, second = _identity(source, x=0), _identity(source, x=1)
    retained = Parallel(first, second).run()
    supplied = _tensor(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16), device)
    first.output_tensors[0] = supplied
    updated = _tensor(torch.full((1, 1, 32, 128), 7, dtype=torch.bfloat16), device)
    first.update(updated)
    second.update(updated)
    returned = Parallel(first, second).run()
    _check_constant(supplied, 7)
    assert returned[0].buffer_address() == supplied.buffer_address()
    for tensor in retained:
        _check_constant(tensor, 3)


def test_all_bound_lazy_slots_release_allocator_capture():
    class Capture:
        pass

    capture = Capture()
    ref = weakref.ref(capture)

    def allocator(slots, retained=capture):
        slots[:] = [retained]

    outputs = LazyOutputList([None], allocator)
    del allocator, capture
    outputs[0] = object()
    op = OpDescriptor(factory_fn=ttnn.ProgramDescriptor, output_tensors=outputs, program_cache_key=84519)
    _ = op.descriptor
    gc.collect()
    assert ref() is None


def test_cleanup_preserves_lazy_slots_for_cached_rebinding(device):
    source = _tensor(torch.full((1, 1, 32, 128), 3, dtype=torch.bfloat16), device)
    op = _identity(x=0)
    op.update(source)
    fused = Sequential(op)
    fused.run()
    op.update(source)
    fused.run()
    assert len(op.output_tensors) == 1
    supplied = _tensor(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16), device)
    op.output_tensors[0] = supplied
    op.update(source)
    [returned] = fused.run()
    _check_constant(supplied, 3)
    assert returned.buffer_address() == supplied.buffer_address()


def test_cold_rebuild_after_lazy_cleanup_has_clear_error(device, expect_error):
    source = _tensor(torch.full((1, 1, 32, 128), 3, dtype=torch.bfloat16), device)
    op = _identity(x=0)
    op.update(source)
    fused = Sequential(op)
    fused.run()
    op.update(source)
    fused.run()
    op.update(source)
    clear_build_cache()
    with expect_error(RuntimeError, "construct a new descriptor before rebuilding"):
        fused.run()


def test_tuple_factory_can_return_same_bound_tensor_in_new_list(device):
    source, supplied = _constant(3, device), _constant(0, device)

    def factory():
        return build_identity_op(source, supplied, _core(0), 1).descriptor, [supplied]

    op = OpDescriptor(factory_fn=factory, input_tensors=[source], output_tensors=[supplied], program_cache_key=41234)
    for _ in range(2):
        [returned] = op.launch()
        _check_constant(supplied, 3)
        assert returned.buffer_address() == supplied.buffer_address()


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("entry", ["run", "build", "launch"])
def test_cold_freed_binding_has_clear_error(device, expect_error, view, entry):
    source, root = _constant(3, device), _constant(0, device)
    supplied = ttnn.reshape(root, (1, 1, 32, 32)) if view else root
    op = build_identity_op(source, supplied, _core(0), 1)
    fused = Sequential(op)
    fused.run()
    ttnn.deallocate(root)
    assert supplied.buffer_unique_id() is None
    clear_build_cache()
    with expect_error(RuntimeError, "Cold fusion output binding has no live local buffer"):
        if entry == "launch":
            op.launch()
        elif entry == "build":
            fused.build()
        else:
            fused.run()


def test_freed_root_rejected_before_address_discovery(device, expect_error, monkeypatch):
    from models.experimental.ops.descriptors.fusion import fusion as impl

    source, root = _constant(3, device), _constant(0, device)
    view = ttnn.reshape(root, (1, 1, 32, 32))
    op = build_identity_op(source, view, _core(0), 1)
    fused = Sequential(op)
    fused.run()
    ttnn.deallocate(root)
    clear_build_cache()

    def stop_before_unsafe_discovery(*args, **kwargs):
        raise RuntimeError("TEST STOP: freed-root tensor reached address discovery")

    monkeypatch.setattr(impl, "_compute_address_slots", stop_before_unsafe_discovery)
    with expect_error(RuntimeError, "Cold fusion output binding has no live local buffer"):
        fused.run()


def test_tuple_factory_preserves_additional_outputs(device):
    source, supplied, additional = (_constant(value, device) for value in [3, 0, 0])

    def factory():
        first = build_identity_op(source, supplied, _core(0), 1).descriptor
        second = build_identity_op(source, additional, _core(1), 1).descriptor
        descriptor = ttnn.ProgramDescriptor()
        descriptor.kernels = list(first.kernels) + list(second.kernels)
        descriptor.cbs = list(first.cbs) + list(second.cbs)
        return descriptor, [supplied, additional]

    op = OpDescriptor(factory_fn=factory, input_tensors=[source], output_tensors=[supplied], program_cache_key=45678)
    returned = op.launch()
    assert len(returned) == 2
    for tensor in returned:
        _check_constant(tensor, 3)


@pytest.mark.parametrize("storage", ["host", "freed"])
def test_eager_construction_does_not_access_unavailable_buffers(device, storage):
    tensor = _constant(0, device) if storage == "freed" else ttnn.from_torch(torch.zeros((32, 32)))
    if storage == "freed":
        ttnn.deallocate(tensor)
    op = OpDescriptor(ttnn.ProgramDescriptor(), [], [tensor])
    assert op.output_tensors[0] is tensor


def test_eager_construction_allows_a_live_root_without_local_buffer(device, monkeypatch):
    # Single-card guard test for the remote-only-view API contract.
    # This does not claim a multi-host mesh reproduction.
    tensor = _constant(0, device)
    assert tensor.buffer_unique_id() is not None

    def no_local_buffer(self):
        raise RuntimeError("no local buffer found")

    monkeypatch.setattr(ttnn.Tensor, "buffer_address", no_local_buffer)
    op = OpDescriptor(ttnn.ProgramDescriptor(), [], [tensor])
    assert op.output_tensors[0] is tensor


@pytest.mark.parametrize("empty", [False, True])
def test_eager_output_slot_filled_after_construction(device, empty):
    source, supplied = _constant(3, device), _constant(0, device)
    descriptor = build_identity_op(source, supplied, _core(0), 1).descriptor
    op = OpDescriptor(descriptor, [source], [] if empty else [None])
    op.output_tensors[:] = [supplied]
    [returned] = Sequential(op).run()
    assert returned.buffer_address() == supplied.buffer_address()
    _check_constant(supplied, 3)
