# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.device_utils import is_blackhole

from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from models.experimental.ops.descriptors.fusion import fusion as impl
from models.experimental.ops.descriptors.op_descriptor import LazyOutputList, OpDescriptor, _DeferredOutput
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.demo.test_fused_demo import _build_noop_op
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


def test_bound_tile_transpose_flags_are_checked(device, expect_error):
    golden = torch.arange(1024, dtype=torch.bfloat16).reshape(32, 32)
    source = ttnn.from_torch(golden, layout=ttnn.TILE_LAYOUT, device=device)
    supplied = _constant(0, device)
    op = build_identity_op(source, supplied, _core(0), 1)
    fused = Sequential(op)
    fused.run()
    transposed = ttnn.from_torch(
        torch.zeros_like(golden), layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([32, 32], True), device=device
    )
    assert transposed.spec == supplied.spec  # TensorSpec equality omits the two tile flags.
    op.output_tensors[0] = transposed
    with expect_error(RuntimeError, "cached tile transpose flags"):
        fused.run()
    assert torch.equal(ttnn.to_torch(transposed), torch.zeros_like(golden))


@pytest.mark.parametrize("lazy", [False, True])
def test_persistent_cleanup_keeps_supplied_outputs(device, lazy):
    source, supplied = _constant(3, device), _constant(0, device)
    op = build_identity_op(source, supplied, _core(0), 1)
    if lazy:
        op.output_tensors = LazyOutputList([None])
        op.output_tensors[0] = supplied
    fused = Sequential(op)
    op.update(source)
    for value in [3, 3, 5]:
        if value == 5:
            op.update(_constant(value, device))
        [returned] = fused.run()
        _check_constant(supplied, value)
        assert returned.buffer_address() == supplied.buffer_address()
    replacement = _constant(0, device)
    op.output_tensors[0] = replacement
    op.update(_constant(7, device))
    [returned] = fused.run()
    _check_constant(replacement, 7)
    assert returned.buffer_address() == replacement.buffer_address()


def test_binding_before_first_update_is_written(device):
    source = _tensor(torch.full((1, 1, 32, 128), 3, dtype=torch.bfloat16), device)
    supplied = _tensor(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16), device)
    op = _identity(x=0)
    op.output_tensors[0] = supplied
    op.update(source)
    [returned] = Sequential(op).run()
    assert returned.buffer_address() == supplied.buffer_address()
    _check_constant(supplied, 3)


@pytest.mark.parametrize("entry", ["descriptor", "launch"])
def test_tuple_factory_cannot_silently_replace_binding(device, expect_error, entry):
    source, supplied, allocated = (_constant(value, device) for value in [3, 0, 0])

    def factory():
        full = build_identity_op(source, allocated, _core(0), 1)
        # The decorator's program-key-cache hit uses this tuple-returning path.
        return full.descriptor, full.output_tensors

    op = OpDescriptor(
        factory_fn=factory, input_tensors=[source], output_tensors=[_DeferredOutput()], program_cache_key=1234
    )
    op.output_tensors[0] = supplied
    with expect_error(RuntimeError, "factory allocates its own outputs"):
        if entry == "descriptor":
            _ = op.descriptor
        else:
            op.launch()
    assert op.output_tensors[0] is supplied
    _check_constant(supplied, 0)


def test_direct_factory_keeps_existing_output_list(device):
    source, supplied = _constant(3, device), _constant(0, device)
    op = OpDescriptor(
        factory_fn=lambda: build_identity_op(source, supplied, _core(0), 1).descriptor,
        input_tensors=[source],
        output_tensors=[supplied],
        program_cache_key=5678,
    )
    op.launch()
    assert op.output_tensors[0] is supplied
    _check_constant(supplied, 3)


def test_new_container_bound_specs_do_not_depend_on_cache_history(device):
    source = _constant(3, device)
    retained = []
    first_prefix = None
    for shape in [(32, 32), (1, 1, 32, 32), (32, 32)]:
        dummy = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
        supplied = _constant(0, device)
        noop = _build_noop_op(_core(1), dummy)
        identity = build_identity_op(source, supplied, _core(0), 1)
        fused = Parallel(noop, identity)
        prefix = impl._build_run_cache_key(fused, fused._cached_ops, device)[:3]
        if first_prefix is None:
            first_prefix = prefix
        assert prefix == first_prefix  # Same topology, program hashes and device.
        outputs = fused.run()
        # A noop does not write main's freshly allocated cached return.
        _check_constant(dummy, 0)
        _check_constant(outputs[1], 3)
        retained.append((dummy, supplied, noop, identity, fused, outputs))


def test_binding_one_lazy_output_preserves_other_allocations(device):
    source, supplied = _constant(3, device), _constant(0, device)

    def allocate(slots):
        slots[:] = [_constant(0, device), _constant(0, device)]

    outputs = LazyOutputList([None, None], allocate)

    def factory():
        first = build_identity_op(source, outputs[0], _core(0), 1).descriptor
        second = build_identity_op(source, outputs[1], _core(1), 1).descriptor
        descriptor = ttnn.ProgramDescriptor()
        descriptor.cbs = list(first.cbs) + list(second.cbs)
        descriptor.kernels = list(first.kernels) + list(second.kernels)
        return descriptor

    op = OpDescriptor(factory_fn=factory, input_tensors=[source], output_tensors=outputs, program_cache_key=9012)
    op.output_tensors[0] = supplied
    assert isinstance(op.output_tensors[1], ttnn.Tensor)
    assert op.output_tensors[0] is supplied
    returned = op.launch()
    assert len(returned) == 2
    for tensor in returned:
        _check_constant(tensor, 3)
