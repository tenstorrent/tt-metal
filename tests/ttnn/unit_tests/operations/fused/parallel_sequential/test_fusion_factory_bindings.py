# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import gc
import pytest
import torch
import ttnn
from models.common.device_utils import is_blackhole
from models.experimental.ops.descriptors import op_descriptor as descriptor_impl
from models.experimental.ops.descriptors.op_descriptor import LazyOutputList, OpDescriptor
from models.experimental.ops.descriptors.fusion import Sequential, clear_build_cache
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.test_fusion_output_ownership import (
    _constant,
    _check_constant,
    _core,
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


def _force_shell(factory, source, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
        factory(source)
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: next(reversed(cache.values())))
        return factory(source)


@pytest.mark.parametrize("style", ["preallocated", "assignment"])
@pytest.mark.parametrize("path", ["miss", "hit"])
def test_decorated_factory_outputs_are_not_bindings(device, monkeypatch, style, path):
    @OpDescriptor.create(name="owned_factory")
    def factory(source):
        out = _constant(0, device)
        slots = [out] if style == "preallocated" else LazyOutputList([None])
        if style == "assignment":
            slots[0] = out
        return OpDescriptor(
            factory_fn=lambda: build_identity_op(source, out, _core(0), 1).descriptor,
            input_tensors={"source": source},
            output_tensors=slots,
            program_cache_key=98234,
        )

    source = _constant(3, device)
    if path == "hit":
        op = _force_shell(factory, source, monkeypatch)
    else:
        op = factory(source)
    fused = Sequential(op)
    [retained] = fused.run()
    _check_constant(retained, 3)
    op.update(_constant(7, device))
    [returned] = fused.run()
    _check_constant(returned, 7)
    _check_constant(retained, 3)
    assert retained.buffer_address() != returned.buffer_address()


@pytest.mark.parametrize("mutation", ["raw", "lazy"])
def test_factory_cannot_replace_shared_list_binding(device, expect_error, mutation):
    source, supplied, allocated = [_constant(v, device) for v in (3, 0, 0)]
    slots = [None]
    overwrite = [True]

    def factory():
        if overwrite[0]:
            if mutation == "raw":
                slots[0] = allocated
            else:
                op.output_tensors[0] = allocated
        return build_identity_op(source, slots[0], _core(0), 1).descriptor

    op = OpDescriptor(factory_fn=factory, input_tensors=[source], output_tensors=slots, program_cache_key=67823)
    op.output_tensors[0] = supplied
    with expect_error(RuntimeError, "factory allocates its own outputs"):
        _ = op.descriptor
    assert op.is_deferred and op.output_tensors[0] is supplied
    assert slots[0] is supplied
    overwrite[0] = False
    op.launch()
    _check_constant(supplied, 3)
    _check_constant(allocated, 0)


def test_factory_grown_slots_stay_unbound(device):
    source, supplied, extra = [_constant(v, device) for v in (3, 0, 0)]
    slots = [None]

    def factory():
        slots[:] = [supplied, extra]
        return build_identity_op(source, supplied, _core(0), 1).descriptor

    op = OpDescriptor(factory_fn=factory, input_tensors=[source], output_tensors=slots, program_cache_key=67483)
    op.output_tensors[0] = supplied
    returned = op.launch()
    assert len(returned) == 2
    _check_constant(supplied, 3)
    _check_constant(extra, 0)
    # Assignment after raw growth must also work without indexing past flags.
    op.output_tensors[1] = extra
    assert op.output_tensors[1] is extra


def test_tuple_factory_outputs_survive_cleanup(device):
    source, out = [_constant(v, device) for v in (3, 0)]
    op = OpDescriptor(
        factory_fn=lambda: (build_identity_op(source, out, _core(0), 1).descriptor, (out,)),
        input_tensors=[source],
        output_tensors=[None],
        program_cache_key=91192,
    )
    op.update(source)
    fused = Sequential(op)
    [retained] = fused.run()
    _check_constant(retained, 3)
    op.update(_constant(7, device))
    [returned] = fused.run()
    _check_constant(returned, 7)
    _check_constant(retained, 3)


def test_shell_factory_outlives_shell(device, monkeypatch):
    @OpDescriptor.create(name="detached_factory")
    def factory(source):
        out = _constant(0, device)
        return build_identity_op(source, out, _core(0), 1)

    shell = _force_shell(factory, _constant(3, device), monkeypatch)
    callback = shell._factory_fn
    del shell
    gc.collect()
    descriptor, outputs = callback()
    assert descriptor is not None and len(outputs) == 1


def test_failed_persistent_materialization_can_retry(device, monkeypatch, expect_error):
    @OpDescriptor.create(name="slice_style")
    def factory(source):
        out = _constant(0, device)
        op = build_identity_op(source, out, _core(0), 1)
        return OpDescriptor(descriptor=op.descriptor, input_tensors={"source": source}, output_tensors=[out])

    source, supplied = [_constant(v, device) for v in (3, 0)]
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
        factory(source=source)
    op = factory()
    op.output_tensors[0] = supplied
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: next(reversed(cache.values())))
        with expect_error(RuntimeError, "factory allocates its own outputs"):
            op.update(source)
    assert op._complete_fn is not None and op.program_cache_key is None
    assert op.output_tensors[0] is supplied
    op.output_tensors[0] = None
    op.update(source)
    [returned] = Sequential(op).run()
    _check_constant(returned, 3)
    _check_constant(supplied, 0)


@pytest.mark.parametrize("kind", ["eager", "lazy"])
@pytest.mark.parametrize("entry", ["run", "launch"])
def test_updated_input_cannot_rebuild_cold(device, expect_error, kind, entry):
    sources = [_constant(value, device) for value in (3, 7, 11)]
    supplied = _constant(0, device)
    if kind == "eager":
        op = build_identity_op(sources[0], supplied, _core(0), 1)
    else:

        @OpDescriptor.create(name="cold_input_identity")
        def factory(source):
            outputs = LazyOutputList([None], lambda slots: slots.__setitem__(0, _constant(0, device)))
            return OpDescriptor(
                factory_fn=lambda: build_identity_op(source, outputs[0], _core(0), 1).descriptor,
                input_tensors={"source": source},
                output_tensors=outputs,
                program_cache_key=45789,
            )

        op = factory()
        op.output_tensors[0] = supplied
        op.update(sources[0])
    fused = Sequential(op)
    fused.run()
    _check_constant(supplied, 3)
    op.update(sources[1])
    fused.run()
    _check_constant(supplied, 7)
    clear_build_cache()
    op.update(sources[2])
    with expect_error(RuntimeError, "changed an input buffer and cannot be rebuilt cold"):
        if entry == "run":
            fused.run()
        else:
            op.launch()
    _check_constant(supplied, 7)
