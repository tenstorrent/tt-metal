# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import ttnn

from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from models.experimental.ops.descriptors.fusion import fusion as impl
from models.experimental.ops.descriptors.op_descriptor import LazyOutputList, OpDescriptor
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.demo.test_fused_demo import _build_noop_op
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.test_parallel_sequential_global_cb import (
    build_identity_op,
)


def _core(x):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, 0))})


def _tensor(value, device):
    return ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, device=device)


def _input(seed, device):
    torch.manual_seed(seed)
    value = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    return _tensor(value, device), value


@OpDescriptor.create(name="factory_identity")
def _identity(input_tensor, x, lazy=True):
    def allocate(slots):
        slots[0] = _tensor(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16), input_tensor.device())

    if not lazy:
        slots = [None]
        allocate(slots)
        op = build_identity_op(input_tensor, slots[0], _core(x), 4)
        return OpDescriptor(
            descriptor=op.descriptor,
            input_tensors={"input_tensor": input_tensor},
            output_tensors=op.output_tensors,
        )
    outputs = LazyOutputList([None], allocate)

    def factory():
        return build_identity_op(input_tensor, outputs[0], _core(x), 4).descriptor

    return OpDescriptor(
        factory_fn=factory,
        input_tensors={"input_tensor": input_tensor},
        output_tensors=outputs,
        program_cache_key=hash(("factory_identity", x)),
    )


@pytest.fixture(autouse=True)
def _clear_fusion_cache():
    clear_build_cache()
    yield
    clear_build_cache()


@pytest.mark.parametrize(
    "kind,effect",
    [
        (kind, effect)
        for kind in ("lazy", "eager")
        for effect in ("sequential_alias", "deallocated_result", "parallel_alias")
    ],
)
@pytest.mark.parametrize("persistent", [False, True])
def test_factory_results_keep_their_values(device, kind, effect, persistent):
    parallel = effect == "parallel_alias"

    def make_fused():
        first = _identity(x=0, lazy=kind == "lazy")
        second = (
            _identity(x=1, lazy=kind == "lazy")
            if parallel
            else _identity(first.output_tensors[0], x=0, lazy=kind == "lazy")
        )
        fused = (Parallel if parallel else Sequential)(first=first, second=second)
        return first, second, fused

    retained = []
    for iteration in range(3):
        if iteration == 0 or not persistent:
            first, second, fused = make_fused()
        tt_input, value = _input(100 + iteration, device)
        first.update(tt_input)
        expected = [value]
        if parallel:
            tt_other, other = _input(200 + iteration, device)
            second.update(tt_other)
            expected.append(other)
        returned = fused.run()
        current = [ttnn.to_torch(t).clone() for t in returned]
        assert len(current) == len(expected)
        for actual, golden in zip(current, expected):
            assert torch.equal(actual, golden)
        for previous, snapshots in retained:
            assert len(previous) == len(snapshots) == len(returned)
            for old, snapshot in zip(previous, snapshots):
                assert torch.equal(ttnn.to_torch(old), snapshot), "later run overwrote an earlier result"
            assert all(old.buffer_address() != new.buffer_address() for old, new in zip(previous, returned))
        if effect == "deallocated_result":
            for result in returned:
                ttnn.deallocate(result)
        else:
            retained.append((returned, current))


def test_deallocated_eager_binding_allocates_a_result(device):
    value = torch.ones(32, 32, dtype=torch.bfloat16)
    source = _tensor(value, device)
    bound = _tensor(torch.zeros_like(value), device)
    op = build_identity_op(source, bound, _core(0), 1)
    fused = Sequential(op)
    assert fused.run()[0].buffer_address() == bound.buffer_address()
    ttnn.deallocate(bound)
    [result] = fused.run()
    assert torch.equal(ttnn.to_torch(result), value)
    assert result.is_allocated() and not bound.is_allocated()


@pytest.mark.parametrize("root_view", [False, True])
def test_explicit_lazy_binding_and_freed_root(device, root_view):
    source, golden = _input(301, device)
    other, golden_other = _input(302, device)
    first, second = _identity(source, x=0), _identity(other, x=1)
    fused = Parallel(first, second)
    fused.run()
    root = _tensor(torch.zeros((32, 128) if root_view else golden.shape, dtype=torch.bfloat16), device)
    target = ttnn.reshape(root, tuple(golden.shape)) if root_view else root
    first.output_tensors[0] = target
    result = fused.run()[0]
    assert result.buffer_address() == target.buffer_address()
    assert torch.equal(ttnn.to_torch(target), golden)
    ttnn.deallocate(root)
    assert root.buffer_unique_id() is None
    if root_view:
        assert target.is_allocated()  # The view holder survives its root.
        assert target.buffer_unique_id() is None
    result, result_other = fused.run()
    assert torch.equal(ttnn.to_torch(result), golden)
    assert torch.equal(ttnn.to_torch(result_other), golden_other)
    assert result.is_allocated()
    assert result.buffer_unique_id() is not None


@pytest.mark.parametrize("sequential", [False, True])
def test_inline_lazy_factories_reuse_compiled_fusion(device, monkeypatch, sequential):
    cold_builds = []
    compute_slots = impl._compute_address_slots

    def count_cold_build(*args, **kwargs):
        cold_builds.append(1)
        return compute_slots(*args, **kwargs)

    monkeypatch.setattr(impl, "_compute_address_slots", count_cold_build)
    sources = [_input(400 + i, device) for i in range(2)]
    compiled_count = None
    for _ in range(3):
        first = _identity(sources[0][0], x=0)
        second = _identity(first.output_tensors[0] if sequential else sources[1][0], x=0 if sequential else 1)
        returned = (Sequential if sequential else Parallel)(first, second).run()
        expected = [sources[0][1]] if sequential else [t[1] for t in sources]
        assert len(returned) == len(expected)
        for actual, golden in zip(returned, expected):
            assert torch.equal(ttnn.to_torch(actual), golden)
        assert len(cold_builds) == 1
        current = device.num_program_cache_entries()
        if compiled_count is None:
            compiled_count = current
        assert current == compiled_count


def test_synthetic_branch_descriptors_stay_deferred_after_cache_hit_build(device, monkeypatch):
    # Exercise a fusion-cache hit with direct lazy factories. A decorator's
    # program-key-cache hit instead returns _DeferredOutput sentinels; direct
    # build()/launch() does not allocate those on this base.
    factory = _identity.__wrapped__
    cold_builds = []
    compute_slots = impl._compute_address_slots

    def count_cold_build(*args, **kwargs):
        cold_builds.append(1)
        return compute_slots(*args, **kwargs)

    monkeypatch.setattr(impl, "_compute_address_slots", count_cold_build)
    sources = [_input(500 + i, device) for i in range(2)]
    Parallel(*[factory(t[0], x=i) for i, t in enumerate(sources)]).run()
    branches = [factory(t[0], x=i) for i, t in enumerate(sources)]
    assert all(isinstance(op.output_tensors, LazyOutputList) for op in branches)
    assert all(op.is_deferred for op in branches)
    fused = Parallel(*branches).build()
    assert all(op.is_deferred for op in branches)
    fused.launch()
    assert len(cold_builds) == 1
    assert all(op.is_deferred for op in branches)
    assert len(fused.output_tensors) == len(sources)
    for actual, (_, golden) in zip(fused.output_tensors, sources):
        assert torch.equal(ttnn.to_torch(actual), golden)


def test_lazy_output_assignment_tracks_only_explicit_slots(expect_error):
    factory_value = object()
    caller_value = object()

    def allocate(slots):
        slots[:] = [factory_value] * 3

    outputs = LazyOutputList([None] * 3, allocate)
    assert outputs[0] is factory_value
    assert outputs._explicit_bindings is None
    assert not outputs._any_explicit
    outputs[-1] = caller_value
    assert outputs._explicit_bindings == [False, False, True]
    assert outputs._any_explicit
    outputs[0:1] = [caller_value, caller_value]
    assert outputs._explicit_bindings == [True, True, False, True]
    outputs[::2] = [caller_value, caller_value]
    assert outputs._explicit_bindings == [True, True, True, True]
    previous = list(outputs._slots)
    with expect_error(ValueError, "attempt to assign sequence of size"):
        outputs[::2] = [caller_value]
    assert outputs._slots == previous
    assert outputs._explicit_bindings == [True, True, True, True]
    outputs[1:3] = []
    assert len(outputs) == 2
    assert outputs._explicit_bindings == [True, True]
    outputs[:] = []
    assert not outputs._any_explicit

    outputs = LazyOutputList([factory_value, factory_value])
    outputs[1] = caller_value
    outputs[1:] = []
    assert outputs._explicit_bindings == [False]
    assert not outputs._any_explicit


def _constant(value, device):
    return _tensor(torch.full((32, 32), value, dtype=torch.bfloat16), device)


def _eager_identity(source, output, x=0):
    return build_identity_op(source, output, _core(x), 1)


def _check_constant(tensor, value):
    assert torch.equal(ttnn.to_torch(tensor), torch.full(tuple(tensor.shape), value, dtype=torch.bfloat16))


@pytest.mark.parametrize("pattern", ["inplace", "unconsumed", "view", "results", "noop", "cross_alias"])
def test_main_correct_written_values_cold_and_cached(device, pattern):
    _eager_pattern(device, pattern, check_returns=pattern not in ("results", "cross_alias"))


def _eager_pattern(device, pattern, check_returns):
    source, other = _constant(3, device), _constant(5, device)
    intermediate, target = _constant(0, device), _constant(0, device)
    results = None
    if pattern == "inplace":
        first, last = _eager_identity(source, source), _eager_identity(source, target)
    elif pattern in ["unconsumed", "results"]:
        first, last = _eager_identity(source, intermediate), _eager_identity(other, target)
        if pattern == "results":
            results = [first, last]
    elif pattern == "view":
        first = _eager_identity(source, intermediate)
        last = _eager_identity(ttnn.reshape(intermediate, (1, 1, 32, 32)), target)
    elif pattern == "noop":
        first, last = _build_noop_op(_core(1), other), _eager_identity(source, target)
        results = [last]
    else:
        first, last = _eager_identity(source, intermediate), _eager_identity(intermediate, source)
        target = source
    fused = Parallel(first, last) if pattern == "noop" else Sequential(first, last)
    for _ in range(3):
        outputs = fused.run(results=results)
        expected = [3, 5] if pattern == "results" else [5 if pattern == "unconsumed" else 3]
        if check_returns:
            assert len(outputs) == len(expected)
            for tensor, value in zip(outputs, expected):
                _check_constant(tensor, value)
        _check_constant(target, expected[-1])
        if pattern in ["unconsumed", "results", "view", "cross_alias"]:
            _check_constant(intermediate, 3)


def test_output_binding_gather_tracks_reassigned_containers(device):
    tensor, _ = _input(601, device)
    entry = SimpleNamespace(output_sources=[(0, 0)])
    op = SimpleNamespace(output_tensors=LazyOutputList([tensor]))
    assert impl._gather_output_bindings([op], entry) == []
    op.output_tensors[0] = tensor
    assert impl._gather_output_bindings([op], entry)[0] is tensor

    op.output_tensors = LazyOutputList([tensor])
    assert impl._gather_output_bindings([op], entry) == []
    op.output_tensors = [tensor]
    assert impl._gather_output_bindings([op], entry)[0] is tensor
    op.output_tensors = []
    assert impl._gather_output_bindings([op], entry) == []

    class DerivedOutputs(LazyOutputList):
        pass

    op.output_tensors = DerivedOutputs([tensor])
    assert impl._gather_output_bindings([op], entry) == []
    op.output_tensors[0] = tensor
    assert impl._gather_output_bindings([op], entry)[0] is tensor
