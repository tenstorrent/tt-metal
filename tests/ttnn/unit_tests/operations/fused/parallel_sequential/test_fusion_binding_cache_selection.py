# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.device_utils import is_blackhole
from models.experimental.ops.descriptors import op_descriptor as descriptor_impl
from models.experimental.ops.descriptors.fusion import Sequential, clear_build_cache
from models.experimental.ops.descriptors.op_descriptor import LazyOutputList, OpDescriptor
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.test_parallel_sequential_global_cb import (
    build_identity_op,
)


pytestmark = pytest.mark.skipif(not is_blackhole(), reason="not yet run on Wormhole")


def _tensor(shape, value, device, tile=None):
    kwargs = {} if tile is None else {"tile": tile}
    return ttnn.from_torch(
        torch.full(shape, value, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, **kwargs
    )


def _core():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})


@OpDescriptor.create(name="binding_cache_identity")
def _identity(input_tensor):
    outputs = LazyOutputList(
        [None], lambda slots: slots.__setitem__(0, _tensor((1, 1, 32, 128), 0, input_tensor.device()))
    )
    return OpDescriptor(
        factory_fn=lambda: build_identity_op(input_tensor, outputs[0], _core(), 4).descriptor,
        input_tensors={"input_tensor": input_tensor},
        output_tensors=outputs,
        program_cache_key=734718,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_build_cache()
    yield
    clear_build_cache()


@pytest.mark.parametrize("construction", ["persistent", "cold", "warm"])
def test_prebound_factory_spec_does_not_change_unbound_output(device, monkeypatch, construction):
    source = _tensor((1, 1, 32, 128), 3, device)
    supplied = _tensor((32, 128), 0, device)
    if construction == "persistent":
        op = _identity()
    elif construction == "cold":
        with monkeypatch.context() as patch:
            patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
            op = _identity(source)
    else:
        with monkeypatch.context() as patch:
            patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
            _identity(source)
        with monkeypatch.context() as patch:
            patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: next(reversed(cache.values())))
            op = _identity(source)
    op.output_tensors[0] = supplied
    if construction == "persistent":
        op.update(source)
    [bound_result] = Sequential(op).run()
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
        unbound = _identity(source)
    [default_result] = Sequential(unbound).run()
    assert tuple(default_result.shape) == (1, 1, 32, 128)
    assert torch.equal(ttnn.to_torch(default_result), torch.full((1, 1, 32, 128), 3, dtype=torch.bfloat16))
    assert tuple(bound_result.shape) == (32, 128)
    assert torch.equal(ttnn.to_torch(supplied), torch.full((32, 128), 3, dtype=torch.bfloat16))


def test_prebound_factory_program_key_includes_tile_flags(device):
    source = _tensor((1, 1, 32, 128), 3, device)
    ops = []
    for transpose in [False, True]:
        op = _identity()
        op.output_tensors[0] = _tensor((1, 1, 32, 128), 0, device, ttnn.Tile([32, 32], transpose))
        op.update(source)
        ops.append(op)
    assert ops[0].output_tensors[0].spec == ops[1].output_tensors[0].spec
    assert ops[0].program_cache_key != ops[1].program_cache_key


@pytest.mark.parametrize("root_view", [False, True])
def test_freed_different_spec_binding_keeps_cached_allocation(device, root_view):
    source, original = (_tensor((32, 32), value, device) for value in [3, 0])
    op = build_identity_op(source, original, _core(), 1)
    Sequential(op).run()
    root = _tensor((32, 32) if root_view else (1, 1, 32, 32), 0, device)
    freed = ttnn.reshape(root, (1, 1, 32, 32)) if root_view else root
    ttnn.deallocate(root)
    assert freed.buffer_unique_id() is None
    if root_view:
        assert freed.is_allocated()
    op.output_tensors[0] = freed
    [returned] = Sequential(op).run()
    assert tuple(returned.shape) == (32, 32)
    assert returned.buffer_unique_id() is not None
    assert torch.equal(ttnn.to_torch(returned), torch.full((32, 32), 3, dtype=torch.bfloat16))


@pytest.mark.parametrize("construction", ["inline", "persistent", "shell"])
@pytest.mark.parametrize("entry", ["run", "build", "launch"])
def test_factory_input_order_matches_its_own_snapshot(device, monkeypatch, construction, entry):
    @OpDescriptor.create(name="reordered_inputs")
    def factory(a, b=None):
        out = _tensor((32, 32), 0, device)
        return OpDescriptor(
            descriptor=build_identity_op(a, out, _core(), 1).descriptor,
            input_tensors={"b": b, "a": a},
            output_tensors=[out],
        )

    a, b = (_tensor((32, 32), value, device) for value in [3, 7])
    if construction == "persistent":
        op = factory(b=b)
        op.update(a)
    elif construction == "shell":
        with monkeypatch.context() as patch:
            patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
            factory(a=a, b=b)
        with monkeypatch.context() as patch:
            patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: next(reversed(cache.values())))
            op = factory(a=a, b=b)
    else:
        op = factory(a=a, b=b)
    if entry == "run":
        [returned] = Sequential(op).run()
    elif entry == "build":
        Sequential(op).build().launch()
        returned = op.output_tensors[0]
    else:
        [returned] = op.launch()
    assert torch.equal(ttnn.to_torch(returned), torch.full((32, 32), 3, dtype=torch.bfloat16))


def test_unbind_before_materialization_keeps_factory_allocator(device, monkeypatch):
    source = _tensor((1, 1, 32, 128), 3, device)
    supplied = _tensor((1, 1, 32, 128), 0, device)
    with monkeypatch.context() as patch:
        patch.setattr(descriptor_impl, "_program_key_cache_get", lambda cache, key: None)
        op = _identity(source)
    op.output_tensors[0] = supplied
    op.output_tensors[0] = None
    [returned] = Sequential(op).run()
    assert torch.equal(ttnn.to_torch(returned), torch.full((1, 1, 32, 128), 3, dtype=torch.bfloat16))
    assert torch.equal(ttnn.to_torch(supplied), torch.zeros((1, 1, 32, 128), dtype=torch.bfloat16))
    assert returned.buffer_address() != supplied.buffer_address()
