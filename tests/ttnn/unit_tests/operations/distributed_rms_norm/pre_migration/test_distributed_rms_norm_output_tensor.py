# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the port adds to ``ttnn.operations.distributed_rms_norm:distributed_rms_norm``:
an optional preallocated output, which the Python op did not take, and, when the port adds one, an optional scratch
tensor with the helper that allocates it.

Written by the migration driver; the test writer filled ``inputs()`` only. It runs on the port alone: on the
untouched Python op every test skips, since that op takes no output tensor.
"""

import importlib
import os

import pytest
import torch
import ttnn

SOURCE_ENTRY = "ttnn.operations.distributed_rms_norm:distributed_rms_norm"
pytestmark = pytest.mark.skipif(
    os.environ.get("TT_PRE_MIGRATION_MODE") != "native", reason="the untouched op takes no output tensor; port only"
)


def _op():
    # On the port, the registered C++ op the native-route guard checks every test reaches (the acceptance suite's
    # own selection); the public path is an alias of it.
    module, _, name = (os.environ.get("TT_PRE_MIGRATION_ENTRY") or SOURCE_ENTRY).partition(":")
    return getattr(importlib.import_module(module), name)


def _scratch_helper():
    """``<op>_scratch_tensor`` beside the op, or None on a port that takes no scratch tensor."""
    module, _, name = (os.environ.get("TT_PRE_MIGRATION_ENTRY") or SOURCE_ENTRY).partition(":")
    return getattr(importlib.import_module(module), f"{name}_scratch_tensor", None)


# --------------------------------------------------------------------------- inputs (the test writer's)

# The fixture every check opens the operation's device with: the one-chip `device`, unless the operation runs on
# something else, such as the mesh a multi-device op needs (the acceptance suite's own fixture, from its conftest.py).
DEVICE_FIXTURE = "drms_mesh"


def inputs(device):
    """2-3 calls of the operation as ``(args, kwargs)`` pairs, taken from the acceptance suite's own cases: one
    single-core call, one multi-core call whose plan has several blocks per core, and one with a non-default option
    or memory config when the op has one. ``device`` is what DEVICE_FIXTURE opens. Device tensors only; never pass an
    output tensor here."""
    from tests.ttnn.unit_tests.operations.distributed_rms_norm.pre_migration import drms_pre_migration_helpers as H

    cases = [
        # single_tile: one tile per device, one core of work
        H.Case((1, 1, 32, 32)),
        # wsplit_two_waves_sticks: many tile-rows over many cores, gamma as ROW_MAJOR sticks, the other axis
        H.Case((1, 1, 640, 1792), cluster_axis=0, gamma=(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)),
        # width_sharded: a sharded input and output (the output defaults to the input's shard)
        H.Case((1, 1, 32, 1792), in_mem="width"),
    ]
    calls = []
    for seed, case in enumerate(cases):
        call = H.build(case, device, seed=seed)
        calls.append(((call.tx,), call.kwargs))
    return calls


# --------------------------------------------------------------------------- checks (the driver's; do not edit)


@pytest.fixture
def op_device(request):
    return request.getfixturevalue(DEVICE_FIXTURE)


def _outs(result):
    return list(result) if isinstance(result, (list, tuple)) else [result]


def _with_outputs(kwargs, expected, outs):
    many = isinstance(expected, (list, tuple))
    return {**kwargs, **({"output_tensors": list(outs)} if many else {"output_tensor": outs[0]})}


def _on_many(t):
    """True when `t` lives on a mesh of several devices (a multi-device op): one shard per device."""
    return t.device().get_num_devices() > 1


def _host(t):
    """`t` on the host as one torch tensor per device (a single one on one device): a tensor spread over a mesh has
    no single host form without the op's own composer."""
    return [ttnn.to_torch(part) for part in ttnn.get_device_tensors(t)]


def _blank_like(t, shape=None, dtype=None, memory_config=None):
    """A fresh device tensor of `t`'s spec, or of that spec with another shape, dtype or memory config; zero-filled.
    On a mesh, every device holds one of `t`'s per-device shape."""
    host = torch.zeros(list(shape or t.shape), dtype=torch.float32)
    mapper = ttnn.ReplicateTensorToMesh(t.device()) if _on_many(t) else None
    return ttnn.from_torch(
        host,
        dtype=dtype or t.dtype,
        layout=t.layout,
        device=t.device(),
        memory_config=memory_config or t.memory_config(),
        mesh_mapper=mapper,
    )


def _wider_fit(memory_config):
    """`memory_config` for a tensor one tile wider: a sharded one's shards one tile wider too, on the same grid, since
    the extra tile column may need a shard more than the grid has cores."""
    spec = memory_config.shard_spec if memory_config.is_sharded() else None
    if spec is None:
        return memory_config
    return ttnn.MemoryConfig(
        memory_config.memory_layout,
        memory_config.buffer_type,
        ttnn.ShardSpec(spec.grid, [spec.shape[0], spec.shape[1] + 32], spec.orientation),
    )


def _other_specs(t):
    """Outputs that differ from `t`'s spec in one field each: shape, memory config, dtype. (A sharded output one tile
    wider also has wider shards: the same layout and grid, the shard shape the wider tensor needs.)"""
    wider = [*list(t.shape)[:-1], list(t.shape)[-1] + 32]
    in_dram = t.memory_config().buffer_type == ttnn.BufferType.DRAM
    other_memory = ttnn.L1_MEMORY_CONFIG if in_dram else ttnn.DRAM_MEMORY_CONFIG
    other_dtype = ttnn.float32 if t.dtype != ttnn.float32 else ttnn.bfloat16
    return {
        "shape": _blank_like(t, shape=wider, memory_config=_wider_fit(t.memory_config())),
        "memory config": _blank_like(t, memory_config=other_memory),
        "dtype": _blank_like(t, dtype=other_dtype),
    }


def _copy(t):
    """A copy of `t` in a buffer of its own; on a mesh, with every device's shard as it was."""
    if _on_many(t):
        return ttnn.clone(t, memory_config=t.memory_config())
    return ttnn.from_torch(
        ttnn.to_torch(t), dtype=t.dtype, layout=t.layout, device=t.device(), memory_config=t.memory_config()
    )


def _same(a, b):
    ha, hb = _host(a), _host(b)
    return len(ha) == len(hb) and all(torch.equal(x, y) for x, y in zip(ha, hb))


def _same_spec(a, b):
    return (
        list(a.shape) == list(b.shape)
        and a.dtype == b.dtype
        and a.layout == b.layout
        and a.memory_config() == b.memory_config()
    )


def test_none_allocates_as_before(op_device):
    op = _op()
    for args, kwargs in inputs(op_device):
        expected = op(*args, **kwargs)
        got = op(*args, **_with_outputs(kwargs, expected, [None] * len(_outs(expected))))
        assert all(_same(g, e) for g, e in zip(_outs(got), _outs(expected)))


def test_the_given_output_is_written_and_returned(op_device):
    op = _op()
    for args, kwargs in inputs(op_device):
        expected = op(*args, **kwargs)
        outs = [_blank_like(e) for e in _outs(expected)]
        got = op(*args, **_with_outputs(kwargs, expected, outs))
        for g, o, e in zip(_outs(got), outs, _outs(expected)):
            assert g.buffer_address() == o.buffer_address(), "the result must be the given output tensor"
            assert _same(g, e)


def test_fresh_outputs_hit_the_program_cache(op_device):
    op = _op()
    for args, kwargs in inputs(op_device):
        expected = op(*args, **kwargs)
        op(*args, **_with_outputs(kwargs, expected, [_blank_like(e) for e in _outs(expected)]))
        before = op_device.num_program_cache_entries()
        outs = [_blank_like(e) for e in _outs(expected)]
        got = op(*args, **_with_outputs(kwargs, expected, outs))
        assert op_device.num_program_cache_entries() == before, "a fresh output must not need a new program"
        for g, o, e in zip(_outs(got), outs, _outs(expected)):
            assert g.buffer_address() == o.buffer_address() and _same(g, e)


def test_an_output_of_another_spec_is_refused(op_device, expect_error):
    # The output must match exactly: its shape, memory config or dtype never replaces the op's own.
    op = _op()
    for args, kwargs in inputs(op_device):
        expected = op(*args, **kwargs)
        rest = [_blank_like(e) for e in _outs(expected)[1:]]
        for field, wrong in _other_specs(_outs(expected)[0]).items():
            with expect_error(RuntimeError, "output_tensor"):
                op(*args, **_with_outputs(kwargs, expected, [wrong, *rest]))


def _aliases(arg, first):
    """`(input copy, output sharing its buffer)` when `arg` can be its own output: it has the output's spec, or a
    reshape of it to the output's shape is a view of the same buffer with that spec. None when it cannot alias."""
    if not isinstance(arg, ttnn.Tensor):
        return None
    if _same_spec(arg, first):
        alias = _copy(arg)
        return alias, alias
    if arg.dtype != first.dtype or arg.layout != first.layout or arg.memory_config() != first.memory_config():
        return None
    if torch.Size(list(arg.shape)).numel() != torch.Size(list(first.shape)).numel():
        return None
    alias = _copy(arg)
    view = ttnn.reshape(alias, first.shape)
    return (alias, view) if view.buffer_address() == alias.buffer_address() and _same_spec(view, first) else None


def test_in_place_is_refused_or_exact(op_device, record_property):
    op = _op()
    tried = 0
    for args, kwargs in inputs(op_device):
        expected = op(*args, **kwargs)
        first = _outs(expected)[0]
        for i, arg in enumerate(args):
            pair = _aliases(arg, first)
            if pair is None:
                continue
            alias, out = pair
            tried += 1
            aliased = [alias if j == i else a for j, a in enumerate(args)]
            outs = [out] + [_blank_like(e) for e in _outs(expected)[1:]]
            try:
                got = op(*aliased, **_with_outputs(kwargs, expected, outs))
            except RuntimeError:
                continue  # the port refuses in place: allowed
            assert _same(_outs(got)[0], first), f"in place on argument {i} must give the out-of-place result"
    # 0 means no input can share the output's buffer (another volume, dtype, layout or memory config): in place
    # cannot happen, since an output of another spec is refused.
    record_property("in_place_cases", tried)


def test_a_held_scratch_tensor_serves_repeated_calls(op_device):
    # A caller that keeps the scratch tensor pays no allocation per call; it must change nothing else.
    helper = _scratch_helper()
    if helper is None:
        pytest.skip("the port takes no scratch tensor")
    op = _op()
    for args, kwargs in inputs(op_device):
        expected = op(*args, **kwargs)
        scratch = helper(*args, **kwargs)
        held = {**kwargs, "scratch_tensor": scratch}
        first = op(*args, **held)
        before = op_device.num_program_cache_entries()
        for _ in range(2):
            got = op(*args, **held)
            same = all(_same(g, e) for g, e in zip(_outs(got), _outs(expected)))
            assert same, "a held scratch tensor changed the result"
        assert op_device.num_program_cache_entries() == before, "calls on one scratch tensor must share one program"
        assert all(_same(g, e) for g, e in zip(_outs(first), _outs(expected)))
        ttnn.deallocate(scratch)


def test_another_calls_scratch_is_refused_or_exact(op_device, record_property):
    # One call's scratch passed to another: refused with a message naming `scratch_tensor`, or exactly that call's
    # plain result. A port that serves a larger scratch must plan around its full size: a plan that does not puts the
    # call's circular buffers where the scratch lies, which tt-metal refuses at dispatch (a message without that name)
    # or which changes the result.
    helper = _scratch_helper()
    if helper is None:
        pytest.skip("the port takes no scratch tensor")
    op = _op()
    calls = list(inputs(op_device))
    served = 0
    for i, (owner_args, owner_kwargs) in enumerate(calls):
        scratch = helper(*owner_args, **owner_kwargs)
        for j, (args, kwargs) in enumerate(calls):
            if j == i:
                continue
            expected = op(*args, **kwargs)
            try:
                got = op(*args, **{**kwargs, "scratch_tensor": scratch})
            except RuntimeError as exc:
                assert "scratch_tensor" in str(exc), f"call {j} on call {i}'s scratch failed without naming it: {exc}"
                continue
            served += 1
            same = all(_same(g, e) for g, e in zip(_outs(got), _outs(expected)))
            assert same, f"call {j} on call {i}'s scratch must give its plain result"
        ttnn.deallocate(scratch)
    # 0 means every call refused the others' scratch (an exact-size port) or inputs() has one call.
    record_property("cross_scratch_served", served)


def test_a_wrong_scratch_tensor_is_refused(op_device, expect_error):
    # One of the op's own inputs is never its scratch: wrong buffer type, layout or size for what the plan needs.
    helper = _scratch_helper()
    if helper is None:
        pytest.skip("the port takes no scratch tensor")
    op = _op()
    for args, kwargs in inputs(op_device):
        wrong = next(a for a in args if isinstance(a, ttnn.Tensor))
        with expect_error(RuntimeError, "scratch_tensor"):
            op(*args, **{**kwargs, "scratch_tensor": wrong})
