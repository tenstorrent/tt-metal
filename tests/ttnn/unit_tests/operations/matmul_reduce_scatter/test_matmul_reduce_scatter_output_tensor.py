# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the port adds to ``ttnn.operations.matmul_reduce_scatter:matmul_reduce_scatter``:
an optional preallocated output, which the Python op did not take.

Written by the migration driver; the test writer filled ``inputs()`` only. It runs on the port alone: on the
untouched Python op every test skips, since that op takes no output tensor.
"""

import importlib
import os

import pytest
import torch
import ttnn

SOURCE_ENTRY = "ttnn.operations.matmul_reduce_scatter:matmul_reduce_scatter"
pytestmark = pytest.mark.skipif(
    os.environ.get("TT_PRE_MIGRATION_MODE") != "native", reason="the untouched op takes no output tensor; port only"
)


def _op():
    # On the port, the registered C++ op the native-route guard checks every test reaches (the acceptance suite's
    # own selection); the public path is an alias of it.
    module, _, name = (os.environ.get("TT_PRE_MIGRATION_ENTRY") or SOURCE_ENTRY).partition(":")
    return getattr(importlib.import_module(module), name)


# --------------------------------------------------------------------------- inputs (the test writer's)

# The fixture every check opens the operation's device with: the one-chip `device`, unless the operation runs on
# something else, such as the mesh a multi-device op needs (the acceptance suite's own fixture, from its conftest.py).
DEVICE_FIXTURE = "mrs_line_mesh"


def inputs(device):
    """2-3 calls of the operation as ``(args, kwargs)`` pairs, taken from the acceptance suite's own cases: one
    single-core call, one multi-core call whose plan has several blocks per core, and one with a non-default option
    or memory config when the op has one. ``device`` is what DEVICE_FIXTURE opens. Device tensors only; never pass an
    output tensor here."""
    # test_acceptance.py's line cases on the system mesh (conftest.py's mrs_line_mesh): one_tile_block_sd-1_ax1,
    # multi_tile_sd-2_ax0_w_bf8b_all_links (with a bf16 weight) and rank3_pos_sd_hifi4_l1_out. Every device holds its
    # own A and W (sharded per device), as the op expects.
    mesh = device
    rows, cols = tuple(mesh.shape)
    axes = [a for a in (1, 0) if (rows, cols)[a] >= 2]
    first, last = axes[0], axes[-1]

    def on_mesh(shape, seed, scale=1.0):
        gen = torch.Generator().manual_seed(seed)
        stacked = torch.randn((rows, cols, *shape), generator=gen) * scale
        whole = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
        return ttnn.from_torch(
            whole.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
        )

    g = (rows, cols)[first]
    hifi4 = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    return [
        (
            (on_mesh((1, 1, 32 * g, 64), 1), on_mesh((64, 32 * g), 2, 64**-0.5)),
            dict(cluster_axis=first, scatter_dim=-1, topology=ttnn.Topology.Linear, num_links=1),
        ),
        (
            (on_mesh((1, 1, 256, 512), 3), on_mesh((512, 1024), 4, 512**-0.5)),
            dict(cluster_axis=last, scatter_dim=-2, topology=ttnn.Topology.Linear, num_links=None),
        ),
        (
            (on_mesh((1, 512, 256), 5), on_mesh((256, 768), 6, 256**-0.5)),
            dict(
                cluster_axis=last,
                scatter_dim=1,
                topology=ttnn.Topology.Linear,
                num_links=1,
                compute_kernel_config=hifi4,
                memory_config=l1,
            ),
        ),
    ]


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


def _other_specs(t):
    """Outputs that differ from `t`'s spec in one field each: shape, memory config, dtype."""
    wider = [*list(t.shape)[:-1], list(t.shape)[-1] + 32]
    in_dram = t.memory_config().buffer_type == ttnn.BufferType.DRAM
    other_memory = ttnn.L1_MEMORY_CONFIG if in_dram else ttnn.DRAM_MEMORY_CONFIG
    other_dtype = ttnn.float32 if t.dtype != ttnn.float32 else ttnn.bfloat16
    return {
        "shape": _blank_like(t, shape=wider),
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
