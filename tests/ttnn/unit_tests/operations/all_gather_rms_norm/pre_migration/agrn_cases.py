# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared case builder of the all_gather_rms_norm pre-migration suite.

Inputs, weights and the reference come from the golden suite's own helpers (eval/golden_tests/all_gather_rms_norm/
helpers.py): ``pytorch_all_gather_rms_norm`` (pure torch, fp32) is the oracle, ``check_output`` with ``TOLERANCES``
the gate (PCC and relative RMS on every device and on every group-position slice of the gathered row). Nothing here
imports the op's planner or program descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
import torch
import ttnn

from eval.golden_tests.all_gather_rms_norm.helpers import (
    LN_ROW_OFFSET_SCALE,
    build_memory_config,
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gathered_shape,
    pytorch_all_gather_rms_norm,
    quantize_like_device,
    random_gamma,
    random_stacked,
    replicate,
)
from eval.golden_tests.all_gather_rms_norm.topology import ring_mock_meshes, system_mesh_shape, wrap_axes

EPS = 1e-6
BF16, F32, BF8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
IL = ttnn.TensorMemoryLayout.INTERLEAVED
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
RING, LINEAR = ttnn.Topology.Ring, ttnn.Topology.Linear
TORCH_DTYPE = {F32: torch.float32, BF16: torch.bfloat16, BF8: torch.bfloat16}


class RegistryGroup(tuple):
    """A ``mesh_device`` param naming a group, not a shape: (topology, cluster_axis), resolved at setup (not at
    collection, which must not touch the cluster) to the mesh the golden suite's ``topology.registry_mesh`` gives."""

    def __new__(cls, topology, cluster_axis):
        return super().__new__(cls, (topology, cluster_axis))


MESH_1x4 = pytest.param((1, 4), id="1x4")
MESH_4x1 = pytest.param((4, 1), id="4x1")


def ring_ok(mesh_shape, cluster_axis):
    """A Ring needs a closed cycle along the group: the box's snake ring (no wrap links) or a real wrap axis."""
    if (tuple(mesh_shape), cluster_axis) in [(tuple(s), a) for s, a in ring_mock_meshes()]:
        return True
    return system_mesh_shape()[cluster_axis] == tuple(mesh_shape)[cluster_axis] and cluster_axis in wrap_axes()


def require(mesh, cluster_axis, topology):
    shape = tuple(mesh.shape)
    if shape[cluster_axis] < 2:
        pytest.skip(f"cluster_axis {cluster_axis} has size {shape[cluster_axis]} on mesh {shape}")
    if topology == RING and not ring_ok(shape, cluster_axis):
        pytest.skip(f"no ring along cluster_axis {cluster_axis} of mesh {shape} on this cluster")


def compute_config(fp32_dest_acc_en=True):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


def entries(mesh):
    return mesh.num_program_cache_entries()


def addresses(t):
    return [d.buffer_address() for d in ttnn.get_device_tensors(t)]


def per_device_torch(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@dataclass
class Built:
    """One call's inputs, kwargs and independent reference."""

    mesh: Any
    shape: tuple
    G: int
    out_shape: tuple
    dtype: Any
    fp32_dest_acc_en: bool
    x: torch.Tensor  # stacked (R, C, *shape), as the device holds it
    tx: Any
    kwargs: dict
    expected: torch.Tensor  # stacked (R, C, *out_shape)
    out_memory_layout: Any
    out_buffer_type: Any
    out_shard_spec: Optional[Any] = None
    keep: list = field(default_factory=list)

    def call(self, op, **overrides):
        return op(self.tx, **{**self.kwargs, **overrides})

    def check(self, out):
        """Values against the reference (golden check_output), then the output's placement."""
        check_output(
            out,
            self.expected,
            shape=self.out_shape,
            dtype=self.dtype,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            expected_layout=TILE,
            group=self.G,
        )
        assert ttnn.is_tensor_storage_on_device(out), "output must stay on the mesh"
        assert len(ttnn.get_device_tensors(out)) == self.mesh.get_num_devices()
        mc = out.memory_config()
        assert mc.memory_layout == self.out_memory_layout, f"output memory layout {mc.memory_layout}"
        assert mc.buffer_type == self.out_buffer_type, f"output buffer type {mc.buffer_type}"
        if self.out_shard_spec is not None:
            assert mc.shard_spec is not None, "a sharded output must carry its shard spec"
            assert list(mc.shard_spec.shape) == list(self.out_shard_spec.shape), f"shard {mc.shard_spec.shape}"
            assert mc.shard_spec.grid == self.out_shard_spec.grid, "output shard grid differs from the requested one"


def build(
    mesh,
    shape,
    *,
    cluster_axis=1,
    dtype=BF16,
    norm_type="rms_norm",
    gamma_layout=None,
    gamma_dtype=None,
    with_beta=False,
    memory_layout=IL,
    output_placement=None,
    topology=RING,
    num_links=2,
    epsilon=EPS,
    fp32_dest_acc_en=True,
    pass_compute_config=True,
    seed=0,
    scale=1.0,
    offset=0.0,
    x=None,
    gamma=None,
):
    """Build per-device x (and optional full-W gamma / beta) on the mesh, the op kwargs and the torch reference.

    ``output_placement``: None -> DRAM interleaved (memory_config not passed) for an interleaved input, the matching
    L1 shard spec of the gathered shape for a sharded input; "l1_interleaved" -> ttnn.L1_MEMORY_CONFIG.
    ``gamma`` / ``x`` override the random tensors (patterns); ``gamma_layout`` None means no gamma.
    """
    require(mesh, cluster_axis, topology)
    shape = tuple(shape)
    G = tuple(mesh.shape)[cluster_axis]
    out_shape = gathered_shape(shape, G)
    ln = norm_type == "layer_norm"
    if x is None:
        x = random_stacked(
            mesh,
            shape,
            TORCH_DTYPE[dtype],
            seed=seed,
            scale=scale,
            offset=offset,
            row_offset_scale=LN_ROW_OFFSET_SCALE if ln else 0.0,
        )
    x = quantize_like_device(x.to(TORCH_DTYPE[dtype]), dtype)
    gdt = gamma_dtype or (dtype if dtype != BF8 else BF16)
    kwargs = {"cluster_axis": cluster_axis, "epsilon": epsilon, "topology": topology, "num_links": num_links}
    g = b = None
    if gamma_layout is not None:
        g = gamma if gamma is not None else random_gamma(out_shape[-1], TORCH_DTYPE[gdt], seed=seed + 1)
        g = g.to(TORCH_DTYPE[gdt])
        kwargs["gamma"] = create_ttnn_gamma(g, mesh, dtype=gdt, layout=gamma_layout)
        if with_beta:
            b = random_gamma(out_shape[-1], TORCH_DTYPE[gdt], seed=seed + 2)
            kwargs["beta"] = create_ttnn_gamma(b, mesh, dtype=gdt, layout=gamma_layout)
    if ln:
        kwargs["norm_type"] = "layer_norm"
    if pass_compute_config:
        kwargs["compute_kernel_config"] = compute_config(fp32_dest_acc_en)
    tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=TILE, memory_layout=memory_layout)
    out_shard = None
    if memory_layout == IL:
        if output_placement == "l1_interleaved":
            kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
            out_ml, out_bt = IL, ttnn.BufferType.L1
        else:
            out_ml, out_bt = IL, ttnn.BufferType.DRAM
    else:
        out_mc = build_memory_config(memory_layout, TILE, out_shape, mesh, dtype=dtype)
        kwargs["memory_config"] = out_mc
        out_ml, out_bt, out_shard = memory_layout, ttnn.BufferType.L1, out_mc.shard_spec
    expected = pytorch_all_gather_rms_norm(x, g, cluster_axis, epsilon, norm_type=norm_type, beta=b)
    return Built(
        mesh=mesh,
        shape=shape,
        G=G,
        out_shape=out_shape,
        dtype=dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        x=x,
        tx=tx,
        kwargs=kwargs,
        expected=expected,
        out_memory_layout=out_ml,
        out_buffer_type=out_bt,
        out_shard_spec=out_shard,
    )


def garbage_output(mesh, out_shape, *, dtype=BF16, memory_config=ttnn.DRAM_MEMORY_CONFIG, fill=7.0):
    """A preallocated output buffer filled with a constant no normalized row can equal."""
    return create_ttnn_input_tensor(
        replicate(torch.full(tuple(out_shape), fill), mesh),
        mesh,
        dtype=dtype,
        layout=TILE,
        memory_config=memory_config,
    )


def assert_input_untouched(built):
    """The op must not write its input: every device still holds its own slice."""
    rows, cols = tuple(built.mesh.shape)
    for idx, got in enumerate(per_device_torch(built.tx)):
        r, c = divmod(idx, cols)
        want = built.x[r, c].to(got.dtype)
        assert torch.equal(got.reshape(want.shape), want), f"input of device ({r},{c}) was modified"
