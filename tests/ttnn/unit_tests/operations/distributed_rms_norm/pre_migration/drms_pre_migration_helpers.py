# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inputs, memory configs and the independent torch reference shared by the pre-migration suite of
distributed_rms_norm (acceptance, host-perf and output-tensor files). No op import: every file calls the operation
through its own route.

A case lives on a mesh of R x C devices; the G devices of a group along `cluster_axis` together hold the full hidden
dim W = G * W_local, and device p returns its slice of torch rms_norm(concat_p x[p], (W,), gamma, eps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
import ttnn

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}  # the golden spec's floors
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
DEFAULT_EPS = 1e-6
# PCC is blind to a per-row scale error (a wrong eps, G or W in the mean): the median got/true ratio must stay
# within 1% of 1.0 (the precision baseline's scale-bug guard).
RATIO_TOL = 0.01


@dataclass(frozen=True)
class Case:
    """One call of the operation. `None` options are not passed (the op's default applies)."""

    shape: tuple
    cluster_axis: int = 1
    dtype: object = ttnn.bfloat16
    gamma: Optional[tuple] = (ttnn.bfloat16, ttnn.TILE_LAYOUT)  # (dtype, layout) or None
    epsilon: Optional[float] = None
    fp32_dest_acc_en: Optional[bool] = None
    in_mem: str = "dram"  # dram | l1 | height | width
    out_mem: Optional[str] = None  # None (op default) | dram | l1 | width_8x3
    topology: object = None
    num_links: Optional[int] = None
    exchange_mode: Optional[str] = None
    mesh: str = "system"  # system | line | line_1d | ring_1d (conftest.mesh_config)

    def but(self, **changes):
        return replace(self, **changes)


# --------------------------------------------------------------------------- memory configs


def _rows(shape):
    return math.prod(shape[:-1])


def _largest_divisor(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def height_sharded(shape, max_cores=10):
    """HEIGHT_SHARDED L1, ROW_MAJOR, whole tile-rows per core on up to `max_cores` cores."""
    tile_rows = -(-_rows(shape) // 32)
    n = _largest_divisor(tile_rows, max_cores)
    grid = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(8, 8), row_wise=True)
    shard = [tile_rows // n * 32, -(-shape[-1] // 32) * 32]
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def width_sharded(shape, max_cores=8, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """WIDTH_SHARDED L1, whole tile-columns per core on up to `max_cores` cores of row 0."""
    tile_cols = -(-shape[-1] // 32)
    n = _largest_divisor(tile_cols, max_cores)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})
    shard = [-(-_rows(shape) // 32) * 32, tile_cols // n * 32]
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, orientation)
    )


def width_on_grid(shape, grid_x, grid_y, x0=0):
    """WIDTH_SHARDED L1 over a grid_x x grid_y rectangle at (x0, 0); the last shard may be ragged."""
    n = grid_x * grid_y
    shard_w = -(-shape[-1] // (n * 32)) * 32
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x0 + grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [-(-_rows(shape) // 32) * 32, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def memory_config(name, shape):
    return {
        "dram": lambda: ttnn.DRAM_MEMORY_CONFIG,
        "l1": lambda: ttnn.L1_MEMORY_CONFIG,
        "height": lambda: height_sharded(shape),
        "width": lambda: width_sharded(shape),
        "width_8x3": lambda: width_on_grid(shape, 8, 3),
    }[name]()


def expected_output_memory_config(case):
    """The op's placement rule: the given memory_config, else DRAM for an interleaved input, else the input's."""
    if case.out_mem is not None:
        return memory_config(case.out_mem, case.shape)
    if case.in_mem in ("dram", "l1"):
        return ttnn.DRAM_MEMORY_CONFIG
    return memory_config(case.in_mem, case.shape)


# --------------------------------------------------------------------------- host <-> mesh


def to_mesh(stacked, mesh, *, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Stacked (R, C, *shape) torch tensor -> mesh tensor with device (r, c) holding stacked[r, c]."""
    rows, cols = stacked.shape[0], stacked.shape[1]
    assert (rows, cols) == tuple(mesh.shape)
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )


def gamma_to_mesh(gamma_stacked, mesh, *, dtype, layout):
    """Stacked gamma (R, C, W_local) -> TILE (1,1,1,W_local) or ROW_MAJOR sticks (1,1,ceil(W/32),32) per device."""
    rows, cols, w_local = gamma_stacked.shape
    g = gamma_stacked.reshape(rows, cols, 1, 1, 1, w_local)
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        padded = -(-w_local // 32) * 32
        if padded != w_local:
            g = torch.nn.functional.pad(g, (0, padded - w_local))
        g = g.reshape(rows, cols, 1, 1, padded // 32, 32)
    return to_mesh(g, mesh, dtype=dtype, layout=layout)


def quantize_like_device(stacked, dtype):
    """bfloat8_b: the values the device holds, so the reference measures the op, not the format."""
    if dtype != ttnn.bfloat8_b:
        return stacked
    flat = stacked.reshape(-1, stacked.shape[-1])
    t = ttnn.from_torch(flat, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).reshape(stacked.shape).to(torch.bfloat16)


def per_device(t, mesh):
    """[[torch tensor of device (r, c)]]."""
    rows, cols = tuple(mesh.shape)
    devs = ttnn.get_device_tensors(t)
    assert len(devs) == rows * cols
    return [[ttnn.to_torch(devs[r * cols + c]) for c in range(cols)] for r in range(rows)]


def compute_config(fp32_dest_acc_en):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


# --------------------------------------------------------------------------- reference


def reference(x_stacked, gamma_stacked, cluster_axis, eps):
    """x_stacked (R, C, ..., M, W_local), gamma_stacked (R, C, W_local) or None -> expected stacked output (fp32)."""
    x = x_stacked.to(torch.float32)
    rows, cols, w_local = x.shape[0], x.shape[1], x.shape[-1]
    out = torch.empty_like(x)
    for o in range(cols if cluster_axis == 0 else rows):
        members = [(r, o) for r in range(rows)] if cluster_axis == 0 else [(o, c) for c in range(cols)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        weight = None
        if gamma_stacked is not None:
            weight = torch.cat([gamma_stacked[r, c].to(torch.float32).reshape(-1) for r, c in members])
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=weight, eps=eps)
        for p, (r, c) in enumerate(members):
            out[r, c] = normed[..., p * w_local : (p + 1) * w_local]
    return out


def pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return 1.0 if den == 0 else float((a * b).sum() / den)


def ratio_median(actual, expected):
    a, e = actual.flatten().to(torch.float64), expected.flatten().to(torch.float64)
    keep = e.abs() > 1e-2 * e.abs().max()
    return float((a[keep] / e[keep]).median())


# --------------------------------------------------------------------------- building a call


@dataclass
class Call:
    case: Case
    x: torch.Tensor  # stacked host input, as the device holds it
    gamma: Optional[torch.Tensor]  # stacked host gamma (R, C, W_local)
    tx: object
    tg: object
    kwargs: dict

    @property
    def expected(self):
        return reference(self.x, self.gamma, self.case.cluster_axis, self.case.epsilon or DEFAULT_EPS)


def host_data(case, mesh_shape, seed):
    torch.manual_seed(seed)
    rows, cols = mesh_shape
    x = torch.randn((rows, cols, *case.shape), dtype=torch.float32).to(TORCH_DTYPE[case.dtype])
    x = quantize_like_device(x, case.dtype)
    gamma = None
    if case.gamma is not None:
        gamma = torch.randn((rows, cols, case.shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[case.gamma[0]])
    return x, gamma


def op_kwargs(case, tg, mesh=None):
    kwargs = {"cluster_axis": case.cluster_axis}
    if tg is not None:
        kwargs["gamma"] = tg
    if case.epsilon is not None:
        kwargs["epsilon"] = case.epsilon
    if case.topology is not None:
        kwargs["topology"] = case.topology
    if case.num_links is not None:
        kwargs["num_links"] = case.num_links
    if case.fp32_dest_acc_en is not None:
        kwargs["compute_kernel_config"] = compute_config(case.fp32_dest_acc_en)
    if case.out_mem is not None:
        kwargs["memory_config"] = memory_config(case.out_mem, case.shape)
    if case.exchange_mode is not None:
        kwargs["exchange_mode"] = case.exchange_mode
    return kwargs


def device_input(x, case, mesh):
    return to_mesh(
        x, mesh, dtype=case.dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config(case.in_mem, case.shape)
    )


def device_gamma(gamma, case, mesh):
    if gamma is None:
        return None
    return gamma_to_mesh(gamma, mesh, dtype=case.gamma[0], layout=case.gamma[1])


def build(case, mesh, seed=0):
    x, gamma = host_data(case, tuple(mesh.shape), seed)
    tx = device_input(x, case, mesh)
    tg = device_gamma(gamma, case, mesh)
    return Call(case, x, gamma, tx, tg, op_kwargs(case, tg))


def check_output(out, call, mesh):
    """Metadata, placement and every device's values against the independent reference. Returns the device outputs."""
    case = call.case
    assert isinstance(out, ttnn.Tensor), f"one tensor is returned, got {type(out)}"
    assert list(out.shape) == list(case.shape), f"shape {list(out.shape)} != {list(case.shape)}"
    assert out.dtype == case.dtype, f"dtype {out.dtype} != {case.dtype}"
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.storage_type() == ttnn.StorageType.DEVICE
    assert out.device().get_num_devices() == mesh.get_num_devices()
    want_mc = expected_output_memory_config(case)
    assert out.memory_config() == want_mc, f"memory_config {out.memory_config()} != {want_mc}"
    expected = call.expected
    got = per_device(out, mesh)
    for r, row in enumerate(got):
        for c, actual in enumerate(row):
            a = actual.to(torch.float32)
            assert torch.isfinite(a).all(), f"inf/nan on device ({r},{c})"
            v = pcc(a, expected[r, c])
            assert v >= PCC[case.dtype], f"PCC {v:.6f} < {PCC[case.dtype]} on device ({r},{c})"
            ratio = ratio_median(a, expected[r, c])
            assert abs(ratio - 1.0) <= RATIO_TOL, f"scale error on device ({r},{c}): median got/true {ratio:.5f}"
    return got


def check_input_untouched(call, mesh):
    """The op does not write its input (bf8b is skipped: its host form is a re-quantization, not a byte copy)."""
    if call.case.dtype == ttnn.bfloat8_b:
        return
    got = per_device(call.tx, mesh)
    for r, row in enumerate(got):
        for c, actual in enumerate(row):
            assert torch.equal(actual.to(call.x.dtype), call.x[r, c]), f"input changed on device ({r},{c})"
