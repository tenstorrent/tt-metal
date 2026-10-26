# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K3 KDA output projection on Galaxy: ttnn.linear + reduce_scatter_minimal_async vs the fused matmul_reduce_scatter."""

from __future__ import annotations

import json
import statistics
import time

import pytest
import torch
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
from ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter_program_descriptor import _LAST_PLAN

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE,
    l1_when_it_fits,
    tuned_projection_matmul_configs,
)
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]

_ROWS = 640
_K = 96 * 128
_N = 7168
_TP_AXIS = 1
_REPLAYS = 50
_SAMPLES = 5


def _traced_us(mesh_device, fn) -> list[float]:
    """Per-call wall of `fn` captured once and replayed _REPLAYS times per sample."""
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    samples = []
    for _ in range(_SAMPLES):
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(_REPLAYS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1e6 / _REPLAYS)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(out)
    return samples


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=1152),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
def test_kda_output_projection_mmrs(mesh_device: ttnn.MeshDevice) -> None:
    tp = tuple(mesh_device.shape)[_TP_AXIS]
    k_local = _K // tp
    torch.manual_seed(0)
    a_host = torch.randn(1, 1, _ROWS, _K) * 0.5
    w_host = torch.randn(_K, _N) / _K**0.5
    a = ttnn.from_torch(
        a_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=tuple(mesh_device.shape)),
    )
    w = ttnn.from_torch(
        w_host,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=tuple(mesh_device.shape)),
    )
    grid = mesh_device.compute_with_storage_grid_size()
    _, program_config = tuned_projection_matmul_configs(grid, _ROWS, 3 * ttnn.TILE_SIZE * grid.x, k_local, _N)
    linear_compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    partial_memory = l1_when_it_fits(mesh_device, _ROWS * _N * 2, KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE)
    tt_ccl = TT_CCL(mesh_device)

    def linear():
        return ttnn.linear(
            a, w, memory_config=partial_memory, program_config=program_config, compute_kernel_config=linear_compute
        )

    def reduce_scatter(partial):
        return ttnn.experimental.reduce_scatter_minimal_async(
            partial,
            dim=-1,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(_TP_AXIS),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(_TP_AXIS),
            num_links=tt_ccl.get_num_links(_TP_AXIS),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=_TP_AXIS,
        )

    def unfused():
        partial = linear()
        out = reduce_scatter(partial)
        ttnn.deallocate(partial)
        return out

    partial_fixed = linear()

    def rs_only():
        return reduce_scatter(partial_fixed)

    def fused():
        return matmul_reduce_scatter(
            a,
            w,
            cluster_axis=_TP_AXIS,
            scatter_dim=-1,
            topology=ttnn.Topology.Ring,
            num_links=tt_ccl.get_num_links(_TP_AXIS),
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    golden = a_host[0, 0].float() @ w_host.float()
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))
    pcc = {}
    for name, fn in (("unfused", unfused), ("fused", fused)):
        out = fn()
        got = ttnn.to_torch(out, mesh_composer=composer)[0, 0, :_ROWS].float()
        ttnn.deallocate(out)
        _, pcc[name] = comp_pcc(golden, got)
        pcc[name] = float(pcc[name])

    times = {}
    for name, fn in (("linear", linear), ("reduce_scatter", rs_only), ("unfused", unfused), ("fused", fused)):
        times[name] = statistics.median(_traced_us(mesh_device, fn))
    ttnn.deallocate(partial_fixed)

    serial = times["linear"] + times["reduce_scatter"]
    result = {
        "shape": f"{_ROWS}x{k_local}x{_N} per device, TP{tp} ring",
        "pcc": pcc,
        "median_us": times,
        "fused_plan": _LAST_PLAN,
        "saved_us": times["unfused"] - times["fused"],
        "hidden_comm_fraction": (serial - times["fused"]) / times["reduce_scatter"],
    }
    print("KDA_OUTPUT_MMRS=" + json.dumps(result, sort_keys=True, default=str))
    assert pcc["fused"] > 0.999, pcc


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=1152),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
def test_kda_output_projection_mmrs_balanced_lines(mesh_device: ttnn.MeshDevice) -> None:
    """balance_lines: 9 n-lines with per-block rotating widths (7, 7, 6 x 7) instead of 8 lines of 7. Every output tile
    keeps its K order, but the ring's two halves now split the 9 lines 5/4, so some blocks reduce their partials in a
    different cross-device order than the uniform plan: equal to it within BF16 rounding, not bit-identical. The
    activation in interleaved L1 (the KDA layer's norm output) is read through the same TensorAccessor: bit-identical
    too."""
    tp = tuple(mesh_device.shape)[_TP_AXIS]
    torch.manual_seed(0)
    a_host = torch.randn(1, 1, _ROWS, _K) * 0.5
    a_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=tuple(mesh_device.shape))
    a = {
        name: ttnn.from_torch(
            a_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=a_mapper,
        )
        for name, memory_config in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG))
    }
    w = ttnn.from_torch(
        torch.randn(_K, _N) / _K**0.5,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_ccl = TT_CCL(mesh_device)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))

    def fused(activation, balance_lines):
        return lambda: matmul_reduce_scatter(
            a[activation],
            w,
            cluster_axis=_TP_AXIS,
            scatter_dim=-1,
            topology=ttnn.Topology.Ring,
            num_links=tt_ccl.get_num_links(_TP_AXIS),
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            staging="ephemeral",
            balance_lines=balance_lines,
        )

    cases = {"uniform": ("dram", False), "balanced": ("dram", True), "balanced_l1_activation": ("l1", True)}
    results, plans, times = {}, {}, {}
    for name, (activation, balance_lines) in cases.items():
        out = fused(activation, balance_lines)()
        plans[name] = dict(_LAST_PLAN)
        results[name] = ttnn.to_torch(out, mesh_composer=composer)
        ttnn.deallocate(out)
        times[name] = statistics.median(_traced_us(mesh_device, fused(activation, balance_lines)))
    print(
        "KDA_OUTPUT_MMRS_BALANCED="
        + json.dumps({"tp": tp, "plans": plans, "median_us": times}, sort_keys=True, default=str)
    )
    assert not plans["uniform"]["balanced_lines"] and plans["balanced"]["balanced_lines"], plans
    cores = {name: plan["grid"][0] * plan["grid"][1] for name, plan in plans.items()}
    assert cores["balanced"] > cores["uniform"], plans
    for name in ("balanced", "balanced_l1_activation"):
        assert_with_pcc(results["uniform"].float(), results[name].float(), 0.99999)
