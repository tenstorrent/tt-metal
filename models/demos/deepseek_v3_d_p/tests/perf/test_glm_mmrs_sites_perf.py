# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 matmul -> TP reduce-scatter sites on Galaxy SP8xTP4 (640 local rows): linear +
reduce_scatter_minimal_async vs the fused matmul_reduce_scatter, per site."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import get_program_configs
from models.tt_transformers.tt.ccl import TT_CCL
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]

_ROWS = 640
_TP_AXIS = 1
_REPLAYS = 50
_SAMPLES = 5
# name: (K per device, N, scatter_dim, activation dtype)
_SITES = {
    "mla_q_a_proj": (1536, 2048, -1, ttnn.bfloat16),
    "mla_o_proj": (4096, 6144, -1, ttnn.bfloat16),
    "dense_ffn_down": (3072, 6144, -1, ttnn.bfloat16),
    "shared_expert_down": (512, 6144, -1, ttnn.bfloat16),
    "indexer_wk": (1536, 128, -1, ttnn.bfloat16),
    "indexer_weights_proj": (1536, 32, -2, ttnn.bfloat16),
}


# The matmul config the model itself runs at each site (MLA / indexer: GLM's tuned 640-row entry; dense FFN: auto).
_MLA_KEYS = {
    "mla_q_a_proj": "q_a_proj",
    "mla_o_proj": "o_proj",
    "indexer_wk": "indexer.wk",
    "indexer_weights_proj": "indexer.weights_proj",
}


def _model_linear_kwargs(site: str, mesh_device) -> dict:
    if site in _MLA_KEYS:
        entries = MLA_MATMUL_CONFIG[_MLA_KEYS[site]][_ROWS]
        entries = entries if isinstance(entries, list) else [entries]
        (cfg,) = [e for e in entries if e.get("num_heads") == 64 and e.get("q_lora_rank") == 2048]
        return {"program_config": cfg["program_config"], "memory_config": cfg["out_mem_config"], "dtype": cfg["out_dtype"]}
    if site == "shared_expert_down":
        grid = mesh_device.compute_with_storage_grid_size()
        down = get_program_configs(grid, _ROWS // 32, 1536 // 32, 512 // 32, 512 // 32, 6144 // 32)[2]
        return {"program_config": down, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
    return {"memory_config": ttnn.DRAM_MEMORY_CONFIG}


def _traced_us(mesh_device, fn) -> float:
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
    return statistics.median(samples)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=1536),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("site", list(_SITES))
def test_glm_mmrs_site(mesh_device: ttnn.MeshDevice, site: str) -> None:
    k_local, n, scatter_dim, a_dtype = _SITES[site]
    tp = tuple(mesh_device.shape)[_TP_AXIS]
    mesh_shape = tuple(mesh_device.shape)
    torch.manual_seed(0)
    a_host = torch.randn(1, 1, _ROWS, k_local * tp) * 0.5
    w_host = torch.randn(k_local * tp, n) / (k_local * tp) ** 0.5
    a = ttnn.from_torch(
        a_host,
        dtype=a_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=mesh_shape),
    )
    w = ttnn.from_torch(
        w_host,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_shape),
    )
    # MLA's default projection numerics (HiFi2, bf16 DEST, packer L1 accumulation).
    linear_compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    tt_ccl = TT_CCL(mesh_device)
    rs_dim = 3 if scatter_dim == -1 else 2

    linear_kwargs = _model_linear_kwargs(site, mesh_device)

    def linear():
        return ttnn.linear(a, w, compute_kernel_config=linear_compute, **linear_kwargs)

    def reduce_scatter(partial):
        return ttnn.experimental.reduce_scatter_minimal_async(
            partial,
            dim=rs_dim,
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
            scatter_dim=scatter_dim,
            topology=ttnn.Topology.Ring,
            num_links=tt_ccl.get_num_links(_TP_AXIS),
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    golden = a_host[0, 0].float() @ w_host.float()
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3) if scatter_dim == -1 else (3, 2), mesh_shape=mesh_shape)
    pcc = {}
    for name, fn in (("unfused", unfused), ("fused", fused)):
        out = fn()
        got = ttnn.to_torch(out, mesh_composer=composer)[0, 0].float()
        got = got[:_ROWS] if scatter_dim == -1 else got[:, :n]
        ttnn.deallocate(out)
        pcc[name] = float(comp_pcc(golden, got)[1])

    ablate = os.environ.get("MRS_ABLATE", "")
    times = {name: _traced_us(mesh_device, fn) for name, fn in (("linear", linear), ("reduce_scatter", rs_only), ("unfused", unfused), ("fused", fused))}
    ttnn.deallocate(partial_fixed)
    serial = times["linear"] + times["reduce_scatter"]
    result = {
        "site": site,
        "shape": f"{_ROWS}x{k_local}x{n} per device, scatter {scatter_dim}, TP{tp} ring",
        "ablate": ablate,
        "pcc": pcc,
        "median_us": {k: round(v, 1) for k, v in times.items()},
        "saved_us": round(times["unfused"] - times["fused"], 1),
        "hidden_rs_fraction": round((serial - times["fused"]) / times["reduce_scatter"], 3),
        "plan": {k: sys.modules["ttnn.operations.matmul_reduce_scatter"]._LAST_PLAN.get(k) for k in ("grid", "P", "kb", "xblock", "chunks")},
    }
    print("GLM_MMRS_SITE=" + json.dumps(result, sort_keys=True, default=str))
    if not ablate:
        assert pcc["fused"] > 0.999, pcc
