# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for matmul_reduce_scatter: the feature_spec LOOSE_CASES on the placement their extras pin.

Run under the device profiler:
    scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_perf.py
Each case does WARMUP calls (compile + cache) then MEASURED calls; read the per-device
"DEVICE KERNEL DURATION [ns]" of the generic op rows in the profiler CSV. The op's latency is the
MAX over the group's devices of one call (the slowest chip finishes the reduce-scatter).
Correctness is checked on the last call against each case's precision gate.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.matmul_reduce_scatter.feature_spec import LOOSE_CASES
from eval.golden_tests.matmul_reduce_scatter.helpers import (
    TOLERANCES,
    create_ttnn_input_tensor,
    pytorch_matmul_reduce_scatter,
    quantize_like_device,
    random_stacked,
)
from eval.golden_tests.matmul_reduce_scatter.topology import pinned_mesh_feasible
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

WARMUP = 2
MEASURED = 3


def _open(mesh_shape, fabric_config):
    from tests.scripts.common import get_updated_device_params

    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        cfg,
    )
    params = get_updated_device_params({"fabric_config": fabric_config})
    params.pop("fabric_config")
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **params)


@pytest.fixture(scope="module")
def perf_mesh():
    ex = LOOSE_CASES[0]["extras"]
    if pinned_mesh_feasible(ex["mesh_shape"], ex["fabric_config"], LOOSE_CASES[0]["topology"], 1) is not None:
        pytest.skip("perf placement not available on this cluster")
    mesh = _open(ex["mesh_shape"], ex["fabric_config"])
    yield mesh
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _id(case):
    (a, w) = case["inputs"]
    return f"M{a[-2]}_K{a[-1]}_N{w[-1]}_sd{case['scatter_dim']}"


@pytest.mark.parametrize("case", LOOSE_CASES, ids=[_id(c) for c in LOOSE_CASES])
def test_perf_case(perf_mesh, case):
    ex = case["extras"]
    a_shape, w_shape = case["inputs"]
    a = random_stacked(perf_mesh, a_shape, torch.bfloat16, seed=0)
    w = quantize_like_device(
        random_stacked(perf_mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5), case["weight_dtype"]
    )
    ta = create_ttnn_input_tensor(a, perf_mesh, dtype=case["dtype"], layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(w, perf_mesh, dtype=case["weight_dtype"], layout=ttnn.TILE_LAYOUT)
    ckc = ttnn.ComputeConfigDescriptor(math_fidelity=ex["math_fidelity"], fp32_dest_acc_en=ex["fp32_dest_acc_en"])
    out = None
    for _ in range(WARMUP + MEASURED):
        out = matmul_reduce_scatter(
            ta,
            tw,
            cluster_axis=case["cluster_axis"],
            scatter_dim=case["scatter_dim"],
            topology=case["topology"],
            num_links=case["num_links"],
            compute_kernel_config=ckc,
        )
    ttnn.synchronize_device(perf_mesh)
    expected = pytorch_matmul_reduce_scatter(a, w, case["cluster_axis"], case["scatter_dim"])
    cols = tuple(perf_mesh.shape)[1]
    pcc_t, rms_t = TOLERANCES[case["weight_dtype"]]
    rms_t = ex.get("rel_rms_threshold", rms_t)
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).double()
        ref = expected[idx // cols, idx % cols].double()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        assert rms <= rms_t, f"device {idx}: rel_rms {rms} > {rms_t}"
