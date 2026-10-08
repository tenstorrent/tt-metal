# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""moe_compute FullCcl calls of different sizes in one process (#59738).

Each program-cache miss of moe_compute FullCcl creates the combine's barrier semaphores. Created in general L1 while the
call's L1 outputs are alive, they stay behind in the middle of L1 once those outputs are freed, and a later call can no
longer allocate its 64 * H * 2-byte double buffer. This runs moe_compute's own end-to-end check
(test_moe_compute_6U._run_model_test) for a DeepSeek-V3-like shape at 1 and then 8 tokens per device in one process.

Layouts: every mesh of test_moe_compute_6U (Galaxy 1x8 / 1x16 torus and linear, 8x1 / 16x1 rows, the Blackhole
LoudBox lines), each run when TT_MESH_GRAPH_DESC_PATH names its descriptor, as in that file; and a 1xN line for other
boxes when MOE_BARRIER_MESH_WIDTH is set (MOE_BARRIER_NUM_LINKS, default 2; TT_MESH_GRAPH_DESC_PATH describes the line).
"""

import os

import pytest
import ttnn
from ttnn.operations.ccl import MoEActivationFunction, Topology

from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
    _MOE_MESH_CONFIGS,
    MOE_DEVICE_PARAMS_LINEAR,
    MoEMeshConfig,
    MoEModelConfig,
    _run_model_test,
    is_mesh_graph_descriptor_set,
)


def _layouts():
    """One case per mesh descriptor and cluster axis of test_moe_compute_6U, plus the 1xN development line."""
    cases, seen = [], set()
    for cfg in _MOE_MESH_CONFIGS:
        key = (cfg.mesh_graph_desc, cfg.cluster_axis)
        if key in seen:
            continue
        seen.add(key)
        skip = pytest.mark.skipif(
            not is_mesh_graph_descriptor_set(cfg.mesh_graph_desc),
            reason=f"{cfg.name} requires TT_MESH_GRAPH_DESC_PATH={cfg.mesh_graph_desc}",
        )
        cases.append(pytest.param(cfg.device_params, cfg.mesh_shape, cfg, marks=skip, id=cfg.name))
    width = int(os.environ.get("MOE_BARRIER_MESH_WIDTH", "4"))
    line = MoEMeshConfig(
        f"1x{width}-linear",
        width,
        os.environ.get("TT_MESH_GRAPH_DESC_PATH", ""),
        (),
        MOE_DEVICE_PARAMS_LINEAR,
        use_linear_topology=True,
        num_links=int(os.environ.get("MOE_BARRIER_NUM_LINKS", "2")),
    )
    skip = pytest.mark.skipif(
        "MOE_BARRIER_MESH_WIDTH" not in os.environ, reason="the 1xN line runs when MOE_BARRIER_MESH_WIDTH is set"
    )
    cases.append(pytest.param(line.device_params, line.mesh_shape, line, marks=skip, id=line.name))
    return cases


@pytest.mark.parametrize("device_params, mesh_device, mesh_cfg", _layouts(), indirect=["device_params", "mesh_device"])
def test_moe_compute_full_ccl_call_sizes_in_one_process(mesh_device, device_params, mesh_cfg):
    for tokens_per_device in (1, 8):
        model_cfg = MoEModelConfig(
            "deepseek_v3",
            N=2048,
            hidden_size=7168,
            selected_experts_k=8,
            tokens_per_device=tokens_per_device,
            num_layers=1,
            num_iterations=1,
        )
        _run_model_test(
            mesh_device,
            mesh_cfg.mesh_shape,
            False,
            model_cfg,
            "correctness",
            False,
            8,
            MoEActivationFunction.SILU,
            num_links=mesh_cfg.num_links,
            topology=Topology.Linear if mesh_cfg.use_linear_topology else None,
            cluster_axis=mesh_cfg.cluster_axis,
        )
        ttnn.synchronize_device(mesh_device)
