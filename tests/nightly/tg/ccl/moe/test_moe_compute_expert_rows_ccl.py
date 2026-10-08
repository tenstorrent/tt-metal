# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""moe_compute FullCcl through the expert rows programs.

Runs moe_compute's own end-to-end check (test_moe_compute_6U._run_model_test: per-expert counts, activation rows, e_t,
the double buffer and the combine output against the goldens, over several layers, eager or traced) on the ring
program and on the expert rows programs (each forced with TT_METAL_MOE_COMPUTE_KERNEL; the expert rows arm on
Blackhole only for now).

Layouts: every mesh of test_moe_compute_6U (Galaxy 1x8 / 1x16 torus and linear, 8x1 / 16x1 rows, the Blackhole
LoudBox lines), each run when TT_MESH_GRAPH_DESC_PATH names its descriptor, as in that file; and a 1xN line for other
boxes when MOE_EXPERT_ROWS_MESH_WIDTH is set (MOE_EXPERT_ROWS_NUM_LINKS, default 2; TT_MESH_GRAPH_DESC_PATH describes
the line).
"""

import os
from contextlib import contextmanager

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

KERNEL_ENV = "TT_METAL_MOE_COMPUTE_KERNEL"

# (model, experts per device, has_bias, activation): moe_compute's 6U models, fewer layers
MODELS = {
    "deepseek_v3": (
        MoEModelConfig(
            "deepseek_v3",
            N=2048,
            hidden_size=7168,
            selected_experts_k=8,
            tokens_per_device=8,
            num_layers=2,
            num_iterations=2,
        ),
        2,
        False,
        MoEActivationFunction.SILU,
    ),
    "deepseek_v3_bias": (
        MoEModelConfig(
            "deepseek_v3",
            N=2048,
            hidden_size=7168,
            selected_experts_k=8,
            tokens_per_device=8,
            num_layers=2,
            num_iterations=2,
        ),
        2,
        True,
        MoEActivationFunction.SILU,
    ),
    "gpt_oss": (
        MoEModelConfig("gpt_oss", N=2880, hidden_size=2880, selected_experts_k=4, num_layers=2, num_iterations=2),
        4,
        True,
        MoEActivationFunction.SWIGLU,
    ),
    "gemma_4_26b": (
        MoEModelConfig("gemma_4_26b", N=704, hidden_size=2816, selected_experts_k=8, num_layers=2, num_iterations=2),
        2,
        False,
        MoEActivationFunction.GELU,
    ),
    # top-10 needs at least 10 experts over the line: 4 per device
    "qwen35_397b": (
        MoEModelConfig("qwen35_397b", N=1024, hidden_size=4096, selected_experts_k=10, num_layers=2, num_iterations=2),
        4,
        False,
        MoEActivationFunction.SILU,
    ),
    "glm5": (
        MoEModelConfig("glm5", N=2048, hidden_size=6144, selected_experts_k=8, num_layers=2, num_iterations=2),
        2,
        False,
        MoEActivationFunction.SILU,
    ),
}


@contextmanager
def _kernel(name):
    old = os.environ.get(KERNEL_ENV)
    os.environ[KERNEL_ENV] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(KERNEL_ENV, None)
        else:
            os.environ[KERNEL_ENV] = old


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
    width = int(os.environ.get("MOE_EXPERT_ROWS_MESH_WIDTH", "4"))
    line = MoEMeshConfig(
        f"1x{width}-linear",
        width,
        os.environ.get("TT_MESH_GRAPH_DESC_PATH", ""),
        (),
        MOE_DEVICE_PARAMS_LINEAR,
        use_linear_topology=True,
        num_links=int(os.environ.get("MOE_EXPERT_ROWS_NUM_LINKS", "2")),
    )
    skip = pytest.mark.skipif(
        "MOE_EXPERT_ROWS_MESH_WIDTH" not in os.environ,
        reason="the 1xN line runs when MOE_EXPERT_ROWS_MESH_WIDTH is set",
    )
    cases.append(pytest.param(line.device_params, line.mesh_shape, line, marks=skip, id=line.name))
    return cases


@pytest.mark.parametrize("device_params, mesh_device, mesh_cfg", _layouts(), indirect=["device_params", "mesh_device"])
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("kernel", ["expert_rows", "ring"])
@pytest.mark.parametrize("model", sorted(MODELS))
def test_moe_compute_expert_rows_full_ccl(mesh_device, device_params, mesh_cfg, enable_trace, kernel, model):
    if kernel == "expert_rows" and mesh_device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("Wormhole keeps the ring path; its forced expert rows FullCcl waits for a run on a healthy Galaxy")
    model_cfg, experts_per_device, has_bias, activation = MODELS[model]
    with _kernel(kernel):
        _run_model_test(
            mesh_device,
            mesh_cfg.mesh_shape,
            enable_trace,
            model_cfg,
            "correctness",
            has_bias,
            experts_per_device,
            activation,
            num_links=mesh_cfg.num_links,
            topology=Topology.Linear if mesh_cfg.use_linear_topology else None,
            cluster_axis=mesh_cfg.cluster_axis,
        )
