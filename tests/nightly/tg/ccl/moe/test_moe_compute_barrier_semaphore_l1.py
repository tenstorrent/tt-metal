# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""moe_compute FullCcl calls of different sizes in one process (#59738).

Each program-cache miss of moe_compute FullCcl creates the combine's barrier semaphores. Created in general L1 while the
call's L1 outputs are alive, they stay behind in the middle of L1 once those outputs are freed, and a later call can no
longer allocate its 64 * H * 2-byte double buffer. This runs moe_compute's own end-to-end check
(test_moe_compute_6U._run_model_test) for a DeepSeek-V3-like shape at 1 and then 8 tokens per device in one process, on
the 1xN line that TT_MESH_GRAPH_DESC_PATH describes (MOE_BARRIER_MESH_WIDTH, default 4; MOE_BARRIER_NUM_LINKS, default
2).
"""

import os

import pytest
import ttnn
from ttnn.operations.ccl import MoEActivationFunction, Topology

from tests.nightly.tg.ccl.moe.test_moe_compute_6U import MOE_DEVICE_PARAMS_LINEAR, MoEModelConfig, _run_model_test

MESH_WIDTH = int(os.environ.get("MOE_BARRIER_MESH_WIDTH", "4"))
NUM_LINKS = int(os.environ.get("MOE_BARRIER_NUM_LINKS", "2"))


@pytest.mark.parametrize("device_params", [MOE_DEVICE_PARAMS_LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, MESH_WIDTH)], indirect=True)
def test_moe_compute_full_ccl_call_sizes_in_one_process(mesh_device, device_params):
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
            (1, MESH_WIDTH),
            False,
            model_cfg,
            "correctness",
            False,
            8,
            MoEActivationFunction.SILU,
            num_links=NUM_LINKS,
            topology=Topology.Linear,
            cluster_axis=1,
        )
        ttnn.synchronize_device(mesh_device)
