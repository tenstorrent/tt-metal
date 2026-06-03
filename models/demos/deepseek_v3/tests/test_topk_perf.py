# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-side perf test for the ttnn.topk op as used by MoEGate (topk_experts).

This captures a trace of repeated ``ttnn.topk`` calls and brackets the main
trace execution with signposts so the perf harness (``perf_topk.py``) can
isolate the relevant device kernels.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor
from models.demos.deepseek_v3.utils.config_helpers import get_fabric_config
from tests.ttnn.utils_for_testing import assert_with_pcc

# Memory config matching the model's decode mode (see MoEGate.model_config).
TOPK_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG

# Sub-core grid matching model usage patterns.
SUB_CORE_GRIDS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 7))])

# MoEGate topk_experts: shape (1, 1, USERS_PER_ROW=32, n_routed_experts=256), k=num_experts_per_tok=8.
TOPK_EXPERTS_SHAPE = [1, 1, 1, 256]
TOPK_EXPERTS_K = 8


@pytest.mark.requires_device(["TG"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 7000000, "fabric_config": get_fabric_config()}],
    indirect=True,
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "warmup_iters, num_iters",
    [
        (5, 10),
    ],
)
def test_topk_perf(mesh_device, dtype, warmup_iters, num_iters, device_params):
    """Capture and execute a trace of ``ttnn.topk`` (topk_experts) for perf measurement."""
    shape = TOPK_EXPERTS_SHAPE
    k = TOPK_EXPERTS_K

    torch_input = random_torch_tensor(dtype, shape)
    torch_values, _ = torch.topk(torch_input, k=k, dim=-1, largest=True, sorted=True)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=TOPK_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_topk():
        tt_values, tt_indices = ttnn.topk(
            tt_input,
            k=k,
            dim=-1,
            largest=True,
            sorted=True,
            memory_config=TOPK_MEMORY_CONFIG,
            sub_core_grids=SUB_CORE_GRIDS,
        )
        return tt_values, tt_indices

    # Compile the op
    tt_values, tt_indices = run_topk()
    ttnn.synchronize_device(mesh_device)

    tt_values_torch = ttnn.to_torch(
        ttnn.get_device_tensors(tt_values)[0],
    )
    assert_with_pcc(torch_values, tt_values_torch, 0.999)

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
