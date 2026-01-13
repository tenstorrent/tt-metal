#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from tests.ttnn.utils_for_testing import assert_with_pcc

# Memory config matching model's decode mode (see MoEGate.model_config)
TOPK_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG

# Sub-core grids for mesh device tests
SUB_CORE_GRIDS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8, 9))])


# k=32 matches the DeepSeek v3 MoE gating configuration, where the gate selects 32 experts per token.
K_VALUE = 32


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 8, 64],  # Note: (1, 32, 32, 64) is the padded shape
        [1, 1, 32, 64],
        [1, 1, 32, 256],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_topk_single_device(shape, dtype, device):
    """
    Single device topk test.

    The model uses L1_MEMORY_CONFIG for decode mode (see MoEGate.model_config).
    """
    torch.manual_seed(1234)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_values, torch_indices = torch.topk(torch_input, k=K_VALUE, dim=-1, largest=True, sorted=True)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=TOPK_MEMORY_CONFIG,
    )

    tt_values, tt_indices = ttnn.topk(
        tt_input,
        k=K_VALUE,
        dim=-1,
        largest=True,
        sorted=True,
        memory_config=TOPK_MEMORY_CONFIG,
    )

    tt_values_torch = ttnn.to_torch(tt_values)
    tt_indices_torch = ttnn.to_torch(tt_indices).to(torch.int64)

    # Check output shapes
    expected_shape = list(shape)
    expected_shape[-1] = K_VALUE
    assert list(tt_values.shape) == expected_shape
    assert list(tt_indices.shape) == expected_shape

    # Check values with PCC
    assert_with_pcc(torch_values, tt_values_torch, 0.999)

    # Check indices using cosine similarity on gathered values
    # (PCC isn't ideal for indices since tied values can produce different orderings)
    tt_gathered = torch.gather(torch_input, -1, tt_indices_torch)
    cosine = torch.nn.CosineSimilarity(dim=-1)
    cosine_sim = torch.mean(cosine(torch_values, tt_gathered))
    assert cosine_sim > 0.99, f"Cosine similarity {cosine_sim} is less than 0.99"


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 8, 64],  # Note: (1, 32, 32, 64) is the padded shape
        [1, 1, 32, 64],
        [1, 1, 32, 256],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_topk_mesh_device(mesh_device, shape, dtype, enable_trace, device_params):
    """
    Mesh device topk test with model-realistic shapes and k values.

    Uses SUB_CORE_GRIDS to limit the cores used, matching model usage patterns.
    """
    torch_input = random_torch_tensor(dtype, shape)
    torch_values, _ = torch.topk(torch_input, k=K_VALUE, dim=-1, largest=True, sorted=True)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=TOPK_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        tt_values, _ = ttnn.topk(
            tt_input,
            k=K_VALUE,
            dim=-1,
            largest=True,
            sorted=True,
            memory_config=TOPK_MEMORY_CONFIG,
            sub_core_grids=SUB_CORE_GRIDS,
        )
        return tt_values

    def check_op(tt_output):
        assert_with_pcc(torch_values, tt_output, 0.999)

    run_test(mesh_device, run_op, check_op, enable_trace)
