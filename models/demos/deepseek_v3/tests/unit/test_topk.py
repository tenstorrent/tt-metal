#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from models.demos.deepseek_v3.utils.config_helpers import TOPK_MIN_WIDTH
from tests.ttnn.utils_for_testing import assert_with_pcc

# Memory config matching model's decode mode (see MoEGate.model_config)
TOPK_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG

# Sub-core grids for mesh device tests
SUB_CORE_GRIDS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8, 9))])


# k=32 matches model usage patterns
K_VALUE = 32


@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 32, 64],
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


@pytest.mark.parametrize(
    "shape, k",
    [
        # Width < TOPK_MIN_WIDTH (64) requires padding in model (see MoEGate.forward)
        # Model pads with -inf: ttnn.pad(..., value=-float("inf"))
        ([1, 1, 32, 32], 2),  # 32 < 64, needs padding
        ([1, 1, 32, 16], 2),  # 16 < 64, needs padding
        ([1, 1, 32, 48], 4),  # 48 < 64, needs padding
    ],
    ids=[
        "width_32_needs_padding",
        "width_16_needs_padding",
        "width_48_needs_padding",
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_topk_with_padding(shape, k, dtype, device):
    """
    Test topk with padding for inputs smaller than TOPK_MIN_WIDTH.

    The model pads inputs with width < TOPK_MIN_WIDTH (64) using -inf padding
    before calling ttnn.topk (see MoEGate.forward):

        if expert_scores_grouped.shape[3] < TOPK_MIN_WIDTH:
            expert_scores_grouped = ttnn.pad(
                expert_scores_grouped,
                [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - expert_scores_grouped.shape[3])],
                value=-float("inf"),
            )

    This test verifies the padding approach works correctly.
    """
    torch.manual_seed(1234)
    original_width = shape[-1]
    assert original_width < TOPK_MIN_WIDTH, f"Test expects width < {TOPK_MIN_WIDTH}"

    torch_input = torch.rand(shape, dtype=torch.bfloat16)

    # Reference: topk on original input
    torch_values, _ = torch.topk(torch_input, k=k, dim=-1, largest=True, sorted=True)

    # Pad input to TOPK_MIN_WIDTH (same as model does)
    padded_shape = list(shape)
    padded_shape[-1] = TOPK_MIN_WIDTH
    torch_input_padded = torch.full(padded_shape, -float("inf"), dtype=torch.bfloat16)
    torch_input_padded[..., :original_width] = torch_input

    tt_input = ttnn.from_torch(
        torch_input_padded,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=TOPK_MEMORY_CONFIG,
    )

    tt_values, tt_indices = ttnn.topk(
        tt_input,
        k=k,
        dim=-1,
        largest=True,
        sorted=True,
        memory_config=TOPK_MEMORY_CONFIG,
    )

    tt_values_torch = ttnn.to_torch(tt_values)
    tt_indices_torch = ttnn.to_torch(tt_indices).to(torch.int64)

    # Check output shapes
    expected_shape = list(padded_shape)
    expected_shape[-1] = k
    assert list(tt_values.shape) == expected_shape
    assert list(tt_indices.shape) == expected_shape

    # Values should match (padded -inf values shouldn't appear in top-k since k < original_width)
    assert_with_pcc(torch_values, tt_values_torch, 0.999)

    # Indices should be within original width (not pointing to padded region)
    assert (tt_indices_torch < original_width).all(), "Indices should not point to padded region"


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, k",
    [
        # Model-realistic shapes for mesh device tests
        ([32, 8, 64], 8),  # batch=32, k=num_experts_per_tok
        ([1, 32, 64], 4),  # k=topk_group
        ([1, 32, 256], 8),  # larger width, k=num_experts_per_tok
    ],
    ids=[
        "batch32_k8",
        "batch1_k4",
        "batch1_width256_k8",
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_topk_mesh_device(mesh_device, shape, k, dtype, enable_trace, device_params):
    """
    Mesh device topk test with model-realistic shapes and k values.

    Uses SUB_CORE_GRIDS to limit the cores used, matching model usage patterns.
    """
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

    def run_op():
        tt_values, _ = ttnn.topk(
            tt_input,
            k=k,
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
