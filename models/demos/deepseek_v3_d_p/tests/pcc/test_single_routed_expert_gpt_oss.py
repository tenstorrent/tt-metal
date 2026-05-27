# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS single-device, single-expert PCC test for TtRoutedExpert.

Mirrors test_single_routed_expert.py but with GPT-OSS dimensions
(emb_dim = hidden_dim = 2880). Lives in its own file so the DeepSeek
test stays untouched.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.gpt_oss_config import GptOssConfig
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [
        (1024, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (2048, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (3200, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (4096, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (8192, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (16384, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (25600, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
    ],
    ids=[
        "gpt-oss-1k",
        "gpt-oss-2k",
        "gpt-oss-3.2k",
        "gpt-oss-4k",
        "gpt-oss-8k",
        "gpt-oss-16k",
        "gpt-oss-25k",
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            id="single-chip",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_single_routed_expert_gpt_oss(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """Simplest test for GPT-OSS dims: 1 chip, 1 expert."""
    experts_per_chip = 1

    signpost(f"SingleRoutedExpertGptOss {num_tokens=} {emb_dim=} {hidden_dim=}")

    logger.debug(f"Testing GPT-OSS single routed expert: {num_tokens=}, {emb_dim=}, {hidden_dim=}")
    logger.debug(f"Mesh: {mesh_device.shape}, num_devices={mesh_device.get_num_devices()}")

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }

    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)
    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)

    with torch.no_grad():
        torch_output = torch_expert(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([num_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )

    _, pcc = comp_pcc(torch_output, tt_output_torch)
    logger.debug(f"PCC: {pcc:.6f}")

    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    logger.debug("Test PASSED!")


@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [
        (4096, 2048, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (25 * 1024, 2048, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (25 * 1024, 4096, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (16384, 2048, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
        (16384, 4096, GptOssConfig.EMB_SIZE, GptOssConfig.MOE_INTERMEDIATE_SIZE),
    ],
    ids=[
        "gpt-oss-4k-alloc-2k-active",
        "gpt-oss-25k-alloc-2k-active",
        "gpt-oss-25k-alloc-4k-active",
        "gpt-oss-16k-alloc-2k-active",
        "gpt-oss-16k-alloc-4k-active",
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            id="single-chip",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_single_routed_expert_faked_token_count_gpt_oss(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Same dispatch-buffer/count-sparsity check as the DeepSeek variant,
    but with GPT-OSS dims.
    """
    experts_per_chip = 1

    signpost(f"SingleRoutedExpertFakedGptOss {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=}")

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active

    with torch.no_grad():
        torch_output_active = torch_expert(torch_active)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([active_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    import time as _time

    for _i in range(5):
        _t0 = _time.time()
        tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
        ttnn.synchronize_device(mesh_device)
        _dt_ms = (_time.time() - _t0) * 1000
        logger.warning(f"  faked iter {_i}: {_dt_ms:.2f} ms (alloc={allocated_tokens}, active={active_tokens})")

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    tt_output_active = tt_output_torch[:active_tokens]

    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows): {pcc:.6f}")

    assert pcc >= 0.97, f"PCC {pcc:.6f} below threshold 0.97"
    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"
