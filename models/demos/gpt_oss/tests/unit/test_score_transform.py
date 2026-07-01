# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def _weight_and_reduce(combine_tt, scores_tt):
    combine_tiled = ttnn.to_layout(combine_tt, ttnn.TILE_LAYOUT)
    combine_4d = ttnn.unsqueeze_to_4D(combine_tiled)
    scores_tile = ttnn.to_layout(scores_tt, ttnn.TILE_LAYOUT)
    scores_t1 = ttnn.transpose(scores_tile, 0, 3)
    scores_t2 = ttnn.transpose(scores_t1, 2, 3)
    weighted = ttnn.mul(combine_4d, scores_t2)
    return ttnn.sum(weighted, dim=0, keepdim=True)


def _assert_close(actual, expected, label):
    actual = actual.float()
    expected = expected.float()
    max_abs = torch.max(torch.abs(actual - expected)).item()
    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2), f"{label}: max_abs={max_abs}"


def test_gpt_oss_score_transform_tile_to_row_major_l1_vs_dram(device):
    torch.manual_seed(0)
    tokens_per_device = 32
    num_experts_per_token = 4
    hidden_size = 64
    scores = torch.randn((tokens_per_device, num_experts_per_token), dtype=torch.bfloat16)
    combine = torch.randn((num_experts_per_token, tokens_per_device, hidden_size), dtype=torch.bfloat16)
    actual_by_memory = {}

    scores_tile = ttnn.from_torch(
        scores,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    scores_rm = ttnn.to_layout(scores_tile, ttnn.ROW_MAJOR_LAYOUT)
    scores_reshaped = ttnn.reshape(scores_rm, (tokens_per_device, 1, 1, num_experts_per_token))

    for label, scores_tt in (
        ("DRAM", ttnn.to_memory_config(scores_reshaped, ttnn.DRAM_MEMORY_CONFIG)),
        ("L1", scores_reshaped),
    ):
        combine_tt = ttnn.from_torch(
            combine,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        actual_by_memory[label] = ttnn.to_torch(_weight_and_reduce(combine_tt, scores_tt))

    _assert_close(actual_by_memory["L1"], actual_by_memory["DRAM"], "L1_vs_DRAM")


def test_gpt_oss_score_transform_from_host_l1_vs_dram(device):
    torch.manual_seed(0)
    tokens_per_device = 32
    num_experts_per_token = 4
    hidden_size = 64
    scores = torch.randn((tokens_per_device, 1, 1, num_experts_per_token), dtype=torch.bfloat16)
    combine = torch.randn((num_experts_per_token, tokens_per_device, hidden_size), dtype=torch.bfloat16)
    actual_by_memory = {}

    for label, memory_config in (("DRAM", ttnn.DRAM_MEMORY_CONFIG), ("L1", ttnn.L1_MEMORY_CONFIG)):
        scores_tt = ttnn.from_torch(
            scores,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
        )
        combine_tt = ttnn.from_torch(
            combine,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        actual = ttnn.to_torch(_weight_and_reduce(combine_tt, scores_tt))
        actual_by_memory[label] = actual

    _assert_close(actual_by_memory["L1"], actual_by_memory["DRAM"], "L1_vs_DRAM")
