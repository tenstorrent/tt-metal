# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 schedule pins: the per-plan exchange transport and the pipelined row blocks.

* Back-to-back calls on one cached program with DIFFERENT data. The single-round op-layout plans exchange with
  concurrent_unicast (one arrival semaphore, no handshake), so a stale semaphore count or gather slot from the previous
  call would let a core finalize the previous call's statistics; same-data repeats cannot see that.
* The plan: single-block shards whose rows split evenly into >= 2 PIPELINE_BLOCK_ROWS-row blocks run pipelined, and
  the exchange mode follows the round count.
"""

import importlib

import pytest
import torch
import ttnn

from .test_nomic_residual_layer_norm import EPSILON, check, make_inputs, to_device, torch_reference

op_module = importlib.import_module("ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm")


@pytest.mark.parametrize(
    "shape",
    [(2, 1, 512, 768), (4, 1, 512, 768), (8, 1, 512, 768), (4, 1, 288, 768)],
    ids=lambda s: f"{s[0]}x{s[2]}",
)
def test_back_to_back_calls_with_different_data(device, shape):
    first = make_inputs(shape, True)
    torch.manual_seed(7)
    second_x = (3.0 * torch.randn(shape) + 1.5).to(torch.bfloat16)
    second_r = torch.randn(shape).to(torch.bfloat16)
    gamma, beta = first[2], first[3]

    tt_first = to_device(*first, device)
    tt_second = to_device(second_x, second_r, gamma, beta, device)
    outputs = []
    for tt_x, tt_r, tt_g, tt_b in (tt_first, tt_second, tt_first):
        outputs.append(
            op_module.nomic_residual_layer_norm(
                tt_x,
                tt_g,
                tt_b,
                residual=tt_r,
                epsilon=EPSILON,
                compute_kernel_config=op_module.default_compute_kernel_config(),
            )
        )
    expected_first = torch_reference(first[0], gamma, beta, first[1])
    expected_second = torch_reference(second_x, gamma, beta, second_r)
    check(ttnn.to_torch(outputs[0]).float(), expected_first)
    check(ttnn.to_torch(outputs[1]).float(), expected_second)
    check(ttnn.to_torch(outputs[2]).float(), expected_first)


def test_plan_pipelines_single_block_shards(device):
    plan = op_module._plan((8, 1, 512, 768), device)  # 4-row op-layout shards -> 2 blocks of 2 rows
    assert (plan.rows_per_shard, plan.block_rows, plan.num_row_blocks) == (4, 2, 2)
    plan = op_module._plan((4, 1, 512, 768), device)  # 2 rows: a split would give one block, so none
    assert (plan.rows_per_shard, plan.block_rows) == (2, 2)
    plan = op_module._plan((8, 1, 512, 768), device, "model_layout")  # L1 already splits: unchanged
    assert (plan.rows_per_shard, plan.block_rows, plan.num_row_blocks) == (12, 4, 3)
    plan = op_module._plan((2, 1, 512, 768), device, "model_layout")  # bh 3: odd, unchanged
    assert (plan.rows_per_shard, plan.block_rows) == (3, 3)
