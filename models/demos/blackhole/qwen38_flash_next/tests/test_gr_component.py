# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidual, Qwen38GatedResidualWeights

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")


@pytest.fixture(scope="module")
def component():
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    weights = Qwen38GatedResidualWeights.from_checkpoint(checkpoint, placement, layer_index=0, block="attn")
    return Qwen38GatedResidual(weights)


def test_exact_gr_tp4_shards_reassemble_without_replication(component):
    weights = component.weights
    shards = [weights.device_shard(device) for device in range(4)]

    assert all(shard.norm.shape == (4, 640) for shard in shards)
    assert all(shard.down.shape == (320, 4, 640) for shard in shards)
    assert all(shard.up.shape == (4, 640, 320) for shard in shards)
    assert all(shard.inject.shape == (4, 4, 640) for shard in shards)
    assert torch.equal(torch.cat([shard.norm for shard in shards], dim=1).flatten(), weights.norm)
    assert torch.equal(torch.cat([shard.down for shard in shards], dim=2).flatten(1), weights.down)
    assert torch.equal(torch.cat([shard.up for shard in shards], dim=1).flatten(0, 1), weights.up)
    assert torch.equal(torch.cat([shard.inject for shard in shards], dim=2).flatten(1), weights.inject)


@pytest.mark.skipif(not TRANSFORMERS_SRC, reason="set QWEN38_TRANSFORMERS_SRC to pinned Transformers source")
def test_exact_checkpoint_gr_and_tp4_decomposition_match_pinned_transformers(component):
    sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    config = Qwen4ExpTextConfig.from_pretrained(CHECKPOINT)
    oracle = Qwen4ExpTextGatedResidual(config).to(torch.bfloat16).eval()
    oracle.load_state_dict(component.weights.transformers_state_dict(), strict=True)

    generator = torch.Generator().manual_seed(41)
    residual = (torch.randn((1, 3, 10240), generator=generator) * 0.02).to(torch.bfloat16)
    block_output = (torch.randn((1, 3, 2560), generator=generator) * 0.02).to(torch.bfloat16)
    with torch.no_grad():
        expected_input, expected_residual, expected_injection = oracle(residual)
        expected_output = expected_residual + (expected_injection.unsqueeze(-1) * block_output.unsqueeze(-2)).flatten(
            -2
        )
        got_input, state = component.read(residual)
        got_output = component.write(block_output, state)

    torch.testing.assert_close(got_input, expected_input, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state.injection, expected_injection, rtol=0.0, atol=0.0)
    torch.testing.assert_close(got_output, expected_output, rtol=0.0, atol=0.0)

    residual_shards = component.shard_residual(residual)
    input_shards, tp_state = component.read_tp4(residual_shards)
    output_shards = component.write_tp4(component.shard_hidden(block_output), tp_state)
    torch.testing.assert_close(torch.cat(input_shards, dim=-1), expected_input, rtol=0.01, atol=0.01)
    torch.testing.assert_close(component.combine_residual_shards(output_shards), expected_output, rtol=0.01, atol=0.01)
