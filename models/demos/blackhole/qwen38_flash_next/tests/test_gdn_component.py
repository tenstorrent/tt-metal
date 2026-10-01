# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDN, Qwen38GDNWeights

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")


@pytest.fixture(scope="module")
def weights():
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    return Qwen38GDNWeights.from_checkpoint(checkpoint, layer_idx=0)


def test_exact_checkpoint_gdn_tp4_shards_reassemble_without_replication(weights):
    assert weights.dimensions.q_heads == 16
    assert weights.dimensions.value_heads == 48
    assert weights.dimensions.q_heads_per_device == 4
    assert weights.dimensions.value_heads_per_device == 12
    assert weights.dimensions.qkv_width_per_device == 2560
    assert weights.dimensions.value_width_per_device == 1536

    shards = [weights.device_shard(index) for index in range(4)]
    assert all(shard.qkv.shape == (2560, 2560) for shard in shards)
    assert all(shard.z.shape == (1536, 2560) for shard in shards)
    assert all(shard.a.shape == (12, 2560) for shard in shards)
    assert all(shard.b.shape == (12, 2560) for shard in shards)
    assert all(shard.out.shape == (2560, 1536) for shard in shards)
    assert all(shard.conv.shape == (2560, 1, 4) for shard in shards)
    assert all(shard.recurrent_state_shape(2) == (2, 12, 128, 128) for shard in shards)

    q, k, v = zip(*(shard.split_qkv() for shard in shards))
    torch.testing.assert_close(torch.cat(q), weights.qkv[:2048], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(k), weights.qkv[2048:4096], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(v), weights.qkv[4096:], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([shard.z for shard in shards]), weights.z, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([shard.a for shard in shards]), weights.a, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([shard.b for shard in shards]), weights.b, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([shard.out for shard in shards], dim=1), weights.out, rtol=0, atol=0)

    conv_q, conv_k, conv_v = zip(*(shard.split_conv() for shard in shards))
    torch.testing.assert_close(torch.cat(conv_q), weights.conv[:2048], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(conv_k), weights.conv[2048:4096], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(conv_v), weights.conv[4096:], rtol=0, atol=0)


@pytest.mark.skipif(not TRANSFORMERS_SRC, reason="set QWEN38_TRANSFORMERS_SRC to pinned Transformers source")
def test_exact_checkpoint_gdn_matches_pinned_transformers(weights):
    sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

    config = Qwen4ExpTextConfig.from_pretrained(CHECKPOINT)
    oracle = Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).to(torch.bfloat16).eval()
    oracle.load_state_dict(weights.transformers_state_dict(), strict=True)

    torch.manual_seed(211)
    hidden = torch.randn(1, 4, 2560, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = oracle(hidden, cache_params=None)
        got, state = Qwen38GDN(weights).forward(hidden)

    torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
    assert state.recurrent.dtype == torch.float32
    assert state.recurrent.shape == (1, 48, 128, 128)
    assert state.conv.shape == (1, 10240, 3)


def test_exact_checkpoint_gdn_prefill_equals_tokenwise_state_transition(weights):
    torch.manual_seed(223)
    hidden = torch.randn(1, 5, 2560, dtype=torch.bfloat16)
    component = Qwen38GDN(weights)

    full_output, full_state = component.forward(hidden)
    state = None
    pieces = []
    for position in range(hidden.shape[1]):
        output, state = component.forward(hidden[:, position : position + 1], state)
        pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full_output, rtol=0.02, atol=0.02)
    # BF16 GEMM may accumulate the [B,S,H] projection differently from five
    # separate [B,1,H] projections.  The recurrence itself is FP32; bound the
    # resulting projection-rounding envelope instead of demanding bit identity.
    torch.testing.assert_close(state.recurrent, full_state.recurrent, rtol=0.02, atol=5e-4)
    torch.testing.assert_close(state.conv, full_state.conv, rtol=0.01, atol=1e-3)
