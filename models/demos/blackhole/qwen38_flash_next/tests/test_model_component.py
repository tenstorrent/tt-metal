# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.model import (
    Qwen38FinalMixer,
    Qwen38FinalMixerWeights,
    Qwen38ModelIO,
    text_rope,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import _host_rope, _inverse_frequency

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")


@pytest.fixture(scope="module")
def checkpoint_and_placement():
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    return checkpoint, placement


@pytest.mark.skipif(not TRANSFORMERS_SRC, reason="set QWEN38_TRANSFORMERS_SRC to pinned Transformers source")
def test_exact_final_mixer_and_rope_match_pinned_transformers(checkpoint_and_placement):
    checkpoint, placement = checkpoint_and_placement
    sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual, Qwen4ExpTextRotaryEmbedding

    config = Qwen4ExpTextConfig.from_pretrained(CHECKPOINT)
    weights = Qwen38FinalMixerWeights.from_checkpoint(checkpoint, placement)
    mixer = Qwen38FinalMixer(weights)
    oracle = Qwen4ExpTextGatedResidual(config, use_combine=False).to(torch.bfloat16).eval()
    oracle.load_state_dict(weights.transformers_state_dict(), strict=True)
    generator = torch.Generator().manual_seed(53)
    residual = (torch.randn((1, 3, 10240), generator=generator) * 0.02).to(torch.bfloat16)

    with torch.no_grad():
        expected = oracle(residual)
        got = mixer(residual)
        reference_rope = Qwen4ExpTextRotaryEmbedding(config)(
            torch.zeros((1, 7, 2560), dtype=torch.bfloat16), torch.arange(7).view(1, 7)
        )

    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    actual_rope = text_rope(checkpoint.config, batch=1, length=7, dtype=torch.bfloat16)
    torch.testing.assert_close(actual_rope[0], reference_rope[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_rope[1], reference_rope[1], rtol=0.0, atol=0.0)


def test_model_io_uses_sparse_embedding_lookup_and_explicit_vocab_shards(checkpoint_and_placement):
    checkpoint, placement = checkpoint_and_placement
    model_io = Qwen38ModelIO(checkpoint, placement)
    token_ids = torch.tensor([[17, checkpoint.config.eos_token_id]], dtype=torch.long)
    embedded = model_io.embed(token_ids)

    assert embedded.shape == (1, 2, 2560)
    assert embedded.dtype == torch.bfloat16
    assert model_io.vocab_ranges == ((0, 62080), (62080, 124160), (124160, 186240), (186240, 248320))
    assert model_io.embedding_metadata.shape == (248320, 2560)
    assert model_io.lm_head_metadata.shape == (248320, 2560)
    assert model_io.embedding_metadata.shard != model_io.lm_head_metadata.shard

    selected = torch.cat(
        (
            checkpoint.tensor_slice("model.language_model.embed_tokens.weight", (slice(17, 18), slice(None))),
            checkpoint.tensor_slice(
                "model.language_model.embed_tokens.weight",
                (slice(checkpoint.config.eos_token_id, checkpoint.config.eos_token_id + 1), slice(None)),
            ),
        )
    )
    torch.testing.assert_close(embedded.view(2, 2560), selected, rtol=0.0, atol=0.0)


def test_final_mixer_tp4_decomposition_matches_full_path(checkpoint_and_placement):
    checkpoint, placement = checkpoint_and_placement
    mixer = Qwen38FinalMixer(Qwen38FinalMixerWeights.from_checkpoint(checkpoint, placement))
    generator = torch.Generator().manual_seed(59)
    residual = (torch.randn((1, 2, 10240), generator=generator) * 0.02).to(torch.bfloat16)

    expected = mixer(residual)
    residual_shards = mixer.shard_residual(residual)
    output_shards = mixer.forward_tp4(residual_shards)

    assert all(shard.shape == (1, 2, 640) for shard in output_shards)
    torch.testing.assert_close(torch.cat(output_shards, dim=-1), expected, rtol=0.01, atol=0.01)


def test_resident_rope_table_rows_match_the_cpu_oracle_text_rope_bitwise(checkpoint_and_placement):
    # The device RoPE tables are built one _host_rope row per position; the CPU
    # oracle computes every position at once.  Both must agree bitwise so a
    # table lookup at P is the oracle's row P.
    checkpoint, _ = checkpoint_and_placement
    length = 4096
    oracle_cos, oracle_sin = text_rope(checkpoint.config, batch=1, length=length, dtype=torch.bfloat16)
    inverse_frequency = _inverse_frequency(checkpoint.config)
    rows = [_host_rope(position, inverse_frequency) for position in range(length)]
    table_cos = torch.cat([cos for cos, _ in rows], dim=2).reshape(length, 64)
    table_sin = torch.cat([sin for _, sin in rows], dim=2).reshape(length, 64)
    torch.testing.assert_close(table_cos, oracle_cos[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(table_sin, oracle_sin[0], rtol=0.0, atol=0.0)
