# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""First-layer real inputs and decay statistics, on a tiny synthetic checkpoint. CPU only."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.tests.kda.real_input import FirstLayerInput, chunk_decay_statistics


def write_checkpoint(root: Path, embedding: torch.Tensor, norm_weight: torch.Tensor) -> Path:
    tensors = {
        "language_model.model.embed_tokens.weight": embedding,
        "language_model.model.layers.0.input_layernorm.weight": norm_weight,
    }
    save_file(tensors, root / "shard.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "shard.safetensors")})
    )
    (root / "config.json").write_text(json.dumps({"text_config": {"rms_norm_eps": 0.0}}))
    return root


def test_first_layer_input_is_the_rms_normed_embedding_in_bf16(tmp_path: Path) -> None:
    # Row 1 is (3, 4): RMS sqrt(12.5), so it normalizes to (3, 4) / sqrt(12.5), then scales by (2, 1).
    embedding = torch.tensor([[1.0, 1.0], [3.0, 4.0], [0.0, 2.0]], dtype=torch.bfloat16)
    norm_weight = torch.tensor([2.0, 1.0], dtype=torch.bfloat16)
    build = FirstLayerInput(write_checkpoint(tmp_path, embedding, norm_weight))

    hidden = build([1, 0, 1])

    assert hidden.shape == (1, 3, 2) and hidden.dtype == torch.bfloat16
    rms = 12.5**0.5
    normalized = torch.tensor([3.0 / rms, 4.0 / rms]).bfloat16()  # rounded to BF16 before the weight
    torch.testing.assert_close(hidden[0, 0], norm_weight * normalized, rtol=0, atol=0)
    torch.testing.assert_close(hidden[0, 1], norm_weight * torch.ones(2, dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(hidden[0, 2], hidden[0, 0], rtol=0, atol=0)


def test_chunk_decay_statistics_counts_ceiling_and_weak_chunks() -> None:
    # Two 32-token chunks, one head, two channels; a trailing partial chunk of 5 tokens is ignored.
    gate = torch.zeros(69, 1, 2)
    gate[:32, 0, 0] = -5.0  # chunk 0, channel 0: |G_last| = 160, at the ceiling
    gate[:32, 0, 1] = -1e-5  # chunk 0, channel 1: 3.2e-4, weak (below 2^-9)
    gate[32:64, 0, :] = -1.0  # chunk 1: 32 for both channels
    gate[64:, 0, :] = -5.0  # partial chunk

    stats = chunk_decay_statistics(gate)

    assert stats["chunk_channels"] == 4
    assert stats["max"] == 160.0
    assert stats["at_ceiling"] == 1
    assert stats["above_144"] == 1
    assert stats["weak_below_2^-9"] == 1
    assert stats["p50"] == 32.0
