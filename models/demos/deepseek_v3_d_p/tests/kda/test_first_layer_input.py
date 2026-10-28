# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The first-layer input on a tiny synthetic checkpoint. CPU only."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FirstLayerInput


def write_checkpoint(root: Path, embedding: torch.Tensor, norm_weight: torch.Tensor) -> Path:
    root.mkdir(parents=True, exist_ok=True)
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


def test_first_layer_input_is_the_rms_normed_embedding_rounded_once(tmp_path: Path) -> None:
    # Row 1 is (1, 5): RMS sqrt(13), so it normalizes to (1, 5) / sqrt(13), then scales by (7, 1).
    embedding = torch.tensor([[1.0, 1.0], [1.0, 5.0], [0.0, 2.0]], dtype=torch.bfloat16)
    norm_weight = torch.tensor([7.0, 1.0], dtype=torch.bfloat16)
    build = FirstLayerInput(write_checkpoint(tmp_path, embedding, norm_weight))

    hidden = build([1, 0, 1])

    assert hidden.shape == (1, 3, 2) and hidden.dtype == torch.bfloat16
    # Normalized and scaled in FP32, rounded to BF16 once: 7/sqrt(13) = 1.94145 -> 1.9453125 and
    # 5/sqrt(13) = 1.38675 -> 1.390625. Rounding 1/sqrt(13) to BF16 first (HF KimiRMSNorm) would give
    # 7 * 0.27734375 = 1.94140625, a tie that rounds to 1.9375.
    assert hidden[0, 0].tolist() == [1.9453125, 1.390625]
    assert hidden[0, 1].tolist() == [7.0, 1.0]  # (1, 1) has RMS 1
    assert torch.equal(hidden[0, 2], hidden[0, 0])


def test_missing_shard_names_the_tensor(tmp_path: Path, expect_error) -> None:
    root = write_checkpoint(tmp_path, torch.ones(2, 2, dtype=torch.bfloat16), torch.ones(2, dtype=torch.bfloat16))
    (root / "shard.safetensors").unlink()
    with expect_error(FileNotFoundError, "embed_tokens.weight lives in shard.safetensors"):
        FirstLayerInput(root)


def test_weights_identity_covers_embedding_and_norm(tmp_path: Path) -> None:
    embedding = torch.ones(3, 2, dtype=torch.bfloat16)
    first = FirstLayerInput(write_checkpoint(tmp_path / "a", embedding, torch.ones(2, dtype=torch.bfloat16)))
    other_norm = FirstLayerInput(
        write_checkpoint(tmp_path / "b", embedding, torch.full((2,), 2.0, dtype=torch.bfloat16))
    )
    other_embedding = FirstLayerInput(
        write_checkpoint(tmp_path / "c", embedding * 2, torch.ones(2, dtype=torch.bfloat16))
    )
    assert len({first.weights_sha256(), other_norm.weights_sha256(), other_embedding.weights_sha256()}) == 3
