# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""First-layer real inputs and decay statistics, on a tiny synthetic checkpoint. CPU only."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from models.demos.deepseek_v3_d_p.tests.kda.real_input import (
    FirstLayerInput,
    chunk_decay_statistics,
    trace_token_ids,
    trace_token_source,
)


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
    # Row 1 is (1, 5): RMS sqrt(13), so it normalizes to (1, 5) / sqrt(13), then scales by (7, 1).
    embedding = torch.tensor([[1.0, 1.0], [1.0, 5.0], [0.0, 2.0]], dtype=torch.bfloat16)
    norm_weight = torch.tensor([7.0, 1.0], dtype=torch.bfloat16)
    build = FirstLayerInput(write_checkpoint(tmp_path, embedding, norm_weight))

    hidden = build([1, 0, 1])

    assert hidden.shape == (1, 3, 2) and hidden.dtype == torch.bfloat16
    # Normalized and scaled in FP32, rounded to BF16 once: 7/sqrt(13) = 1.94145 -> 1.9453125, 5/sqrt(13) =
    # 1.38675 -> 1.390625. Rounding 1/sqrt(13) to BF16 first (HF KimiRMSNorm) would give 7 · 0.27734375 =
    # 1.94140625, a tie that rounds to 1.9375.
    assert hidden[0, 0].tolist() == [1.9453125, 1.390625]
    torch.testing.assert_close(hidden[0, 1], norm_weight * torch.ones(2, dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(hidden[0, 2], hidden[0, 0], rtol=0, atol=0)


def test_first_layer_input_matches_sampled_vllm_trace_rows() -> None:
    """Rebuilds 1,024 rows of the R1 trace's ``kda_input_layer_0`` from their token ids (see the trace README).

    vLLM's CUDA RMSNorm reduces in a different order and uses an approximate rsqrt, so a few elements per
    million land one BF16 ulp away (38 of 7,340,032 when this was written); everything else is bit-identical.
    """
    trace = Path(os.environ.get("KDA_R1_TRACE", Path.home() / ".cache/traces/flashKDA/k3_4_layers_moe_kda_io"))
    sample_file = trace / "kda_input_layer_0.sample1024_seed20261013.safetensors"
    checkpoint = os.environ.get("KIMI_K3_CKPT")
    if not sample_file.is_file() or not checkpoint:
        pytest.skip("needs the R1 layer-0 input sample (KDA_R1_TRACE) and KIMI_K3_CKPT")
    sample = load_file(sample_file)
    expected = sample["kda_input_layer_0"]

    actual = FirstLayerInput(Path(checkpoint))(sample["token_ids"])[0]

    ulps = (actual.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
    assert int(ulps.max()) <= 1, f"rebuilt rows differ by up to {int(ulps.max())} BF16 ulp"
    assert int((ulps > 0).sum()) <= 100, f"{int((ulps > 0).sum())} of {ulps.numel()} elements differ by one ulp"


def test_trace_token_ids_are_the_metadata_prompt_ids(tmp_path: Path) -> None:
    trace = tmp_path / "k3_trace"
    trace.mkdir()
    (trace / "metadata.json").write_text(json.dumps({"token_ids": [5, 0, 163577], "completion_token_ids": [9]}))

    assert trace_token_ids(trace) == [5, 0, 163577]
    assert trace_token_source(trace) == "vllm trace k3_trace"


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


def test_chunk_decay_statistics_without_a_full_chunk_reports_nothing() -> None:
    assert chunk_decay_statistics(torch.full((31, 1, 2), -5.0)) == {"chunk_channels": 0}
