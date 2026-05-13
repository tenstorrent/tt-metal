# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import semantic_codebook_quantize_indices_reference
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    VoxtralTTAudioTokenizer,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


def _bts_to_tt_b1ts(device, x_bts: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        x_bts.unsqueeze(1).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
@pytest.mark.parametrize("time_len", [4, 16])
def test_semantic_codebook_quantize_matches_reference(device, reset_seeds, time_len):
    """TT semantic VQ argmin matches CPU reference (exact indices)."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    if "quantizer.semantic_codebook.embedding_sum" not in sd or "quantizer.semantic_codebook.cluster_usage" not in sd:
        pytest.skip("Semantic quantizer buffers not in audio_tokenizer state_dict.")

    try:
        tok = VoxtralTTAudioTokenizer(device, state_dict=sd, tokenizer_cfg=cfg)
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")
    if tok.semantic_codebook_quantizer is None:
        pytest.skip("Semantic quantizer not constructed.")

    sem = int(cfg.semantic_dim)
    b = 1
    torch.manual_seed(1)
    x = torch.randn(b, time_len, sem, dtype=torch.bfloat16)
    ref_idx = semantic_codebook_quantize_indices_reference(x, sd)

    x_tt = _bts_to_tt_b1ts(device, x)
    tt_idx = tok.semantic_codebook_quantize_forward(x_tt)
    ttnn.deallocate(x_tt)

    tt_idx_cpu = ttnn.to_torch(tt_idx).to(torch.int64)
    ttnn.deallocate(tt_idx)

    assert tt_idx_cpu.shape == ref_idx.shape
    assert torch.equal(ref_idx.cpu(), tt_idx_cpu), "semantic quantizer indices differ from reference"
