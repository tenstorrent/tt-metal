# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-only paths: multi-codebook MM sum + ``latent_from_codes_tt`` vs CPU references."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import (
    audio_tokenizer_encode_tokens_reference,
    audio_tokenizer_latent_from_codes,
)
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    VoxtralTTAudioTokenizer,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


def _codes_to_tt_b37t(device, codes: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        codes.to(torch.uint32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
@pytest.mark.parametrize("time_len", [4, 16])
def test_mm_audio_encode_tokens_summed_pcc(device, reset_seeds, time_len):
    """``mm_audio_encode_tokens_summed_forward`` vs ``audio_tokenizer_encode_tokens_reference``."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    emb_key = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
    if emb_key not in full:
        pytest.skip(f"Missing {emb_key} in checkpoint.")

    cfg = load_voxtral_config(model_name)
    tok_cfg = cfg.audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tok = VoxtralTTAudioTokenizer(device, state_dict=sd, tokenizer_cfg=tok_cfg, full_checkpoint=full)
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")
    if tok.mm_audio_codebook_embedding is None or tok._mm_offsets_tt is None:
        pytest.skip("MM embedding / offsets not initialized.")

    torch.manual_seed(0)
    b = 1
    semantic = torch.randint(0, tok_cfg.semantic_codebook_size, (b, 1, time_len))
    acoustic = torch.randint(0, tok_cfg.acoustic_codebook_size, (b, tok_cfg.acoustic_dim, time_len))
    codes = torch.cat([semantic, acoustic], dim=1).long()

    ref = audio_tokenizer_encode_tokens_reference(codes, full[emb_key], cfg.audio_model_args)

    codes_tt = _codes_to_tt_b37t(device, codes)
    out_tt = tok.mm_audio_encode_tokens_summed_forward(codes_tt)
    ttnn.deallocate(codes_tt)

    tt_td = ttnn.to_torch(out_tt).float()
    assert ref.shape == tt_td.shape, f"shape ref={tuple(ref.shape)} tt={tuple(tt_td.shape)}"
    passing, msg = comp_pcc(ref.float(), tt_td, pcc=0.99)
    assert passing, f"MM encode tokens summed PCC failed: {msg}"


@torch.no_grad()
@pytest.mark.parametrize("time_len", [8, 16])
def test_latent_from_codes_tt_pcc(device, reset_seeds, time_len):
    """``latent_from_codes_tt`` vs ``audio_tokenizer_latent_from_codes`` (same layout as ``decode_full_forward`` input)."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    tok_cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tok = VoxtralTTAudioTokenizer(device, state_dict=sd, tokenizer_cfg=tok_cfg, full_checkpoint=full)
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")
    if tok.semantic_codebook_quantizer is None:
        pytest.skip("Semantic quantizer / centroids not in checkpoint.")

    torch.manual_seed(1)
    b = 1
    semantic = torch.randint(0, tok_cfg.semantic_codebook_size, (b, 1, time_len))
    acoustic = torch.randint(0, tok_cfg.acoustic_codebook_size, (b, tok_cfg.acoustic_dim, time_len))
    codes = torch.cat([semantic, acoustic], dim=1).long()

    ref_ncl = audio_tokenizer_latent_from_codes(codes, sd, n_acoustic_levels=tok_cfg.acoustic_codebook_size).to(
        torch.bfloat16
    )
    ref_b1tc = ref_ncl.permute(0, 2, 1).unsqueeze(1).contiguous()

    codes_tt = _codes_to_tt_b37t(device, codes)
    latent_tt = tok.latent_from_codes_tt(codes_tt)
    ttnn.deallocate(codes_tt)

    tt_b1tc = ttnn.to_torch(latent_tt).float()
    ttnn.deallocate(latent_tt)
    assert ref_b1tc.shape == tt_b1tc.shape, f"shape ref={tuple(ref_b1tc.shape)} tt={tuple(tt_b1tc.shape)}"
    passing, msg = comp_pcc(ref_b1tc.float(), tt_b1tc, pcc=0.99)
    assert passing, f"latent_from_codes_tt PCC failed: {msg}"
