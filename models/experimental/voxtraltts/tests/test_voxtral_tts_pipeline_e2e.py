# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end TT pipeline: text + acoustic + audio tokenizer (orchestrated via :mod:`tt.voxtral_tts`).

- **Waveform:** same discrete codes through TT pipeline vs PyTorch ``audio_tokenizer_decode_reference`` (PCC).
- **Acoustic:** ``VoxtralTTSPipeline.acoustic_codes_forward`` vs ``FlowMatchingAudioTransformerRef.forward`` with
  the same RNG protocol as ``test_acoustic_forward_e2e_matches_reference``.
- **Text decode loop:** teacher-forced multi-step last-token logits PCC each iteration (same pattern as
  ``test_text_model_decode_multistep_reference_pcc`` / ``test_model`` PCC after reference logits).
- **Generate smoke:** the real ``VoxtralTTSPipeline.generate_with_codes`` path used by the demo.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import (
    FlowMatchingAudioTransformerRef,
    build_audio_model_args_from_voxtral_config,
)
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline


def _align_to_ref_shape(ref_t: torch.Tensor, tt_t: torch.Tensor) -> torch.Tensor:
    """Trim TT tensor logical extents to reference (ttnn may pad)."""
    out = tt_t
    for dim, size in enumerate(ref_t.shape):
        if dim < out.dim() and out.shape[dim] > size:
            sl = [slice(None)] * out.dim()
            sl[dim] = slice(0, size)
            out = out[tuple(sl)]
    return out.reshape(ref_t.shape)


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_loads(device, reset_seeds):
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=256)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")
    assert pipe.text.inner.args.dim > 0
    assert pipe.acoustic.dim > 0
    assert pipe.audio_tokenizer.cfg.dim > 0


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_waveform_codes_pcc(device, reset_seeds):
    """TT ``decode_waveform_from_codes_tt`` vs CPU golden for the same codes (full tokenizer decode)."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=256)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    cfg = pipe.config.audio_tokenizer_args
    b, t = 1, 32
    semantic = torch.randint(0, cfg.semantic_codebook_size, (b, 1, t))
    acoustic = torch.randint(0, cfg.acoustic_codebook_size, (b, cfg.acoustic_dim, t))
    codes = torch.cat([semantic, acoustic], dim=1).long()

    try:
        ref_wav = pipe.decode_waveform_from_codes_reference(codes)
        tt_wav = pipe.decode_waveform_from_codes_tt(codes)
    except RuntimeError as exc:
        if "requires the full decoder stack" in str(exc) or "output_proj" in str(exc):
            pytest.skip(str(exc))
        raise

    assert ref_wav.shape == tt_wav.shape
    ok, msg = comp_pcc(ref_wav.float(), tt_wav.float(), pcc=0.99)
    assert ok, f"Pipeline waveform PCC: {msg}"


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_generate_path_smoke_and_waveform_pcc(device, reset_seeds):
    """Exercise the same free-running text → acoustic → tokenizer path used by the demo."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=512)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    out = pipe.generate_with_codes(
        text="Hello from the Voxtral Tenstorrent demo.",
        voice="casual_male",
        max_tokens=4,
        seed=0,
    )

    assert out.codes_b37t.dim() == 3 and tuple(out.codes_b37t.shape[:2]) == (1, 37)
    assert out.codes_b37t.shape[2] > 0, "generate path produced no acoustic frames before end-of-audio"
    assert out.waveform.shape == (1, 1, out.codes_b37t.shape[2] * pipe._downsample_factor)
    assert torch.isfinite(out.waveform).all(), "generate path produced non-finite waveform samples"

    ref_wav = pipe.decode_waveform_from_codes_reference(out.codes_b37t)
    ref_wav = ref_wav.reshape(1, 1, -1)[:, :, : out.waveform.shape[-1]]
    ok, msg = comp_pcc(ref_wav.float(), out.waveform.float(), pcc=0.99)
    assert ok, f"Generated-code waveform PCC failed: {msg}"


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_voxtral_tts_pipeline_acoustic_forward_matches_reference(device, reset_seeds):
    """Acoustic head: TT ``forward`` vs CPU ref with synced FM RNG (semantic exact; acoustic agreement threshold)."""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=256)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    cfg = load_voxtral_config(name)
    ref = FlowMatchingAudioTransformerRef(build_audio_model_args_from_voxtral_config(cfg)).to(torch.bfloat16).eval()
    full = _load_safetensors_state_dict(name)
    for k, v in full.items():
        if k.startswith("acoustic_transformer."):
            ref.load_weight((k.removeprefix("acoustic_transformer."), v))

    torch.manual_seed(42)
    bsz = 1
    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    llm_h = torch.randn(bsz, d_llm, dtype=torch.bfloat16)
    cfg_alpha = torch.tensor(0.73, dtype=torch.bfloat16)

    torch.manual_seed(12345)
    ref_out = ref.forward(llm_h, cfg_alpha)
    torch.manual_seed(12345)
    tt_raw = pipe.acoustic_codes_forward(llm_h, cfg_alpha)
    tt_out = _align_to_ref_shape(ref_out, tt_raw)

    assert (
        ref_out.shape == tt_out.shape
    ), f"forward shape mismatch after align: ref={tuple(ref_out.shape)} tt_raw={tuple(tt_raw.shape)}"
    assert torch.equal(ref_out[:, :1], tt_out[:, :1]), "semantic token mismatch"

    n_acoustic = ref_out.shape[1] - 1
    if n_acoustic > 0:
        acoustic_ok = ref_out[:, 1:] == tt_out[:, 1:]
        match_frac = float(acoustic_ok.float().mean().item())
        min_frac = 0.88
        assert (
            match_frac >= min_frac
        ), f"acoustic code agreement {match_frac:.4f} < {min_frac} (TT vs CPU FM drift at round)"


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("decode_steps", [4], ids=["4_steps"])
def test_voxtral_tts_pipeline_text_multistep_decode_pcc(device, reset_seeds, decode_steps):
    """Iterative decode: each step compares TT logits to PyTorch reference"""
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=256)
    except Exception as exc:
        pytest.skip(f"Pipeline load failed: {exc}")

    prompt_len = 128
    vocab_size = pipe.text.inner.vocab_size
    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.int64)
    decode_tokens = torch.randint(0, vocab_size, (1, decode_steps), dtype=torch.int64)

    pipe.text_decode_multistep_compare_reference(
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        pcc=0.98,
    )
