# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Teacher-forced end-to-end Word Error Rate (WER): TTNN VibeVoice vs the fp32 PyTorch reference.

Isolates TT's per-step audio-rendering fidelity, with no free-running feedback drift. The fp32
reference free-runs the 4-speaker ``resources/text/<TF_DEMO_ID>.txt`` script with voice cloning
(Speaker 1-4 → en-Alice_woman / en-Carter_man / en-Frank_man / en-Maya_woman) and we capture its
per-frame re-encoded embedding (``acoustic_connector + semantic_connector`` = ``diffusion_embeds``).
TT then replays the reference token stream (``forced_token_ids``) with ``_post_diffusion_embeds``
hooked so TT's LM is fed the REFERENCE embedding each step (no drift), while TT still renders its
own audio.

Whisper (``WHISPER_MODEL``) transcribes both waveforms and WER (TT vs reference) is reported by
cumulative transcript-word prefix (32, 64, 128, …). Report-only: asserts audio is finite and
transcripts non-empty. ``TF_MAX_NEW_TOKENS`` (env ``VV_WER_MAX_NEW_TOKENS``, default 512) caps AR steps.
"""

import contextlib
import json
import os
import re
import sys
from pathlib import Path

import pytest
import torch

from models.experimental.vibevoice.common.config import (
    MODEL_PATH,
    TEXT_EXAMPLES_DIR,
)
from models.experimental.vibevoice.common.resource_utils import build_voice_samples, load_script
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
SR = 24000  # VibeVoice sample rate
WHISPER_SR = 16000  # Whisper feature-extractor sample rate
WHISPER_MODEL = "openai/whisper-medium"
# Cumulative transcript-word prefixes at which WER is reported.
WER_PREFIX_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
_OUT_DIR = _VIBEVOICE_ROOT / "output" / "e2e_wer"

# 4-speaker climate script + voice cloning.
TF_DEMO_ID = "4p_climate_45min"
# AR-step cap (override with VV_WER_MAX_NEW_TOKENS; the fp32 CPU reference free-runs the same
# input, so keep it modest — e.g. 32 for a quick smoke test).
TF_MAX_NEW_TOKENS = int(os.environ.get("VV_WER_MAX_NEW_TOKENS", "512"))


@contextlib.contextmanager
def _quiet_load_report():
    """Silence transformers 5.x's cosmetic model LOAD REPORT during a from_pretrained call.

    The vendored reference ties lm_head to embed_tokens in tie_weights() (post_init), so
    lm_head.weight is intentionally absent from the checkpoint. transformers 5.x cannot see the tie
    at load time (list-form _tied_weights_keys + config.tie_word_embeddings unset), so it logs a
    spurious "lm_head.weight | MISSING" report before the tie is applied. The weight is correct
    after load; only the report is wrong. Raise transformers' verbosity to ERROR for the load only.
    """
    from transformers.utils import logging as hf_logging

    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(prev)


def _normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation (keep intra-word apostrophes), split to words."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def _wer(ref_words: list[str], hyp_words: list[str]) -> float:
    """Word Error Rate = word-level Levenshtein(ref, hyp) / len(ref)."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return float(m > 0)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / n


def _load_whisper():
    from transformers import pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device=device)


def _transcribe(asr, audio_24k: torch.Tensor) -> str:
    import librosa

    wav = audio_24k.detach().to(torch.float32).cpu().clamp(-1.0, 1.0).numpy()
    wav16 = librosa.resample(wav, orig_sr=SR, target_sr=WHISPER_SR)
    result = asr(
        {"raw": wav16, "sampling_rate": WHISPER_SR},
        generate_kwargs={"language": "en", "task": "transcribe"},
        chunk_length_s=30,
        return_timestamps=True,
    )
    return result["text"].strip()


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_wer_teacher_forced(mesh_device):
    """Teacher-forced parity: bf16 reference re-encoded embedding fed into TT's LM each frame.

    Both backends bf16, 45-min script, cap 512 frames. The reference
    free-runs and we capture its per-frame re-encoded embedding (``acoustic_connector +
    semantic_connector`` = ``diffusion_embeds``). TT then replays the reference token stream
    (``forced_token_ids``) with its ``_post_diffusion_embeds`` hooked so TT's LM is fed the
    REFERENCE embedding each step (no feedback drift), while TT still renders its own audio. WER is
    TT audio vs reference audio (Whisper) — isolates TT's per-step rendering fidelity.
    """
    try:
        asr = _load_whisper()
    except Exception as exc:
        pytest.skip(f"Whisper ASR unavailable ({WHISPER_MODEL}): {exc}")

    from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from processor.vibevoice_processor import VibeVoiceProcessor

    from models.experimental.vibevoice.tt.ttnn_vibevoice_generator import _host_2d_to_embeds

    text_path = TEXT_EXAMPLES_DIR / f"{TF_DEMO_ID}.txt"
    assert text_path.is_file(), f"Missing script: {text_path}"
    script = load_script(text_path)
    voice_samples, _ = build_voice_samples(script, TF_DEMO_ID)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    inputs = processor(
        text=[script], voice_samples=[voice_samples], padding=True, return_tensors="pt", return_attention_mask=True
    )
    prefill_len = inputs["input_ids"].shape[1]

    # ── bf16 reference ──
    # fp32 reference on CPU: bf16 matmul has no CPU hardware acceleration (~5-20x slower); fp32 is far
    # faster and the injected embeds are cast to bf16 anyway, so tt_vs_ref (word-level WER) is unaffected.
    with _quiet_load_report():
        ref = VibeVoiceForConditionalGenerationInference.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
        )
    ref.eval()
    ref.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)
    ref.model.acoustic_tokenizer.std_dist_type = "none"

    # Pass 1: reference free-run — capture per-frame diffusion_embeds (semantic_connector fires only in
    # the decode loop, so its hook is exactly one capture per diffusion frame; acoustic is summed in).
    ref_embeds: list[torch.Tensor] = []
    _acc: dict = {}
    _sc_h = ref.model.semantic_connector.register_forward_hook(
        lambda m, i, o: ref_embeds.append((_acc["a"] + o).detach().reshape(-1).to(torch.float32).cpu())
    )
    _ac_h = ref.model.acoustic_connector.register_forward_hook(lambda m, i, o: _acc.update(a=o))
    torch.manual_seed(0)
    ref_out = ref.generate(
        **inputs,
        max_new_tokens=TF_MAX_NEW_TOKENS,
        cfg_scale=CFG_SCALE,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        is_prefill=True,
    )
    _ac_h.remove()
    _sc_h.remove()
    ref_speech = ref_out.speech_outputs[0].to(torch.float32).reshape(-1)
    ref_tokens = ref_out.sequences[0, prefill_len:].reshape(-1)
    assert len(ref_embeds) > 0 and ref_tokens.numel() > 0, "reference produced no diffusion frames"
    print(f"[tf] reference: {len(ref_embeds)} diffusion frames, {ref_speech.numel() / SR:.1f}s audio", flush=True)

    with torch.no_grad():
        _, prefill_speech_embeds = ref._process_speech_inputs(
            inputs["speech_tensors"].to(ref.dtype), inputs["speech_masks"]
        )
    prefill_speech_embeds = prefill_speech_embeds.to(torch.float32)

    # ── bf16 TT, teacher-forced ──
    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device, MODEL_PATH, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS
    )
    tt_model.set_speech_scale_bias(ref.model.speech_scaling_factor.item(), ref.model.speech_bias_factor.item())
    tt_gen = tt_model._make_generator(
        processor.tokenizer, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS, max_new_tokens=None
    )

    # Pass 2: hook _post_diffusion_embeds so TT's LM is fed the reference embedding (TT keeps its own audio).
    inj = [0]
    _orig_post = tt_gen._post_diffusion_embeds

    def _post(lat):
        _, tt_audio = _orig_post(lat)
        e = ref_embeds[inj[0]] if inj[0] < len(ref_embeds) else ref_embeds[-1]
        inj[0] += 1
        return _host_2d_to_embeds(e.reshape(1, -1), tt_gen.device), tt_audio

    tt_gen._post_diffusion_embeds = _post

    torch.manual_seed(0)
    tt_out = tt_gen.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_input_mask=inputs["speech_input_mask"],
        prefill_speech_embeds=prefill_speech_embeds,
        max_new_tokens=None,
        forced_token_ids=ref_tokens,
    )
    tt_speech = tt_out.speech_outputs[0].to(torch.float32).reshape(-1)
    print(f"[tf] TT teacher-forced: injected {inj[0]} ref embeds, {tt_speech.numel() / SR:.1f}s audio", flush=True)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(_OUT_DIR / f"{TF_DEMO_ID}_tf_ref.wav"), ref_speech.clamp(-1.0, 1.0).numpy(), SR)
    sf.write(str(_OUT_DIR / f"{TF_DEMO_ID}_tf_tt.wav"), tt_speech.clamp(-1.0, 1.0).numpy(), SR)
    assert torch.isfinite(ref_speech).all() and torch.isfinite(tt_speech).all(), "non-finite audio"

    ref_words = _normalize(_transcribe(asr, ref_speech))
    tt_words = _normalize(_transcribe(asr, tt_speech))
    assert ref_words and tt_words, "empty transcript"

    lengths = [n for n in WER_PREFIX_LENGTHS if n < max(len(ref_words), len(tt_words))]
    lengths.append(max(len(ref_words), len(tt_words)))
    table = [{"n": n, "tt_vs_ref": round(_wer(ref_words[:n], tt_words[:n]), 4)} for n in lengths]
    overall = round(_wer(ref_words, tt_words), 4)

    metrics = {
        "mode": "teacher_forced_ref_embedding",
        "demo": TF_DEMO_ID,
        "max_new_tokens": TF_MAX_NEW_TOKENS,
        "ref_frames": len(ref_embeds),
        "ref_audio_sec": round(ref_speech.numel() / SR, 2),
        "tt_audio_sec": round(tt_speech.numel() / SR, 2),
        "ref_word_count": len(ref_words),
        "tt_word_count": len(tt_words),
        "overall_tt_vs_ref": overall,
        "tt_vs_ref_by_length": table,
        "ref_transcript": " ".join(ref_words),
        "tt_transcript": " ".join(tt_words),
    }
    (_OUT_DIR / f"{TF_DEMO_ID}_tf_wer.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [f"{'N':>7} | {'tt_vs_ref':>9}", "-" * 20]
    for r in table:
        lines.append(f"{r['n']:>7} | {r['tt_vs_ref']:>9.4f}")
    print(
        f"\n[tf] TEACHER-FORCED WER (bf16 ref embedding -> TT LM), demo={TF_DEMO_ID} cap={TF_MAX_NEW_TOKENS}\n"
        f"[tf] ref {ref_speech.numel() / SR:.1f}s ({len(ref_words)}w) | tt {tt_speech.numel() / SR:.1f}s ({len(tt_words)}w)\n"
        + "\n".join(lines)
        + f"\n[tf] overall tt_vs_ref={overall:.4f}  artifacts -> {_OUT_DIR}"
    )
