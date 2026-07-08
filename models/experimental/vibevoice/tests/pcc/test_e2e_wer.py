# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end Word Error Rate (WER): TTNN VibeVoice vs the golden reference audio, by Whisper.

Runs ``demo_ttnn.py --demo 4p_climate_45min`` inputs — the 4-speaker
``resources/text/4p_climate_45min.txt`` script with voice cloning (Speaker 1-4 →
en-Alice_woman / en-Carter_man / en-Frank_man / en-Maya_woman) — generating the FULL audio on
TTNN. The **reference** side is NOT regenerated (the fp32 CPU reference would take ~8-12 h for a
45-min script, and there is no GPU); instead we use the shipped golden clip
``resources/golden/4p_climate_45min.wav`` (~43 min, the microsoft.github.io/VibeVoice reference).

Whisper (``WHISPER_MODEL``) transcribes both waveforms, and WER is reported **by sequence
length** — at cumulative transcript-word prefixes (32, 64, 128, 256, …): for each length N we
compare the first N words of a hypothesis against the first N words of the target. Comparing
aligned prefixes keeps every point a fair local measure (a token cap only shortens the curve, it
does not inflate WER with trailing deletions), so the curve shows how WER drifts as generation
proceeds. Target text = the ground-truth transcript ``…_gt_timestamp.json`` (the actual golden
words). Three series per length:
  * ``golden_vs_gt`` — Whisper's own error on the golden clip (a floor / sanity baseline).
  * ``tt_vs_gt``     — TT synthesis intelligibility vs the true words.
  * ``tt_vs_golden`` — TT transcript vs the golden transcript (direct divergence).

Report-only: asserts both waveforms are finite / non-silent and both transcripts non-empty.
``MAX_NEW_TOKENS = None`` free-runs the full script (heavy: ~2 h TT eager + long Whisper passes on
43-min clips); set it to an int to bound the run.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest
import torch

from models.experimental.vibevoice.common.config import (
    GOLDEN_DIR,
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
SR = 24000  # VibeVoice / golden sample rate
WHISPER_SR = 16000  # Whisper feature-extractor sample rate
WHISPER_MODEL = "openai/whisper-medium"
# Match `demo_ttnn.py --demo 4p_climate_45min`: 4-speaker climate script + voice cloning.
DEMO_ID = "4p_climate_45min"
_TEXT_PATH = TEXT_EXAMPLES_DIR / f"{DEMO_ID}.txt"
_GOLDEN_WAV = GOLDEN_DIR / f"{DEMO_ID}.wav"
_GT_JSON = GOLDEN_DIR / "transcripts" / f"{DEMO_ID}_gt_timestamp.json"
# None → free-run the full script (heavy). Set an int to cap AR steps (~4096 ≈ ~9 min audio).
MAX_NEW_TOKENS = 4096
# Cumulative transcript-word prefixes at which WER is reported.
WER_PREFIX_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
_OUT_DIR = _VIBEVOICE_ROOT / "output" / "e2e_wer"


def _normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation (keep intra-word apostrophes), split to words."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def _target_words() -> list[str]:
    """Ground-truth spoken words: the golden transcript JSON (fallback: the input script)."""
    if _GT_JSON.is_file():
        segs = json.loads(_GT_JSON.read_text(encoding="utf-8"))
        text = " ".join(s.get("text", "") for s in segs)
    else:
        script = load_script(_TEXT_PATH)
        text = " ".join(re.sub(r"(?i)^\s*speaker\s+\d+\s*:\s*", "", ln.strip()) for ln in script.split("\n"))
    return _normalize(text)


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


def _cumulative_wer(target: list[str], golden: list[str], tt: list[str]) -> list[dict]:
    """WER over the first N words for N in WER_PREFIX_LENGTHS (+ a final full-length row).

    Each series compares aligned prefixes: golden↔target, tt↔target, tt↔golden.
    """
    rows = []
    lengths = [n for n in WER_PREFIX_LENGTHS if n < max(len(golden), len(tt))]
    lengths.append(max(len(golden), len(tt)))  # final full-length point
    for n in lengths:
        rows.append(
            {
                "n": n,
                "golden_vs_gt": round(_wer(target[:n], golden[:n]), 4),
                "tt_vs_gt": round(_wer(target[:n], tt[:n]), 4),
                "tt_vs_golden": round(_wer(golden[:n], tt[:n]), 4),
            }
        )
    return rows


def _build_inputs():
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    assert _TEXT_PATH.is_file(), f"Missing script: {_TEXT_PATH}"
    script = load_script(_TEXT_PATH)
    # Same 4 speaker voices demo_ttnn uses for this demo (Alice/Carter/Frank/Maya).
    voice_samples, _ = build_voice_samples(script, DEMO_ID)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    inputs = processor(
        text=[script],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return processor, inputs


def _load_whisper():
    from transformers import pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device=device)


def _load_wav(path: Path) -> torch.Tensor:
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert file_sr == SR, f"{path.name} sample rate {file_sr} != {SR}"
    return torch.from_numpy(data)


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


def _sanity_check(name: str, speech: torch.Tensor) -> None:
    assert speech.numel() > SR // 2, f"{name} audio too short: {speech.numel()} samples"
    assert torch.isfinite(speech).all(), f"{name} audio contains NaN/Inf"
    # Only flag a real blow-up/divergence, not mild >1.0 overshoot (clamped before transcription).
    peak = speech.abs().max().item()
    assert peak < 50.0, f"{name} audio blew up (peak {peak:.1f}), not mild clipping"
    assert (speech.abs() > 1e-3).float().mean().item() > 0.3, f"{name} audio is mostly silent"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_wer_tt_vs_golden(mesh_device):
    """Generate the full script on TT, transcribe TT + golden with Whisper, report WER by length."""
    # Load Whisper up front so an unavailable ASR skips fast (before slow generation).
    try:
        asr = _load_whisper()
    except Exception as exc:
        pytest.skip(f"Whisper ASR unavailable ({WHISPER_MODEL}): {exc}")

    assert _GOLDEN_WAV.is_file(), f"Missing golden reference audio: {_GOLDEN_WAV}"
    processor, inputs = _build_inputs()

    # TT free-run — full on-device TTNN pipeline (the demo_ttnn path). Save the wav immediately
    # so a later failure never costs the (~30 min) regeneration; VV_WER_REUSE_TT=1 reuses it.
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    tt_wav_path = _OUT_DIR / f"{DEMO_ID}_tt.wav"
    if os.environ.get("VV_WER_REUSE_TT") == "1" and tt_wav_path.is_file():
        tt_speech = _load_wav(tt_wav_path)
    else:
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh_device,
            MODEL_PATH,
            cfg_scale=CFG_SCALE,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
        )
        torch.manual_seed(0)
        tt_out = tt_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speech_tensors=inputs["speech_tensors"],
            speech_masks=inputs["speech_masks"],
            speech_input_mask=inputs["speech_input_mask"],
            tokenizer=processor.tokenizer,
            cfg_scale=CFG_SCALE,
            num_diffusion_steps=NUM_DIFFUSION_STEPS,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        assert tt_out.speech_outputs and tt_out.speech_outputs[0] is not None
        tt_speech = tt_out.speech_outputs[0].to(torch.float32).reshape(-1)
        sf.write(str(tt_wav_path), tt_speech.clamp(-1.0, 1.0).numpy(), SR)
    _sanity_check("tt", tt_speech)

    golden_full = _load_wav(_GOLDEN_WAV)
    _sanity_check("golden", golden_full)
    golden_full_sec = golden_full.numel() / SR
    # TT is capped (~9 min) but the golden is ~43 min. Transcribe only the overlapping golden
    # span (+30 s margin) so the WER curve compares aligned coverage and Whisper isn't run over
    # the ~34 min of golden the TT side never reached.
    golden_span = min(golden_full.numel(), tt_speech.numel() + 30 * SR)
    golden_speech = golden_full[:golden_span]

    # Transcribe both.
    golden_text = _transcribe(asr, golden_speech)
    tt_text = _transcribe(asr, tt_speech)
    assert golden_text, "Whisper produced an empty golden transcript"
    assert tt_text, "Whisper produced an empty TT transcript"

    target = _target_words()
    golden_words = _normalize(golden_text)
    tt_words = _normalize(tt_text)

    table = _cumulative_wer(target, golden_words, tt_words)
    overall = {
        "golden_vs_gt": round(_wer(target, golden_words), 4),
        "tt_vs_gt": round(_wer(target, tt_words), 4),
        "tt_vs_golden": round(_wer(golden_words, tt_words), 4),
    }

    # Persist transcripts + metrics (TT wav already saved above; golden is shipped).
    metrics = {
        "whisper_model": WHISPER_MODEL,
        "demo": DEMO_ID,
        "max_new_tokens": MAX_NEW_TOKENS,
        "golden_full_audio_sec": round(golden_full_sec, 1),
        "golden_transcribed_sec": round(golden_speech.numel() / SR, 2),
        "tt_audio_sec": round(tt_speech.numel() / SR, 2),
        "target_word_count": len(target),
        "golden_word_count": len(golden_words),
        "tt_word_count": len(tt_words),
        "overall_wer": overall,
        "wer_by_length": table,
        "golden_transcript": golden_text,
        "tt_transcript": tt_text,
    }
    (_OUT_DIR / f"{DEMO_ID}_wer.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    header = f"{'N':>7} | {'golden_vs_gt':>12} | {'tt_vs_gt':>9} | {'tt_vs_golden':>12}"
    lines = [header, "-" * len(header)]
    for r in table:
        lines.append(f"{r['n']:>7} | {r['golden_vs_gt']:>12.4f} | {r['tt_vs_gt']:>9.4f} | {r['tt_vs_golden']:>12.4f}")
    print(
        f"\n[e2e_wer] whisper={WHISPER_MODEL} demo={DEMO_ID} max_new_tokens={MAX_NEW_TOKENS}\n"
        f"[e2e_wer] golden {golden_speech.numel() / SR:.1f}s ({len(golden_words)}w) | "
        f"tt {tt_speech.numel() / SR:.1f}s ({len(tt_words)}w) | target {len(target)}w\n"
        f"[e2e_wer] WER by cumulative transcript length:\n" + "\n".join(lines) + "\n"
        f"[e2e_wer] overall: golden_vs_gt={overall['golden_vs_gt']:.4f} "
        f"tt_vs_gt={overall['tt_vs_gt']:.4f} tt_vs_golden={overall['tt_vs_golden']:.4f}\n"
        f"[e2e_wer] artifacts -> {_OUT_DIR}"
    )
