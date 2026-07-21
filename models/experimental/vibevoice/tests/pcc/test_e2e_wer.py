# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end Word Error Rate (WER): TTNN VibeVoice vs the golden reference audio, by Whisper.

Runs ``demo_ttnn.py --demo <DEMO_ID>`` inputs — the 4-speaker ``resources/text/<DEMO_ID>.txt``
script with voice cloning (Speaker 1-4 → en-Alice_woman / en-Carter_man / en-Frank_man /
en-Maya_woman) — generating the FULL audio on TTNN. The **reference** side is NOT regenerated (the
fp32 CPU reference would take many hours, and there is no GPU); instead we use the shipped golden
clip ``resources/golden/<DEMO_ID>.wav`` (the microsoft.github.io/VibeVoice reference).

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
``MAX_NEW_TOKENS`` (env ``VV_WER_MAX_NEW_TOKENS``) caps AR steps; the default renders the full
script. Decode uses the whole-segment fused-frame trace (``VV_TRACE_SEGMENT=1``, ~4.4x faster).
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
# Match `demo_ttnn.py --demo <DEMO_ID>`: 4-speaker climate script + voice cloning.
DEMO_ID = os.environ.get("VV_WER_DEMO_ID", "4p_climate_100min")
_TEXT_PATH = TEXT_EXAMPLES_DIR / f"{DEMO_ID}.txt"
_GOLDEN_WAV = GOLDEN_DIR / f"{DEMO_ID}.wav"
_GT_JSON = GOLDEN_DIR / "transcripts" / f"{DEMO_ID}_gt_timestamp.json"
# AR-step cap. Default renders the full 100-min script: prefill is 23,038 tokens, so 42,498 frames
# fit under the 65,536 max_position ceiling (94.4 min); the ~42,071-frame script EOSes first.
# Override with VV_WER_MAX_NEW_TOKENS (e.g. 32 for a quick traced smoke test).
MAX_NEW_TOKENS = int(os.environ.get("VV_WER_MAX_NEW_TOKENS", "42400"))
# Cumulative transcript-word prefixes at which WER is reported.
WER_PREFIX_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
_OUT_DIR = _VIBEVOICE_ROOT / "output" / "e2e_wer"
_OUT_TAG = os.environ.get("VV_WER_OUT_TAG", "")  # suffix to avoid clobbering baseline artifacts


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
    from processor.vibevoice_processor import VibeVoiceProcessor

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
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1_400_000_000, "num_command_queues": 2}],
    indirect=True,
)
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

    tt_wav_path = _OUT_DIR / f"{DEMO_ID}{_OUT_TAG}_tt.wav"
    if os.environ.get("VV_WER_REUSE_TT") == "1" and tt_wav_path.is_file():
        tt_speech = _load_wav(tt_wav_path)
    else:
        # Whole-segment fused-frame trace (~4.4x faster decode; PCC-1.0 vs eager). Read by the
        # generator at construction; the mesh is opened above with a trace region + 2nd queue.
        os.environ.setdefault("VV_TRACE_SEGMENT", "1")
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
        print(
            f"[e2e_wer] TT audio SAVED -> {tt_wav_path} ({tt_speech.numel() / SR:.1f}s) "
            f"before any reference/Whisper work (rerun with VV_WER_REUSE_TT=1 to skip regen)",
            flush=True,
        )
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
    (_OUT_DIR / f"{DEMO_ID}{_OUT_TAG}_wer.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

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


TF_DEMO_ID = "4p_climate_45min"
TF_MAX_NEW_TOKENS = 512


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_wer_teacher_forced(mesh_device):
    """Teacher-forced parity: bf16 reference re-encoded embedding fed into TT's LM each frame.

    Both backends bf16, 45-min script, cap 512 frames. The reference free-runs and we capture its
    per-frame re-encoded embedding (``acoustic_connector + semantic_connector`` = ``diffusion_embeds``).
    TT then replays the reference token stream (``forced_token_ids``) with its ``_post_diffusion_embeds``
    hooked so TT's LM is fed the REFERENCE embedding each step (no feedback drift), while TT still renders
    its own audio. WER is TT audio vs reference audio (Whisper) — isolates TT's per-step rendering fidelity.
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
