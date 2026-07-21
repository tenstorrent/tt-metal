# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end Speaker Similarity (SIM): does TTNN VibeVoice preserve the cloned speaker's identity?

Standard speaker-verification (SIM-O) methodology applied to voice cloning:
  1. Take the target speaker's *reference* speech — a voice sample from ``resources/voices/``
     (the same enrollment audio VibeVoice clones from).
  2. Generate speech on TTNN, voice-cloning that target speaker from a single-speaker script
     (``resources/text/1p_abs.txt``, Speaker 1).
  3. Embed both waveforms with a speaker-verification model (``microsoft/wavlm-base-plus-sv``,
     a WavLM x-vector head — ships with transformers, no extra deps).
  4. Compare the L2-normalized embeddings by cosine similarity.

The headline metric is ``sim_target`` = cos(embed(TT-generated), embed(target reference)). To make
it a *verification* (not just "a positive number"), we also embed a panel of the other demo voices
as impostors and check the generated speech is closer to the intended target than to any of them —
``sim_target > best_impostor + margin`` — i.e. the SV model would pick the right speaker. The
opposite-gender impostors are the clean negative control.

Report context: the ``microsoft/wavlm-base-plus-sv`` model card suggests 0.86 as an example
same-speaker decision threshold (it notes the optimal threshold is dataset-dependent). Cosine
scores from this model are compressed (different speakers still land ~0.5-0.7), so the *relative*
target-vs-impostor margin is the robust signal, not the absolute value alone.

Two tests:
  * ``test_e2e_sim_tt_voice_clone`` — one target voice vs an impostor panel (the headline SIM).
  * ``test_e2e_sim_4speaker`` — generates a separate single-speaker TT clip for each of the four
    climate-demo voices (Alice/Carter/Frank/Maya) and builds a 4x4 confusion matrix; each clip
    must self-identify (be closest to its OWN reference), a proper multi-speaker verification.

SV backend is selectable via ``VV_SIM_SV_BACKEND``:
  * ``base_plus`` (default) — ``microsoft/wavlm-base-plus-sv`` (ships with transformers).
  * ``wavlm_large_ft`` — the WavLM-large fine-tuned SV model the VibeVoice technical report uses for
    SIM (UniSpeech ``wavlm_large_finetune.pth``, vendored torch-only under ``common/wavlm_sv``).
Thresholds are env-overridable (``VV_SIM_TARGET_FLOOR``, ``VV_SIM_MARGIN``; backend-aware defaults);
``VV_SIM_MAX_NEW_TOKENS`` caps AR steps (default renders ~20-25 s). ``VV_SIM_REUSE_TT=1`` reuses the
saved TT wavs (e.g. to rescore the same audio with a different SV backend, no device). Artifacts
(wav + metrics JSON) land under ``output/e2e_sim/``.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from models.experimental.vibevoice.common.config import MODEL_PATH, TEXT_EXAMPLES_DIR, VOICES_DIR
from models.experimental.vibevoice.common.resource_utils import (
    CLIMATE_4P_SPEAKER_NAMES,
    CLIMATE_4P_VOICE_FILES,
    load_script,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
SR = 24000  # VibeVoice sample rate
SV_SR = 16000  # WavLM feature-extractor sample rate
# Selectable speaker-verification backend:
#   "base_plus"      — microsoft/wavlm-base-plus-sv x-vector head (ships with transformers, no extra
#                      deps). Compressed cosine scale (different speakers ~0.5-0.7).
#   "wavlm_large_ft" — the WavLM-large fine-tuned SV model the VibeVoice technical report uses for
#                      SIM (UniSpeech wavlm_large_finetune.pth, vendored under common/wavlm_sv,
#                      torch-only auto-download). Clean scale (same speaker ~0.9, impostors ~0).
SV_BACKEND = os.environ.get("VV_SIM_SV_BACKEND", "base_plus")
# Per backend: (label, same-speaker reference threshold, default target floor, default margin).
# The reference threshold is context-only ("optimal threshold is dataset-dependent"); the test
# asserts a relative target-vs-impostor margin, not the absolute value.
_SV_PROFILES = {
    "base_plus": ("microsoft/wavlm-base-plus-sv", 0.86, 0.5, 0.05),
    "wavlm_large_ft": ("wavlm_large_finetune (UniSpeech, VibeVoice-report SIM)", 0.5, 0.3, 0.15),
}
assert SV_BACKEND in _SV_PROFILES, f"VV_SIM_SV_BACKEND must be one of {list(_SV_PROFILES)}"
SV_MODEL, SV_SAME_SPEAKER_THRESHOLD, _FLOOR_DEFAULT, _MARGIN_DEFAULT = _SV_PROFILES[SV_BACKEND]

# Single-speaker script cloned to the target voice.
TEXT_ID = os.environ.get("VV_SIM_TEXT_ID", "1p_abs")
_TEXT_PATH = TEXT_EXAMPLES_DIR / f"{TEXT_ID}.txt"
# Target speaker whose voice we clone; the generated speech must resemble this reference.
TARGET_VOICE = os.environ.get("VV_SIM_TARGET_VOICE", "en-Carter_man.wav")
# Impostor voices for the negative control (opposite gender = cleanest separation).
IMPOSTOR_VOICES = ["en-Alice_woman.wav", "en-Maya_woman.wav", "zh-Xinran_woman.wav"]

# AR-step cap. ~7.5 frames/s of audio, so 200 frames ≈ 27 s — ample for a stable x-vector.
MAX_NEW_TOKENS = int(os.environ.get("VV_SIM_MAX_NEW_TOKENS", "200"))
# Pass thresholds (backend-aware defaults, calibrated from correct runs; override to explore).
SIM_TARGET_FLOOR = float(os.environ.get("VV_SIM_TARGET_FLOOR", str(_FLOOR_DEFAULT)))
SIM_MARGIN = float(os.environ.get("VV_SIM_MARGIN", str(_MARGIN_DEFAULT)))

_OUT_DIR = _VIBEVOICE_ROOT / "output" / "e2e_sim"
_OUT_TAG = os.environ.get("VV_SIM_OUT_TAG", "")


def _load_wav_16k(path: Path) -> torch.Tensor:
    """Load a wav (any sr, mono) and resample to the SV model's 16 kHz."""
    import librosa
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != SV_SR:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=SV_SR)
    return torch.from_numpy(data)


class _BasePlusVerifier:
    """microsoft/wavlm-base-plus-sv x-vector embedder → L2-normalized embeddings."""

    def __init__(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.model = WavLMForXVector.from_pretrained(SV_MODEL).eval()

    def embed(self, wav_16k: torch.Tensor) -> torch.Tensor:
        wav = wav_16k.detach().to(torch.float32).cpu().clamp(-1.0, 1.0).numpy()
        inp = self.fe(wav, sampling_rate=SV_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = self.model(**inp).embeddings
        return torch.nn.functional.normalize(emb, dim=-1)

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a, b).item()


class _WavLMLargeFtVerifier:
    """WavLM-large fine-tuned SV (UniSpeech wavlm_large_finetune.pth) — the VibeVoice-report SIM."""

    def __init__(self):
        from models.experimental.vibevoice.common.wavlm_sv.wavlm_standalone import init_model

        self.model = init_model()  # auto-downloads + caches the checkpoint

    def embed(self, wav_16k: torch.Tensor) -> torch.Tensor:
        from models.experimental.vibevoice.common.wavlm_sv.wavlm_standalone import embed as _embed

        return _embed(self.model, wav_16k)

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a, b).item()


def _make_verifier():
    """Build the selected SV backend (VV_SIM_SV_BACKEND)."""
    if SV_BACKEND == "wavlm_large_ft":
        return _WavLMLargeFtVerifier()
    return _BasePlusVerifier()


def _make_processor():
    from processor.vibevoice_processor import VibeVoiceProcessor

    return VibeVoiceProcessor.from_pretrained(MODEL_PATH)


def _process_inputs(processor, target_voice_path: Path):
    """Build processor inputs: the single-speaker script cloned to ``target_voice_path``."""
    assert _TEXT_PATH.is_file(), f"Missing script: {_TEXT_PATH}"
    script = load_script(_TEXT_PATH)  # single speaker → cloned to target_voice
    return processor(
        text=[script],
        voice_samples=[[str(target_voice_path)]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )


def _tt_generate_speech(tt_model, tokenizer, inputs, max_new_tokens) -> torch.Tensor:
    """Run TT voice-clone generation; return the 1-D 24 kHz waveform."""
    torch.manual_seed(0)
    tt_out = tt_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_tensors=inputs["speech_tensors"],
        speech_masks=inputs["speech_masks"],
        speech_input_mask=inputs["speech_input_mask"],
        tokenizer=tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=max_new_tokens,
    )
    assert tt_out.speech_outputs and tt_out.speech_outputs[0] is not None
    return tt_out.speech_outputs[0].to(torch.float32).reshape(-1)


def _to_16k(speech_24k: torch.Tensor) -> torch.Tensor:
    import librosa

    return torch.from_numpy(librosa.resample(speech_24k.numpy(), orig_sr=SR, target_sr=SV_SR))


def _sanity_check(name: str, speech: torch.Tensor) -> None:
    assert speech.numel() > SR // 2, f"{name} audio too short: {speech.numel()} samples"
    assert torch.isfinite(speech).all(), f"{name} audio contains NaN/Inf"
    peak = speech.abs().max().item()
    assert peak < 50.0, f"{name} audio blew up (peak {peak:.1f}), not mild clipping"
    assert (speech.abs() > 1e-3).float().mean().item() > 0.3, f"{name} audio is mostly silent"


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_sim_tt_voice_clone(mesh_device):
    """Generate cloned speech on TT, embed it + the reference/impostor voices, assert SIM."""
    # Load the speaker-verification model up front so an unavailable SV skips fast (before generation).
    try:
        sv = _make_verifier()
    except Exception as exc:
        pytest.skip(f"Speaker-verification model unavailable ({SV_MODEL}): {exc}")

    target_path = VOICES_DIR / TARGET_VOICE
    assert target_path.is_file(), f"Missing target voice: {target_path}"

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    tt_wav_path = _OUT_DIR / f"{TEXT_ID}_{Path(TARGET_VOICE).stem}{_OUT_TAG}_tt.wav"
    if os.environ.get("VV_SIM_REUSE_TT") == "1" and tt_wav_path.is_file():
        gen_16k = _load_wav_16k(tt_wav_path)
        tt_sec = gen_16k.numel() / SV_SR
    else:
        processor = _make_processor()
        inputs = _process_inputs(processor, target_path)
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh_device, MODEL_PATH, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS
        )
        tt_speech = _tt_generate_speech(tt_model, processor.tokenizer, inputs, MAX_NEW_TOKENS)
        _sanity_check("tt", tt_speech)
        sf.write(str(tt_wav_path), tt_speech.clamp(-1.0, 1.0).numpy(), SR)
        tt_sec = tt_speech.numel() / SR
        print(f"[e2e_sim] TT audio SAVED -> {tt_wav_path} ({tt_sec:.1f}s)", flush=True)
        gen_16k = _to_16k(tt_speech)

    # Embed the generated speech and every candidate voice; cosine vs each.
    gen_emb = sv.embed(gen_16k)
    candidates = [TARGET_VOICE] + IMPOSTOR_VOICES
    sims: dict[str, float] = {}
    for voice in candidates:
        vpath = VOICES_DIR / voice
        if not vpath.is_file():
            continue
        sims[voice] = round(sv.cosine(gen_emb, sv.embed(_load_wav_16k(vpath))), 4)

    sim_target = sims[TARGET_VOICE]
    impostors = {v: s for v, s in sims.items() if v != TARGET_VOICE}
    best_impostor_voice = max(impostors, key=impostors.get)
    best_impostor = impostors[best_impostor_voice]
    margin = round(sim_target - best_impostor, 4)
    rank1 = max(sims, key=sims.get) == TARGET_VOICE

    metrics = {
        "sv_backend": SV_BACKEND,
        "sv_model": SV_MODEL,
        "same_speaker_threshold": SV_SAME_SPEAKER_THRESHOLD,
        "text_id": TEXT_ID,
        "target_voice": TARGET_VOICE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tt_audio_sec": round(tt_sec, 2),
        "sim_target": sim_target,
        "best_impostor_voice": best_impostor_voice,
        "best_impostor_sim": best_impostor,
        "target_minus_best_impostor": margin,
        "rank1_is_target": rank1,
        "all_sims": sims,
        "thresholds": {"target_floor": SIM_TARGET_FLOOR, "margin": SIM_MARGIN},
    }
    (_OUT_DIR / f"{TEXT_ID}_{Path(TARGET_VOICE).stem}{_OUT_TAG}_{SV_BACKEND}_sim.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    rows = "\n".join(
        f"{'  * ' if v == TARGET_VOICE else '    '}{v:<24} {s:.4f}" + ("  <- target" if v == TARGET_VOICE else "")
        for v, s in sorted(sims.items(), key=lambda kv: -kv[1])
    )
    print(
        f"\n[e2e_sim] sv={SV_MODEL} text={TEXT_ID} target={TARGET_VOICE} "
        f"tt_audio={tt_sec:.1f}s max_new_tokens={MAX_NEW_TOKENS}\n"
        f"[e2e_sim] cosine similarity (generated vs each voice):\n{rows}\n"
        f"[e2e_sim] sim_target={sim_target:.4f}  best_impostor={best_impostor:.4f} ({best_impostor_voice})  "
        f"margin={margin:+.4f}  rank1_is_target={rank1}\n"
        f"[e2e_sim] same-speaker ref threshold ({SV_MODEL})={SV_SAME_SPEAKER_THRESHOLD}  "
        f"artifacts -> {_OUT_DIR}"
    )

    # Assertions: generated speech resembles the intended target more than any impostor.
    assert sim_target >= SIM_TARGET_FLOOR, (
        f"sim_target {sim_target:.4f} < floor {SIM_TARGET_FLOOR}: generated speech does not resemble "
        f"the cloned target speaker ({TARGET_VOICE})"
    )
    assert margin >= SIM_MARGIN, (
        f"target-vs-impostor margin {margin:+.4f} < {SIM_MARGIN}: generated speech is not clearly "
        f"closer to the target ({sim_target:.4f}) than to impostor {best_impostor_voice} ({best_impostor:.4f})"
    )


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_sim_4speaker(mesh_device):
    """4-speaker SIM confusion matrix: each climate speaker's own TT clip must self-identify.

    The 4-speaker climate demo casts Alice / Carter / Frank / Maya, each cloned from its own
    reference voice. Here we generate a *separate* single-speaker TT clip per voice (holding the
    text constant so only speaker identity varies — its own TT), embed all four generated clips
    and all four reference voices, and build a 4x4 cosine matrix ``M[gen_i][ref_j]``.

    Verification is per speaker: each generated clip must be closest to its OWN reference
    (rank-1 self-identification), above the floor, and clear of the best impostor by a margin —
    i.e. the SV model correctly tells the four TT-cloned speakers apart. Same-gender pairs
    (Carter/Frank, Alice/Maya) are the hard case this exercises.
    """
    try:
        sv = _make_verifier()
    except Exception as exc:
        pytest.skip(f"Speaker-verification model unavailable ({SV_MODEL}): {exc}")

    # Speaker → voice file, in speaker-id order (Alice, Carter, Frank, Maya).
    names = [CLIMATE_4P_SPEAKER_NAMES[i] for i in sorted(CLIMATE_4P_VOICE_FILES)]
    voice_files = {CLIMATE_4P_SPEAKER_NAMES[i]: CLIMATE_4P_VOICE_FILES[i] for i in CLIMATE_4P_VOICE_FILES}
    for name in names:
        assert (VOICES_DIR / voice_files[name]).is_file(), f"Missing voice for {name}: {voice_files[name]}"

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    import gc

    import soundfile as sf

    # Reference (enrollment) embedding per speaker.
    ref_emb = {name: sv.embed(_load_wav_16k(VOICES_DIR / voice_files[name])) for name in names}

    # One TT clip per speaker. Model loaded lazily (once) only if something needs generating, so
    # VV_SIM_REUSE_TT=1 rescoring (e.g. a different SV backend on the same audio) needs no device.
    reuse = os.environ.get("VV_SIM_REUSE_TT") == "1"
    processor = None
    tt_model = None
    gen_emb: dict[str, torch.Tensor] = {}
    gen_sec: dict[str, float] = {}
    for name in names:
        wav_path = _OUT_DIR / f"4speaker_{name}{_OUT_TAG}_tt.wav"
        if reuse and wav_path.is_file():
            speech_16k = _load_wav_16k(wav_path)
            gen_sec[name] = round(speech_16k.numel() / SV_SR, 2)
        else:
            if tt_model is None:
                processor = _make_processor()
                tt_model = TTVibeVoiceModel.from_checkpoint(
                    mesh_device, MODEL_PATH, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS
                )
            inputs = _process_inputs(processor, VOICES_DIR / voice_files[name])
            tt_speech = _tt_generate_speech(tt_model, processor.tokenizer, inputs, MAX_NEW_TOKENS)
            _sanity_check(name, tt_speech)
            sf.write(str(wav_path), tt_speech.clamp(-1.0, 1.0).numpy(), SR)
            gen_sec[name] = round(tt_speech.numel() / SR, 2)
            speech_16k = _to_16k(tt_speech)
            gc.collect()  # drop the per-generate KV caches before the next speaker
        gen_emb[name] = sv.embed(speech_16k)
        print(f"[e2e_sim_4p] {name}: TT clip {gen_sec[name]:.1f}s -> {wav_path}", flush=True)

    # 4x4 cosine matrix: rows = generated clip, cols = reference voice.
    matrix = {gi: {rj: round(sv.cosine(gen_emb[gi], ref_emb[rj]), 4) for rj in names} for gi in names}

    per_speaker = []
    for name in names:
        row = matrix[name]
        self_sim = row[name]
        off = {r: s for r, s in row.items() if r != name}
        best_off_name = max(off, key=off.get)
        best_off = off[best_off_name]
        row_margin = round(self_sim - best_off, 4)
        rank1 = max(row, key=row.get) == name
        ok = rank1 and self_sim >= SIM_TARGET_FLOOR and row_margin >= SIM_MARGIN
        per_speaker.append(
            {
                "speaker": name,
                "voice_file": voice_files[name],
                "self_sim": self_sim,
                "best_impostor": best_off_name,
                "best_impostor_sim": best_off,
                "margin": row_margin,
                "rank1_is_self": rank1,
                "pass": ok,
            }
        )

    metrics = {
        "sv_backend": SV_BACKEND,
        "sv_model": SV_MODEL,
        "same_speaker_threshold": SV_SAME_SPEAKER_THRESHOLD,
        "text_id": TEXT_ID,
        "speakers": names,
        "voice_files": voice_files,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tt_audio_sec": gen_sec,
        "matrix": matrix,
        "per_speaker": per_speaker,
        "thresholds": {"target_floor": SIM_TARGET_FLOOR, "margin": SIM_MARGIN},
        "all_pass": all(r["pass"] for r in per_speaker),
    }
    (_OUT_DIR / f"4speaker{_OUT_TAG}_{SV_BACKEND}_sim.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    # Print the confusion matrix (diagonal = self; should dominate each row).
    col_hdr = "gen\\ref".ljust(10) + " ".join(f"{n:>10}" for n in names)
    lines = [col_hdr, "-" * len(col_hdr)]
    for gi in names:
        cells = " ".join((f"[{matrix[gi][rj]:.3f}]" if rj == gi else f" {matrix[gi][rj]:.3f} ") for rj in names)
        lines.append(f"{gi:<10}{cells}")
    verdicts = "\n".join(
        f"    {r['speaker']:<8} self={r['self_sim']:.4f}  best_impostor={r['best_impostor_sim']:.4f} "
        f"({r['best_impostor']})  margin={r['margin']:+.4f}  rank1={r['rank1_is_self']}  "
        f"{'PASS' if r['pass'] else 'FAIL'}"
        for r in per_speaker
    )
    print(
        f"\n[e2e_sim_4p] sv={SV_MODEL} text={TEXT_ID} max_new_tokens={MAX_NEW_TOKENS}\n"
        f"[e2e_sim_4p] 4x4 cosine similarity (diagonal [x.xxx] = generated vs its OWN reference):\n"
        + "\n".join(lines)
        + "\n[e2e_sim_4p] per-speaker verdict:\n"
        + verdicts
        + f"\n[e2e_sim_4p] same-speaker ref threshold={SV_SAME_SPEAKER_THRESHOLD}  artifacts -> {_OUT_DIR}"
    )

    failed = [r["speaker"] for r in per_speaker if not r["pass"]]
    assert not failed, (
        f"{len(failed)}/4 speakers failed self-identification: {failed}. "
        f"Each generated clip must match its own reference best (rank-1), above floor "
        f"{SIM_TARGET_FLOOR} with margin ≥ {SIM_MARGIN}. See matrix above."
    )
