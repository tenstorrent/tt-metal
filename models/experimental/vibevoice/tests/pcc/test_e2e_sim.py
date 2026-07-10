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

Report context: ``microsoft/wavlm-base-plus-sv``'s published same-speaker (EER) threshold is 0.86 —
cosine scores from this model are compressed (different speakers still land ~0.5-0.7), so the
*relative* target-vs-impostor margin is the robust signal, not the absolute value alone.

Thresholds are env-overridable (``VV_SIM_TARGET_FLOOR``, ``VV_SIM_MARGIN``); ``VV_SIM_MAX_NEW_TOKENS``
caps AR steps (default renders ~20-25 s, plenty for a stable embedding). Artifacts (wav + metrics
JSON) land under ``output/e2e_sim/``.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from models.experimental.vibevoice.common.config import MODEL_PATH, TEXT_EXAMPLES_DIR, VOICES_DIR
from models.experimental.vibevoice.common.resource_utils import load_script
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
SR = 24000  # VibeVoice sample rate
SV_MODEL = "microsoft/wavlm-base-plus-sv"
SV_SR = 16000  # WavLM feature-extractor sample rate
SV_SAME_SPEAKER_THRESHOLD = 0.86  # published EER threshold for wavlm-base-plus-sv (context only)

# Single-speaker script cloned to the target voice.
TEXT_ID = os.environ.get("VV_SIM_TEXT_ID", "1p_abs")
_TEXT_PATH = TEXT_EXAMPLES_DIR / f"{TEXT_ID}.txt"
# Target speaker whose voice we clone; the generated speech must resemble this reference.
TARGET_VOICE = os.environ.get("VV_SIM_TARGET_VOICE", "en-Carter_man.wav")
# Impostor voices for the negative control (opposite gender = cleanest separation).
IMPOSTOR_VOICES = ["en-Alice_woman.wav", "en-Maya_woman.wav", "zh-Xinran_woman.wav"]

# AR-step cap. ~7.5 frames/s of audio, so 200 frames ≈ 27 s — ample for a stable x-vector.
MAX_NEW_TOKENS = int(os.environ.get("VV_SIM_MAX_NEW_TOKENS", "200"))
# Pass thresholds (calibrated from a correct run; override to explore).
SIM_TARGET_FLOOR = float(os.environ.get("VV_SIM_TARGET_FLOOR", "0.5"))
SIM_MARGIN = float(os.environ.get("VV_SIM_MARGIN", "0.05"))

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


class _SpeakerVerifier:
    """WavLM x-vector speaker embedder → L2-normalized embeddings + cosine similarity."""

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


def _build_inputs(target_voice_path: Path):
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    assert _TEXT_PATH.is_file(), f"Missing script: {_TEXT_PATH}"
    script = load_script(_TEXT_PATH)  # single speaker → cloned to target_voice
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    inputs = processor(
        text=[script],
        voice_samples=[[str(target_voice_path)]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return processor, inputs


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
        sv = _SpeakerVerifier()
    except Exception as exc:
        pytest.skip(f"Speaker-verification model unavailable ({SV_MODEL}): {exc}")

    target_path = VOICES_DIR / TARGET_VOICE
    assert target_path.is_file(), f"Missing target voice: {target_path}"
    processor, inputs = _build_inputs(target_path)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    tt_wav_path = _OUT_DIR / f"{TEXT_ID}_{Path(TARGET_VOICE).stem}{_OUT_TAG}_tt.wav"
    if os.environ.get("VV_SIM_REUSE_TT") == "1" and tt_wav_path.is_file():
        gen_16k = _load_wav_16k(tt_wav_path)
        tt_sec = gen_16k.numel() / SV_SR
    else:
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh_device, MODEL_PATH, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS
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
        _sanity_check("tt", tt_speech)
        sf.write(str(tt_wav_path), tt_speech.clamp(-1.0, 1.0).numpy(), SR)
        tt_sec = tt_speech.numel() / SR
        print(f"[e2e_sim] TT audio SAVED -> {tt_wav_path} ({tt_sec:.1f}s)", flush=True)
        import librosa

        gen_16k = torch.from_numpy(librosa.resample(tt_speech.numpy(), orig_sr=SR, target_sr=SV_SR))

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
    (_OUT_DIR / f"{TEXT_ID}_{Path(TARGET_VOICE).stem}{_OUT_TAG}_sim.json").write_text(
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
