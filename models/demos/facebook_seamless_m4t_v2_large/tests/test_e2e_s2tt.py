# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end WER validation for the SeamlessM4T-v2 S2TT / ASR TTNN demo.

Runs the TTNN ``SpeechToTextModel.translate`` over the sample WAVs under
``demo/inputs/`` (in both S2TT and ASR modes), computes WER against the
HuggingFace reference output on the same audio, and asserts the TTNN
WER is at most 5 percentage points above the HF WER (which serves as
the floor, since the same audio passes through the same architecture
and we use greedy decoding). The HF reference is run in-process with
``SeamlessM4Tv2ForSpeechToText.generate(do_sample=False, num_beams=1)``.

Run with::

    pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_s2tt.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import SpeechToTextModel, _load_wav_to_16k_mono

# Sample WAVs (committed under demo/inputs, all < 200 KB and <= 5 s, 16 kHz mono).
INPUTS_DIR = Path(__file__).resolve().parent.parent / "demo" / "inputs"

# S2TT: each entry is (wav_filename, src_lang, tgt_lang)
S2TT_SAMPLES = [
    ("sample_hello.wav", "eng", "fra"),
    ("sample_jim.wav", "eng", "spa"),
]

# ASR: each entry is (wav_filename, lang)
ASR_SAMPLES = [
    ("sample_hello.wav", "eng"),
    ("sample_jim.wav", "eng"),
]

MAX_NEW_TOKENS = 32
MAX_AUDIO_SECONDS = 5.0

# WER drift tolerance: TTNN must be within +0.05 of HF WER on the same set.
WER_TOLERANCE = 0.05


def _wer(hyps: List[str], refs: List[str]) -> float:
    """Word error rate via ``jiwer.wer``."""
    import jiwer

    if len(hyps) != len(refs):
        raise ValueError(f"hyps ({len(hyps)}) and refs ({len(refs)}) must be the same length")
    if not hyps:
        raise ValueError("_wer requires at least one (hyp, ref) pair")
    return float(jiwer.wer(refs, hyps))


def _hf_reference(wav_path: Path, tgt_lang: str, processor) -> str:
    from transformers import SeamlessM4Tv2ForSpeechToText

    audio = _load_wav_to_16k_mono(str(wav_path))
    audio = audio[: int(MAX_AUDIO_SECONDS * 16000)]
    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_features=feats["input_features"],
            attention_mask=feats["attention_mask"],
            tgt_lang=tgt_lang,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
        )
    if hasattr(out, "sequences"):
        out = out.sequences
    text = processor.decode(out[0].tolist(), skip_special_tokens=True)
    del model
    return text


@pytest.fixture(scope="module")
def hf_sd():
    return wl.load_hf_state_dict()


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(wl.HF_PATH)


@pytest.fixture(scope="module")
def hf_translations(processor) -> List[str]:
    return [_hf_reference(INPUTS_DIR / wav, tgt, processor) for (wav, _src, tgt) in S2TT_SAMPLES]


@pytest.fixture(scope="module")
def hf_transcriptions(processor) -> List[str]:
    return [_hf_reference(INPUTS_DIR / wav, lang, processor) for (wav, lang) in ASR_SAMPLES]


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def test_s2tt_wer_matches_hf(hf_sd, processor, hf_translations, device):
    """TTNN S2TT WER on a short sample set must be within +5% of HF WER."""
    refs = list(hf_translations)
    assert all(Path(INPUTS_DIR / w).is_file() for w, _, _ in S2TT_SAMPLES), "sample WAVs missing"

    model = SpeechToTextModel(device=device, hf_state_dict=hf_sd, processor=processor)
    ttnn_outs: List[str] = []
    for wav, src, tgt in S2TT_SAMPLES:
        out = model.translate(
            audio_path=str(INPUTS_DIR / wav),
            src_lang=src,
            tgt_lang=tgt,
            max_new_tokens=MAX_NEW_TOKENS,
            max_audio_seconds=MAX_AUDIO_SECONDS,
        )
        ttnn_outs.append(out)

    ttnn_wer = _wer(ttnn_outs, refs)
    hf_wer = _wer(hf_translations, refs)  # same list -> 0.0 by construction

    print("")
    for i, ((wav, src, tgt), hf_t, tt_t) in enumerate(zip(S2TT_SAMPLES, hf_translations, ttnn_outs)):
        print(f"[s2tt {i}] {wav} ({src}->{tgt})")
        print(f"   HF:   {hf_t}")
        print(f"   TTNN: {tt_t}")
    print(f"S2TT TTNN_WER = {ttnn_wer:.4f}")
    print(f"S2TT HF_WER   = {hf_wer:.4f}")
    print(f"drift         = {ttnn_wer - hf_wer:.4f}  (tolerance = {WER_TOLERANCE})")

    assert (
        ttnn_wer <= hf_wer + WER_TOLERANCE
    ), f"TTNN S2TT WER {ttnn_wer:.4f} exceeded HF WER {hf_wer:.4f} by more than {WER_TOLERANCE}"


def test_asr_wer_matches_hf(hf_sd, processor, hf_transcriptions, device):
    """TTNN ASR (src_lang==tgt_lang) WER must be within +5% of HF WER."""
    refs = list(hf_transcriptions)
    assert all(Path(INPUTS_DIR / w).is_file() for w, _ in ASR_SAMPLES), "sample WAVs missing"

    model = SpeechToTextModel(device=device, hf_state_dict=hf_sd, processor=processor)
    ttnn_outs: List[str] = []
    for wav, lang in ASR_SAMPLES:
        out = model.transcribe(
            audio_path=str(INPUTS_DIR / wav),
            lang=lang,
            max_new_tokens=MAX_NEW_TOKENS,
            max_audio_seconds=MAX_AUDIO_SECONDS,
        )
        ttnn_outs.append(out)

    ttnn_wer = _wer(ttnn_outs, refs)
    hf_wer = _wer(hf_transcriptions, refs)  # 0.0

    print("")
    for i, ((wav, lang), hf_t, tt_t) in enumerate(zip(ASR_SAMPLES, hf_transcriptions, ttnn_outs)):
        print(f"[asr {i}] {wav} ({lang})")
        print(f"   HF:   {hf_t}")
        print(f"   TTNN: {tt_t}")
    print(f"ASR  TTNN_WER = {ttnn_wer:.4f}")
    print(f"ASR  HF_WER   = {hf_wer:.4f}")
    print(f"drift         = {ttnn_wer - hf_wer:.4f}  (tolerance = {WER_TOLERANCE})")

    assert (
        ttnn_wer <= hf_wer + WER_TOLERANCE
    ), f"TTNN ASR WER {ttnn_wer:.4f} exceeded HF WER {hf_wer:.4f} by more than {WER_TOLERANCE}"
