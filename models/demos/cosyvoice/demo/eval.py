"""C8 Audio quality evaluation — WER + Speaker Similarity.

Measures:
  - ASR WER < 3.0 (whisper-large-v3 transcription vs input text)
  - Speaker similarity > 60 (CAM++ cosine × 100)

Usage:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal
    python -m pytest models/demos/cosyvoice/demo/eval.py -v -s
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

DEMO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
CV_SRC = DEMO_ROOT / "model_data" / "CosyVoice_src"
ASSET_DIR = CV_SRC / "asset"

ZERO_SHOT_PROMPT_WAV = str(ASSET_DIR / "zero_shot_prompt.wav")
ZERO_SHOT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

EVAL_TEXTS_ZH = [
    "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
    "在他讲述那个荒诞故事的过程中，他突然停下来，因为他自己也被逗笑了。",
]

SAMPLE_RATE = 24000


@pytest.fixture(scope="module")
def pipeline():
    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024, trace_region_size=5000000)

    sys.path.insert(0, str(DEMO_ROOT))
    from models.demos.cosyvoice.tt.pipeline import TtnnCosyVoice

    pipe = TtnnCosyVoice(device, model_dir=str(CKPT_DIR))
    pipe.add_zero_shot_spk(ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV, "test_spk")
    yield pipe
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def whisper_model():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return model, processor


def _normalize_zh(text: str) -> str:
    text = re.sub(r"[，。！？、；：" "''（）《》【】\s]", "", text)
    return text.lower()


def _compute_wer(reference: str, hypothesis: str) -> float:
    ref_chars = list(_normalize_zh(reference))
    hyp_chars = list(_normalize_zh(hypothesis))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    import jiwer

    return jiwer.wer(" ".join(ref_chars), " ".join(hyp_chars))


def _transcribe(waveform: torch.Tensor, model, processor) -> str:
    audio = waveform.squeeze().numpy()
    if len(audio) > 30 * 16000:
        audio = audio[: 30 * 16000]
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        generated_ids = model.generate(input_features, language="zh")
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


def _extract_spk_embedding(pipe, wav_path: str) -> np.ndarray:
    embedding = pipe.frontend._extract_spk_embedding(wav_path)
    return embedding.squeeze().numpy()


def _wav_tensor_to_temp(waveform: torch.Tensor, suffix=".wav") -> str:
    import tempfile

    import soundfile as sf

    audio = waveform.squeeze().numpy()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    return tmp.name


def test_wer_zero_shot(pipeline, whisper_model):
    """C8: ASR WER < 3.0 on generated Chinese audio."""
    model, processor = whisper_model
    import torchaudio

    wers = []
    for text in EVAL_TEXTS_ZH:
        waveform = pipeline.inference_zero_shot(text, ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)
        assert waveform.shape[1] > 0, f"No audio generated for: {text[:20]}..."

        audio_16k = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=16000)(waveform)

        transcription = _transcribe(audio_16k, model, processor)
        wer = _compute_wer(text, transcription)
        wers.append(wer)
        print(f"\n[C8] WER={wer:.3f} | ref: {text[:30]}... | hyp: {transcription[:30]}...")

    mean_wer = np.mean(wers)
    print(f"\n[C8] Mean WER: {mean_wer:.3f} (target < 3.0)")
    assert mean_wer < 3.0, f"Mean WER {mean_wer:.3f} >= 3.0"


def test_speaker_similarity(pipeline):
    """C8: Speaker similarity > 60 (CAM++ cosine × 100)."""
    prompt_emb = _extract_spk_embedding(pipeline, ZERO_SHOT_PROMPT_WAV)

    waveform = pipeline.inference_zero_shot(EVAL_TEXTS_ZH[0], ZERO_SHOT_PROMPT_TEXT, ZERO_SHOT_PROMPT_WAV)
    assert waveform.shape[1] > 0

    gen_wav_path = _wav_tensor_to_temp(waveform)
    gen_emb = _extract_spk_embedding(pipeline, gen_wav_path)

    cosine_sim = np.dot(prompt_emb, gen_emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(gen_emb))
    similarity = cosine_sim * 100

    print(f"\n[C8] Speaker similarity: {similarity:.1f} (target > 60)")
    assert similarity > 60, f"Speaker similarity {similarity:.1f} <= 60"
