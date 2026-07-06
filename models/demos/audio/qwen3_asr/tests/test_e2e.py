# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Real-weights end-to-end test for Qwen3-ASR on P150.

Exercises the full shipped pipeline on device with the REAL model weights (no golden
short-circuit): raw waveform -> processor (mel + prompt input_ids) -> ttnn AuT encoder
-> splice audio embeds at the audio-token positions -> ttnn Qwen3-1.7B decoder greedy
decode -> text. This is the same path `demo/demo_wav.py` and `server/qwen3_asr_server._infer`
run, asserted against an expected transcription.

Needs (all via env; the test ``pytest.skip``s cleanly when any is absent):
  QWEN3ASR_E2E_WAV    path to a 16 kHz mono wav to transcribe
  QWEN3ASR_E2E_TEXT   expected substring of the transcription (case/punct-insensitive)
  QWEN3ASR_SNAP       HF snapshot of Qwen/Qwen3-ASR-1.7B (audio-tower weights + processor)
  QWEN3ASR_TEXT_DECODER / HF_MODEL   extracted Qwen3-1.7B text-decoder checkpoint
and the ``qwen_asr`` processor package importable. See tests/conftest.py for the fixtures.

Run on a P150 (single Blackhole) box, e.g.:
  QWEN3ASR_E2E_WAV=/data/jfk.wav QWEN3ASR_E2E_TEXT="ask not what your country" \
  pytest models/demos/audio/qwen3_asr/tests/test_e2e.py -s
"""
import os
import re

import numpy as np
import pytest
import torch

import ttnn
from models.demos.audio.qwen3_asr.reference import audio_encoder_ref as ref
from models.demos.audio.qwen3_asr.tt import audio_encoder as tt_enc
from models.demos.audio.qwen3_asr.tt.qwen3_asr_decoder import Qwen3ASRDecoder
from models.tt_transformers.tt.model_config import ModelArgs

AUDIO_TOKEN_ID = 151676
SR = 16000
FIXED_INFER_SEC = 14.0  # server pins every request to this length; mirror it here

E2E_DEVICE_PARAMS = {"l1_small_size": 32768, "trace_region_size": 200000000}


def _normalize(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _parse(text):
    m = re.search(r"language\s*(.*?)<asr_text>(.*)", text, flags=re.DOTALL)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", text.strip())


@pytest.mark.parametrize("device_params", [E2E_DEVICE_PARAMS], indirect=True)
def test_e2e_wav_transcription(device, snap_dir, text_decoder_ckpt):
    """Raw wav -> TT encoder -> splice -> TT decoder -> text; assert expected words appear."""
    import soundfile as sf

    wav_path = os.environ.get("QWEN3ASR_E2E_WAV")
    expected = os.environ.get("QWEN3ASR_E2E_TEXT")
    if not wav_path or not os.path.isfile(wav_path):
        pytest.skip("set QWEN3ASR_E2E_WAV to a 16 kHz mono wav to run the e2e test")
    if not expected:
        pytest.skip("set QWEN3ASR_E2E_TEXT to the expected transcription substring")
    try:
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
    except Exception as e:  # noqa: BLE001 - external processor package is optional
        pytest.skip(f"qwen_asr processor not importable: {e}")

    # --- load + pad the waveform to the fixed inference length (shipped behavior) ---
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    n = int(FIXED_INFER_SEC * SR)
    wav = wav[:n] if len(wav) >= n else np.concatenate([wav, np.zeros(n - len(wav), dtype=np.float32)])

    # --- processor: prompt (auto language) + mel ---
    proc = Qwen3ASRProcessor.from_pretrained(snap_dir, fix_mistral_regex=True)
    prompt = proc.apply_chat_template(
        [{"role": "system", "content": ""},
         {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
        add_generation_prompt=True, tokenize=False,
    )
    inputs = proc(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"][0].long()
    mel = inputs["input_features"][0].float() if inputs["input_features"].dim() == 3 else inputs["input_features"].float()

    # --- build the real TT encoder + decoder ---
    w = ref.load_audio_tower_weights(snap_dir=snap_dir, dtype=torch.float32)
    enc_params = tt_enc.preprocess_weights(w, device)
    args = ModelArgs(device, max_batch_size=1, max_seq_len=2048)
    model = Qwen3ASRDecoder(
        args, ttnn.bfloat16, device, args.load_state_dict(),
        args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False,
    )
    from safetensors import safe_open

    with safe_open(os.path.join(text_decoder_ckpt, "model.safetensors"), "pt") as h:
        embed = h.get_tensor("model.embed_tokens.weight").float()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(text_decoder_ckpt)

    # --- full pipeline: encode mel -> splice at audio-token rows -> decode ---
    audio_embeds = tt_enc.encode_mel(mel, enc_params, device).float()
    inp = embed[input_ids].clone()
    mask = input_ids == AUDIO_TOKEN_ID
    n_mask = int(mask.sum())
    if audio_embeds.shape[0] > n_mask:
        audio_embeds = audio_embeds[:n_mask]
    elif audio_embeds.shape[0] < n_mask:
        audio_embeds = torch.cat([audio_embeds, torch.zeros(n_mask - audio_embeds.shape[0], audio_embeds.shape[1])], 0)
    inp[mask] = audio_embeds

    ids = model.generate(inp.unsqueeze(0), max_new_tokens=200)
    lang, text = _parse(tok.decode(ids, skip_special_tokens=False))
    print(f"\n[e2e] lang={lang!r} text={text!r}")

    assert text, "e2e produced an empty transcript"
    assert _normalize(expected) in _normalize(text), (
        f"expected substring not found.\n  expected ~ {expected!r}\n  got        {text!r}"
    )
