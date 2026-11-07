# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host audio backend (tt/audio/host_backend.py): no device.

* Its tokenizer encoder (fp32, ``is_causal`` attention on the full layers) gives the reference's RVQ codes exactly,
  on every clip from reference/make_audio_clips.py (5 s to 600 s, one to ten 6000-frame segments), at the backend's
  thread count (8; the reference runs at MIMO_AUDIO_THREADS).
* ``embed_audio`` returns bf16 [audio_token_len(T), 4096]: with an fp32 patch encoder bit for bit the reference's
  embeddings cast to bf16, with the default bf16 patch encoder PCC >= 0.9995 against them.
* It restores the caller's torch thread count and validates the mel's shape.
"""

import os

import pytest
import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.tests.test_audio_reference_host import (
    audio_model_path,
    clips_dir,
    load_wav_24k,
    pcc,
    vllm_processor_or_none,
)
from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import HostAudioBackend

CLIPS = ("speech_5s", "speech_30s", "speech_65s", "speech_120s", "speech_300s", "speech_600s")
THREADS = int(os.environ.get("MIMO_AUDIO_THREADS", "16"))


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(THREADS)
    path = audio_model_path()
    root = clips_dir()
    proc = vllm_processor_or_none()
    if proc is None:
        pytest.skip("vllm or torchaudio not importable (the clips are resampled with vLLM's resampler)")
    cfg = ref.MiMoAudioConfig.from_model_path(path)
    weights = ref.load_audio_weights(path)
    mels = {}
    for c in CLIPS:
        if os.path.exists(os.path.join(root, f"{c}.wav")):
            mels[c] = ref.log_mel_spectrogram(load_wav_24k(os.path.join(root, f"{c}.wav"), proc[1]), cfg)
    return {"path": path, "cfg": cfg, "weights": weights, "mels": mels, "proc": proc[0]}


@pytest.fixture(scope="module")
def backends(setup):
    exact = HostAudioBackend(setup["path"], num_threads=8, patch_dtype=torch.float32, weights=setup["weights"])
    default = HostAudioBackend(setup["path"], num_threads=8, weights=setup["weights"])
    return exact, default


@pytest.mark.parametrize("clip", CLIPS)
def test_codes_identical_to_reference(setup, backends, clip):
    if clip not in setup["mels"]:
        pytest.skip(f"{clip}.wav not in the clips directory")
    mel = setup["mels"][clip]
    enc = ref.AudioTokenizerEncoder(setup["weights"], setup["cfg"], torch.float32)
    want = ref.tokenize_audio_batch([mel], enc)[0]
    got = backends[0].codes(mel)
    assert torch.get_num_threads() == THREADS
    assert torch.equal(got, want), (got != want).any(1).float().mean()
    logger.info(f"{clip}: {mel.shape[0]} mel frames, {got.shape[0]} code frames identical to the reference")


@pytest.mark.parametrize("clip", ("speech_5s", "speech_65s"))
def test_embed_audio(setup, backends, clip):
    mel, cfg = setup["mels"][clip], setup["cfg"]
    want = ref.MiMoAudioReference(setup["weights"], cfg).embed([mel])[0]
    exact, default = backends
    e = exact.embed_audio(mel)
    d = default.embed_audio(mel[None])  # vLLM's batched field shape [1, T, 128] is accepted too
    n = ref.audio_token_len(mel.shape[0], cfg)
    assert (
        e.shape == d.shape == (n, 4096) and e.dtype == d.dtype == torch.bfloat16 and n == exact.num_tokens(mel.shape[0])
    )
    assert torch.equal(e, want.to(torch.bfloat16))
    p = pcc(d, want)
    logger.info(f"{clip}: {n} rows; fp32 patch bit-identical to the reference, bf16 patch PCC {p:.6f}")
    assert p >= 0.9995


def test_bad_mel_shape(setup, backends, expect_error):
    with expect_error(ValueError, "audio_features must be"):
        backends[1].embed_audio(torch.zeros(100, 80))
    with expect_error(ValueError, "audio_features must be"):
        backends[1].embed_audio(torch.zeros(2, 100, 128))
