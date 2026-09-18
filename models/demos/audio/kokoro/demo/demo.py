# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kokoro-82M text-to-speech demo (single-chip p150).

This demo uses the hybrid path: the plbert encoder on device (TTNN
``OptimizedDecoder``) with the prosody + ISTFTNet vocoder stages on the host (torch),
via the reference ``kokoro`` package with ``KModel.bert`` swapped for the TT encoder.
It mirrors the serving runner in tt-inference-server
(``tt-media-server/tt_model_runners/kokoro_runner.py``). For the fully-on-device path
(plbert + prosody + text encoder + ISTFTNet vocoder all in TTNN) see
``KokoroDevicePipeline.synthesize_device`` in ``tt/device_pipeline.py``.

Requires a Blackhole p150 and the host deps in ``requirements.txt`` plus
``espeak-ng``. Grapheme-to-phoneme uses ``misaki.espeak.EspeakFallback`` so that
spaCy (numpy-2 ABI) is never imported into the numpy-1.26 TTNN process.

Run:
    pytest --disable-warnings models/demos/audio/kokoro/demo/demo.py::test_demo
"""
import json
import sys
import types

import numpy as np
import pytest
import torch

import ttnn
from models.demos.audio.kokoro.tt.optimized_decoder import OptimizedDecoder

# Stub spaCy before any ``import kokoro`` (misaki.en imports spacy, whose numpy-2
# ABI conflicts with the numpy 1.26 TTNN is built against). We only use the
# espeak G2P path, which never touches spaCy.
if "spacy" not in sys.modules:
    sys.modules["spacy"] = types.ModuleType("spacy")

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000  # Kokoro / ISTFTNet output rate
DEFAULT_VOICE = "af_heart"
MAX_PHONEME_TOKENS = 510  # plbert context 512, minus bos/eos


class _Utt:
    """Minimal token object exposing ``.text`` for EspeakFallback."""

    def __init__(self, text: str):
        self.text = text


class _TTBert(torch.nn.Module):
    """Drop-in replacement for ``KModel.bert`` (``CustomAlbert``).

    ``forward`` returns the same ``last_hidden_state`` the stock encoder would,
    computed on device by ``encode``.
    """

    def __init__(self, encode):
        super().__init__()
        self._encode = encode

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask=None):
        return self._encode(input_ids)


def _build_single_chip_encode(mesh_device):
    """Build the single-chip TT plbert encoder; return ``encode(ids) -> [1,S,768]``."""
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    decoder = OptimizedDecoder.from_state_dict(sd, hf_config=config, mesh_device=mesh_device)

    def encode(ids):
        prep = OptimizedDecoder.prepare_inputs(ids, mesh_device, attention_mask=None)
        out = decoder.prefill_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )
        return ttnn.to_torch(out)[:, : prep["seq_len"], :].float()

    return encode


@pytest.mark.parametrize(
    "text",
    ["Kokoro is a lightweight text to speech model running on Tenstorrent hardware."],
)
def test_demo(device, text, tmp_path):
    from huggingface_hub import hf_hub_download
    from kokoro.model import KModel
    from misaki.espeak import EspeakFallback

    encode = _build_single_chip_encode(device)

    kmodel = KModel(repo_id=MODEL_ID).eval()
    kmodel.bert = _TTBert(encode)  # TT plbert in the loop
    g2p = EspeakFallback(british=False)
    pack = torch.load(hf_hub_download(MODEL_ID, f"voices/{DEFAULT_VOICE}.pt"), weights_only=True)

    phonemes, _ = g2p(_Utt(text))
    ids = [i for i in (kmodel.vocab.get(p) for p in phonemes) if i is not None][:MAX_PHONEME_TOKENS]
    assert ids, "no pronounceable phonemes produced from input text"

    ref_s = pack[max(0, min(len(ids) - 1, pack.shape[0] - 1))]
    audio = kmodel(phonemes[: len(ids)], ref_s, 1.0).detach().cpu().numpy().astype(np.float32)
    assert audio.size > 0

    out_path = tmp_path / "kokoro_demo.wav"
    import soundfile as sf

    sf.write(str(out_path), audio, SAMPLE_RATE)
    print(f"Wrote {audio.size / SAMPLE_RATE:.2f}s of audio to {out_path}")
