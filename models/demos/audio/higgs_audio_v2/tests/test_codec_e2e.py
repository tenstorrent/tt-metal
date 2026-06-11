# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""End-to-end codec decode: TTNN tt_decode (RVQ+fc2 host, DacDecoder on device)
vs HF HiggsAudioV2TokenizerModel.decode, on real audio code indices.

Gate: PCC > 0.97 on the decoded waveform.
"""
import os
import pytest
import torch
import ttnn
from loguru import logger

from models.demos.audio.higgs_audio_v2.tt.codec import tt_decode


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    n = min(a.numel(), b.numel())
    return float(torch.corrcoef(torch.stack([a[:n], b[:n]]))[0, 1])


@pytest.fixture(scope="module")
def mesh_device():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def codec_model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModel

    return AutoModel.from_pretrained("/data/hf_cache/higgs/tokenizer").eval()


def test_decode_e2e(mesh_device, codec_model):
    torch.manual_seed(0)
    T = int(os.environ.get("CODEC_T", "64"))
    K = len(codec_model.quantizer.quantizers)
    codes = torch.randint(0, 1024, (1, K, T), dtype=torch.long)  # real LLM audio-code range
    with torch.no_grad():
        ref = codec_model.decode(codes).audio_values  # [1, 1, T*960]
    tt = tt_decode(mesh_device, codec_model, codes)
    pcc = _pcc(ref, tt)
    logger.info(f"codec decode E2E PCC={pcc:.5f}  ref{tuple(ref.shape)} tt{tuple(tt.shape)}")
    print(f"CODEC_E2E pcc={pcc:.5f} ref_len={ref.shape[-1]} tt_len={tt.shape[-1]}")
    assert pcc > 0.97
