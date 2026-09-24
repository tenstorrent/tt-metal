# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI-only check of PR #49699's claims on the real KoeAI checkpoint (not part of the PR).

Loads ``KoeAI/llvc`` ``G_500000.pth`` from the Hugging Face cache (pre-staged on
/mnt/MLPerf), the KoeAI ``config.json`` (fetched from GitHub, embedded fallback), and a
KoeAI ``test_wavs`` sample (GitHub, synthetic fallback). Asserts TTNN-vs-reference PCC
and the streaming RTF / latency targets the PR body quotes for the full-size model.
"""
import json
import os
import struct
import urllib.request
import wave

import pytest
import torch
from loguru import logger

from models.demos.llvc.tests.pcc.test_llvc import compute_pcc
from models.demos.llvc.tt.model import LLVCModel
from models.demos.llvc.tt.state_io import load_llvc_config_and_model

CONFIG_URL = "https://raw.githubusercontent.com/KoeAI/LLVC/main/experiments/llvc/config.json"
WAV_URL = "https://raw.githubusercontent.com/KoeAI/LLVC/main/test_wavs/1919-142785-0000.wav"
HF_REPO, HF_FILE = "KoeAI/llvc", "models/checkpoints/llvc/G_500000.pth"
# experiments/llvc/config.json as of KoeAI/LLVC main on 2026-09-24.
FALLBACK_CONFIG = {
    "model_params": {
        "label_len": 1, "L": 16, "enc_dim": 512, "num_enc_layers": 8, "dec_dim": 256, "num_dec_layers": 1,
        "dec_buf_len": 13, "dec_chunk_size": 13, "out_buf_len": 4, "use_pos_enc": True, "decoder_dropout": 0.1,
        "convnet_config": {
            "convnet_prenet": True, "out_channels": [1] * 12, "kernel_sizes": [3] * 12, "dilations": [1] * 12,
            "dropout": 0.5, "combine_residuals": None, "skip_connection": "add", "use_residual_blocks": True,
        },
    },
    "data": {"sr": 16000},
}
TARGET_PCC, TARGET_RTF, TARGET_LATENCY_MS = 0.99, 0.3, 100.0


def _fetch(url: str, dest: str) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return os.path.getsize(dest) > 0
    except Exception as e:  # noqa: BLE001
        logger.warning("could not fetch {}: {}", url, e)
        return False


def _checkpoint_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, HF_FILE)


def _config_path(tmp_path) -> str:
    p = str(tmp_path / "config.json")
    if not _fetch(CONFIG_URL, p):
        with open(p, "w") as f:
            json.dump(FALLBACK_CONFIG, f)
        logger.warning("using the embedded copy of KoeAI config.json")
    return p


def _wav(tmp_path, sample_rate: int) -> torch.Tensor:
    p = str(tmp_path / "in.wav")
    if _fetch(WAV_URL, p):
        with wave.open(p, "rb") as w:
            sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        assert width == 2, f"expected 16-bit PCM, got width {width}"
        pcm = torch.tensor(struct.unpack(f"<{n * ch}h", raw), dtype=torch.float32) / 32768.0
        audio = pcm.view(n, ch).mean(dim=1)
        if sr != sample_rate:
            audio = torch.nn.functional.interpolate(
                audio[None, None], scale_factor=sample_rate / sr, mode="linear", align_corners=False
            )[0, 0]
        logger.info("input: KoeAI test wav, {} samples at {} Hz", audio.numel(), sample_rate)
    else:
        t = torch.arange(2 * sample_rate, dtype=torch.float32) / sample_rate
        audio = 0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.1 * torch.sin(2 * torch.pi * 660 * t) + 0.02 * torch.randn_like(t)
        logger.warning("input: synthetic 2 s tone (GitHub unreachable)")
    return audio


def _load(device, tmp_path):
    torch.manual_seed(0)
    config, reference = load_llvc_config_and_model(_config_path(tmp_path), _checkpoint_path())
    model = LLVCModel(config, reference, device=device)
    return config, reference, model


def test_checkpoint_pcc_vs_reference(device, tmp_path):
    config, reference, model = _load(device, tmp_path)
    wav = _wav(tmp_path, config.sample_rate)
    chunk = config.dec_chunk_size * config.L
    wav = wav[: (wav.numel() // chunk) * chunk]
    with torch.no_grad():
        ref_out = reference(wav[None, None])
    tt_out = model(wav)
    assert tt_out.shape == ref_out.shape, f"{tuple(tt_out.shape)} vs {tuple(ref_out.shape)}"
    pcc = compute_pcc(tt_out, ref_out)
    logger.info("full checkpoint: TTNN vs PyTorch reference PCC {:.6f} over {} samples", pcc, wav.numel())
    assert pcc > TARGET_PCC, f"PCC {pcc:.4f} < {TARGET_PCC} (PR claims 0.9997)"


@pytest.mark.parametrize("chunk_factor", [1, 2])
def test_checkpoint_streaming_rtf(device, tmp_path, chunk_factor):
    config, _reference, model = _load(device, tmp_path)
    wav = _wav(tmp_path, config.sample_rate)
    _ = model.stream(wav, chunk_factor=chunk_factor)  # warm-up: compile + trace capture
    out, m = model.stream(wav, chunk_factor=chunk_factor)
    logger.info(
        "full checkpoint chunk_factor={}: e2e_RTF={:.3f} e2e_latency={:.2f}ms device_RTF={:.3f} device_latency={:.2f}ms",
        chunk_factor, m.rtf, m.latency_ms, m.device_rtf, m.device_latency_ms,
    )
    assert torch.isfinite(out).all()
    if chunk_factor == 2:  # the configuration the PR body quotes (RTF 0.217, 33.6 ms)
        assert m.rtf < TARGET_RTF, f"e2e RTF {m.rtf:.3f} >= {TARGET_RTF}"
        assert m.latency_ms < TARGET_LATENCY_MS, f"e2e latency {m.latency_ms:.2f} ms >= {TARGET_LATENCY_MS}"
