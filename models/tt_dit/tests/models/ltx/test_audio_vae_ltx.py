# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Parity test: on-device LTX-2 audio VAE *decoder* vs the CPU reference.

Compares the TTNN audio decoder (latent -> log-mel spectrogram) against the
reference torch ``AudioDecoder`` via PCC. The vocoder is *not* exercised here —
only the conv/attention decoder stack that the port covers.

Requires the LTX-2.3 checkpoint (audio VAE weights live in the same file). The
test skips when the checkpoint is unavailable.

NOTE: This is the gate for iterating the port on hardware. The conv/groupnorm/
memory configs in ``audio_vae_ltx.py`` likely need tuning before PCC passes; a
failure here is expected until that work is done.
"""

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.audio_vae_ltx import TtAudioDecoder

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")


def _default_checkpoint() -> str:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors")
    if os.path.exists(local):
        return local
    return "Lightricks/LTX-2.3:ltx-2.3-22b-distilled-1.1.safetensors"


def _compute_pcc(tt_out: torch.Tensor, ref_out: torch.Tensor) -> float:
    a = tt_out.flatten().float()
    b = ref_out.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((1, 1), {"l1_small_size": 32768})],
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_vae_decoder_pcc(mesh_device: ttnn.MeshDevice):
    """TTNN audio VAE decoder must match the CPU reference decoder (PCC gate)."""
    ckpt = _default_checkpoint()
    if not os.path.exists(ckpt):
        pytest.skip(f"LTX checkpoint not found at {ckpt}")

    from ltx_pipelines.utils.blocks import AudioDecoder as AudioDecoderBlock

    block = AudioDecoderBlock(checkpoint_path=ckpt, dtype=torch.float32, device=torch.device("cpu"))
    ref_decoder = block._decoder_builder.build(device=torch.device("cpu"), dtype=torch.float32).eval()

    # Synthetic audio latent matching the pipeline layout: (1, N, 128) -> (1, 8, N, 16).
    torch.manual_seed(0)
    audio_N = 64
    latent = torch.randn(1, audio_N, 128)
    audio_spatial = latent.reshape(1, audio_N, 8, 16).permute(0, 2, 1, 3).float()

    with torch.no_grad():
        ref_mel = ref_decoder(audio_spatial)

    tt_decoder = TtAudioDecoder.from_reference(ref_decoder, mesh_device=mesh_device)
    with torch.no_grad():
        tt_mel = tt_decoder(audio_spatial)

    assert tt_mel.shape == ref_mel.shape, f"shape mismatch: tt={tt_mel.shape} ref={ref_mel.shape}"
    assert torch.isfinite(tt_mel).all(), "TT mel has NaN/Inf"

    pcc = _compute_pcc(tt_mel, ref_mel)
    logger.info(f"Audio VAE decoder PCC: {pcc:.6f} (shape {tuple(tt_mel.shape)})")
    assert pcc > 0.99, f"Audio VAE decoder PCC {pcc:.6f} below threshold 0.99"
    logger.info("PASSED: on-device audio VAE decoder parity")
