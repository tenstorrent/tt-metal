# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: on-device SigLIP image encoder (i2v) vs the host SiglipVisionModel.

Loads the real SigLIP-so400m encoder from the cached 480p_i2v checkpoint, runs the
TTSiglipImageEncoderAdapter (27 vision-transformer layers on the TT mesh, patch-embed
+ post-LN on host), and asserts `last_hidden_state` matches the host encoder.

    HF_HOME=~/.cache/huggingface pytest tests/pcc/test_siglip.py -s
"""
import glob
import os

import pytest
import torch

_I2V = "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_i2v"
_THRESHOLD = 0.95


def _image_encoder_dir():
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = glob.glob(os.path.join(hub, _I2V, "snapshots", "*", "image_encoder"))
    return snaps[0] if snaps else None


def test_siglip_on_device_pcc(device):
    enc_dir = _image_encoder_dir()
    if not enc_dir:
        pytest.skip(f"{_I2V} image_encoder not cached")

    from transformers import SiglipVisionModel

    from models.demos.hf_eager.hunyuanvideo_1_5.tt.siglip_encoder import TTSiglipImageEncoderAdapter, _pcc

    enc = SiglipVisionModel.from_pretrained(enc_dir, torch_dtype=torch.float32).eval()
    torch.manual_seed(0)
    px = torch.randn(1, 3, enc.config.image_size, enc.config.image_size)

    with torch.no_grad():
        ref = enc(pixel_values=px).last_hidden_state

    # verify_pcc=False here: this test owns the assertion (avoid the adapter's internal
    # one-time self-check double-running the host encoder).
    tt_enc = TTSiglipImageEncoderAdapter(enc, device, verify_pcc=False)
    out = tt_enc(pixel_values=px).last_hidden_state

    pcc = _pcc(ref, out)
    print(f"\nSigLIP on-device PCC vs host = {pcc:.6f}  (shape {tuple(out.shape)})", flush=True)
    assert pcc >= _THRESHOLD, f"PCC {pcc:.6f} < {_THRESHOLD}"
