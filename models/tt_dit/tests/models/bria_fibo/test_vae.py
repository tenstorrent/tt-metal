# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from huggingface_hub import snapshot_download

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, allow_patterns=["scheduler/*", "vae/*"], local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


def _load_ref_vae(dtype=None):
    import torch
    from diffusers import AutoencoderKLWan

    try:
        path = snapshot_download(FIBO_PATH, allow_patterns=["vae/*"], local_files_only=True)
        return AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype or torch.float32).eval()
    except Exception as e:
        pytest.skip(f"FIBO vae unavailable: {e}")


def test_fibo_vae_reference_config():
    m = _load_ref_vae()
    c = m.config
    assert c.z_dim == 48 and c.is_residual is True
    assert c.decoder_base_dim == 256 and c.base_dim == 160
    assert c.dim_mult == [1, 2, 4, 4] and c.out_channels == 12
    assert c.scale_factor_spatial == 16
