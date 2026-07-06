# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _load_ref_transformer(dtype=torch.bfloat16):
    try:
        from diffusers import BriaFiboTransformer2DModel
    except Exception:
        from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
    try:
        # When running offline, resolve the HF repo ID to its local cache path.
        fibo_path = FIBO_PATH
        if not os.path.isdir(fibo_path):
            from huggingface_hub import snapshot_download

            fibo_path = snapshot_download(fibo_path, allow_patterns=["transformer/*"], local_files_only=True)
        return BriaFiboTransformer2DModel.from_pretrained(fibo_path, subfolder="transformer", torch_dtype=dtype).eval()
    except Exception as e:
        pytest.skip(f"FIBO transformer unavailable: {e}")


def test_fibo_transformer_reference_config():
    m = _load_ref_transformer()
    c = m.config
    assert c.num_layers == 8 and c.num_single_layers == 38
    assert c.num_attention_heads == 24 and c.attention_head_dim == 128
    assert c.in_channels == 48 and c.joint_attention_dim == 4096
    assert c.axes_dims_rope == [16, 56, 56]
    assert len(m.caption_projection) == c.num_layers + c.num_single_layers  # 46
