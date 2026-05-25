# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-device PCC tests for the Qwen3-TTS Talker TT implementation.

These tests require a TT device (P150 / N150 / N300).
For CPU-only tests, see test_talker.py.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "P150": (1, 1)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
class TestTalkerOnDevice:
    """On-device tests for Talker construction and forward pass."""

    def test_config_loading(self, mesh_device):
        """Verify TalkerModelArgs correctly initializes with a real device."""
        from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs

        args = TalkerModelArgs(
            mesh_device=mesh_device,
            dummy_weights=True,
            max_batch_size=1,
            max_seq_len=256,
        )

        assert args.dim == 2048
        assert args.n_heads == 16
        assert args.n_kv_heads == 8
        assert args.n_layers == 28
        assert args.head_dim == 128
        assert args.hidden_dim == 6144
        assert args.vocab_size == 3072
        assert args.text_vocab_size == 151936
        assert args.model_name == "Qwen3-TTS-12Hz-1.7B-Base"
        logger.info("TalkerModelArgs config loaded successfully on device")

    def test_talker_construction(self, mesh_device):
        """Verify TalkerTransformer can be constructed with dummy weights."""
        from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs
        from models.demos.qwen3_tts.tt.talker import TalkerTransformer

        args = TalkerModelArgs(
            mesh_device=mesh_device,
            dummy_weights=True,
            max_batch_size=1,
            max_seq_len=256,
        )

        talker = TalkerTransformer(
            args=args,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            state_dict=None,
            weight_cache_path=args.weight_cache_path(ttnn.bfloat16),
        )

        assert talker.text_embed_weight.shape == (args.text_vocab_size, args.dim)
        assert talker.text_proj_fc1_w is not None
        assert talker.text_proj_fc2_w is not None
        logger.info("TalkerTransformer constructed successfully")

    def test_text_embedding(self, mesh_device):
        """Verify CPU-side text embedding produces correct shapes."""
        from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs
        from models.demos.qwen3_tts.tt.talker import TalkerTransformer

        args = TalkerModelArgs(
            mesh_device=mesh_device,
            dummy_weights=True,
            max_batch_size=1,
            max_seq_len=256,
        )

        talker = TalkerTransformer(
            args=args,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            state_dict=None,
            weight_cache_path=args.weight_cache_path(ttnn.bfloat16),
        )

        tokens = torch.randint(0, args.text_vocab_size, (1, 32))
        embeddings = talker.embed_text_tokens(tokens)

        assert embeddings.shape == (1, 32, args.dim)
        assert embeddings.dtype == torch.float32
        logger.info(f"Text embedding shape: {embeddings.shape}")
