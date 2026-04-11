# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test factory and helpers for Gemma4 unit tests.

Uses HF_MODEL env var to determine which model variant to test against.
All HF reference configs and layers are created from the real checkpoint.
"""

import os

import pytest
import torch

import ttnn

from ..config import MeshConfig, ModeConfig
from ..tt.model_config import Gemma4ModelArgs

_DEFAULT_MODEL_PATH = "/proj_sw/user_dev/gemma4/gemma-4-26B-A4B-it"


def _get_model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", _DEFAULT_MODEL_PATH)


def is_moe_model():
    """Check if the current model has MoE enabled."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = getattr(config, "text_config", config)
    return getattr(tc, "enable_moe_block", False)


skip_if_not_moe = pytest.mark.skipif(not is_moe_model(), reason="Model does not use MoE")


def _needs_multi_device():
    """Check if model is too large for single-device tests (hidden_size > 4096)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = getattr(config, "text_config", config)
    return tc.hidden_size > 4096


skip_if_needs_multi_device = pytest.mark.skipif(
    _needs_multi_device(), reason="Model too large for single device (use -k 1x8 for multi-device tests)"
)


class TestFactory:
    """Common test setup for Gemma4 unit tests."""

    BATCH_SEQ_CONFIGS = [
        (1, 1),  # Decode: single token
        (1, 128),  # Prefill: short sequence
    ]

    @staticmethod
    def create_hf_config():
        """Create Gemma4ModelArgs from the real model checkpoint (HF_MODEL env var)."""
        from transformers import AutoConfig

        model_path = _get_model_path()
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return Gemma4ModelArgs.from_hf_config(hf_config)

    @staticmethod
    def create_mesh_config(mesh_shape=(1, 1)):
        """Create a single-device MeshConfig for testing."""
        return MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1]))

    @staticmethod
    def create_random_state_dict(hf_config, prefix=""):
        """Generate random state dict for a given config and module prefix."""
        return {}

    @staticmethod
    def create_hf_text_config(num_experts=None, top_k=None):
        """Create HF Gemma4TextConfig from real model checkpoint.

        Optionally override num_experts/top_k for faster testing.
        """
        from transformers import AutoConfig

        model_path = _get_model_path()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tc = config.text_config
        if num_experts is not None:
            tc.num_experts = num_experts
        if top_k is not None:
            tc.top_k_experts = top_k
        tc._attn_implementation = "eager"
        return tc

    @staticmethod
    def create_hf_reference_layer(hf_text_config, layer_idx=0):
        """Create HF Gemma4TextDecoderLayer with random weights."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer as HFLayer

        hf_layer = HFLayer(hf_text_config, layer_idx=layer_idx)
        with torch.no_grad():
            for name, param in hf_layer.named_parameters():
                if any(k in name for k in ["router", "experts"]):
                    if "scale" in name:
                        param.data.fill_(1.0)
                    else:
                        param.data.normal_(0, 0.02)
            hf_layer.layer_scalar.fill_(1.0)
        hf_layer.eval()
        return hf_layer

    @staticmethod
    def create_hf_rope(hf_text_config, seq_len, layer_idx):
        """Create HF RoPE position embeddings (cos, sin) for torch reference."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        x_dummy = torch.randn(1, seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(seq_len).unsqueeze(0)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        return cos, sin

    @staticmethod
    def create_tt_rope_cache(device, hf_text_config, max_seq_len, layer_idx):
        """Create HF-format cos/sin cache on TT device using HF Gemma4TextRotaryEmbedding.

        Returns (cos_cache, sin_cache) each [1, 1, max_seq_len, head_dim] on device.
        Matches exactly what HF produces (including identity padding for partial RoPE).
        """
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        x_dummy = torch.randn(1, max_seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(max_seq_len).unsqueeze(0)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        # cos, sin: [1, max_seq_len, head_dim] -> [1, 1, max_seq_len, head_dim]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        is_mesh = hasattr(device, "shape")
        cos_tt = ttnn.from_torch(
            cos,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        )
        sin_tt = ttnn.from_torch(
            sin,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        )
        return cos_tt, sin_tt


def compare_tensors(tt_tensor, torch_tensor, mesh_device=None, pcc_threshold=0.99):
    """Compare TT and torch tensors using PCC. Logs the PCC value."""
    from loguru import logger

    from models.common.utility_functions import comp_pcc

    if isinstance(tt_tensor, torch.Tensor):
        tt_torch = tt_tensor
    else:
        tt_torch = ttnn.to_torch(tt_tensor)

    passing, pcc_value = comp_pcc(torch_tensor, tt_torch, pcc_threshold)
    status = "PASS" if passing else "FAIL"
    logger.info(f"PCC check: {pcc_value} (threshold={pcc_threshold}) [{status}]")
    return passing, pcc_value


def parametrize_batch_seq(configs=None, ids=None):
    """Parametrize test with batch/seq combinations."""
    configs = configs or [(1, 1), (1, 128)]
    ids = ids or ["decode" if seq_len == 1 else f"prefill_{seq_len}" for _, seq_len in configs]
    return pytest.mark.parametrize("batch_size, seq_len", configs, ids=ids)


def parametrize_mesh_with_fabric(mesh_shapes=None):
    """Universal mesh parametrization with FABRIC_1D.

    Generates mesh_device + device_params parametrization for tests at any TP factor.
    Only includes mesh shapes that fit on the current system.

    Default shapes: (1,1) single card, (1,2) N300, (1,8) T3K.

    Usage:
        @parametrize_mesh_with_fabric()           # default: all shapes that fit
        @parametrize_mesh_with_fabric([(1,8)])     # explicit shapes

        pytest -k "1x1"   # single card (TP=1)
        pytest -k "1x2"   # N300 (TP=2)
        pytest -k "1x8"   # T3K (TP=8)
    """
    num_devices = ttnn.get_num_devices()

    if mesh_shapes is None:
        all_shapes = [(1, 1), (1, 2), (1, 8)]
        mesh_shapes = [s for s in all_shapes if s[0] * s[1] <= num_devices]

    if not mesh_shapes:
        mesh_shapes = [pytest.param((1, 1), marks=pytest.mark.skip(reason="Not enough devices"))]

    mesh_params = [pytest.param(s, id=f"{s[0]}x{s[1]}") for s in mesh_shapes]
    fabric_params = [pytest.param({"fabric_config": ttnn.FabricConfig.FABRIC_1D})]

    def decorator(func):
        func = pytest.mark.parametrize("mesh_device", mesh_params, indirect=True)(func)
        func = pytest.mark.parametrize("device_params", fabric_params, indirect=True)(func)
        return func

    return decorator
