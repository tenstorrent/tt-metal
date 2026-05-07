# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Dict

import pytest
import torch
from transformers import AutoConfig

import ttnn

from ..config import MeshConfig
from ..tt.ccl import CCLManager
from ..tt.model_config import ModelArgs


class TestFactory:
    # Common test configurations
    MESH_SHAPES = {"4x8": (4, 8), "1x8": (1, 8), "4x4": (4, 4), "2x4": (2, 4), "1x1": (1, 1)}

    BATCH_SEQ_CONFIGS = [
        (1, 1),  # Single token
        (1, 32),  # Small sequence
        (1, 128),  # Medium sequence
    ]

    @staticmethod
    def setup_test(mesh_device, use_real_weights=True, dtype=ttnn.bfloat8_b):
        """Universal test setup - replaces all the duplicated setup code"""

        # Use mesh_device as-is (already created by conftest.py fixture)
        mesh_shape = mesh_device.shape

        # Setup ModelArgs (no import-time loading)
        model_args = ModelArgs(mesh_device=mesh_device, dummy_weights=not use_real_weights)

        # Setup mesh config using actual mesh shape
        from models.demos.gpt_oss.config import ModeConfig

        mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0]))

        # Setup CCL
        ccl_manager = CCLManager(mesh_device, num_links=4 if mesh_shape[0] > 1 else 1)

        config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)

        # Only cache tensors to disk when using real weights; with dummy (random) weights
        # there is nothing to persist and the model_path may not be a writable local directory.
        tensor_cache_path = model_args.weight_cache_path(dtype) if use_real_weights else None

        return {
            "mesh_device": mesh_device,
            "model_args": model_args,
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            "config": config,
            "dtype": dtype,
            "tensor_cache_path": tensor_cache_path,
        }

    @staticmethod
    def _generate_dummy_state_dict(config) -> Dict[str, torch.Tensor]:
        """Generate dummy state dict - scales with any model size"""

        # Extract dimensions from config (works for any model size)
        num_experts = getattr(config, "num_local_experts", 128)
        hidden_size = getattr(config, "hidden_size", 2048)
        intermediate_size = getattr(config, "intermediate_size", 5632)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", 32)
        head_dim = getattr(config, "head_dim", 64)
        vocab_size = getattr(config, "vocab_size", 201088)

        return {
            # Expert weights - scales with model size
            "gate_up_proj": torch.randn(num_experts, hidden_size, 2 * intermediate_size),
            "gate_up_proj_bias": torch.randn(num_experts, 2 * intermediate_size),
            "down_proj": torch.randn(num_experts, intermediate_size, hidden_size),
            "down_proj_bias": torch.randn(num_experts, hidden_size),
            # Router weights - scales with experts and hidden size
            "router": {
                "weight": torch.randn(num_experts, hidden_size),
                "bias": torch.randn(num_experts),
            },
            # Attention weights - scales with hidden dimensions
            "q_proj": {
                "weight": torch.randn(hidden_size, hidden_size),
                "bias": torch.randn(hidden_size),
            },
            "k_proj": {
                "weight": torch.randn(hidden_size, num_kv_heads * head_dim),
                "bias": torch.randn(num_kv_heads * head_dim),
            },
            "v_proj": {
                "weight": torch.randn(hidden_size, num_kv_heads * head_dim),
                "bias": torch.randn(num_kv_heads * head_dim),
            },
            "o_proj": {
                "weight": torch.randn(hidden_size, hidden_size),
                "bias": torch.randn(hidden_size),
            },
            # Norm weights - scales with hidden size
            "weight": torch.ones(hidden_size),
            # KV cache and attention - scales with num heads
            "sinks": torch.randn(num_attention_heads),
            # Additional embeddings for full model tests
            "embed_tokens": {"weight": torch.randn(vocab_size, hidden_size)},
            "lm_head": {"weight": torch.randn(vocab_size, hidden_size)},
        }


def parametrize_mesh_with_fabric(mesh_shapes=None):
    """Mesh parametrization that works across all SKUs (N150, N300, T3K, Galaxy).

    Opens the mesh *directly* at the requested shape — tests should not
    create submeshes on top of this fixture. Pass the shapes the test
    wants to exercise; default is every shape that fits on the system.

    Pairs each shape with the appropriate fabric config:
      (1, 1) → no fabric   (single device; fabric needs all system devices open)
      (1, 8) → FABRIC_1D_RING
      (4, 8) → FABRIC_1D_RING

    In CI (CI=true env var) only the largest fitting shape is kept so that one
    yaml entry runs correctly on every SKU without per-SKU ``-k`` filters.

    Usage:
        @parametrize_mesh_with_fabric()                  # default: all shapes that fit
        @parametrize_mesh_with_fabric([(1, 1), (4, 8)])  # explicit shapes

        pytest -k "1x8"   # T3K only           (manual / non-CI)
    """
    # Ordered largest → smallest; each entry is (mesh_shape, device_params_dict).
    _all_configs = [
        ((4, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100000000}),
        ((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100000000}),
        ((1, 1), {"fabric_config": None}),
    ]
    _config_by_shape = dict(_all_configs)

    num_devices = ttnn.get_num_devices()
    candidates = [s for s, _ in _all_configs] if mesh_shapes is None else list(mesh_shapes)
    fitting = [s for s in candidates if s[0] * s[1] <= num_devices]

    # CI: keep only the largest fitting shape so one yaml entry covers all SKUs.
    if os.getenv("CI") == "true" and len(fitting) > 1:
        fitting = [max(fitting, key=lambda s: s[0] * s[1])]

    if not fitting:
        # No fitting shape on this host — emit a single skipped entry so the
        # test still collects (instead of pytest erroring on empty parametrize).
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None},
                id="1x1",
                marks=pytest.mark.skip(reason="No requested mesh shape fits the available device count"),
            )
        ]
    else:
        params = [
            pytest.param(s, _config_by_shape.get(s, {"fabric_config": None}), id=f"{s[0]}x{s[1]}") for s in fitting
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator


def parametrize_batch_seq(configs=None, ids=None):
    """Universal batch/seq parametrization"""
    configs = configs or [(1, 1), (1, 32)]
    ids = ids or [
        f"prefill_{seq_len//1024 if seq_len > 1024 else seq_len}" + ("k" if seq_len > 1024 else "")
        if seq_len > 1
        else "decode_mode"
        for batch_size, seq_len in configs
    ]
    return pytest.mark.parametrize("batch_size, seq_len", configs, ids=ids)


def parametrize_weights(use_real=False):
    """Universal weight parametrization"""
    return pytest.mark.parametrize("use_real_weights", [use_real], ids=["real" if use_real else "random"])


# Test helper functions
def compare_tensors(tt_tensor, torch_tensor, mesh_device, pcc_threshold=0.99):
    """Universal tensor comparison - handles both TT tensors and already-converted torch tensors"""
    from models.common.utility_functions import comp_pcc

    # Check if tt_tensor is already a torch tensor
    if isinstance(tt_tensor, torch.Tensor):
        # Already converted, use directly
        tt_torch = tt_tensor
    else:
        # Convert TT tensor to torch
        mesh_shape = mesh_device.shape
        tt_torch = ttnn.to_torch(
            tt_tensor, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1))
        )

    return comp_pcc(torch_tensor, tt_torch, pcc_threshold)
