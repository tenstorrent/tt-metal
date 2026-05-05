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
        # state_dict = TestFactory._generate_dummy_state_dict(config)

        return {
            "mesh_device": mesh_device,
            "model_args": model_args,
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            "config": config,
            # "state_dict": state_dict,
            "dtype": dtype,
            "tensor_cache_path": model_args.weight_cache_path(dtype),
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
    """Universal mesh + fabric parametrization for gpt_oss tests.

    Generates a paired ``(mesh_device, device_params)`` parametrize. Each
    case opens a mesh of the requested shape directly (no submesh carving
    in test bodies) and configures the appropriate fabric for that shape.
    Each parametrize case has a single id like ``1x1`` / ``1x2`` / ``1x4`` /
    ``1x8`` / ``4x8``, so ``pytest -k 1x2`` (or ``-k 4x8``) filters cleanly
    without dual-id confusion from a separate inner ``mesh_shape`` parametrize.

    Auto-filters to the shapes that fit on the current system. Default
    shapes: (1,1) single card, (1,2) 2xP150, (1,4) QuietBox 2 (2xP300),
    (1,8) LoudBox / T3K, (4,8) Galaxy. Pass an explicit ``mesh_shapes`` list
    to override (useful for tests that only make sense at one TP factor).

    Fabric: ``(1,1)`` disables fabric (no inter-chip topology to ring
    around). Multi-device shapes use ``FABRIC_1D_RING`` — gpt_oss's CCL
    operations (reduce_scatter, all_gather, all_reduce) all use the ring
    topology.

    When ``CI=true`` is set in the environment, only the largest mesh shape
    that fits on the current system is parametrized. This lets one yaml
    entry target multiple SKUs without per-SKU ``-k "1xN"`` filters: each
    runner picks the largest mesh its device count supports. Manual / non-CI
    invocations are unchanged (all-shapes-that-fit, ``-k`` available for fast
    iteration).

    Usage:
        @parametrize_mesh_with_fabric()              # all shapes that fit
        @parametrize_mesh_with_fabric([(1, 8)])      # 1x8 only

        pytest -k 1x1   # single card             (manual / non-CI)
        pytest -k 1x2   # 2xP150                  (manual / non-CI)
        pytest -k 1x4   # QuietBox 2 (2xP300)     (manual / non-CI)
        pytest -k 1x8   # LoudBox / T3K           (manual / non-CI)
        pytest -k 4x8   # Galaxy                  (manual / non-CI)
    """
    num_devices = ttnn.get_num_devices()
    if mesh_shapes is None:
        all_shapes = [(1, 1), (1, 2), (1, 4), (1, 8), (4, 8)]
        mesh_shapes = [s for s in all_shapes if s[0] * s[1] <= num_devices]
    else:
        # User-provided shapes: still filter to those that fit, so an explicit
        # mesh_shapes=[(1,8)] decorator gracefully skips on smaller systems.
        mesh_shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]

    # CI mode: pick only the largest fitting shape so that one yaml entry can
    # target multiple SKUs and let each runner select the appropriate mesh.
    if os.getenv("CI") == "true" and len(mesh_shapes) > 1:
        mesh_shapes = [max(mesh_shapes, key=lambda s: s[0] * s[1])]

    if not mesh_shapes:
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None, "trace_region_size": 100000000},
                id="1x1",
                marks=pytest.mark.skip(reason="No supported gpt_oss mesh shape fits on this system"),
            )
        ]
    else:
        params = [
            pytest.param(
                shape,
                {
                    "fabric_config": (None if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D_RING),
                    "trace_region_size": 100000000,
                },
                id=f"{shape[0]}x{shape[1]}",
            )
            for shape in mesh_shapes
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
