# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Dict

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0

from ..config import MeshConfig
from ..tt.ccl import CCLManager
from ..tt.model_config import ModelArgs


def _config_pretrained_id(model_path: str) -> str:
    """Prefer a local snapshot when it contains config.json; otherwise use the Hub id for known GPT-OSS dirs.

    If `model_path` exists as a directory but has no config.json (e.g. empty mount or weights-only tree),
    `transformers` falls through to Hub logic and mis-treats an absolute path as a repo id (HFValidationError).
    """
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path
    basename = Path(model_path).name
    known = {"gpt-oss-20b": "openai/gpt-oss-20b", "gpt-oss-120b": "openai/gpt-oss-120b"}
    return known.get(basename, model_path)


class TestFactory:
    # Common test configurations
    MESH_SHAPES = {"4x8": (4, 8), "1x8": (1, 8), "1x4": (1, 4), "4x4": (4, 4), "2x4": (2, 4), "1x1": (1, 1)}

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

        config = AutoConfig.from_pretrained(
            _config_pretrained_id(model_args.model_path),
            trust_remote_code=True,
        )
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


def parametrize_mesh_with_fabric():
    """Mesh + fabric for GPT-OSS demos/tests: WH 1×8 or 4×8, or BH T3K (Blackhole) 4- or 8-chip."""
    try:
        num_devices = ttnn.get_num_devices()
    except RuntimeError as e:
        err = str(e).lower()
        if "tlb" in err or "tt_tlb_alloc" in err or "hung process" in err:
            raise RuntimeError(
                "Device probe during test collection failed (TLB allocation). Usually another Python/pytest "
                "still holds the Tenstorrent device, or a prior run exited without releasing hardware. "
                "Try: pkill -9 -f 'pytest.*gpt_oss'; pkill -9 -f 'python.*tt-metal'; sleep 2; "
                "then tt-smi -r if your machine supports a device reset; retry."
            ) from e
        raise
    if num_devices == 32:
        mesh_params = [pytest.param((4, 8))]
    elif num_devices == 8 and is_blackhole():
        shape_env = os.environ.get("GPT_OSS_MESH_SHAPE", "1,8").strip()
        try:
            r_s, c_s = shape_env.split(",", maxsplit=1)
            r, c = int(r_s.strip()), int(c_s.strip())
        except ValueError as e:
            raise ValueError(f"GPT_OSS_MESH_SHAPE must be 'rows,cols' (got {shape_env!r})") from e
        if r * c != num_devices:
            raise ValueError(f"GPT_OSS_MESH_SHAPE {r},{c} does not multiply to device count {num_devices}")
        mesh_params = [pytest.param((r, c))]
    elif num_devices == 8 and is_wormhole_b0():
        mesh_params = [pytest.param((1, 8))]
    elif num_devices == 4 and is_blackhole():
        shape_env = os.environ.get("GPT_OSS_MESH_SHAPE", "1,4").strip()
        try:
            r_s, c_s = shape_env.split(",", maxsplit=1)
            r, c = int(r_s.strip()), int(c_s.strip())
        except ValueError as e:
            raise ValueError(f"GPT_OSS_MESH_SHAPE must be 'rows,cols' (got {shape_env!r})") from e
        if r * c != num_devices:
            raise ValueError(f"GPT_OSS_MESH_SHAPE {r},{c} does not multiply to device count {num_devices}")
        mesh_params = [pytest.param((r, c))]
    else:
        raise ValueError(
            f"Invalid number of devices for GPT-OSS mesh harness: {num_devices} "
            f"(supported: 8 or 32 Wormhole B0, or 4/8 Blackhole BH T3K). Arch={ttnn.get_arch_name()}"
        )
    trace_region_size = 30000000
    if is_blackhole() and num_devices in (4, 8):
        trace_region_size = 90000000
    fabric_params = [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": trace_region_size},
            id="fabric_1d_ring",
        ),
    ]

    # Return a single decorator that combines both parametrizations
    def decorator(func):
        func = pytest.mark.parametrize("mesh_device", mesh_params, indirect=True)(func)
        func = pytest.mark.parametrize("device_params", fabric_params, indirect=True)(func)
        return func

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
