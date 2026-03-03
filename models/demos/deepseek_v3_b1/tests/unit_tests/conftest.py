# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for deepseek_v3_b1 unit tests (e.g. real HuggingFace weights)."""

import os
from pathlib import Path
from typing import Literal, NamedTuple

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.prepare_weights import SharedExpertWeights, prepare_shared_expert_weights

# If False, load from DEEPSEEK_V3_HF_MODEL when set; otherwise skip. If True, use deterministic random weights.
USE_RANDOM_WEIGHTS = True

# HF state dict shapes (out_features, in_features) for reference random weights. Align with test_prepare_weights.
_REF_HF_Q_B = (24576, 1536)
_REF_HF_O_PROJ = (7168, 16384)
_REF_HF_KV_B = (32768, 512)
_REF_HF_SHARED_GATE_UP = (2048, 7168)
_REF_K = 7168


def _reference_layer_state_dict(
    layer_idx: int,
    *,
    is_moe: bool,
    seed: int = 42,
    num_routed_experts: int = 4,
) -> dict[str, torch.Tensor]:
    """Build one layer state_dict in HF key convention with deterministic random weights (Generator)."""
    g = torch.Generator().manual_seed(seed)
    state = {
        f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": torch.randn(
            1536, _REF_K, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": torch.randn(
            *_REF_HF_Q_B, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(
            576, _REF_K, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": torch.randn(
            *_REF_HF_KV_B, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": torch.randn(
            *_REF_HF_O_PROJ, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.input_layernorm.weight": torch.randn(_REF_K, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": torch.randn(
            1536, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": torch.randn(
            512, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": torch.randn(
            _REF_K, generator=g, dtype=torch.bfloat16
        ),
    }
    if is_moe:
        state[f"model.layers.{layer_idx}.mlp.gate.weight"] = torch.randn(256, _REF_K, generator=g, dtype=torch.bfloat16)
        state[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = torch.randn(
            256, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            *_REF_HF_SHARED_GATE_UP, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(
            *_REF_HF_SHARED_GATE_UP, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.randn(
            7168, _REF_HF_SHARED_GATE_UP[0], generator=g, dtype=torch.bfloat16
        )
        # Expert weights: deterministic per-expert seeds (gate_seed, up_seed, down_seed) for golden parity
        gate_seed = seed + 0
        up_seed = seed + 256
        down_seed = seed + 512
        for e in range(num_routed_experts):
            g_gate = torch.Generator().manual_seed(gate_seed + e)
            g_up = torch.Generator().manual_seed(up_seed + e)
            g_down = torch.Generator().manual_seed(down_seed + e)
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                2048, _REF_K, generator=g_gate, dtype=torch.bfloat16
            ).clamp(-2, 2)
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                2048, _REF_K, generator=g_up, dtype=torch.bfloat16
            ).clamp(-2, 2)
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                7168, 2048, generator=g_down, dtype=torch.bfloat16
            ).clamp(-2, 2)
    else:
        state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.randn(
            18432, _REF_K, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.randn(
            18432, _REF_K, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.randn(
            _REF_K, 18432, generator=g, dtype=torch.bfloat16
        )
    return state


def _add_reference_global_weights(state: dict[str, torch.Tensor], seed: int = 42) -> None:
    """Add embedding, final norm, and lm_head to state (in place). Deterministic via Generator."""
    g = torch.Generator().manual_seed(seed)
    state["model.embed_tokens.weight"] = torch.randn(129280, _REF_K, generator=g, dtype=torch.bfloat16)
    state["model.norm.weight"] = torch.randn(_REF_K, generator=g, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn(129280, _REF_K, generator=g, dtype=torch.bfloat16)


def _resolve_hf_model_path() -> Path:
    """Resolve HuggingFace model path from DEEPSEEK_V3_HF_MODEL; skip if unset or invalid."""
    raw = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if not raw or not raw.strip():
        pytest.skip("DEEPSEEK_V3_HF_MODEL is not set; use random_weights=True or set env for real-weights tests")
    path = Path(raw.strip()).resolve()
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip(f"model.safetensors.index.json not found at {index_path}")
    return path


def _state_dict_key(layer_idx: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_idx}."""
    return f"model.layers.{layer_idx}.{suffix}"


class SharedExpertWeightBundle(NamedTuple):
    """Bundle of prepared shared-expert weights on device and raw torch tensors for golden."""

    weights: SharedExpertWeights
    torch_gate: torch.Tensor
    torch_up: torch.Tensor
    torch_down: torch.Tensor


@pytest.fixture(scope="session")
def get_reference_model_state_dict():
    """Return a callable that yields a reference state_dict in HF key convention.

    When random_weights=True (or USE_RANDOM_WEIGHTS=True): returns a dict of deterministic
    random weights. When random_weights=False: loads from DEEPSEEK_V3_HF_MODEL via
    LazyStateDict (read-only; tests never mutate it).

    Usage:
        get = get_reference_model_state_dict
        state = get(layer_idx=0, is_moe=True, num_routed_experts=256, include_global=False)
    """

    def get(
        layer_idx: int = 0,
        *,
        is_moe: bool = True,
        seed: int = 42,
        include_global: bool = False,
        num_routed_experts: int = 4,
        random_weights: bool = USE_RANDOM_WEIGHTS,
    ):
        if random_weights:
            state = _reference_layer_state_dict(
                layer_idx,
                is_moe=is_moe,
                seed=seed,
                num_routed_experts=num_routed_experts,
            )
            if include_global:
                _add_reference_global_weights(state, seed=seed)
            return state
        path = _resolve_hf_model_path()
        return LazyStateDict(path)

    return get


@pytest.fixture(scope="session")
def get_model_decoder_weight(hf_state_dict):
    """Return a callable to load model decoder weights onto a mesh device by layer and weight name.

    Weights are always prepared and moved to the given mesh_device.

    Usage:
        get = get_model_decoder_weight
        bundle = get(layer_idx=3, weight_name="shared_expert", mesh_device=submesh)

    Supported weight_name: "shared_expert".
    """
    state_dict = hf_state_dict

    def get(
        layer_idx: int,
        weight_name: Literal["shared_expert"],
        mesh_device: ttnn.MeshDevice,
    ):
        if weight_name == "shared_expert":
            torch_gate = state_dict[_state_dict_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T.contiguous()
            torch_up = state_dict[_state_dict_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T.contiguous()
            torch_down = state_dict[_state_dict_key(layer_idx, "mlp.shared_experts.down_proj.weight")].T.contiguous()
            bdw = BlitzDecodeWeights(mesh_device)
            weights = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=True, move_to_device=True)
            return SharedExpertWeightBundle(
                weights=weights,
                torch_gate=torch_gate,
                torch_up=torch_up,
                torch_down=torch_down,
            )
        raise ValueError(f"Unknown weight_name: {weight_name!r}")

    return get


@pytest.fixture(scope="session")
def hf_model_path():
    """Path to HuggingFace model directory (model.safetensors.index.json).

    Skips when DEEPSEEK_V3_HF_MODEL is not set or when the index file is missing.
    """
    raw = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if not raw or not raw.strip():
        pytest.skip("DEEPSEEK_V3_HF_MODEL is not set; skip real-weights tests")
    path = Path(raw.strip()).resolve()
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip(f"model.safetensors.index.json not found at {index_path}")
    return path


@pytest.fixture(scope="session")
def hf_state_dict(hf_model_path):
    """Session-scoped LazyStateDict over the HuggingFace model for real-weights tests."""
    return LazyStateDict(hf_model_path)


@pytest.fixture(scope="session")
def cache_path():
    """Path to pre-generated weight cache (layer_XXX/ with manifest + tensorbin).

    Skips when DEEPSEEK_V3_CACHE_PATH is not set or when the directory is missing.
    """
    raw = os.getenv("DEEPSEEK_V3_CACHE_PATH")
    if not raw or not raw.strip():
        pytest.skip("DEEPSEEK_V3_CACHE_PATH is not set")
    path = Path(raw.strip()).resolve()
    if not path.is_dir():
        pytest.skip(f"Cache directory not found: {path}")
    return path
