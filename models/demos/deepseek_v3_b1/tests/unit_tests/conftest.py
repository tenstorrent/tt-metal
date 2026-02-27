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
