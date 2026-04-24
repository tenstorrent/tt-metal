# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest


class TtSharedExpertMLP(LightweightModule):
    """Single-device TTNN SwiGLU path for DeepSeek V4 Flash shared experts."""

    def __init__(
        self,
        *,
        device,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        swiglu_limit: float = 0.0,
    ):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.swiglu_limit = float(swiglu_limit)
        self.w1 = _to_tt_linear_weight(w1, device=device, dtype=dtype, memory_config=memory_config)
        self.w2 = _to_tt_linear_weight(w2, device=device, dtype=dtype, memory_config=memory_config)
        self.w3 = _to_tt_linear_weight(w3, device=device, dtype=dtype, memory_config=memory_config)

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        device,
        layer: int = 0,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        swiglu_limit: float | None = None,
    ) -> "TtSharedExpertMLP":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        weights = _load_shared_expert_weights(preprocessed_path, manifest, layer)
        if swiglu_limit is None:
            swiglu_limit = float(manifest["config"].get("swiglu_limit", 0.0))
        return cls(
            device=device,
            w1=weights["w1"],
            w2=weights["w2"],
            w3=weights["w3"],
            dtype=dtype,
            memory_config=memory_config,
            swiglu_limit=swiglu_limit,
        )

    def forward(self, hidden_states):
        gate = ttnn.linear(hidden_states, self.w1, memory_config=self.memory_config)
        up = ttnn.linear(hidden_states, self.w3, memory_config=self.memory_config)
        if self.swiglu_limit > 0:
            gate = ttnn.clamp(gate, min=None, max=self.swiglu_limit, memory_config=self.memory_config)
            up = ttnn.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit, memory_config=self.memory_config)

        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=self.memory_config,
        )
        output = ttnn.linear(hidden, self.w2, memory_config=self.memory_config)
        return output


def _to_tt_linear_weight(
    weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    torch_weight = weight.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return ttnn.from_torch(
        torch_weight,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _load_shared_expert_weights(preprocessed_path: Path, manifest: dict, layer: int) -> dict[str, torch.Tensor]:
    keys = {projection: f"layers.{layer}.ffn.shared_experts.{projection}.weight" for projection in ("w1", "w2", "w3")}
    loaded: dict[str, torch.Tensor] = {}
    for artifact in manifest["artifacts"]["non_expert_safetensors"]:
        artifact_path = preprocessed_path / artifact
        with safe_open(artifact_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for projection, key in keys.items():
                if projection not in loaded and key in available:
                    loaded[projection] = handle.get_tensor(key).contiguous()
        if len(loaded) == len(keys):
            return loaded

    missing = sorted(key for projection, key in keys.items() if projection not in loaded)
    raise KeyError(f"Missing shared expert weights for layer {layer}: {missing}")
