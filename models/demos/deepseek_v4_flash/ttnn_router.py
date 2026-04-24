# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest


@dataclass(frozen=True)
class RouterWeights:
    gate_weight: torch.Tensor
    bias: torch.Tensor | None = None
    tid2eid: torch.Tensor | None = None


class TtRouter(LightweightModule):
    """TTNN gate projection with host DeepSeek route scoring and selection."""

    def __init__(
        self,
        *,
        device,
        gate_weight: torch.Tensor,
        topk: int,
        route_scale: float,
        scoring_func: str,
        bias: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        validate_router_config(
            gate_weight,
            topk=topk,
            route_scale=route_scale,
            scoring_func=scoring_func,
            bias=bias,
            tid2eid=tid2eid,
        )
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.hidden_size = int(gate_weight.shape[-1])
        self.num_experts = int(gate_weight.shape[0])
        self.topk = int(topk)
        self.route_scale = float(route_scale)
        self.scoring_func = scoring_func
        self.bias = None if bias is None else bias.float().contiguous()
        self.tid2eid = None if tid2eid is None else tid2eid.to(torch.long).contiguous()
        self.gate_weight = _to_tt_linear_weight(
            gate_weight,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        device,
        layer: int = 0,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> "TtRouter":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        weights = load_router_weights(preprocessed_path, manifest=manifest, layer=layer)
        config = manifest["config"]
        return cls(
            device=device,
            gate_weight=weights.gate_weight,
            bias=weights.bias,
            tid2eid=weights.tid2eid,
            topk=int(config["num_experts_per_tok"]),
            route_scale=float(config["routed_scaling_factor"]),
            scoring_func=str(config["scoring_func"]),
            dtype=dtype,
            memory_config=memory_config,
        )

    def forward(self, hidden_states, *, input_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_ttnn_hidden_states(hidden_states, hidden_size=self.hidden_size)
        scores = ttnn.to_torch(ttnn.linear(hidden_states, self.gate_weight, memory_config=self.memory_config))
        scores = _ttnn_scores_to_host(scores, num_experts=self.num_experts)
        return select_router_scores(
            scores,
            topk=self.topk,
            route_scale=self.route_scale,
            scoring_func=self.scoring_func,
            bias=self.bias,
            input_ids=input_ids,
            tid2eid=self.tid2eid,
        )


def load_router_weights(
    preprocessed_path: str | Path, *, manifest: dict | None = None, layer: int = 0
) -> RouterWeights:
    preprocessed_path = Path(preprocessed_path)
    if manifest is None:
        manifest = load_tt_manifest(preprocessed_path)

    keys = {
        "gate_weight": f"layers.{layer}.ffn.gate.weight",
        "bias": f"layers.{layer}.ffn.gate.bias",
        "tid2eid": f"layers.{layer}.ffn.gate.tid2eid",
    }
    loaded: dict[str, torch.Tensor] = {}
    for artifact in manifest["artifacts"]["non_expert_safetensors"]:
        artifact_path = preprocessed_path / artifact
        with safe_open(artifact_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name, key in keys.items():
                if name not in loaded and key in available:
                    loaded[name] = handle.get_tensor(key).contiguous()
        if "gate_weight" in loaded and ("bias" in loaded or "tid2eid" in loaded):
            break

    if "gate_weight" not in loaded:
        raise KeyError(f"Missing router gate weight for layer {layer}: {keys['gate_weight']}")
    return RouterWeights(
        gate_weight=loaded["gate_weight"],
        bias=loaded.get("bias"),
        tid2eid=loaded.get("tid2eid"),
    )


def select_router_scores(
    scores: torch.Tensor,
    *,
    topk: int,
    route_scale: float,
    scoring_func: str = "sqrtsoftplus",
    bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply DeepSeek V4 Flash router scoring to host logits from the TT gate projection."""

    _validate_router_scores(scores, topk=topk, route_scale=route_scale, bias=bias, tid2eid=tid2eid)
    if scoring_func == "softmax":
        scored = scores.float().softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scored = scores.float().sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scored = F.softplus(scores.float()).sqrt()
    else:
        raise ValueError(f"Unsupported DeepSeek V4 Flash scoring_func {scoring_func!r}")

    original_scores = scored
    selection_scores = scored if bias is None else scored + bias.float()
    if tid2eid is not None:
        indices = _hash_route_indices(tid2eid, input_ids, scores.shape[:-1])
    else:
        indices = selection_scores.topk(topk, dim=-1).indices

    weights = original_scores.gather(-1, indices)
    if scoring_func != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * float(route_scale), indices


def validate_router_config(
    gate_weight: torch.Tensor,
    *,
    topk: int,
    route_scale: float,
    scoring_func: str,
    bias: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> None:
    if gate_weight.ndim != 2:
        raise ValueError(f"gate_weight must have shape [num_experts, hidden], got {tuple(gate_weight.shape)}")
    if gate_weight.shape[0] <= 0 or gate_weight.shape[1] <= 0:
        raise ValueError(f"gate_weight dimensions must be positive, got {tuple(gate_weight.shape)}")
    if topk <= 0 or topk > gate_weight.shape[0]:
        raise ValueError(f"topk must be in [1, {gate_weight.shape[0]}], got {topk}")
    if route_scale <= 0:
        raise ValueError(f"route_scale must be positive, got {route_scale}")
    if scoring_func not in ("softmax", "sigmoid", "sqrtsoftplus"):
        raise ValueError(f"Unsupported DeepSeek V4 Flash scoring_func {scoring_func!r}")
    if bias is not None and tuple(bias.shape) != (gate_weight.shape[0],):
        raise ValueError(f"bias must have shape {(gate_weight.shape[0],)}, got {tuple(bias.shape)}")
    if tid2eid is not None:
        if tid2eid.ndim != 2:
            raise ValueError(f"tid2eid must have shape [vocab, topk], got {tuple(tid2eid.shape)}")
        if tid2eid.shape[-1] != topk:
            raise ValueError(f"tid2eid topk dim must be {topk}, got {tid2eid.shape[-1]}")
        if torch.any(tid2eid < 0) or torch.any(tid2eid >= gate_weight.shape[0]):
            raise ValueError(f"tid2eid values must be in [0, {gate_weight.shape[0]})")


def _validate_router_scores(
    scores: torch.Tensor,
    *,
    topk: int,
    route_scale: float,
    bias: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
) -> None:
    if scores.ndim < 2:
        raise ValueError(f"scores must have shape [..., num_experts], got {tuple(scores.shape)}")
    num_experts = scores.shape[-1]
    if topk <= 0 or topk > num_experts:
        raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")
    if route_scale <= 0:
        raise ValueError(f"route_scale must be positive, got {route_scale}")
    if bias is not None and tuple(bias.shape) != (num_experts,):
        raise ValueError(f"bias must have shape {(num_experts,)}, got {tuple(bias.shape)}")
    if tid2eid is not None:
        if tid2eid.ndim != 2:
            raise ValueError(f"tid2eid must have shape [vocab, topk], got {tuple(tid2eid.shape)}")
        if tid2eid.shape[-1] != topk:
            raise ValueError(f"tid2eid topk dim must be {topk}, got {tid2eid.shape[-1]}")
        if torch.any(tid2eid < 0) or torch.any(tid2eid >= num_experts):
            raise ValueError(f"tid2eid values must be in [0, {num_experts})")


def _hash_route_indices(
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor | None,
    score_prefix_shape: torch.Size,
) -> torch.Tensor:
    if input_ids is None:
        raise ValueError("input_ids is required for hash-routed layers")
    if tuple(input_ids.shape) != tuple(score_prefix_shape):
        raise ValueError(f"input_ids must have shape {tuple(score_prefix_shape)}, got {tuple(input_ids.shape)}")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"input_ids dtype must be int32 or int64, got {input_ids.dtype}")
    if torch.any(input_ids < 0) or torch.any(input_ids >= tid2eid.shape[0]):
        raise ValueError(f"input_ids values must be in [0, {tid2eid.shape[0]})")
    indices = tid2eid[input_ids.reshape(-1).to(torch.long)].to(torch.long)
    return indices.reshape(*score_prefix_shape, indices.shape[-1])


def _validate_ttnn_hidden_states(hidden_states, *, hidden_size: int) -> None:
    shape = tuple(hidden_states.shape)
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(f"Expected router hidden_states shape [batch, 1, tokens, hidden], got {shape}")
    if shape[-1] != hidden_size:
        raise ValueError(f"Expected router hidden dim {hidden_size}, got {shape[-1]}")
    if shape[-2] == 0:
        raise ValueError("router hidden_states must contain at least one token")


def _ttnn_scores_to_host(scores: torch.Tensor, *, num_experts: int) -> torch.Tensor:
    if scores.ndim != 4 or scores.shape[1] != 1:
        raise ValueError(f"Expected router TTNN scores shape [batch, 1, tokens, experts], got {tuple(scores.shape)}")
    if scores.shape[-1] != num_experts:
        raise ValueError(f"Expected router scores expert dim {num_experts}, got {scores.shape[-1]}")
    if scores.shape[-2] == 0:
        raise ValueError("router scores must contain at least one token")
    return scores[:, 0].contiguous()


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
