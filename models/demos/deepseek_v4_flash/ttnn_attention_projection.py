# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest


@dataclass(frozen=True)
class AttentionProjectionWeights:
    wq_a: torch.Tensor
    q_norm: torch.Tensor
    wq_b: torch.Tensor
    wo_a: torch.Tensor | None = None
    wo_b: torch.Tensor | None = None


class TtAttentionProjection(LightweightModule):
    """Single-device DeepSeek V4 Flash attention projection stepping stone.

    The q path runs on TTNN as ``wq_a -> RMSNorm(q_rank) -> wq_b`` and returns a
    TTNN tensor shaped ``[batch, 1, tokens, num_heads * head_dim]``.

    The optional output projection is intentionally split: grouped ``wo_a`` runs
    on host because this slice does not add a grouped TTNN linear primitive, then
    ``wo_b`` runs as a TTNN linear projection back to hidden size.
    """

    def __init__(
        self,
        *,
        device,
        weights: AttentionProjectionWeights,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        norm_eps: float = 1e-6,
        o_groups: int | None = None,
        o_lora_rank: int | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        validate_attention_projection_weights(
            weights,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            num_heads=num_heads,
            head_dim=head_dim,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
        )
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.hidden_size = int(hidden_size)
        self.q_lora_rank = int(q_lora_rank)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.q_output_dim = self.num_heads * self.head_dim
        self.norm_eps = float(norm_eps)
        self.o_groups = None if o_groups is None else int(o_groups)
        self.o_lora_rank = None if o_lora_rank is None else int(o_lora_rank)

        self.wq_a = _to_tt_linear_weight(weights.wq_a, device=device, dtype=dtype, memory_config=memory_config)
        self.q_norm = _to_tt_norm_weight(weights.q_norm, device=device, dtype=dtype, memory_config=memory_config)
        self.wq_b = _to_tt_linear_weight(weights.wq_b, device=device, dtype=dtype, memory_config=memory_config)
        self.wo_a = None if weights.wo_a is None else weights.wo_a.float().contiguous()
        self.wo_b = (
            None
            if weights.wo_b is None
            else _to_tt_linear_weight(weights.wo_b, device=device, dtype=dtype, memory_config=memory_config)
        )

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        device,
        layer: int = 0,
        include_output_projection: bool = True,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> "TtAttentionProjection":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        config = manifest["config"]
        return cls(
            device=device,
            weights=load_attention_projection_weights(
                preprocessed_path,
                manifest=manifest,
                layer=layer,
                include_output_projection=include_output_projection,
            ),
            hidden_size=int(config["hidden_size"]),
            q_lora_rank=int(config["q_lora_rank"]),
            num_heads=int(config["num_attention_heads"]),
            head_dim=int(config["head_dim"]),
            norm_eps=float(config["rms_norm_eps"]),
            o_groups=int(config["o_groups"]) if include_output_projection else None,
            o_lora_rank=int(config["o_lora_rank"]) if include_output_projection else None,
            dtype=dtype,
            memory_config=memory_config,
        )

    def forward(self, hidden_states):
        return self.project_q(hidden_states)

    def project_q(self, hidden_states):
        _validate_ttnn_projection_input(hidden_states, expected_width=self.hidden_size, label="hidden_states")
        q_rank = ttnn.linear(hidden_states, self.wq_a, memory_config=self.memory_config)
        q_rank = ttnn.rms_norm(
            q_rank,
            weight=self.q_norm,
            epsilon=self.norm_eps,
            memory_config=self.memory_config,
        )
        return ttnn.linear(q_rank, self.wq_b, memory_config=self.memory_config)

    def project_output(self, attention_output):
        _validate_ttnn_projection_input(
            attention_output,
            expected_width=self.q_output_dim,
            label="attention_output",
        )
        if self.wo_a is None or self.wo_b is None or self.o_groups is None:
            raise RuntimeError("Output projection weights were not loaded")

        host_output = _ttnn_projection_to_torch_3d(
            attention_output,
            expected_width=self.q_output_dim,
            label="attention_output",
        )
        output_rank = grouped_output_projection_a(host_output, self.wo_a, o_groups=self.o_groups)
        tt_output_rank = ttnn.from_torch(
            output_rank.unsqueeze(1).to(torch.bfloat16),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        return ttnn.linear(tt_output_rank, self.wo_b, memory_config=self.memory_config)


def load_attention_projection_weights(
    preprocessed_path: str | Path,
    *,
    manifest: dict | None = None,
    layer: int = 0,
    include_output_projection: bool = True,
) -> AttentionProjectionWeights:
    preprocessed_path = Path(preprocessed_path)
    if manifest is None:
        manifest = load_tt_manifest(preprocessed_path)

    prefix = f"layers.{layer}.attn"
    keys = {
        "wq_a": f"{prefix}.wq_a.weight",
        "q_norm": f"{prefix}.q_norm.weight",
        "wq_b": f"{prefix}.wq_b.weight",
    }
    if include_output_projection:
        keys["wo_a"] = f"{prefix}.wo_a.weight"
        keys["wo_b"] = f"{prefix}.wo_b.weight"

    loaded: dict[str, torch.Tensor] = {}
    for artifact in manifest["artifacts"]["non_expert_safetensors"]:
        with safe_open(preprocessed_path / artifact, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name, key in keys.items():
                if name not in loaded and key in available:
                    loaded[name] = handle.get_tensor(key).contiguous()
        if len(loaded) == len(keys):
            break

    missing = sorted(key for name, key in keys.items() if name not in loaded)
    if missing:
        raise KeyError(f"Missing attention projection weights for layer {layer}: {missing}")

    return AttentionProjectionWeights(
        wq_a=loaded["wq_a"],
        q_norm=loaded["q_norm"],
        wq_b=loaded["wq_b"],
        wo_a=loaded.get("wo_a"),
        wo_b=loaded.get("wo_b"),
    )


def validate_attention_projection_weights(
    weights: AttentionProjectionWeights,
    *,
    hidden_size: int,
    q_lora_rank: int,
    num_heads: int,
    head_dim: int,
    o_groups: int | None = None,
    o_lora_rank: int | None = None,
) -> None:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if q_lora_rank <= 0:
        raise ValueError(f"q_lora_rank must be positive, got {q_lora_rank}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")

    expected_q_output_dim = num_heads * head_dim
    _expect_shape(weights.wq_a, (q_lora_rank, hidden_size), "wq_a")
    _expect_shape(weights.q_norm, (q_lora_rank,), "q_norm")
    _expect_shape(weights.wq_b, (expected_q_output_dim, q_lora_rank), "wq_b")

    output_projection_requested = weights.wo_a is not None or weights.wo_b is not None
    if not output_projection_requested:
        if o_groups is not None or o_lora_rank is not None:
            raise ValueError("o_groups and o_lora_rank require wo_a and wo_b weights")
        return

    if weights.wo_a is None or weights.wo_b is None:
        raise ValueError("wo_a and wo_b must either both be present or both be omitted")
    if o_groups is None or o_lora_rank is None:
        raise ValueError("o_groups and o_lora_rank are required when output projection weights are present")
    if o_groups <= 0:
        raise ValueError(f"o_groups must be positive, got {o_groups}")
    if o_lora_rank <= 0:
        raise ValueError(f"o_lora_rank must be positive, got {o_lora_rank}")
    if expected_q_output_dim % o_groups != 0:
        raise ValueError(f"num_heads * head_dim {expected_q_output_dim} must be divisible by o_groups {o_groups}")

    group_input_dim = expected_q_output_dim // o_groups
    output_rank_dim = o_groups * o_lora_rank
    _expect_shape(weights.wo_a, (output_rank_dim, group_input_dim), "wo_a")
    _expect_shape(weights.wo_b, (hidden_size, output_rank_dim), "wo_b")


def grouped_output_projection_a(
    attention_output: torch.Tensor,
    wo_a: torch.Tensor,
    *,
    o_groups: int,
) -> torch.Tensor:
    """Run DeepSeek V4 Flash grouped ``wo_a`` on host.

    ``attention_output`` is shaped ``[batch, tokens, num_heads * head_dim]``.
    ``wo_a`` is shaped ``[o_groups * o_lora_rank, group_input_dim]`` and is
    applied independently to each contiguous attention-output group.
    """

    if attention_output.ndim != 3:
        raise ValueError(
            "attention_output must have shape [batch, tokens, num_heads * head_dim], "
            f"got {tuple(attention_output.shape)}"
        )
    if wo_a.ndim != 2:
        raise ValueError(f"wo_a must have shape [o_groups * o_lora_rank, group_input_dim], got {tuple(wo_a.shape)}")
    if o_groups <= 0:
        raise ValueError(f"o_groups must be positive, got {o_groups}")

    batch_size, tokens, attention_dim = attention_output.shape
    if attention_dim % o_groups != 0:
        raise ValueError(f"attention dim {attention_dim} must be divisible by o_groups {o_groups}")
    if wo_a.shape[0] % o_groups != 0:
        raise ValueError(f"wo_a output dim {wo_a.shape[0]} must be divisible by o_groups {o_groups}")

    group_input_dim = attention_dim // o_groups
    o_lora_rank = wo_a.shape[0] // o_groups
    if wo_a.shape[1] != group_input_dim:
        raise ValueError(f"wo_a group input dim must be {group_input_dim}, got {wo_a.shape[1]}")

    grouped_attention = attention_output.float().reshape(batch_size, tokens, o_groups, group_input_dim)
    grouped_weight = wo_a.float().reshape(o_groups, o_lora_rank, group_input_dim)
    output_rank = torch.einsum("btgi,gri->btgr", grouped_attention, grouped_weight)
    return output_rank.reshape(batch_size, tokens, o_groups * o_lora_rank).to(attention_output.dtype)


def _expect_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected {name} shape {expected_shape}, got {tuple(tensor.shape)}")


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


def _to_tt_norm_weight(
    weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    return ttnn.from_torch(
        weight.contiguous().to(torch.bfloat16),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _validate_ttnn_projection_input(tensor, *, expected_width: int, label: str) -> None:
    shape = tuple(tensor.shape)
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(f"Expected {label} shape [batch, 1, tokens, width], got {shape}")
    if shape[-1] != expected_width:
        raise ValueError(f"Expected {label} width {expected_width}, got {shape[-1]}")
    if shape[-2] == 0:
        raise ValueError(f"{label} must contain at least one token")


def _ttnn_projection_to_torch_3d(tensor, *, expected_width: int, label: str) -> torch.Tensor:
    torch_tensor = ttnn.to_torch(tensor)
    if torch_tensor.ndim != 4 or torch_tensor.shape[1] != 1:
        raise ValueError(f"Expected {label} TTNN shape [batch, 1, tokens, width], got {tuple(torch_tensor.shape)}")
    if torch_tensor.shape[-1] != expected_width:
        raise ValueError(f"Expected {label} width {expected_width}, got {torch_tensor.shape[-1]}")
    if torch_tensor.shape[-2] == 0:
        raise ValueError(f"{label} must contain at least one token")
    return torch_tensor[:, 0].contiguous()
