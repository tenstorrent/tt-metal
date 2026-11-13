# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact Qwen4Exp gated-residual equations and TP4 ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.reference import gated_residual_read

TP_SIZE = 4


@dataclass(frozen=True)
class Qwen38GatedResidualDeviceShard:
    norm: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor
    inject: torch.Tensor


@dataclass(frozen=True)
class Qwen38GatedResidualWeights:
    placement: Qwen38Placement
    layer_index: int
    block: str
    norm: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor
    inject: torch.Tensor

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        *,
        layer_index: int,
        block: Literal["attn", "mlp"],
    ) -> "Qwen38GatedResidualWeights":
        config = checkpoint.config
        if placement.config != config:
            raise ValueError("checkpoint and placement configurations differ")
        if not 0 <= layer_index < config.num_hidden_layers:
            raise ValueError(f"layer index is outside the 48-layer backbone: {layer_index}")
        return cls._from_prefix(
            checkpoint,
            placement,
            layer_index=layer_index,
            block=block,
            prefix=f"model.language_model.layers.{layer_index}.{block}_hyper_connection.",
            source=f"backbone layer {layer_index}",
        )

    @classmethod
    def from_mtp_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        *,
        mtp_layer_index: int = 0,
        block: Literal["attn", "mlp"],
    ) -> "Qwen38GatedResidualWeights":
        """Load one released MTP layer's real hyper-connection weights."""

        config = checkpoint.config
        if placement.config != config:
            raise ValueError("checkpoint and placement configurations differ")
        if not 0 <= mtp_layer_index < config.mtp_layers:
            raise ValueError(f"MTP layer index is outside [0, {config.mtp_layers}): {mtp_layer_index}")
        return cls._from_prefix(
            checkpoint,
            placement,
            layer_index=mtp_layer_index,
            block=block,
            prefix=f"mtp.layers.{mtp_layer_index}.{block}_hyper_connection.",
            source=f"MTP layer {mtp_layer_index}",
        )

    @classmethod
    def _from_prefix(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        *,
        layer_index: int,
        block: Literal["attn", "mlp"],
        prefix: str,
        source: str,
    ) -> "Qwen38GatedResidualWeights":
        config = checkpoint.config
        if placement.config != config:
            raise ValueError("checkpoint and placement configurations differ")
        if block not in ("attn", "mlp"):
            raise ValueError(f"gated residual block must be attn or mlp, got {block!r}")
        tensors = {
            field: checkpoint.tensor(prefix + checkpoint_name)
            for field, checkpoint_name in {
                "norm": "hc_norm.weight",
                "down": "input_mix_weight_down.weight",
                "up": "input_mix_weight_up.weight",
                "inject": "block_inject_weight.weight",
            }.items()
        }
        expected = {
            "norm": (config.residual_width,),
            "down": (config.residual_rank, config.residual_width),
            "up": (config.residual_width, config.residual_rank),
            "inject": (config.residual_branches, config.residual_width),
        }
        for name, shape in expected.items():
            tensor = tensors[name]
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"unexpected {source} GR {name}: {tensor.dtype} {tuple(tensor.shape)}, expected BF16 {shape}"
                )
        return cls(placement=placement, layer_index=layer_index, block=block, **tensors)

    @property
    def config(self):
        return self.placement.config

    def device_shard(self, device_index: int) -> Qwen38GatedResidualDeviceShard:
        if not 0 <= device_index < TP_SIZE:
            raise IndexError(f"device index is outside TP4: {device_index}")
        branches = self.config.residual_branches
        hidden = self.config.hidden_size
        start, end = self.placement.hidden_ranges[device_index]
        return Qwen38GatedResidualDeviceShard(
            norm=self.norm.unflatten(0, (branches, hidden))[:, start:end].contiguous(),
            down=self.down.unflatten(1, (branches, hidden))[:, :, start:end].contiguous(),
            up=self.up.unflatten(0, (branches, hidden))[:, start:end, :].contiguous(),
            inject=self.inject.unflatten(1, (branches, hidden))[:, :, start:end].contiguous(),
        )

    def transformers_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "hc_norm.weight": self.norm,
            "input_mix_weight_down.weight": self.down,
            "input_mix_weight_up.weight": self.up,
            "block_inject_weight.weight": self.inject,
        }


@dataclass(frozen=True)
class Qwen38GatedResidualState:
    residual: torch.Tensor
    injection: torch.Tensor


@dataclass(frozen=True)
class Qwen38GatedResidualTP4State:
    residual_shards: tuple[torch.Tensor, ...]
    injection: torch.Tensor


class Qwen38GatedResidual:
    def __init__(self, weights: Qwen38GatedResidualWeights):
        self.weights = weights
        self.config = weights.config

    def read(self, residual: torch.Tensor) -> tuple[torch.Tensor, Qwen38GatedResidualState]:
        block_input, normalized = gated_residual_read(
            residual,
            self.weights.norm,
            self.weights.down,
            self.weights.up,
            self.config.residual_branches,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )
        injection = 2 * torch.sigmoid(F.linear(normalized, self.weights.inject) / self.config.residual_branches)
        return block_input, Qwen38GatedResidualState(residual=residual, injection=injection)

    @staticmethod
    def write(block_output: torch.Tensor, state: Qwen38GatedResidualState) -> torch.Tensor:
        branches = state.residual.shape[-1] // block_output.shape[-1]
        if state.injection.shape != (*block_output.shape[:-1], branches):
            raise ValueError("GR injection coefficients do not match the block output")
        injection = state.injection.unsqueeze(-1) * block_output.unsqueeze(-2)
        return state.residual + injection.flatten(-2)

    def shard_residual(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if residual.shape[-1] != self.config.residual_width:
            raise ValueError(f"GR residual width must be {self.config.residual_width}")
        branches = residual.unflatten(-1, (self.config.residual_branches, self.config.hidden_size))
        return tuple(
            branches[..., start:end].flatten(-2).contiguous() for start, end in self.weights.placement.hidden_ranges
        )

    def combine_residual_shards(self, shards: tuple[torch.Tensor, ...]) -> torch.Tensor:
        local_width = self.config.hidden_size // TP_SIZE
        self._validate_residual_shards(shards)
        branches = [shard.unflatten(-1, (self.config.residual_branches, local_width)) for shard in shards]
        return torch.cat(branches, dim=-1).flatten(-2)

    def shard_hidden(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if hidden.shape[-1] != self.config.hidden_size:
            raise ValueError(f"GR block output width must be {self.config.hidden_size}")
        return tuple(hidden[..., start:end].contiguous() for start, end in self.weights.placement.hidden_ranges)

    def read_tp4(
        self, residual_shards: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], Qwen38GatedResidualTP4State]:
        self._validate_residual_shards(residual_shards)
        branches = self.config.residual_branches
        hidden = self.config.hidden_size
        local_width = hidden // TP_SIZE
        local_residuals = [shard.unflatten(-1, (branches, local_width)) for shard in residual_shards]

        variance = sum(local.float().square().sum(dim=-1, keepdim=True) for local in local_residuals) / hidden
        inverse_rms = torch.rsqrt(variance + self.config.rms_norm_eps)
        normalized = []
        for device_index, local in enumerate(local_residuals):
            shard = self.weights.device_shard(device_index)
            normalized.append((local.float() * inverse_rms * (1.0 + shard.norm.float())).to(local.dtype))

        down = sum(
            F.linear(local.flatten(-2), self.weights.device_shard(device_index).down.flatten(1))
            for device_index, local in enumerate(normalized)
        )
        low_rank = F.silu(down / branches)
        block_shards = []
        inject_parts = []
        for device_index, local in enumerate(normalized):
            shard = self.weights.device_shard(device_index)
            gate = torch.sigmoid(F.linear(low_rank, shard.up.flatten(0, 1))).unflatten(-1, (branches, local_width))
            block_shards.append((gate * local).mean(dim=-2))
            inject_parts.append(F.linear(local.flatten(-2), shard.inject.flatten(1)))
        injection = 2 * torch.sigmoid(sum(inject_parts) / branches)
        state = Qwen38GatedResidualTP4State(tuple(residual_shards), injection)
        return tuple(block_shards), state

    def write_tp4(
        self, block_output_shards: tuple[torch.Tensor, ...], state: Qwen38GatedResidualTP4State
    ) -> tuple[torch.Tensor, ...]:
        if len(block_output_shards) != TP_SIZE or len(state.residual_shards) != TP_SIZE:
            raise ValueError("GR TP4 write requires exactly four shards")
        local_width = self.config.hidden_size // TP_SIZE
        outputs = []
        for block_output, residual in zip(block_output_shards, state.residual_shards):
            if block_output.shape[-1] != local_width or block_output.shape[:-1] != residual.shape[:-1]:
                raise ValueError("GR local block output is not aligned with its residual shard")
            residual_branches = residual.unflatten(-1, (self.config.residual_branches, local_width))
            injection = state.injection.unsqueeze(-1) * block_output.unsqueeze(-2)
            outputs.append((residual_branches + injection).flatten(-2))
        return tuple(outputs)

    def _validate_residual_shards(self, shards: tuple[torch.Tensor, ...]) -> None:
        expected_width = self.config.residual_width // TP_SIZE
        if len(shards) != TP_SIZE:
            raise ValueError(f"GR TP4 requires four residual shards, got {len(shards)}")
        prefix = shards[0].shape[:-1]
        if any(shard.shape[:-1] != prefix or shard.shape[-1] != expected_width for shard in shards):
            raise ValueError(f"GR residual shards must share a prefix and width {expected_width}")
