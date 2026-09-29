# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lazy exact-checkpoint text-model integration oracle and TP4 I/O placement.

This module intentionally executes on CPU. It proves the complete 48-layer
state and tensor-name contract without loading the 360-GB checkpoint twice.
It is not a substitute for the required four-P150 TT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint, TensorMetadata
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Config, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.reference import gated_residual_read
from models.demos.blackhole.qwen38_flash_next.tt.layer import (
    Qwen38DecoderLayer,
    Qwen38DecoderLayerAux,
    Qwen38DecoderLayerState,
)

TP_SIZE = 4
EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"


def text_rope(
    config: Qwen38Config, *, batch: int, length: int, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact text-only Qwen4Exp RoPE for absolute positions ``[0,length)``."""

    if batch <= 0 or not 0 < length <= config.max_position_embeddings:
        raise ValueError("text RoPE batch/length is outside the pinned configuration")
    if dtype != torch.bfloat16:
        raise ValueError(f"the exact checkpoint oracle requires BF16 RoPE output, got {dtype}")
    rotary_width = config.qsa_rope_dim
    inverse_frequency = 1.0 / (
        config.rope_theta ** (torch.arange(0, rotary_width, 2, dtype=torch.float32) / rotary_width)
    )
    frequencies = torch.outer(torch.arange(length, dtype=torch.float32), inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0).expand(batch, -1, -1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


@dataclass(frozen=True)
class Qwen38FinalMixerDeviceShard:
    norm: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor


@dataclass(frozen=True)
class Qwen38FinalMixerWeights:
    placement: Qwen38Placement
    norm: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor

    @classmethod
    def from_checkpoint(cls, checkpoint: Qwen38Checkpoint, placement: Qwen38Placement) -> "Qwen38FinalMixerWeights":
        return cls._from_prefix(
            checkpoint,
            placement,
            prefix="model.language_model.hyper_connection_mixer.",
            source="backbone",
        )

    @classmethod
    def from_mtp_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
    ) -> "Qwen38FinalMixerWeights":
        """Load the released MTP stack's real terminal hyper-connection mixer."""

        return cls._from_prefix(
            checkpoint,
            placement,
            prefix="mtp.hyper_connection_mixer.",
            source="MTP",
        )

    @classmethod
    def _from_prefix(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        *,
        prefix: str,
        source: str,
    ) -> "Qwen38FinalMixerWeights":
        if checkpoint.config != placement.config:
            raise ValueError("checkpoint and placement configurations differ")
        tensors = {
            field: checkpoint.tensor(prefix + checkpoint_name)
            for field, checkpoint_name in {
                "norm": "hc_norm.weight",
                "down": "input_mix_weight_down.weight",
                "up": "input_mix_weight_up.weight",
            }.items()
        }
        config = checkpoint.config
        expected = {
            "norm": (config.residual_width,),
            "down": (config.residual_rank, config.residual_width),
            "up": (config.residual_width, config.residual_rank),
        }
        for name, shape in expected.items():
            tensor = tensors[name]
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"unexpected {source} final mixer {name}: "
                    f"{tensor.dtype} {tuple(tensor.shape)}, expected BF16 {shape}"
                )
        return cls(placement=placement, **tensors)

    @property
    def config(self) -> Qwen38Config:
        return self.placement.config

    def device_shard(self, device_index: int) -> Qwen38FinalMixerDeviceShard:
        if not 0 <= device_index < TP_SIZE:
            raise IndexError(f"device index is outside TP4: {device_index}")
        branches = self.config.residual_branches
        hidden = self.config.hidden_size
        start, end = self.placement.hidden_ranges[device_index]
        return Qwen38FinalMixerDeviceShard(
            norm=self.norm.unflatten(0, (branches, hidden))[:, start:end].contiguous(),
            down=self.down.unflatten(1, (branches, hidden))[:, :, start:end].contiguous(),
            up=self.up.unflatten(0, (branches, hidden))[:, start:end, :].contiguous(),
        )

    def transformers_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "hc_norm.weight": self.norm,
            "input_mix_weight_down.weight": self.down,
            "input_mix_weight_up.weight": self.up,
        }


class Qwen38FinalMixer:
    def __init__(self, weights: Qwen38FinalMixerWeights):
        self.weights = weights
        self.config = weights.config

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        output, _ = gated_residual_read(
            residual,
            self.weights.norm,
            self.weights.down,
            self.weights.up,
            self.config.residual_branches,
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )
        return output

    def shard_residual(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if residual.shape[-1] != self.config.residual_width:
            raise ValueError(f"final residual width must be {self.config.residual_width}")
        branches = residual.unflatten(-1, (self.config.residual_branches, self.config.hidden_size))
        return tuple(
            branches[..., start:end].flatten(-2).contiguous() for start, end in self.weights.placement.hidden_ranges
        )

    def forward_tp4(self, residual_shards: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(residual_shards) != TP_SIZE:
            raise ValueError(f"final mixer TP4 requires four shards, got {len(residual_shards)}")
        branches = self.config.residual_branches
        hidden = self.config.hidden_size
        local_width = hidden // TP_SIZE
        prefix = residual_shards[0].shape[:-1]
        if any(shard.shape != (*prefix, branches * local_width) for shard in residual_shards):
            raise ValueError("final mixer residual shards have incompatible shapes")
        local_residuals = [shard.unflatten(-1, (branches, local_width)) for shard in residual_shards]
        variance = sum(local.float().square().sum(dim=-1, keepdim=True) for local in local_residuals) / hidden
        inverse_rms = torch.rsqrt(variance + self.config.rms_norm_eps)
        normalized = []
        for device_index, local in enumerate(local_residuals):
            shard = self.weights.device_shard(device_index)
            normalized.append((local.float() * inverse_rms * (1.0 + shard.norm.float())).to(local.dtype))
        # Preserve the full dot product's FP32 accumulation across TP partials,
        # then round once to the BF16 activation contract after the all-reduce.
        down = sum(
            F.linear(local.flatten(-2).float(), self.weights.device_shard(device_index).down.flatten(1).float())
            for device_index, local in enumerate(normalized)
        ).to(residual_shards[0].dtype)
        low_rank = F.silu(down / branches)
        outputs = []
        for device_index, local in enumerate(normalized):
            shard = self.weights.device_shard(device_index)
            gate = torch.sigmoid(F.linear(low_rank, shard.up.flatten(0, 1))).unflatten(-1, (branches, local_width))
            outputs.append((gate * local).mean(dim=-2))
        return tuple(outputs)


class Qwen38ModelIO:
    """Sparse host embedding lookup and explicit vocabulary-parallel LM head."""

    def __init__(self, checkpoint: Qwen38Checkpoint, placement: Qwen38Placement):
        if checkpoint.config != placement.config:
            raise ValueError("checkpoint and placement configurations differ")
        self.checkpoint = checkpoint
        self.placement = placement
        self.config = checkpoint.config
        for metadata in (self.embedding_metadata, self.lm_head_metadata):
            if metadata.dtype != "BF16" or metadata.shape != (self.config.vocab_size, self.config.hidden_size):
                raise ValueError(f"unexpected model I/O tensor {metadata.name}: {metadata.dtype} {metadata.shape}")
        if self.embedding_metadata.name == self.lm_head_metadata.name:
            raise ValueError("the pinned untied embedding and LM head must be distinct tensors")

    @property
    def vocab_ranges(self) -> tuple[tuple[int, int], ...]:
        return self.placement.vocab_ranges

    @property
    def embedding_metadata(self) -> TensorMetadata:
        return self.checkpoint.metadata(EMBEDDING_NAME)

    @property
    def lm_head_metadata(self) -> TensorMetadata:
        return self.checkpoint.metadata(LM_HEAD_NAME)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.device.type != "cpu":
            raise ValueError("model input IDs must be a CPU torch.long [batch, sequence] tensor")
        if input_ids.numel() and (int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size):
            raise IndexError("input token is outside the pinned vocabulary")
        return self.checkpoint.tensor_rows(EMBEDDING_NAME, input_ids)

    def embedding_weight_shard(self, device_index: int) -> torch.Tensor:
        start, end = self._vocab_range(device_index)
        return self.checkpoint.tensor_slice(EMBEDDING_NAME, (slice(start, end), slice(None)))

    def lm_head_weight_shard(self, device_index: int) -> torch.Tensor:
        start, end = self._vocab_range(device_index)
        return self.checkpoint.tensor_slice(LM_HEAD_NAME, (slice(start, end), slice(None)))

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(f"LM input must be [batch, sequence, {self.config.hidden_size}]")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"LM input must be BF16, got {hidden_states.dtype}")
        shards = [F.linear(hidden_states, self.lm_head_weight_shard(device)) for device in range(TP_SIZE)]
        return torch.cat(shards, dim=-1)

    def greedy_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[1] != 1:
            raise ValueError("greedy decode expects exactly one current position")
        best_values = []
        best_indices = []
        for device_index, (start, _) in enumerate(self.vocab_ranges):
            local_logits = F.linear(hidden_states, self.lm_head_weight_shard(device_index))
            value, index = local_logits.max(dim=-1)
            best_values.append(value)
            best_indices.append(index + start)
        values = torch.stack(best_values, dim=-1)
        indices = torch.stack(best_indices, dim=-1)
        owner = values.argmax(dim=-1, keepdim=True)
        return indices.gather(-1, owner).squeeze(-1)

    def _vocab_range(self, device_index: int) -> tuple[int, int]:
        if not 0 <= device_index < TP_SIZE:
            raise IndexError(f"device index is outside TP4: {device_index}")
        return self.vocab_ranges[device_index]


@dataclass(frozen=True)
class Qwen38TextModelState:
    position: int
    layers: tuple[Qwen38DecoderLayerState | None, ...]

    def validate(self, config: Qwen38Config) -> None:
        if self.position < 0 or len(self.layers) != config.num_hidden_layers:
            raise ValueError("text-model state does not cover the exact 48-layer backbone")


@dataclass(frozen=True)
class Qwen38TextModelOutput:
    hidden_states: torch.Tensor
    logits: torch.Tensor | None
    state: Qwen38TextModelState
    layer_aux: tuple[Qwen38DecoderLayerAux, ...]


class Qwen38TextModelOracle:
    """Complete lazy 48-layer CPU oracle for ordinary prefill and decode."""

    def __init__(self, checkpoint: Qwen38Checkpoint, placement: Qwen38Placement):
        self.checkpoint = checkpoint
        self.placement = placement
        self.config = checkpoint.config
        self.model_io = Qwen38ModelIO(checkpoint, placement)
        self.final_mixer = Qwen38FinalMixer(Qwen38FinalMixerWeights.from_checkpoint(checkpoint, placement))
        self._layers: dict[int, Qwen38DecoderLayer] = {}

    def layer(self, layer_index: int) -> Qwen38DecoderLayer:
        if not 0 <= layer_index < self.config.num_hidden_layers:
            raise IndexError(f"layer is outside the exact backbone: {layer_index}")
        layer = self._layers.get(layer_index)
        if layer is None:
            layer = Qwen38DecoderLayer.from_checkpoint(self.checkpoint, self.placement, layer_index=layer_index)
            self._layers[layer_index] = layer
        return layer

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        state: Qwen38TextModelState | None = None,
        return_logits: bool = True,
        logits_to_keep: int = 1,
        layer_observer: Callable[[int, torch.Tensor, Qwen38DecoderLayerState, Qwen38DecoderLayerAux], None]
        | None = None,
    ) -> Qwen38TextModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("text-model input IDs must contain at least one token")
        batch, sequence = input_ids.shape
        if state is None:
            previous_position = 0
            previous_layers: tuple[Qwen38DecoderLayerState | None, ...] = (None,) * self.config.num_hidden_layers
        else:
            state.validate(self.config)
            previous_position = state.position
            previous_layers = state.layers
        total_length = previous_position + sequence
        if total_length > self.config.max_position_embeddings:
            raise ValueError("ordinary decode exceeds the pinned native context")

        hidden_states = self.model_io.embed(input_ids).repeat(1, 1, self.config.residual_branches)
        positions = text_rope(self.config, batch=batch, length=total_length, dtype=hidden_states.dtype)
        query_positions = torch.arange(previous_position, total_length).view(sequence, 1)
        key_positions = torch.arange(total_length).view(1, total_length)
        visible = key_positions <= query_positions
        attention_mask = torch.where(
            visible,
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(torch.finfo(torch.float32).min, dtype=torch.float32),
        ).view(1, 1, sequence, total_length)
        attention_mask = attention_mask.expand(batch, -1, -1, -1)

        next_layers = []
        layer_aux = []
        for layer_index in range(self.config.num_hidden_layers):
            hidden_states, next_state, aux = self.layer(layer_index).forward(
                hidden_states,
                input_ids=input_ids,
                position_embeddings=positions,
                attention_mask=attention_mask,
                state=previous_layers[layer_index],
            )
            next_layers.append(next_state)
            layer_aux.append(aux)
            if layer_observer is not None:
                layer_observer(layer_index, hidden_states, next_state, aux)
        hidden_states = self.final_mixer(hidden_states)
        if logits_to_keep <= 0 or logits_to_keep > sequence:
            raise ValueError("logits_to_keep must select at least one current input position")
        logits = self.model_io.logits(hidden_states[:, -logits_to_keep:]) if return_logits else None
        next_model_state = Qwen38TextModelState(position=total_length, layers=tuple(next_layers))
        return Qwen38TextModelOutput(hidden_states, logits, next_model_state, tuple(layer_aux))
