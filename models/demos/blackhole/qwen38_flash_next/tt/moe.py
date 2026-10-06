# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact Qwen4Exp MoE semantics and non-replicated four-device placement.

The CPU implementation is the numerical oracle for both candidate device
paths.  ``moe_compute`` produces unweighted expert outputs; its scores become
arithmetic only in ``deepseek_moe_fast_reduce_nc_fused``.  The alternative
decode design keeps 128 routed experts per P150, localizes the global top-10 on
each device, and reduce-scatters the four partial outputs.  It never stores all
512 experts on a device.

The current all-to-all dispatch operation defines global batch as
``tokens_per_device * dispatch_devices``.  It therefore cannot represent true
global-B=1 with four dispatch devices.  ``admit_moe_compute_global_batch``
records that limitation and fails closed; the shard-aware path is the required
B=1 decode route until that kernel contract changes and is device-proven.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement

TP_SIZE = 4
BF4_TILE_BYTES = 576
TILE_ELEMENTS = 32 * 32


@dataclass(frozen=True)
class Qwen38ExpertWeights:
    gate_up: torch.Tensor
    down: torch.Tensor
    intermediate_size: int

    @property
    def gate(self) -> torch.Tensor:
        return self.gate_up[: self.intermediate_size]

    @property
    def up(self) -> torch.Tensor:
        return self.gate_up[self.intermediate_size :]


@dataclass(frozen=True)
class Qwen38SharedExpertShard:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    scalar_gate: torch.Tensor


@dataclass(frozen=True)
class Qwen38RoutedDeviceShard:
    """One device's sparse-matmul weight layout before BF4_B upload."""

    expert_range: tuple[int, int]
    gate_up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class Qwen38MoERouting:
    logits: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class Qwen38EP4MoEResult:
    hidden_shards: tuple[torch.Tensor, ...]
    local_selected_experts: tuple[tuple[int, ...], ...]
    routing: Qwen38MoERouting


def admit_moe_compute_global_batch(*, global_batch: int, dispatch_devices: int) -> int:
    """Return tokens/device only when all-to-all dispatch represents the batch.

    Current TTNN source computes the operation's global batch by multiplying the
    local input row count by the number of dispatch devices.  Padding a B=1
    interactive request into B=4 would change the workload and is forbidden.
    """

    if global_batch <= 0 or dispatch_devices <= 0:
        raise ValueError("global_batch and dispatch_devices must be positive")
    if global_batch == 1 and dispatch_devices > 1:
        raise ValueError(
            "moe_compute all-to-all dispatch cannot represent true global batch 1 across multiple dispatch devices"
        )
    if global_batch % dispatch_devices:
        raise ValueError(f"global batch {global_batch} must divide evenly over {dispatch_devices} dispatch devices")
    return global_batch // dispatch_devices


def weighted_routed_reduce(expert_outputs: torch.Tensor, scores: torch.Tensor | None) -> torch.Tensor:
    """Apply the real normalized top-k scores and reduce the expert-slot axis."""

    if scores is None:
        raise ValueError("weighted routed reduction requires real top-k scores")
    if expert_outputs.ndim < 2 or scores.shape != expert_outputs.shape[:-1]:
        raise ValueError(f"expert outputs/scores disagree: {tuple(expert_outputs.shape)} versus {tuple(scores.shape)}")
    score_values = scores.float()
    if not torch.isfinite(score_values).all() or (score_values < 0).any():
        raise ValueError("real top-k scores must be finite and nonnegative")
    row_sums = score_values.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=0.0, atol=0.02):
        raise ValueError("real top-k scores must be normalized per token")
    return (expert_outputs * scores.unsqueeze(-1)).sum(dim=-2)


class Qwen38MoEWeights:
    """Lazy exact-checkpoint MoE access with an explicit EP4/TP4 split."""

    def __init__(
        self,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        *,
        layer_index: int | None = None,
        mtp_layer_index: int | None = None,
    ):
        if (layer_index is None) == (mtp_layer_index is None):
            raise ValueError("select exactly one regular layer or MTP layer")
        if placement.config != checkpoint.config:
            raise ValueError("checkpoint and placement configurations differ")
        if layer_index is not None:
            if not 0 <= layer_index < checkpoint.config.num_hidden_layers:
                raise ValueError(f"layer_index is outside the 48-layer backbone: {layer_index}")
            prefix = f"model.language_model.layers.{layer_index}.mlp"
        else:
            if not 0 <= mtp_layer_index < checkpoint.config.mtp_layers:
                raise ValueError(f"mtp_layer_index is outside the one-layer MTP stack: {mtp_layer_index}")
            prefix = f"mtp.layers.{mtp_layer_index}.mlp"

        self.checkpoint = checkpoint
        self.placement = placement
        self.config = checkpoint.config
        self.prefix = prefix
        self._expert_cache: dict[int, Qwen38ExpertWeights] = {}
        self._validate_metadata()

    @property
    def expert_ranges(self) -> tuple[tuple[int, int], ...]:
        return self.placement.expert_ranges

    @property
    def _gate_up_name(self) -> str:
        return f"{self.prefix}.experts.gate_up_proj"

    @property
    def _down_name(self) -> str:
        return f"{self.prefix}.experts.down_proj"

    def _validate_metadata(self) -> None:
        h = self.config.hidden_size
        intermediate = self.config.expert_intermediate_size
        expected = {
            self._gate_up_name: (self.config.num_experts, 2 * intermediate, h),
            self._down_name: (self.config.num_experts, h, intermediate),
            f"{self.prefix}.gate.weight": (self.config.num_experts, h),
            f"{self.prefix}.shared_expert.gate_proj.weight": (intermediate, h),
            f"{self.prefix}.shared_expert.up_proj.weight": (intermediate, h),
            f"{self.prefix}.shared_expert.down_proj.weight": (h, intermediate),
            f"{self.prefix}.shared_expert_gate.weight": (1, h),
        }
        for name, shape in expected.items():
            metadata = self.checkpoint.metadata(name)
            if metadata.dtype != "BF16" or metadata.shape != shape:
                raise ValueError(
                    f"unexpected MoE tensor {name}: {metadata.dtype} {metadata.shape}, expected BF16 {shape}"
                )

    @cached_property
    def router_weight(self) -> torch.Tensor:
        return self.checkpoint.tensor(f"{self.prefix}.gate.weight")

    @cached_property
    def shared_gate_proj(self) -> torch.Tensor:
        return self.checkpoint.tensor(f"{self.prefix}.shared_expert.gate_proj.weight")

    @cached_property
    def shared_up_proj(self) -> torch.Tensor:
        return self.checkpoint.tensor(f"{self.prefix}.shared_expert.up_proj.weight")

    @cached_property
    def shared_down_proj(self) -> torch.Tensor:
        return self.checkpoint.tensor(f"{self.prefix}.shared_expert.down_proj.weight")

    @cached_property
    def shared_scalar_gate(self) -> torch.Tensor:
        return self.checkpoint.tensor(f"{self.prefix}.shared_expert_gate.weight")

    @property
    def bf4_routed_payload_bytes_per_device(self) -> int:
        gate_up = self.checkpoint.metadata(self._gate_up_name)
        down = self.checkpoint.metadata(self._down_name)
        elements = (gate_up.elements + down.elements) // TP_SIZE
        if elements % TILE_ELEMENTS:
            raise ValueError("routed expert shard is not a whole number of TT tiles")
        return elements // TILE_ELEMENTS * BF4_TILE_BYTES

    def owner(self, expert_index: int) -> int:
        if not 0 <= expert_index < self.config.num_experts:
            raise IndexError(f"expert index is outside [0, {self.config.num_experts}): {expert_index}")
        for device_index, (start, end) in enumerate(self.expert_ranges):
            if start <= expert_index < end:
                return device_index
        raise AssertionError("validated expert placement has no owner")

    def router_shard(self, device_index: int) -> torch.Tensor:
        start, end = self._device_expert_range(device_index)
        return self.checkpoint.tensor_slice(f"{self.prefix}.gate.weight", (slice(start, end), slice(None)))

    def shared_shard(self, device_index: int) -> Qwen38SharedExpertShard:
        start, end = self._device_intermediate_range(device_index)
        hidden_start, hidden_end = self.placement.hidden_ranges[device_index]
        return Qwen38SharedExpertShard(
            gate_proj=self.checkpoint.tensor_slice(
                f"{self.prefix}.shared_expert.gate_proj.weight", (slice(start, end), slice(None))
            ),
            up_proj=self.checkpoint.tensor_slice(
                f"{self.prefix}.shared_expert.up_proj.weight", (slice(start, end), slice(None))
            ),
            down_proj=self.checkpoint.tensor_slice(
                f"{self.prefix}.shared_expert.down_proj.weight", (slice(None), slice(start, end))
            ),
            scalar_gate=self.checkpoint.tensor_slice(
                f"{self.prefix}.shared_expert_gate.weight", (slice(None), slice(hidden_start, hidden_end))
            ),
        )

    def expert(self, expert_index: int) -> Qwen38ExpertWeights:
        self.owner(expert_index)
        cached = self._expert_cache.get(expert_index)
        if cached is None:
            gate_up = self.checkpoint.tensor_slice(
                self._gate_up_name, (slice(expert_index, expert_index + 1), slice(None), slice(None))
            ).squeeze(0)
            down = self.checkpoint.tensor_slice(
                self._down_name, (slice(expert_index, expert_index + 1), slice(None), slice(None))
            ).squeeze(0)
            cached = Qwen38ExpertWeights(gate_up, down, self.config.expert_intermediate_size)
            self._expert_cache[expert_index] = cached
        return cached

    def routed_device_shard(self, device_index: int) -> Qwen38RoutedDeviceShard:
        """Load one 128-expert host shard in ``sparse_matmul`` orientation.

        This allocates roughly 1.17 GiB of BF16 staging memory and is intended for
        sequential conversion into BF4_B cache tensors, not unit-test setup.
        """

        start, end = self._device_expert_range(device_index)
        gate_up = self.checkpoint.tensor_slice(self._gate_up_name, (slice(start, end), slice(None), slice(None)))
        down = self.checkpoint.tensor_slice(self._down_name, (slice(start, end), slice(None), slice(None)))
        return Qwen38RoutedDeviceShard(
            expert_range=(start, end),
            gate_up=gate_up.transpose(1, 2).unsqueeze(0).contiguous(),
            down=down.transpose(1, 2).unsqueeze(0).contiguous(),
        )

    def _device_expert_range(self, device_index: int) -> tuple[int, int]:
        if not 0 <= device_index < TP_SIZE:
            raise IndexError(f"device index is outside TP4: {device_index}")
        return self.expert_ranges[device_index]

    def _device_intermediate_range(self, device_index: int) -> tuple[int, int]:
        if not 0 <= device_index < TP_SIZE:
            raise IndexError(f"device index is outside TP4: {device_index}")
        width = self.config.shared_expert_intermediate_size // TP_SIZE
        return device_index * width, (device_index + 1) * width


class Qwen38MoE:
    """CPU oracle for exact target MoE and its four-device decomposition."""

    def __init__(self, weights: Qwen38MoEWeights):
        self.weights = weights
        self.config = weights.config

    def route(self, hidden_states: torch.Tensor) -> Qwen38MoERouting:
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(f"MoE hidden width must be {self.config.hidden_size}, got {hidden_states.shape[-1]}")
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        logits = F.linear(flat, self.weights.router_weight)
        probabilities = torch.softmax(logits, dtype=torch.float32, dim=-1)
        scores, indices = torch.topk(probabilities, self.config.top_k, dim=-1)
        if self.config.norm_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
        return Qwen38MoERouting(logits, scores.to(logits.dtype), indices)

    def __call__(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, Qwen38MoERouting]:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        routing = self.route(flat)
        routed = self._routed(flat, routing)
        output = routed + self._shared(flat)
        return output.reshape(original_shape), routing

    def expert_parallel_forward(self, hidden_states: torch.Tensor) -> Qwen38EP4MoEResult:
        """Simulate the true EP4/TP4 B=1 device path without weight replication.

        Device execution all-gathers the hidden input, computes router/expert and
        shared-intermediate shards locally, then reduce-scatters the summed full-H
        partials.  This CPU form preserves those ownership boundaries.
        """

        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        routing = self.route(flat)
        scalar_gate_parts = []
        shared_shards = []
        for device_index in range(TP_SIZE):
            hidden_start, hidden_end = self.weights.placement.hidden_ranges[device_index]
            shared = self.weights.shared_shard(device_index)
            scalar_gate_parts.append(F.linear(flat[:, hidden_start:hidden_end], shared.scalar_gate))
            intermediate = F.silu(F.linear(flat, shared.gate_proj)) * F.linear(flat, shared.up_proj)
            shared_shards.append(F.linear(intermediate, shared.down_proj))
        scalar_gate = torch.sigmoid(sum(scalar_gate_parts))

        local_selected: list[tuple[int, ...]] = []
        partial_outputs = []
        for device_index in range(TP_SIZE):
            selected = tuple(
                int(expert)
                for expert in routing.indices.flatten().tolist()
                if self.weights.owner(int(expert)) == device_index
            )
            local_selected.append(selected)
            routed_partial = self._routed(flat, routing, owner=device_index)
            partial_outputs.append(routed_partial + scalar_gate * shared_shards[device_index])

        full_output = sum(partial_outputs).reshape(original_shape)
        hidden_shards = tuple(full_output[..., start:end] for start, end in self.weights.placement.hidden_ranges)
        return Qwen38EP4MoEResult(hidden_shards, tuple(local_selected), routing)

    def _shared(self, flat: torch.Tensor) -> torch.Tensor:
        intermediate = F.silu(F.linear(flat, self.weights.shared_gate_proj)) * F.linear(
            flat, self.weights.shared_up_proj
        )
        output = F.linear(intermediate, self.weights.shared_down_proj)
        return output * torch.sigmoid(F.linear(flat, self.weights.shared_scalar_gate))

    def _routed(
        self,
        flat: torch.Tensor,
        routing: Qwen38MoERouting,
        *,
        owner: int | None = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(flat)
        with torch.no_grad():
            expert_mask = F.one_hot(routing.indices, num_classes=self.config.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_tensor in expert_hit:
            expert_index = int(expert_tensor[0])
            if owner is not None and self.weights.owner(expert_index) != owner:
                continue
            top_k_position, token_index = torch.where(expert_mask[expert_index])
            current = flat[token_index]
            expert = self.weights.expert(expert_index)
            gate, up = F.linear(current, expert.gate_up).chunk(2, dim=-1)
            current = F.silu(gate) * up
            current = F.linear(current, expert.down)
            current = current * routing.scores[token_index, top_k_position, None]
            output.index_add_(0, token_index, current.to(output.dtype))
        return output
