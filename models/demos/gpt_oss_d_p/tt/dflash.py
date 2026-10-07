# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash feature generation contracts for GPT-OSS prefill.

The GPT-OSS DFlash checkpoint projects five target-layer MoE outputs with one
``nn.Linear`` weight::

    fc(cat([h_1, h_9, h_17, h_25, h_33]))

True prefill sees those activations one layer at a time.  By linearity it can
produce the same pre-norm feature without retaining the five activations:

    sum(h_i @ fc.weight[:, i * H : (i + 1) * H].T)

``hidden_norm`` intentionally is not applied here.  The reload/DFlash consumer
owns that operation when it bulk-primes the drafter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

GPT_OSS_DFLASH_TARGET_LAYERS = (1, 9, 17, 25, 33)
GPT_OSS_HIDDEN_SIZE = 2880
GPT_OSS_NUM_TARGET_LAYERS = 36
DFLASH_FC_KEY = "fc.weight"


def _validate_target_layer_ids(target_layer_ids: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(int(x) for x in target_layer_ids)
    if not ids:
        raise ValueError("DFlash target_layer_ids must not be empty")
    if len(set(ids)) != len(ids) or tuple(sorted(ids)) != ids:
        raise ValueError(f"DFlash target_layer_ids must be unique and increasing, got {ids}")
    return ids


@dataclass(frozen=True)
class DFlashPrefillConfig:
    """Validated target-side subset of a GPT-OSS DFlash checkpoint."""

    checkpoint_path: Path
    hidden_size: int = GPT_OSS_HIDDEN_SIZE
    num_target_layers: int = GPT_OSS_NUM_TARGET_LAYERS
    target_layer_ids: tuple[int, ...] = GPT_OSS_DFLASH_TARGET_LAYERS
    fc_key: str = DFLASH_FC_KEY

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        ids = _validate_target_layer_ids(self.target_layer_ids)
        object.__setattr__(self, "target_layer_ids", ids)
        if self.hidden_size <= 0:
            raise ValueError(f"DFlash hidden_size must be positive, got {self.hidden_size}")
        if self.num_target_layers <= max(ids):
            raise ValueError(f"DFlash target layer {max(ids)} is outside a {self.num_target_layers}-layer target")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        expected_hidden_size: int = GPT_OSS_HIDDEN_SIZE,
        expected_num_target_layers: int = GPT_OSS_NUM_TARGET_LAYERS,
        expected_target_layer_ids: Sequence[int] = GPT_OSS_DFLASH_TARGET_LAYERS,
    ) -> "DFlashPrefillConfig":
        """Validate config metadata before loading the large FC tensor."""

        path = Path(checkpoint_path)
        config_file = path / "config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"DFlash checkpoint is missing {config_file}")
        data = json.loads(config_file.read_text())
        dflash = data.get("dflash_config") or {}
        layer_ids = dflash.get("target_layer_ids") or data.get("target_layer_ids")
        if layer_ids is None:
            raise ValueError(f"DFlash checkpoint {config_file} does not declare target_layer_ids")

        hidden_size = int(data.get("hidden_size", -1))
        num_target_layers = int(data.get("num_target_layers", -1))
        expected_ids = _validate_target_layer_ids(expected_target_layer_ids)
        actual_ids = _validate_target_layer_ids(layer_ids)
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"DFlash checkpoint hidden_size={hidden_size}, expected target hidden_size={expected_hidden_size}"
            )
        if num_target_layers != expected_num_target_layers:
            raise ValueError(
                f"DFlash checkpoint num_target_layers={num_target_layers}, " f"expected {expected_num_target_layers}"
            )
        if actual_ids != expected_ids:
            raise ValueError(f"DFlash checkpoint target_layer_ids={actual_ids}, expected ordered layers={expected_ids}")

        cfg = cls(
            checkpoint_path=path,
            hidden_size=hidden_size,
            num_target_layers=num_target_layers,
            target_layer_ids=actual_ids,
        )
        # Fail at startup, rather than after a 36-layer prefill, if weights are absent or malformed.
        validate_fc_weight(load_dflash_fc_weight(cfg), cfg)
        return cfg


def load_dflash_fc_weight(config: DFlashPrefillConfig) -> torch.Tensor:
    """Load only ``fc.weight`` from the explicitly selected checkpoint."""

    from safetensors import safe_open

    weights_file = config.checkpoint_path / "model.safetensors"
    if not weights_file.is_file():
        raise FileNotFoundError(f"DFlash checkpoint is missing {weights_file}")
    with safe_open(str(weights_file), framework="pt", device="cpu") as handle:
        if config.fc_key not in handle.keys():
            raise KeyError(f"DFlash checkpoint {weights_file} is missing key {config.fc_key!r}")
        return handle.get_tensor(config.fc_key)


def validate_fc_weight(fc_weight: torch.Tensor, config: DFlashPrefillConfig) -> None:
    expected = (config.hidden_size, len(config.target_layer_ids) * config.hidden_size)
    if fc_weight.ndim != 2 or tuple(fc_weight.shape) != expected:
        raise ValueError(
            f"DFlash {config.fc_key} has shape {tuple(fc_weight.shape)}, expected {expected} "
            "([out=hidden, in=num_features*hidden])"
        )


def slice_dflash_fc_weight(
    fc_weight: torch.Tensor,
    *,
    hidden_size: int,
    target_layer_ids: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Return ordered ``[in, out]`` blocks for TTNN/torch ``activation @ weight``."""

    ids = _validate_target_layer_ids(target_layer_ids)
    expected = (hidden_size, len(ids) * hidden_size)
    if fc_weight.ndim != 2 or tuple(fc_weight.shape) != expected:
        raise ValueError(f"fc.weight shape {tuple(fc_weight.shape)} does not match expected {expected}")
    return {
        layer_id: fc_weight[:, i * hidden_size : (i + 1) * hidden_size].T.contiguous() for i, layer_id in enumerate(ids)
    }


def reference_accumulate_reduced_hidden(
    activations: Mapping[int, torch.Tensor],
    fc_weight: torch.Tensor,
    *,
    target_layer_ids: Sequence[int],
) -> torch.Tensor:
    """Pure-torch reference for the streamed pre-norm feature accumulation."""

    ids = _validate_target_layer_ids(target_layer_ids)
    if tuple(activations.keys()) != ids:
        raise ValueError(
            f"activation keys must exactly follow target_layer_ids order {ids}, got {tuple(activations.keys())}"
        )
    first = activations[ids[0]]
    if first.ndim < 2:
        raise ValueError(f"target activation must have at least two dimensions, got {tuple(first.shape)}")
    hidden_size = first.shape[-1]
    slices = slice_dflash_fc_weight(fc_weight, hidden_size=hidden_size, target_layer_ids=ids)
    prefix = first.shape[:-1]
    result = None
    for layer_id in ids:
        activation = activations[layer_id]
        if activation.shape[:-1] != prefix or activation.shape[-1] != hidden_size:
            raise ValueError(
                f"activation for layer {layer_id} has shape {tuple(activation.shape)}, "
                f"expected {tuple(prefix) + (hidden_size,)}"
            )
        projected = activation @ slices[layer_id]
        result = projected if result is None else result + projected
    assert result is not None
    return result


def reference_linear_reduced_hidden(
    activations: Mapping[int, torch.Tensor],
    fc_weight: torch.Tensor,
    *,
    target_layer_ids: Sequence[int],
) -> torch.Tensor:
    """Unsliced reference used by CPU and Galaxy proof tests."""

    ids = _validate_target_layer_ids(target_layer_ids)
    if tuple(activations.keys()) != ids:
        raise ValueError(
            f"activation keys must exactly follow target_layer_ids order {ids}, got {tuple(activations.keys())}"
        )
    return F.linear(torch.cat([activations[layer_id] for layer_id in ids], dim=-1), fc_weight)


@dataclass(frozen=True)
class DFlashFeatureLayout:
    """Physical layout of ``reduced_hidden`` at the prefill/P-D boundary."""

    mesh_shape: tuple[int, int]
    sp_axis: int
    tp_axis: int
    sequence: str = "sp_block_cyclic"
    feature: str = "tp_width_sharded"

    def __post_init__(self) -> None:
        if len(self.mesh_shape) != 2:
            raise ValueError(f"DFlash handoff requires a 2D mesh, got {self.mesh_shape}")
        if {self.sp_axis, self.tp_axis} != {0, 1}:
            raise ValueError(f"sp_axis/tp_axis must be distinct 2D axes, got {self.sp_axis}/{self.tp_axis}")


@dataclass
class DFlashPrefillResult:
    """Typed, opt-in handoff from true prefill to the P/D owner.

    ``reduced_hidden`` remains device-resident and includes the chunk's padded
    storage rows.  ``actual_start``/``actual_end`` are the authoritative
    half-open range of real rows; consumers must not migrate the padded tail.
    ``logits`` is the host fp32 vector for the final real token.
    """

    slot_id: int
    actual_start: int
    actual_end: int
    chunk_size: int
    reduced_hidden: Any
    layout: DFlashFeatureLayout
    logits: Optional[torch.Tensor] = None
    y0: Optional[int] = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError(f"slot_id must be non-negative, got {self.slot_id}")
        if not (0 <= self.actual_start < self.actual_end <= self.actual_start + self.chunk_size):
            raise ValueError(
                f"real range [{self.actual_start}, {self.actual_end}) is outside "
                f"chunk [{self.actual_start}, {self.actual_start + self.chunk_size})"
            )
        if self.y0 is not None and self.logits is None:
            raise ValueError("y0 requires final-token logits")

    @property
    def num_real_tokens(self) -> int:
        return self.actual_end - self.actual_start


DFlashHandoffSink = Callable[[DFlashPrefillResult], None]


class TtDFlashFeatureAccumulator:
    """On-device streamed FC accumulator for true prefill.

    The verifier's MoE output is already sequence-sharded over SP and replicated
    over TP.  Each ``[in, out]`` FC block is therefore sharded only on ``out``
    over TP and replicated over SP.  ``ttnn.linear`` directly produces
    ``[seq/SP, hidden/TP]`` and the five additions preserve that layout; no CCL
    is needed.
    """

    def __init__(
        self,
        mesh_device,
        config: DFlashPrefillConfig,
        fc_weight: torch.Tensor,
        *,
        sp_axis: int = 0,
        tp_axis: int = 1,
        dtype=None,
    ):
        import ttnn

        if {sp_axis, tp_axis} != {0, 1}:
            raise ValueError(f"sp_axis/tp_axis must be distinct 2D axes, got {sp_axis}/{tp_axis}")
        if config.hidden_size % mesh_device.shape[tp_axis] != 0:
            raise ValueError(f"hidden_size={config.hidden_size} must be divisible by TP={mesh_device.shape[tp_axis]}")
        validate_fc_weight(fc_weight, config)
        self.mesh_device = mesh_device
        self.config = config
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.dtype = dtype or ttnn.bfloat16
        self._accumulator = None
        self._tapped: list[int] = []
        self.fc_enqueue_ms: dict[int, float] = {}

        dims = [None, None]
        dims[tp_axis] = 1  # weight [in, out]: shard output columns over TP
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
        host_slices = slice_dflash_fc_weight(
            fc_weight,
            hidden_size=config.hidden_size,
            target_layer_ids=config.target_layer_ids,
        )
        self.fc_slices = {
            layer_id: ttnn.as_tensor(
                weight,
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            for layer_id, weight in host_slices.items()
        }
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def reset(self) -> None:
        import ttnn

        if self._accumulator is not None:
            ttnn.deallocate(self._accumulator)
        self._accumulator = None
        self._tapped = []
        self.fc_enqueue_ms = {}

    def is_target_layer(self, global_layer_idx: int) -> bool:
        return global_layer_idx in self.fc_slices

    def tap(self, moe_output, global_layer_idx: int) -> None:
        """Project a pre-residual MoE output and add it to the running feature."""

        import time

        import ttnn

        if global_layer_idx not in self.fc_slices:
            return
        expected_position = len(self._tapped)
        if expected_position >= len(self.config.target_layer_ids):
            raise RuntimeError(f"DFlash target layer {global_layer_idx} tapped twice without reset/export")
        expected_layer = self.config.target_layer_ids[expected_position]
        if global_layer_idx != expected_layer:
            raise RuntimeError(
                f"DFlash taps arrived out of order: got layer {global_layer_idx}, "
                f"expected layer {expected_layer} after {tuple(self._tapped)}"
            )
        if moe_output.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"DFlash layer {global_layer_idx} activation width={moe_output.shape[-1]}, "
                f"expected {self.config.hidden_size}"
            )

        started = time.perf_counter()
        projected = ttnn.linear(
            moe_output,
            self.fc_slices[global_layer_idx],
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc_enqueue_ms[global_layer_idx] = (time.perf_counter() - started) * 1000.0
        if self._accumulator is None:
            self._accumulator = projected
        else:
            summed = ttnn.add(
                self._accumulator,
                projected,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(self._accumulator)
            ttnn.deallocate(projected)
            self._accumulator = summed
        self._tapped.append(global_layer_idx)

    def export(self):
        """Transfer ownership of the complete TP-width/SP-sequence shard."""

        expected = self.config.target_layer_ids
        if tuple(self._tapped) != expected:
            raise RuntimeError(f"DFlash accumulator tapped {tuple(self._tapped)}, expected {expected}")
        result = self._accumulator
        if result is None:
            raise RuntimeError("DFlash accumulator exported before any target layer")
        self._accumulator = None
        self._tapped = []
        return result
