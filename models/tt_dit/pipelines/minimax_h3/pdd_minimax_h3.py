# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The sampling contract of a FastVideo PDD (parallel decoding distillation) export, e.g. FastH3 OmniRef PDD-8.

A PDD student widens ``proj_out`` and ``audio_proj_out`` to ``pdd_steps`` heads, one per interval of a fine grid
``u_0 = 0.999 > ... > u_N = 0`` on the unshifted clock. A forward runs one block ``[start, end)`` of consecutive
heads fused into their integration-weighted mean, and an ordinary Euler step between the block's two node sigmas
applies it: the weights are ``sigma(u_{j+1}) - sigma(u_j)`` and telescope, so the step multiplies the mean by the
block's total. Each modality reaches its sigma through its own rational shift ``f_s(u) = s u M / (u (s - 1) + M)``
with ``M = 0.999`` a fixed point, so both modalities start at 0.999.

Host-side and pure torch: the grid, the node sigmas and the head fusion are computed once per export in float64 /
float32 exactly as FastVideo's ``fastvideo/layers/pdd.py`` does, and only the fused heads go to the device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

INFERENCE_CONTRACT_FILE = "fastvideo_inference.json"
SCHEMA_VERSION = "fasth3-inference-contract-v1"

#: The widened output projections, keyed by modality.
PDD_HEADS = {"video": "proj_out", "audio": "audio_proj_out"}


def shifted_noise_amount(base: torch.Tensor, shift: float, max_t: float) -> torch.Tensor:
    if shift == 1.0:
        return base
    return shift * base * max_t / (base * (shift - 1.0) + max_t)


def shifted_noise_delta(start: torch.Tensor, end: torch.Tensor, shift: float, max_t: float) -> torch.Tensor:
    """``f_s(end) - f_s(start)`` in the cancellation-free rational form FastVideo trains with."""
    if shift == 1.0:
        return end - start
    return shift * max_t**2 * (end - start) / ((max_t + (shift - 1.0) * start) * (max_t + (shift - 1.0) * end))


@dataclass(frozen=True)
class MiniMaxH3PDDContract:
    """``fastvideo_inference.json`` of a PDD export, checked against its transformer config."""

    pdd_steps: int
    step_indices: tuple[int, ...]
    grid_max_t: float
    video_shift: float
    audio_shift: float
    transformer_component: str
    vsa_sparsity: float
    vsa_tile_size: int
    vsa_ref_keep_rate: float

    @classmethod
    def from_export(cls, export_dir: str | Path) -> MiniMaxH3PDDContract:
        export_dir = Path(export_dir)
        raw = json.loads((export_dir / INFERENCE_CONTRACT_FILE).read_text())
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{INFERENCE_CONTRACT_FILE}: schema_version {raw.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION!r}"
            )
        if raw.get("model_type") != "ref2va" or raw.get("transformer_component") != "transformer_ref":
            raise ValueError(f"{INFERENCE_CONTRACT_FILE}: only ref2va PDD exports on transformer_ref are supported")
        if raw.get("vsa_ref_policy") != "p2_multi_region":
            raise ValueError(f"{INFERENCE_CONTRACT_FILE}: vsa_ref_policy {raw.get('vsa_ref_policy')!r} is unsupported")
        if float(raw.get("guidance_scale", 1.0)) != 1.0:
            raise ValueError(f"{INFERENCE_CONTRACT_FILE}: PDD students sample without guidance")
        contract = cls(
            pdd_steps=int(raw["pdd_steps"]),
            step_indices=tuple(int(i) for i in raw["pdd_step_indices"]),
            grid_max_t=float(raw["grid_max_t"]),
            video_shift=float(raw["video_scheduler_shift"]),
            audio_shift=float(raw["audio_scheduler_shift"]),
            transformer_component=str(raw["transformer_component"]),
            vsa_sparsity=float(raw["vsa_sparsity"]),
            vsa_tile_size=int(raw["vsa_tile_size"]),
            vsa_ref_keep_rate=float(raw["vsa_ref_keep_rate"]),
        )
        indices = contract.step_indices
        if indices[0] != 0 or indices[-1] != contract.pdd_steps or any(a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError(
                f"pdd_step_indices {list(indices)} is not a strictly increasing partition of "
                f"[0, {contract.pdd_steps}]"
            )
        if {int(raw["num_inference_steps"]), int(raw["transformer_forwards"])} != {contract.num_forwards}:
            raise ValueError(f"{INFERENCE_CONTRACT_FILE}: step counts disagree with pdd_step_indices")
        config = json.loads((export_dir / contract.transformer_component / "config.json").read_text())
        if int(config.get("pdd_steps", -1)) != contract.pdd_steps:
            raise ValueError(
                f"{contract.transformer_component}/config.json pdd_steps {config.get('pdd_steps')} != "
                f"contract {contract.pdd_steps}"
            )
        return contract

    @property
    def num_forwards(self) -> int:
        return len(self.step_indices) - 1

    def shift(self, modality: str) -> float:
        return {"video": self.video_shift, "audio": self.audio_shift}[modality]

    def fine_grid(self) -> torch.Tensor:
        return torch.linspace(self.grid_max_t, 0.0, self.pdd_steps + 1, dtype=torch.float64).clamp(max=self.grid_max_t)

    def integration_weights(self, modality: str) -> torch.Tensor:
        """``sigma(u_{j+1}) - sigma(u_j)`` per fine interval, float64, negative on the descending grid."""
        grid = self.fine_grid()
        return shifted_noise_delta(grid[:-1], grid[1:], self.shift(modality), self.grid_max_t)

    def node_sigmas(self, modality: str) -> torch.Tensor:
        """The scheduler grid of one modality: one sigma per block boundary, float32, ending at exactly 0."""
        nodes = self.fine_grid()[list(self.step_indices)]
        return shifted_noise_amount(nodes, self.shift(modality), self.grid_max_t).to(torch.float32)

    def block(self, step: int) -> tuple[int, int]:
        return self.step_indices[step], self.step_indices[step + 1]

    def fuse_head(self, modality: str, step: int, parameter: torch.Tensor) -> torch.Tensor:
        """The normal-width parameter of `step`'s fused block, from a ``[pdd_steps * C, ...]`` widened one.

        Heads are channel-major (``[grid, C]`` rows). The block weights are normalized in float64, the head
        accumulation runs in float32 and the result returns to the parameter's dtype, as FastVideo fuses them.
        """
        if parameter.shape[0] % self.pdd_steps:
            raise ValueError(f"{PDD_HEADS[modality]}: {parameter.shape[0]} rows are not {self.pdd_steps} heads")
        start, end = self.block(step)
        block_weights = self.integration_weights(modality)[start:end]
        normalized = (block_weights / block_weights.sum()).to(torch.float32)
        heads = parameter.unflatten(0, (self.pdd_steps, -1))[start:end].to(torch.float32)
        return torch.einsum("n,n...->...", normalized, heads).to(parameter.dtype)

    def fused_head_states(self, widened: dict[str, torch.Tensor]) -> list[dict[str, dict[str, torch.Tensor]]]:
        """Per forward, ``{"proj_out": {"weight", "bias"}, "audio_proj_out": {...}}`` from the widened tensors."""
        return [
            {
                head: {
                    name: self.fuse_head(modality, step, widened[f"{head}.{name}"]).float()
                    for name in ("weight", "bias")
                }
                for modality, head in PDD_HEADS.items()
            }
            for step in range(self.num_forwards)
        ]


__all__ = ["INFERENCE_CONTRACT_FILE", "PDD_HEADS", "MiniMaxH3PDDContract"]
