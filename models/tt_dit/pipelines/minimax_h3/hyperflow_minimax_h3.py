# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sampling contract a HyperFlow adapter publishes in its safetensors header.

A fixed raw sigma grid (9 points, 8 forwards; any other step count is refused), interval conditioning
on ``(t, r)`` with ``r_i = 1 - sigma_{i+1}`` via a second adapted ``time_embedder``, and a blend gate
``emb_t + gate * (emb_r - emb_t)``. The shifts are asserted against the pipeline's, never applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .scheduler import shift_sigmas

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Header entry marking a sampling contract; without it the adapter runs at a caller-supplied step count.
MARKER_KEY = "hyperflow"

_REQUIRED_KEYS = ("hyperflow_version", "hyperflow_gate", "hyperflow_sigmas")


def validate_sigmas(sigmas: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """``sigmas`` as a flat float32 CPU tensor, checked as a rectified-flow grid at header-read time."""
    grid = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
    if grid.numel() < 2:
        raise ValueError(f"a sigma grid needs at least two points, got {grid.numel()}")
    if not bool((grid[1:] < grid[:-1]).all()):
        raise ValueError(f"the sigma grid must be strictly decreasing, got {grid.tolist()}")
    if grid[0].item() > 1.0 or grid[-1].item() != 0.0:
        raise ValueError(f"the sigma grid must start at or below 1.0 and end at exactly 0.0, got {grid.tolist()}")
    return grid


@dataclass(frozen=True)
class MiniMaxH3HyperFlow:
    """One adapter's sampling contract. ``sigmas`` is the raw grid, before either modality's shift."""

    version: str
    gate: float
    sigmas: tuple[float, ...]
    video_shift: float
    audio_shift: float
    tasks: tuple[str, ...] = ()
    subfolders: tuple[str, ...] = ()

    @classmethod
    def from_adapter_metadata(
        cls,
        metadata: Mapping[str, str] | None,
        *,
        video_shift: float,
        audio_shift: float,
    ) -> MiniMaxH3HyperFlow | None:
        """Parse an adapter's header, or ``None`` when :data:`MARKER_KEY` is absent.

        A partial contract raises: a two-time adapter on the default grid silently produces bad video.
        """
        metadata = dict(metadata or {})
        if metadata.get(MARKER_KEY, "").lower() != "true":
            return None

        missing = [key for key in _REQUIRED_KEYS if key not in metadata]
        if missing:
            msg = f"adapter claims `{MARKER_KEY} = true` but its header is missing {missing}"
            raise ValueError(msg)

        sigmas = json.loads(metadata["hyperflow_sigmas"])
        if not isinstance(sigmas, list):
            msg = f"`hyperflow_sigmas` must be a JSON list, got {metadata['hyperflow_sigmas']!r}"
            raise ValueError(msg)
        validate_sigmas(sigmas)

        contract = cls(
            version=metadata["hyperflow_version"],
            gate=float(metadata["hyperflow_gate"]),
            sigmas=tuple(float(sigma) for sigma in sigmas),
            video_shift=float(metadata.get("hyperflow_video_shift", video_shift)),
            audio_shift=float(metadata.get("hyperflow_audio_shift", audio_shift)),
            tasks=tuple(json.loads(metadata["tasks"])) if "tasks" in metadata else (),
            subfolders=(
                tuple(json.loads(metadata["compatible_subfolders"])) if "compatible_subfolders" in metadata else ()
            ),
        )
        contract.assert_shifts(video_shift=video_shift, audio_shift=audio_shift)
        return contract

    @property
    def num_grid_points(self) -> int:
        return len(self.sigmas)

    @property
    def num_forwards(self) -> int:
        """One fewer than the grid points: the terminal sigma has no forward."""
        return len(self.sigmas) - 1

    def assert_shifts(self, *, video_shift: float, audio_shift: float) -> None:
        """Refuse a grid trained at other shifts: every step would land at a noise level never trained on."""
        if (self.video_shift, self.audio_shift) != (video_shift, audio_shift):
            msg = (
                f"adapter was distilled at scheduler shifts (video {self.video_shift:g}, audio "
                f"{self.audio_shift:g}) but this pipeline serves (video {video_shift:g}, audio "
                f"{audio_shift:g}); its grid cannot be reproduced here"
            )
            raise ValueError(msg)

    def assert_supports_task(self, task: str) -> None:
        """Refuse a task the adapter does not list. An empty list claims every task."""
        if self.tasks and task not in self.tasks:
            msg = f"adapter supports {list(self.tasks)}, not {task!r}"
            raise ValueError(msg)

    def assert_supports_subfolder(self, subfolder: str) -> None:
        """Refuse a checkpoint partition the adapter does not list. An empty list claims every one.

        Separate from the task gate: a ``ref2va`` adapter may only carry deltas for ``transformer/``.
        """
        if self.subfolders and subfolder not in self.subfolders:
            msg = f"adapter is compatible with {list(self.subfolders)}, not {subfolder!r}"
            raise ValueError(msg)

    def assert_forwards(self, num_inference_steps: int | None) -> None:
        """Refuse a caller-supplied step count that is not the adapter's own; ``None`` defers to it."""
        if num_inference_steps is not None and int(num_inference_steps) != self.num_grid_points:
            msg = (
                f"this adapter samples a fixed {self.num_grid_points}-point grid "
                f"({self.num_forwards} forwards); num_inference_steps={num_inference_steps} cannot be honoured"
            )
            raise ValueError(msg)

    def modality_sigmas(self, shift: float) -> torch.Tensor:
        """The grid under one modality's shift, for ``set_timesteps(sigmas=...)``.

        No ``unique_consecutive``: dropping a point would change the distilled step count.
        """
        return shift_sigmas(validate_sigmas(self.sigmas), shift)

    def endpoints(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Each step's endpoint ``r_i = 1 - sigma_{i+1}`` on a shifted grid: the next step's timestep, 1.0 last."""
        return 1.0 - sigmas[1:].to(torch.float32)


__all__ = ["MARKER_KEY", "MiniMaxH3HyperFlow", "validate_sigmas"]
