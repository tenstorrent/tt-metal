# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sampling contract a HyperFlow adapter publishes in its own safetensors header.

A distillation adapter that only replaces weights needs nothing from the pipeline but a step count.
HyperFlow needs three more things, and all three are facts about the *adapter*, not about the
request, so they travel in the weights file rather than in a call argument:

**A fixed sigma grid.** The adapter was distilled to sample on one 9-point grid -- 8 forwards -- and
any other grid is a different problem than the one it solves. ``num_inference_steps`` therefore stops
being a request parameter: it is read off the file, and a caller asking for another value is refused
rather than quietly served a grid the weights were never trained on.

**Interval conditioning.** Every step is conditioned on the ``(t, r)`` interval it integrates, not on
the point ``t``, with ``r_i = 1 - sigma_{i+1}``. Under ``precomputed_adaln`` that is entirely a
host-side change -- see :mod:`.adaln_precompute` -- because ``time_embedder`` never reaches the
device.

**A blend gate.** ``emb_t + gate * (emb_r - emb_t)``, where ``gate`` comes from the header. A gate of
0 is the base model's single-time conditioning, which is why :func:`.adaln_precompute.blend_two_time`
treats it as the short-circuit rather than as a value to multiply by.

The shifts are read and *asserted*, not applied: this pipeline serves MiniMax's own shifts and has no
override, so a file trained at different ones is the wrong file and not a warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .scheduler import shift_sigmas

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Header entry that marks a file as carrying a sampling contract at all. Its absence means a plain
#: adapter -- FastH3 and friends -- which the pipeline runs at a caller-supplied step count.
MARKER_KEY = "hyperflow"

_REQUIRED_KEYS = ("hyperflow_version", "hyperflow_gate", "hyperflow_sigmas")


def validate_sigmas(sigmas: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Return ``sigmas`` as a flat float32 CPU tensor after checking it is a rectified-flow grid.

    The same contract :meth:`MiniMaxH3Scheduler.set_timesteps` enforces on an explicit grid, checked
    here so a malformed header fails while reading the file rather than eight stack frames later.
    """
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
    """One adapter's sampling contract, parsed and validated.

    ``sigmas`` is the *raw* grid, before either modality's shift. Both shifts are applied here rather
    than stored pre-shifted so the grid in the header stays the one the adapter was distilled on.
    """

    version: str
    gate: float
    sigmas: tuple[float, ...]
    video_shift: float
    audio_shift: float
    tasks: tuple[str, ...] = ()

    @classmethod
    def from_adapter_metadata(
        cls,
        metadata: Mapping[str, str] | None,
        *,
        video_shift: float,
        audio_shift: float,
    ) -> MiniMaxH3HyperFlow | None:
        """Parse an adapter's header, or ``None`` when it publishes no sampling contract.

        Only the *absence* of :data:`MARKER_KEY` yields ``None``. A file that claims the contract and
        then omits part of it raises: the half-read alternative is a two-time adapter sampled on a
        50-point single-time grid, which produces video rather than an error.
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
        )
        contract.assert_shifts(video_shift=video_shift, audio_shift=audio_shift)
        return contract

    @property
    def num_grid_points(self) -> int:
        return len(self.sigmas)

    @property
    def num_forwards(self) -> int:
        """Model evaluations, one fewer than the grid points -- the terminal sigma has none."""
        return len(self.sigmas) - 1

    def assert_shifts(self, *, video_shift: float, audio_shift: float) -> None:
        """Refuse a grid trained at shifts this pipeline cannot reproduce.

        The shift is what maps the raw grid onto each modality's schedule, so training and inference
        disagreeing on it means every step lands at a noise level the adapter never saw. There is no
        shift override here to reconcile them with, which makes this the wrong file, not a warning.
        """
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

    def assert_forwards(self, num_inference_steps: int | None) -> None:
        """Refuse a caller-supplied step count that is not the adapter's own.

        ``None`` means the caller deferred to the adapter, which is the normal path.
        """
        if num_inference_steps is not None and int(num_inference_steps) != self.num_grid_points:
            msg = (
                f"this adapter samples a fixed {self.num_grid_points}-point grid "
                f"({self.num_forwards} forwards); num_inference_steps={num_inference_steps} cannot be honoured"
            )
            raise ValueError(msg)

    def modality_sigmas(self, shift: float) -> torch.Tensor:
        """The adapter's grid under one modality's shift, ready for ``set_timesteps(sigmas=...)``.

        No ``unique_consecutive`` here, unlike the scheduler's own ``linspace`` path: an 8-forward
        grid is far too coarse for the shift to collapse neighbours, and silently dropping a point
        would change the step count the adapter was distilled for.
        """
        return shift_sigmas(validate_sigmas(self.sigmas), shift)

    def endpoints(self, sigmas: torch.Tensor) -> torch.Tensor:
        """The endpoint of every step, ``r_i = 1 - sigma_{i+1}``, on a shifted grid.

        Pairs with ``MiniMaxH3Scheduler.timesteps == 1 - sigmas[:-1]``: step ``i`` carries its rows
        from ``t_i`` to ``r_i``, so the endpoint of one step is the next step's timestep -- and the
        last step's endpoint is 1.0, a clean sample.
        """
        return 1.0 - sigmas[1:].to(torch.float32)

    def identity(self) -> str:
        """A cache-key term covering everything about this contract that changes a table's rows."""
        grid = ",".join(f"{sigma:.9g}" for sigma in self.sigmas)
        return f"hyperflow={self.version}@gate{self.gate:g}@[{grid}]"


__all__ = ["MARKER_KEY", "MiniMaxH3HyperFlow", "validate_sigmas"]
