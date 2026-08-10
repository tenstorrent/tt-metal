# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read-only views of ``golden/solver/s<S>/`` and ``golden/pipeline/s<S>/`` (Block 4).

Companion to :mod:`dit_reference`, which covers ``golden/dit/``. Both dumps come from
``reference/dump_goldens.py``.

Key layouts differ between the two directories, which is the only real gotcha:

* **solver** — ``solver.transformer.call{i}.{key}.pt`` where ``key`` is one of
  ``kw_hidden_states`` (``x_t``, ``[1, T, 64]``), ``kw_timestep``/``kw_timestep_r`` (scalars,
  always equal -- see the solver module docstring) and ``out0`` (the velocity, ``[1, T, 64]``).
  ``encoder_hidden_states``/``context_latents`` are step-invariant and deliberately **not**
  re-dumped here; take them from ``golden/dit/s<S>``.
* **pipeline** — flat names: ``timesteps.pt``, ``step_latents.call{i}.pt`` (the latent *after*
  step ``i``), ``final_latents.pt`` (== ``step_latents.call{N-1}``) and ``audio.pt``
  (``[1, 2, samples]``).
"""

from __future__ import annotations

from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent / "golden"
GOLDEN_SOLVER_DIR = _ROOT / "solver"
GOLDEN_PIPELINE_DIR = _ROOT / "pipeline"


def _load(path: Path) -> torch.Tensor:
    if not path.exists():
        msg = f"golden tensor {path} not found"
        raise KeyError(msg)
    return torch.load(path, map_location="cpu", weights_only=False).to(torch.float32)


class _Base:
    _dir_root: Path
    _label: str

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self.dir = self._dir_root / f"s{seq_len}"
        if not (self.dir / "meta.pt").exists():
            msg = (
                f"golden directory {self.dir} is missing or has no meta.pt. Block 0 "
                f"(`reference/dump_goldens.py --blocks {self._label}`) owns the dump."
            )
            raise FileNotFoundError(msg)
        self.meta = torch.load(self.dir / "meta.pt", map_location="cpu", weights_only=False)


class SolverGoldens(_Base):
    """``golden/solver/s<S>`` -- the per-step DiT in/out of the denoising loop."""

    _dir_root = GOLDEN_SOLVER_DIR
    _label = "solver"

    @property
    def timesteps(self) -> list[float]:
        """The schedule actually used by the reference run, from ``meta.pt``."""
        return list(self.meta["timesteps"])

    @property
    def num_steps(self) -> int:
        return len(self.timesteps)

    def has(self, step: int, key: str) -> bool:
        return (self.dir / f"solver.transformer.call{step}.{key}.pt").exists()

    def get(self, step: int, key: str) -> torch.Tensor:
        return _load(self.dir / f"solver.transformer.call{step}.{key}.pt")

    def x_at(self, step: int) -> torch.Tensor:
        """``x_t`` entering step ``step``. ``x_at(0)`` is the initial noise."""
        return self.get(step, "kw_hidden_states")

    def velocity(self, step: int) -> torch.Tensor:
        return self.get(step, "out0")


class PipelineGoldens(_Base):
    """``golden/pipeline/s<S>`` -- per-step latents, final latents and the waveform."""

    _dir_root = GOLDEN_PIPELINE_DIR
    _label = "pipeline"

    @property
    def timesteps(self) -> torch.Tensor:
        return _load(self.dir / "timesteps.pt")

    def step_latents(self, step: int) -> torch.Tensor:
        return _load(self.dir / f"step_latents.call{step}.pt")

    @property
    def final_latents(self) -> torch.Tensor:
        return _load(self.dir / "final_latents.pt")

    @property
    def audio(self) -> torch.Tensor:
        return _load(self.dir / "audio.pt")
