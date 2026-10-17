# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the FLUX 3 Image Klein bringup tests.

Deliberately free of ``ttnn`` so the host-only tests (packing, scheduler, CPU reference) collect and run without
a device build; the mesh/device dicts live in ``device_params.py``. Resolves the checkpoint and goldens from the
environment and documents the golden layout the CPU reference test reads.

Environment:
  ``FLUX3_KLEIN_MODEL_PATH``      explicit checkpoint root (see ``pipelines/flux3_klein/weights_flux3_klein.py``).
  ``FLUX3_KLEIN_GOLDENS``         a directory holding golden cases (any ancestor of the case directories).
  ``FLUX3_KLEIN_REFERENCE_DEPTH`` ``"<mode blocks>,<single blocks>"``: shrink the vendored DiT for smoke runs
                                  against a matching random checkpoint; never set for real goldens.
  ``FLUX3_KLEIN_GOLDEN_PCC``      per-step PCC floor of the CPU reference test (default 0.995). CPU and GPU runs
                                  of the same bf16 code are close, not identical: CPU uses the MATH SDPA backend
                                  (CUDA: cuDNN), and CPU autocast keeps ``layer_norm``/``softmax`` in bf16 where
                                  CUDA autocast runs them in fp32, so CPU goldens differ from GPU goldens by more
                                  than the SDPA backend. The B300 run is the only PSNR-grade golden (PLAN.md
                                  decisions 4 and 8); a CPU golden is the interim oracle.

Golden layout, the format ``generate_goldens.py`` (the GPU harness, PLAN.md section 4) writes and its
``compare_steps.py`` reads: one directory per case under ``<root>/<set>/<case>/`` plus a root ``index.json``
(``<root>`` is ``gpu_traces/flux3_klein/<weights_rev>`` for the real goldens):
  ``record.json``            ``upstream_record`` (the replayed ``examples/<set>/<case>.json`` record) and
                             ``replay``: ``model`` (``distilled`` | ``base``), ``guidance`` (``None`` for the flow
                             map), ``alpha``, ``steps``, ``size`` [w, h] actually rendered, ``seed``, token counts,
                             the noise sha256 and the per-file sha256s.
  ``inputs.safetensors``     ``text_tokens`` (L, 20480) bf16, ``text_ids`` (L, 4) int64, ``cond_tokens`` (N, 128)
                             bf16, ``cond_ids`` (N, 4) fp32 (N may be 0), ``noise`` (h*w, 128) fp32 (bf16-valued),
                             ``image_ids`` (h*w, 4) int64 (the unscaled latent ids the VAE decode takes),
                             ``image_ids_scaled`` (h*w, 4) fp32 (what ``pack`` receives), ``schedule`` (steps+1,)
                             fp32, ``mode_indices.<mode>`` (samples, 2) int64. For the base model
                             ``text_null_tokens``/``text_null_ids`` hold the empty prompt's encoding.
  ``steps.safetensors``      per step k: ``step{k:02d}.x_t`` (h*w, 128) fp32, the state ENTERING the step
                             (``step00.x_t == noise``), ``step{k:02d}.velocity`` (h*w, 128) bf16 (for the base
                             model the CFG-combined velocity; ``.velocity_uncond`` / ``.velocity_cond`` beside it);
                             ``t`` (steps,) fp32, ``t_next`` (steps,) fp32; ``final_tokens`` (h*w, 128) fp32, the
                             state after the last step; ``image_f32`` (3, H, W) fp32, the VAE decode (fp32, outside
                             the DiT's autocast).
  ``blocks_step00.safetensors`` optional per-module captures at step 0 (not read here).
  ``index.json`` (root)      ``weights.<model>.revision``, ``reference.upstream_commit``, ``device`` (the producer),
                             ``cases.<set>/<case>`` with sizes, token counts, PSNR and per-file sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from ....pipelines.flux3_klein.weights_flux3_klein import MODEL_PATH_ENV, WeightsNotFoundError, resolve_weights_dir

GOLDENS_ENV = "FLUX3_KLEIN_GOLDENS"
REFERENCE_DEPTH_ENV = "FLUX3_KLEIN_REFERENCE_DEPTH"
GOLDEN_PCC_ENV = "FLUX3_KLEIN_GOLDEN_PCC"
DEFAULT_GOLDEN_PCC = 0.995

# The vendored reference and its example records (``models/tt_dit/reference/flux3_klein``).
REFERENCE_DIR = Path(__file__).resolve().parents[3] / "reference" / "flux3_klein"
EXAMPLES_DIR = REFERENCE_DIR / "examples"

# Canonical record shapes (PLAN.md section 4): (set, case) -> (text length, reference sizes (w, h), canvas (w, h)).
# Text lengths were reproduced with transformers 5.12.1; sizes are the records' post-preparation sizes.
CANONICAL_RECORDS = {
    ("distilled", "man_dog"): (320, (), (1024, 1024)),
    ("distilled", "2_coffee"): (240, ((1024, 1024),), (1024, 1024)),
    ("distilled", "3_bandana"): (320, ((1024, 1024),), (1024, 1024)),
    ("distilled", "2_3_mugprint"): (320, ((1024, 1024), (1024, 1024)), (1024, 1024)),
    ("distilled", "1_figurine"): (320, ((896, 1120),), (896, 1120)),
    ("distilled", "balloons"): (560, (), (1024, 1024)),
    ("distilled", "cactus"): (320, (), (1024, 1024)),
}
RECORDS = tuple(CANONICAL_RECORDS)
EMPTY_PROMPT_TEXT_LENGTH = 80


def load_record(record_set: str, name: str) -> dict:
    """One vendored ``examples/<set>/<name>.json`` record."""
    return json.loads((EXAMPLES_DIR / record_set / f"{name}.json").read_text())


def checkpoint_dir(variant: str = "distilled", required: tuple[str, ...] | None = None) -> Path | None:
    """The checkpoint snapshot of ``variant`` or ``None`` when it cannot be resolved (callers skip)."""
    try:
        return resolve_weights_dir(variant, required=required)
    except WeightsNotFoundError:
        return None


def require_checkpoint_dir(variant: str = "distilled", required: tuple[str, ...] | None = None) -> Path:
    try:
        return resolve_weights_dir(variant, required=required)
    except WeightsNotFoundError as error:
        pytest.skip(f"{MODEL_PATH_ENV} unresolved: {error}")


def goldens_root() -> Path | None:
    """``$FLUX3_KLEIN_GOLDENS`` when it names an existing directory."""
    value = os.environ.get(GOLDENS_ENV)
    if not value:
        return None
    root = Path(value)
    return root if root.is_dir() else None


def golden_cases(root: Path | None = None) -> list[Path]:
    """Every case directory (one holding ``steps.safetensors``) under ``root``, sorted."""
    root = goldens_root() if root is None else root
    if root is None:
        return []
    return sorted(p.parent for p in root.rglob("steps.safetensors"))


@dataclass
class GoldenCase:
    path: Path
    record: dict
    inputs: dict[str, torch.Tensor]
    steps: dict[str, torch.Tensor]

    @property
    def name(self) -> str:
        return f"{self.path.parent.name}/{self.path.name}"

    @property
    def replay(self) -> dict:
        """The generator's ``replay`` block. A golden without it is malformed: the model is never defaulted."""
        assert "replay" in self.record, f"{self.path}/record.json has no 'replay' block (generate_goldens.py format)"
        return self.record["replay"]

    @property
    def model(self) -> str:
        model = self.replay["model"]
        assert model in ("distilled", "base"), f"{self.path}: unknown model {model!r}"
        return model

    @property
    def guidance(self) -> float | None:
        """CFG scale of the base model; ``None`` for the distilled flow map (and asserted to be so)."""
        guidance = self.replay["guidance"]
        assert (guidance is None) == (self.model == "distilled"), f"{self.path}: {self.model} with guidance {guidance}"
        return None if guidance is None else float(guidance)

    @property
    def alpha(self) -> float:
        return float(self.replay["alpha"])

    @property
    def num_steps(self) -> int:
        steps = int(self.steps["t"].shape[0])
        assert steps == int(
            self.replay["steps"]
        ), f"{self.path}: {steps} recorded steps vs replay.steps {self.replay['steps']}"
        return steps

    def x_t(self, i: int) -> torch.Tensor:
        """The fp32 latent state entering step ``i``; ``i == num_steps`` is the state after the last step."""
        return self.steps["final_tokens"] if i == self.num_steps else self.steps[f"step{i:02d}.x_t"]

    def velocity(self, i: int) -> torch.Tensor:
        """The velocity the Euler update of step ``i`` used (CFG-combined for the base model)."""
        return self.steps[f"step{i:02d}.velocity"]

    def text_null(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The empty-prompt encoding of a CFG golden (``[unconditional, conditional]`` pack order)."""
        return self.inputs["text_null_tokens"], self.inputs["text_null_ids"]


def load_golden_case(case_dir: Path) -> GoldenCase:
    from safetensors.torch import load_file

    record = json.loads((case_dir / "record.json").read_text()) if (case_dir / "record.json").is_file() else {}
    return GoldenCase(
        path=case_dir,
        record=record,
        inputs=load_file(str(case_dir / "inputs.safetensors")),
        steps=load_file(str(case_dir / "steps.safetensors")),
    )


def golden_index(root: Path | None = None) -> dict:
    """The nearest ``index.json`` at or above ``root`` (empty when none)."""
    root = goldens_root() if root is None else root
    for directory in (root, *root.parents) if root is not None else ():
        index = directory / "index.json"
        if index.is_file():
            return json.loads(index.read_text())
    return {}


def golden_pcc_floor() -> float:
    return float(os.environ.get(GOLDEN_PCC_ENV, DEFAULT_GOLDEN_PCC))


def reference_depth() -> tuple[int, int] | None:
    """``(DEPTH, DEPTH_SINGLE)`` override for smoke runs, or ``None`` for the real model."""
    value = os.environ.get(REFERENCE_DEPTH_ENV)
    if not value:
        return None
    mode_blocks, single_blocks = (int(v) for v in value.split(","))
    return mode_blocks, single_blocks


def patch_reference_depth(monkeypatch, dit_module, depth: tuple[int, int] | None) -> None:
    """Shrink the vendored DiT (module-level ``DEPTH``/``DEPTH_SINGLE`` are read at construction and forward)."""
    if depth is None:
        return
    monkeypatch.setattr(dit_module, "DEPTH", depth[0])
    monkeypatch.setattr(dit_module, "DEPTH_SINGLE", depth[1])


def pcc(reference: torch.Tensor, test: torch.Tensor) -> float:
    """Pearson correlation of two tensors, in fp64."""
    a = reference.detach().double().flatten()
    b = test.detach().double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return 1.0 if denominator == 0 else float((a @ b) / denominator)


def sha256_of(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
