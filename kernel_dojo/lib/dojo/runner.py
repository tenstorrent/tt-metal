# SPDX-License-Identifier: Apache-2.0
"""Loads exercises, puts tensors on device, runs the learner's kernels."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from . import harness
from .exercise import Case, Comparison, Exercise

EXERCISES_DIR = Path(__file__).resolve().parent.parent.parent / "exercises"

#: Fixed seed so a failing case is reproducible and the golden is stable.
SEED = 1234


@dataclass
class Ctx:
    """Everything an exercise's `program()` needs from the runtime."""

    device: object
    kernel_dir: Path
    #: Set by `bench` so exercises can honour a learner-supplied override.
    overrides: dict = None

    def opt(self, name, default):
        if self.overrides:
            return self.overrides.get(name, default)
        return default


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def list_exercise_dirs() -> list[Path]:
    if not EXERCISES_DIR.is_dir():
        return []
    return sorted(p for p in EXERCISES_DIR.iterdir() if (p / "task.py").is_file())


def resolve_exercise_dir(key: str) -> Path:
    """Accept '3', '03', '03_eltwise_binary', or a full path."""
    key = key.strip().rstrip("/")
    candidate = Path(key)
    if (candidate / "task.py").is_file():
        return candidate.resolve()

    dirs = list_exercise_dirs()
    if not dirs:
        raise SystemExit(f"no exercises found under {EXERCISES_DIR}")

    exact = [d for d in dirs if d.name == key]
    if exact:
        return exact[0]

    padded = key.zfill(2)
    prefixed = [d for d in dirs if d.name.startswith(padded + "_")]
    if len(prefixed) == 1:
        return prefixed[0]

    partial = [d for d in dirs if key.lower() in d.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(d.name for d in partial)
        raise SystemExit(f"'{key}' is ambiguous: {names}")

    names = ", ".join(d.name for d in dirs)
    raise SystemExit(f"unknown exercise '{key}'. Available: {names}")


def load_exercise(exercise_dir: Path) -> Exercise:
    """Import the exercise's task.py and instantiate its EXERCISE class."""
    task_py = exercise_dir / "task.py"
    spec = importlib.util.spec_from_file_location(f"dojo_task_{exercise_dir.name}", task_py)
    module = importlib.util.module_from_spec(spec)
    # Let task.py do `from dojo import harness` without a sys.path dance.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "EXERCISE"):
        raise SystemExit(f"{task_py} does not define EXERCISE")
    ex = module.EXERCISE
    ex = ex() if isinstance(ex, type) else ex
    ex.id = exercise_dir.name
    return ex


def kernel_dir_for(exercise_dir: Path, solution: bool) -> Path:
    d = exercise_dir / ("solution" if solution else "kernels")
    if not d.is_dir():
        raise SystemExit(f"missing kernel directory {d}")
    return d


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _to_device(device, t: torch.Tensor, dtype):
    import ttnn

    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def prepare(ex: Exercise, case: Case, ctx: Ctx, dtype=None):
    """Build device tensors and the program descriptor for one case.

    Returns (io_tensors, program_descriptor, torch_inputs, torch_golden).
    """
    import ttnn

    dtype = dtype or getattr(ex, "dtype", ttnn.bfloat16)

    torch.manual_seed(SEED)
    inputs = ex.make_inputs(case)
    ref = ex.golden(case, inputs)

    harness.set_kernel_dir(str(ctx.kernel_dir))

    dev_inputs = [_to_device(ctx.device, t, dtype) for t in inputs]
    out_shape = ex.output_shape(case, inputs)
    dev_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_shape)),
        dtype,
        ttnn.TILE_LAYOUT,
        ctx.device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    io_tensors = [*dev_inputs, dev_out]
    prog = ex.program(case, io_tensors, ctx)
    return io_tensors, prog, inputs, ref


def run_case(ex: Exercise, case: Case, ctx: Ctx) -> tuple[torch.Tensor, torch.Tensor, Comparison]:
    """Execute one case and compare against the golden."""
    import ttnn

    io_tensors, prog, _inputs, ref = prepare(ex, case, ctx)
    result = ttnn.generic_op(io_tensors, prog)
    out = ttnn.to_torch(result)
    return out, ref, ex.compare(out, ref, case)


# --------------------------------------------------------------------------
# Device lifecycle
# --------------------------------------------------------------------------


class DeviceSession:
    """Opens one device for the lifetime of a command."""

    def __init__(self, device_id: int = 0, l1_small_size: int = 0):
        self.device_id = device_id
        self.l1_small_size = l1_small_size
        self.device = None

    def __enter__(self):
        import ttnn

        self.device = ttnn.open_device(device_id=self.device_id, l1_small_size=self.l1_small_size)
        return self.device

    def __exit__(self, *exc):
        import ttnn

        if self.device is not None:
            ttnn.close_device(self.device)
        return False


def require_tt_metal_home() -> None:
    """tt-metal resolves its own runtime assets through TT_METAL_HOME."""
    if not os.environ.get("TT_METAL_HOME"):
        # The dojo lives at <repo>/kernel_dojo, so the repo root is one up.
        repo = Path(__file__).resolve().parent.parent.parent.parent
        os.environ["TT_METAL_HOME"] = str(repo)
