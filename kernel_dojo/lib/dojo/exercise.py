# SPDX-License-Identifier: Apache-2.0
"""The contract every exercise implements.

An exercise owns everything except the kernels: it describes the test cases,
generates inputs, computes the golden result on the CPU with torch, and builds
the ttnn program that invokes the learner's kernels. Subclass `Exercise` in an
exercise's `task.py` and assign the subclass to a module-level `EXERCISE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class Case:
    """One test configuration.

    `params` is free-form and exercise-specific — number of tiles, matrix
    dimensions, core grid, whatever the exercise varies.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Whether `bench` should time this case. Keep it to one or two of the
    #: larger cases; timing a single tile measures dispatch, not the kernel.
    perf: bool = False

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass
class Workload:
    """How much work a case represents, for deriving rates from the timing."""

    bytes_moved: int = 0
    flops: int = 0


class Exercise:
    """Base class. Override the pieces your exercise needs."""

    #: Directory name, e.g. "01_tile_copy". Set by the loader.
    id: str = ""
    title: str = ""
    blurb: str = ""

    #: Kernel filenames the learner edits, relative to the exercise's kernels/.
    kernels: tuple[str, ...] = ()

    #: Correctness thresholds. bfloat16 has ~8 bits of mantissa, so exact
    #: equality is the wrong test for anything but a pure copy.
    min_pcc: float = 0.9999
    atol: float = 1e-2
    rtol: float = 1e-2

    # -- test definition ---------------------------------------------------

    def cases(self) -> list[Case]:
        raise NotImplementedError

    def make_inputs(self, case: Case) -> list[torch.Tensor]:
        """Host-side input tensors. Called with the RNG already seeded."""
        raise NotImplementedError

    def golden(self, case: Case, inputs: list[torch.Tensor]) -> torch.Tensor:
        """The reference result, computed on the CPU."""
        raise NotImplementedError

    # -- device side -------------------------------------------------------

    def output_shape(self, case: Case, inputs: list[torch.Tensor]) -> tuple[int, ...]:
        """Shape of the output tensor. Defaults to the first input's shape."""
        return tuple(inputs[0].shape)

    def program(self, case: Case, tensors: list, ctx) -> Any:
        """Build the ttnn.ProgramDescriptor.

        `tensors` is [*device_inputs, device_output]. `ctx` carries the device
        and the resolved kernel directory.
        """
        raise NotImplementedError

    # -- perf --------------------------------------------------------------

    def workload(self, case: Case) -> Workload:
        return Workload()

    #: Target device time in microseconds per perf case name. The grader shows
    #: how far off you are; beating it is the exercise's "hard mode".
    perf_targets: dict[str, float] = {}

    # -- comparison --------------------------------------------------------

    def compare(self, out: torch.Tensor, ref: torch.Tensor, case: Case | None = None) -> "Comparison":
        """Compare against the golden, honouring any per-case tolerance override.

        A case may loosen the bar via `min_pcc` / `atol` / `rtol` in its params
        — reduced-precision modes like LoFi are expected to be less accurate,
        and holding them to the full-precision threshold would report a
        hardware behaviour as a bug.
        """
        get = case.get if case is not None else (lambda _k, d: d)
        return compare_tensors(
            out,
            ref,
            get("min_pcc", self.min_pcc),
            get("atol", self.atol),
            get("rtol", self.rtol),
        )


@dataclass
class Comparison:
    passed: bool
    pcc: float
    max_abs_err: float
    mismatch_frac: float
    detail: str = ""

    def summary(self) -> str:
        return (
            f"pcc={self.pcc:.6f} max_abs_err={self.max_abs_err:.4g} "
            f"mismatched={self.mismatch_frac * 100:.2f}%"
        )


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """PCC, the standard accuracy metric in tt-metal.

    Two identical constant tensors have undefined correlation; treat that as a
    perfect match rather than NaN.
    """
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    if torch.equal(a, b):
        return 1.0
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = a_c.norm() * b_c.norm()
    if denom == 0:
        # At least one side is constant. Equal constants were caught above.
        return 0.0
    return float((a_c @ b_c) / denom)


def compare_tensors(
    out: torch.Tensor, ref: torch.Tensor, min_pcc: float, atol: float, rtol: float
) -> Comparison:
    if out.shape != ref.shape:
        return Comparison(
            passed=False,
            pcc=0.0,
            max_abs_err=float("inf"),
            mismatch_frac=1.0,
            detail=f"shape mismatch: got {tuple(out.shape)}, expected {tuple(ref.shape)}",
        )

    out_f = out.to(torch.float32)
    ref_f = ref.to(torch.float32)

    if not torch.isfinite(out_f).all():
        n_bad = int((~torch.isfinite(out_f)).sum())
        return Comparison(
            passed=False,
            pcc=float("nan"),
            max_abs_err=float("inf"),
            mismatch_frac=n_bad / out_f.numel(),
            detail=f"output contains {n_bad} non-finite values (NaN/Inf)",
        )

    diff = (out_f - ref_f).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    close = torch.isclose(out_f, ref_f, atol=atol, rtol=rtol)
    mismatch = 1.0 - float(close.sum()) / close.numel()
    pcc = pearson(out_f, ref_f)

    passed = (pcc >= min_pcc) and (mismatch == 0.0)
    detail = ""
    if not passed:
        detail = _first_mismatch(out_f, ref_f, close)
    return Comparison(passed, pcc, max_abs, mismatch, detail)


def _first_mismatch(out: torch.Tensor, ref: torch.Tensor, close: torch.Tensor) -> str:
    """Point at a concrete bad element — far more useful than an aggregate."""
    bad = (~close).nonzero()
    if bad.numel() == 0:
        return ""
    idx = tuple(int(i) for i in bad[0])
    return (
        f"first mismatch at {idx}: got {float(out[idx]):.6g}, expected {float(ref[idx]):.6g}"
        f"  ({int((~close).sum())} elements differ)"
    )
