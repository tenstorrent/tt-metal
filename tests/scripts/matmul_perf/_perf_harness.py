# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for matmul/SDPA Tracy perf scripts.

Every script in this directory is a top-level, no-arg, standalone Python file
so that ``python -m tracy -r -v <script>.py`` works without the subprocess
arg-splitting issues seen with parametrized pytest IDs (see
``docs/claude/profiling.md``).

Keep this module dependency-light: it is imported by scripts that also run
under Tracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import ttnn


DEFAULT_WARMUP_ITERS = 5
DEFAULT_MEASURE_ITERS = 20


@dataclass(frozen=True)
class HarnessConfig:
    """Run parameters for the warmup + measured loop."""

    warmup_iters: int = DEFAULT_WARMUP_ITERS
    measure_iters: int = DEFAULT_MEASURE_ITERS


def pick_compute_kernel_config(
    math_fidelity: "ttnn.MathFidelity | None" = None,
    *,
    math_approx_mode: bool = True,
    fp32_dest_acc_en: bool = False,
    packer_l1_acc: bool = False,
):
    """Return the arch-appropriate ComputeKernelConfig.

    Mirrors the pattern in ``tests/didt/test_lm_head_matmul.py`` and friends:
    the Blackhole and Wormhole configs accept the same knobs but live on
    different classes.
    """
    arch = ttnn.get_arch_name()
    if math_fidelity is None:
        math_fidelity = ttnn.MathFidelity.HiFi2
    if "blackhole" in arch:
        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


def run_warmup_and_measure(
    run_once: Callable[[], object],
    *,
    device,
    config: "HarnessConfig | None" = None,
    label: str = "op",
) -> None:
    """Run ``run_once`` for warmup + measured iterations, synchronizing each time.

    Tracy captures device kernel durations across ALL iterations, but a warmup
    phase is still essential so the program cache is populated before
    measurement begins. We synchronize on every iteration so the profiler
    stream flushes in a predictable order.

    ``run_once`` receives no args and should perform exactly one op invocation,
    returning the ttnn output tensor (which we discard after sync).
    """
    cfg = config if config is not None else HarnessConfig()
    print(f"[{label}] warmup iters={cfg.warmup_iters}")
    for _ in range(cfg.warmup_iters):
        _ = run_once()
        ttnn.synchronize_device(device)
    print(f"[{label}] measure iters={cfg.measure_iters}")
    for _ in range(cfg.measure_iters):
        _ = run_once()
        ttnn.synchronize_device(device)
    print(f"[{label}] done")
