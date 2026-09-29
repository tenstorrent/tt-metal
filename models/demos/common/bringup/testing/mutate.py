# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mistake injection for proving tests on the CPU (F49): ``BRINGUP_IMPL=mutate:<kind>``.

The module under test is the CPU reference step with its output altered, for the step named by
``BRINGUP_MUTATE_STEP`` only (every other swapped step runs the plain reference; unset = every swapped step). A test
that is meant to catch a wrong device module must FAIL under each mutation that applies to the step's output: float
kinds alter float outputs, integer kinds integer outputs; a kind that does not apply leaves the output unchanged.

    scale1.02, scale0.98   the whole output scaled (a wrong constant, a missing / extra normalization factor)
    halfswap               the two halves of the last dim swapped (TP column halves gathered in the wrong order)
    rowshift               rows rolled by 1 (an off-by-one in the sequence position)
    quarterzero            the first quarter of the rows zeroed (one chip's share of a 4-way SP split missing)
    noise1e-2              Gaussian noise of 1 % of the output's std (a precision loss, seeded)
    sign                   a fixed 1/8 of the columns negated (every 8th column: a sign error in one slice)
    idxshift               integer outputs: every index + 1, clamped to the output's maximum (an off-by-one index)

Control (not a mistake; every check must still PASS): ``bf16`` runs the CPU step on bf16-rounded float inputs and
rounds its float output to bf16, a stand-in for a correct device step's precision (it is not the device's error:
bfp8 / bfp4 weights and bf16 accumulation are not modeled).
"""

from __future__ import annotations

import os

import torch

MUTATE_STEP_ENV = "BRINGUP_MUTATE_STEP"
FLOAT_KINDS = ("scale1.02", "scale0.98", "halfswap", "rowshift", "quarterzero", "noise1e-2", "sign")
INT_KINDS = ("idxshift",)
KINDS = FLOAT_KINDS + INT_KINDS
CONTROL_KINDS = ("bf16",)
PREFIX = "mutate:"


def kind_of(mode: str) -> str | None:
    """The mutation kind of an impl mode (``mutate:<kind>``), else None."""
    if not mode.startswith(PREFIX):
        return None
    kind = mode[len(PREFIX) :]
    if kind not in KINDS + CONTROL_KINDS:
        raise ValueError(f"unknown mutation {kind!r}; kinds: {', '.join(KINDS + CONTROL_KINDS)}")
    return kind


def target_step() -> str | None:
    return os.environ.get(MUTATE_STEP_ENV) or None


def applies(kind: str, t: torch.Tensor) -> bool:
    return (kind in FLOAT_KINDS + CONTROL_KINDS) == t.is_floating_point()


def bf16(t):
    return t.to(torch.bfloat16).float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t


def mutated_step(cpu, kind: str):
    """fn(ctx, *inputs): the CPU step ``cpu`` with the mutation (or the bf16 control) applied."""
    if kind == "bf16":
        return lambda ctx, *x: bf16(cpu(ctx, *[bf16(t) for t in x]))
    return lambda ctx, *x: mutate(cpu(ctx, *x), kind)


def mutate(t: torch.Tensor, kind: str) -> torch.Tensor:
    """A mutated copy of ``t`` (never in place); ``t`` itself when the kind does not apply to its dtype."""
    if not applies(kind, t):
        return t
    if kind == "idxshift":
        return (t + 1).clamp(max=t.max())
    x = t.clone()
    if kind == "scale1.02":
        return x * 1.02
    if kind == "scale0.98":
        return x * 0.98
    if kind == "halfswap":
        n = x.shape[-1]
        return torch.cat([x[..., n // 2 :], x[..., : n // 2]], -1)
    if kind == "rowshift":
        return torch.roll(x, 1, dims=0)
    if kind == "quarterzero":
        x[: max(1, x.shape[0] // 4)] = 0
        return x
    if kind == "noise1e-2":
        g = torch.Generator().manual_seed(0)
        return x + 0.01 * x.float().std() * torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.dtype)
    if kind == "sign":
        x[..., ::8] = -x[..., ::8]
        return x
    raise ValueError(kind)
