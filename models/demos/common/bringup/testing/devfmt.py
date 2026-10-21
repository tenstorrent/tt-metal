# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F137 (owner): "a logic bug or a precision gap?" is answered on the CPU, never by a device run at higher precision.

When a per-step (component) test fails on the device, the step's CPU reference runs at the device's real formats on
the same inputs and is recorded next to the gate, never gated:
    <tag>_devfmt_cpu   rel L2 of the device-format CPU step vs the full-precision CPU step (what the formats cost)
    <tag>_devfmt_dev   rel L2 of the device output vs the device-format CPU step
    <tag>_devfmt_precision   1 when the device sits much closer to the device-format step than to full precision
                       (a precision gap: write the known issue with the numbers), 0 otherwise (a logic bug: fix it)
The device formats come from the model's hook ``devfmt_component(spec, layer, step) -> fn(ctx, *inputs)`` (its CPU
step with the weights packed through their stored block formats, bfp8 / bfp4 round trips, bf16 activations where the
device keeps bf16, the configured fidelity as far as the CPU can model it). Without the hook: the inputs and the
output rounded to bf16 (the activations only; the printout says the weights were not modelled).
"""

from __future__ import annotations

import torch

from models.demos.common.bringup.core import metrics

PRECISION_SHARE = 0.5  # the device is "close" to the device-format step when within half its distance to full precision


def _bf16(t):
    return t.to(torch.bfloat16).float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t


def step_fn(s, ref, layer: int, step: str):
    """(fn(ctx, *inputs), how the device formats were modelled)."""
    hooks = s.hooks()
    if hasattr(hooks, "devfmt_component"):
        return hooks.devfmt_component(s, layer, step), "hooks.devfmt_component"
    cpu = ref.component(layer, step)
    return (lambda ctx, *x: _bf16(cpu(ctx, *[_bf16(t) for t in x]))), "bf16 activations only (no devfmt_component)"


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    b = b.double()
    return float((a.double().reshape(b.shape) - b).norm() / b.norm().clamp_min(1e-300))


def diagnose(s, ref, layer: int, step: str, case, cpu_out: torch.Tensor, dev_out: torch.Tensor, tag: str) -> str:
    """Records the diagnosis of a failing float step on ``case``'s inputs; returns 'precision' or 'logic'."""
    from models.demos.common.bringup.testing.component_checks import _clone

    fn, how = step_fn(s, ref, layer, step)
    y = fn(case.rctx(), *_clone(case.inputs))
    a, b, c = rel(y, cpu_out), rel(dev_out, y), rel(dev_out, cpu_out)
    verdict = "precision" if b <= PRECISION_SHARE * c else "logic"
    metrics.record(f"{tag}_devfmt_cpu", a)
    metrics.record(f"{tag}_devfmt_dev", b)
    metrics.record(f"{tag}_devfmt_precision", int(verdict == "precision"))
    print(
        f"note {tag} (diagnostic, not gated; F137): the CPU step at the device's formats ({how}) is {a:.3g} from full "
        f"precision; the device is {b:.3g} from it and {c:.3g} from full precision: "
        + (
            "a precision gap (write the known issue with these numbers)"
            if verdict == "precision"
            else "a logic bug (the formats do not explain it)"
        )
    )
    return verdict
