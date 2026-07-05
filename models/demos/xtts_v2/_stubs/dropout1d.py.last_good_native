# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `dropout1d` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_dropout` (a `nn.Dropout`-family module).
Dropout is stochastic ONLY in training; in eval (`model.eval()`, which the PCC
harness enforces) it is the identity map — it neither scales nor zeros any
element. The faithful native forward is therefore a passthrough of the ttnn
input tensor. Captured shapes: in/out `[1, 64, 768]`.
"""

from __future__ import annotations

import ttnn  # noqa: F401  (kept for interface consistency across stubs)


def build(device, torch_module):
    """Return the identity forward (eval-mode dropout is a no-op)."""

    def forward(x, *args, **kwargs):
        return x

    return forward


def dropout1d(x, *args, **kwargs):
    """Bare module-level identity callable (eval-mode dropout)."""
    return x
