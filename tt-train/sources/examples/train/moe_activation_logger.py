# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sparse per-step per-expert activation-probability logger for DeepSeek MoE.

Usage from the training loop::

    from moe_activation_logger import should_log_step, log_step_expert_balance

    if args.log_expert_activations and should_log_step(global_step):
        log_step_expert_balance(args.log_expert_activations, global_step, model.get_moe_layers())

The logger is intentionally minimal and side-effect free beyond the CSV
append: no device calls, no torch imports, no global state. It relies on
``MoE.read_activation_probabilities()`` which must be called *before*
``update_expert_bias()`` (which resets the underlying ``_token_counts``).

CSV schema
----------
One row per (step, layer, expert)::

    step,layer,expert,prob

where ``prob`` is the fraction of tokens that selected this expert during
the step (``[0, 1]``, uniform target ``n_activated / num_experts``).
"""

from __future__ import annotations

import csv
import os
from typing import Iterable


_HEADER = ("step", "layer", "expert", "prob")


def should_log_step(step: int) -> bool:
    """Return True on steps 1..10 inclusive, then on every 100th step.

    Examples: 1, 2, ..., 10, 100, 200, 300, ...
    """
    if step <= 0:
        return False
    if 1 <= step <= 10:
        return True
    return step % 100 == 0


def log_step_expert_balance(csv_path: str, step: int, moe_layers: Iterable) -> None:
    """Append per-expert activation probabilities for ``step`` to ``csv_path``.

    Writes the CSV header on first call (when the file does not yet exist).
    ``moe_layers`` must be an iterable of objects exposing
    ``read_activation_probabilities() -> numpy.ndarray`` (shape
    ``(num_experts,)``), typically the result of ``DeepSeek.get_moe_layers()``.
    """
    # Materialise once -- we iterate twice (to write rows) and want a stable
    # list even if ``get_moe_layers`` returns a generator.
    layers = list(moe_layers)
    if not layers:
        return

    file_exists = os.path.exists(csv_path)
    file_mode = "a" if step > 1 else "w"

    # Line-buffered so partial runs still produce a readable file.
    with open(csv_path, file_mode, newline="", buffering=1) as fh:
        writer = csv.writer(fh)
        if step == 1 or not file_exists:
            writer.writerow(_HEADER)
        for layer_idx, moe in enumerate(layers):
            probs = moe.read_activation_probabilities()
            for expert_idx, prob in enumerate(probs):
                writer.writerow((step, layer_idx, expert_idx, f"{float(prob):.6f}"))
