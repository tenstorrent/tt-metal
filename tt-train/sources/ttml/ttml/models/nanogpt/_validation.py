# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for NanoGPT model configuration."""


def _validate_dropout_probability(probability: float) -> float:
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"dropout probability must be in [0, 1], got {probability!r}")
    return probability
