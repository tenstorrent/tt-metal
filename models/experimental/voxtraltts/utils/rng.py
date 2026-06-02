# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


def acoustic_fm_noise_seed(base_seed: int, step_idx: int) -> int:
    """Derive a deterministic per-step RNG seed for the acoustic flow-matching Euler loop.

    Args:
        base_seed: The top-level generation seed.
        step_idx: The current Euler step index (0-based).

    Returns:
        An integer suitable for ``ttnn.randn(..., seed=...)`` (and legacy ``torch.manual_seed`` on CPU ref).
    """
    return base_seed * 1_000_000 + step_idx
