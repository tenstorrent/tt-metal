# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Architecture-dependent choices of the Quasar fork of Panoptic-DeepLab."""

import ttnn


def is_quasar(device) -> bool:
    return device.arch() == ttnn.device.Arch.QUASAR


def block_float_dtype(device):
    """The dtype the model uses for its bfloat8_b tensors on ``device``.

    Quasar has no block-float formats (``is_supported_quasar`` in tt_backend_api_types.cpp
    rejects Bfp8_b / Bfp4_b), so every bfloat8_b weight and activation runs as bfloat16
    there for now. Wormhole and Blackhole keep bfloat8_b.
    """
    return ttnn.bfloat16 if is_quasar(device) else ttnn.bfloat8_b


def has_tuned_profile(device) -> bool:
    """Whether ``device``'s grid has a hand-tuned profile: 5x4 (20 cores) or 11x10 (110 cores)."""
    grid = device.compute_with_storage_grid_size()
    return (grid.x, grid.y) in ((5, 4), (11, 10))


def uses_20_core_path(device) -> bool:
    """Whether the model takes its 20-core (memory-conservative) forward paths on ``device``.

    The model has two sets of forward paths: the 110-core one (11x10 Blackhole), which keeps
    more activations in L1, and the 20-core one (5x4), which moves them to DRAM between stages.
    Every grid without a tuned profile (e.g. Quasar's 8x4 and the emulator's 2 cores) takes the
    20-core paths, matching ``AnyGridOptimiser``, which inherits from the 20-core profile.
    """
    grid = device.compute_with_storage_grid_size()
    return not (grid.x == 11 and grid.y == 10)
