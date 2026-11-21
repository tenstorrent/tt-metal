# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def enabled(args):
    grid = getattr(args, "dram_grid_size", None)
    return (
        getattr(args, "base_model_name", None) == "Llama-3.1-8B"
        and getattr(args, "device_name", None) == "P100"
        and getattr(args, "num_devices", None) == 1
        and not getattr(args, "is_galaxy", False)
        and (grid is None or getattr(grid, "x", None) == 7)
    )


def has_ops(*names):
    group = getattr(ttnn.experimental, "llama31_8b", None)
    return group is not None and all(hasattr(group, n) for n in names)
