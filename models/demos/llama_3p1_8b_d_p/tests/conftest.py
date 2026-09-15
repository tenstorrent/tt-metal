# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Skip the mesh arms the box in hand cannot open, so one suite runs everywhere.

The device arms of these tests span a loudbox (1x8, 4x2), a QuietBox (2x2) and a Galaxy (4x8).
Only one of those is ever the machine under test, and the stock ``mesh_device`` fixture skips only
when a mesh asks for *more* chips than exist -- so every other mismatch used to run and fail in
fabric router sync, which reads as unhealthy ethernet and hides real failures behind noise.

On Blackhole a fabric only comes up across the whole allocation:
``deepseek_v3_d_p/tests/conftest.py`` states it as "BH: only supports all available devices
configs", and a descriptor/fabric sweep on bh-glx-120-c10u08 agrees exactly -- a 32-chip Galaxy
opens 4x8 and 8x4 and refuses 4x2, 2x2, 1x8 and 8x1, while an 8-chip loudbox opens 1x8 and 4x2 and
refuses 2x2. So an arm is feasible when its chip count equals the chip count of the box.

Single-card arms are exempt: with ``fabric_config`` DISABLED no fabric is initialised, so a 1x1
mesh runs on one chip of a 32-chip Galaxy just as well as on a single card -- confirmed by the
1x1 arms passing on both an 8-chip loudbox and a Galaxy.
"""

from __future__ import annotations

import pytest

import ttnn


def _requested_fabric(item):
    """The arm's requested fabric, or None when it never opens one."""
    params = getattr(getattr(item, "callspec", None), "params", {})
    device_params = params.get("device_params")
    if not isinstance(device_params, dict):
        return None
    fabric = device_params.get("fabric_config")
    return None if fabric == ttnn.FabricConfig.DISABLED else fabric


def _requested_mesh_shape(item):
    """The arm's requested (rows, cols), or None when it takes no mesh."""
    params = getattr(getattr(item, "callspec", None), "params", {})
    shape = params.get("mesh_device")
    return shape if isinstance(shape, tuple) and len(shape) == 2 else None


def pytest_collection_modifyitems(config, items):
    mesh_arms = [item for item in items if _requested_mesh_shape(item) and _requested_fabric(item)]
    if not mesh_arms:
        # Nothing selected needs a fabric, so do not open a device just to count chips.
        return

    try:
        num_devices = ttnn.get_num_devices()
    except Exception as exc:  # pragma: no cover - a box that cannot be counted skips nothing
        print(f"llama conftest: could not count devices ({exc}); leaving mesh arms alone")
        return

    for item in mesh_arms:
        rows, cols = _requested_mesh_shape(item)
        needed = rows * cols
        if needed == num_devices:
            continue
        item.add_marker(
            pytest.mark.skip(
                reason=(
                    f"{rows}x{cols} needs {needed} chips and this box has {num_devices}; a Blackhole "
                    f"fabric only comes up across the whole allocation"
                )
            )
        )
