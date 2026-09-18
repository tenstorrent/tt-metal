# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware labelling for the tiered pipeline, and skipping the mesh arms a box cannot open.

Two jobs, both at collection time.

**Labels (tt-blaze#4152).** Every cell carries exactly one of ``cpu_only`` or ``device_required``,
so ``-m cpu_only`` is the PR-CI selector and ``-m device_required`` the dispatch one. The label is
*derived* from whether the cell asks for a device fixture rather than written on each test, because
the two must agree and a hand-written label is free to be wrong: a mislabelled device test does not
fail, it gets scheduled on a runner with no silicon and errors in fixture setup, which reads as a
broken test rather than a broken label. Deriving it means a new test is labelled correctly by
construction, and ``test_hardware_labels_partition_the_suite`` asserts the partition holds.



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

# Asking for any of these means the cell needs an allocation. ``mesh_device`` covers every arm in
# this package today; the rest are listed so a future test that takes a plain device fixture is
# labelled correctly instead of landing in the cpu_only set and erroring on a runner with no chips.
DEVICE_FIXTURES = frozenset({"mesh_device", "device", "all_devices", "pcie_devices", "mesh_device_fixture"})


def _needs_device(item) -> bool:
    """True when the cell requests a device fixture, directly or through another fixture."""
    return bool(DEVICE_FIXTURES & set(getattr(item, "fixturenames", ())))


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
    for item in items:
        item.add_marker(pytest.mark.device_required if _needs_device(item) else pytest.mark.cpu_only)

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
