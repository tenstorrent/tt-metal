# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pick the 1x1 submesh a per-chip DIDT test runs on.

galaxy_chip_N / logical_chip_N select device id N, the id a full-mesh run logs, so a rerun lands on exactly the chip a
failed run named. The 1x1 submesh order follows the mesh layout, not the device id (the 21st submesh can be device
26), so the submesh is found by its device id, never by position. DIDT_DEVICE_ID=<device id> overrides the test's id.
"""

import os

from loguru import logger

DEVICE_ID_ENV = "DIDT_DEVICE_ID"


def pick_submesh(submeshes, logical_chip_id):
    by_dev = {s.get_device_ids()[0]: s for s in submeshes}
    wanted = os.environ.get(DEVICE_ID_ENV)
    dev = int(wanted) if wanted else logical_chip_id
    if dev not in by_dev:
        source = f"{DEVICE_ID_ENV}={wanted}" if wanted else f"device {dev}"
        raise ValueError(f"{source} is not in this mesh, device ids {sorted(by_dev)}")
    if wanted:
        logger.info(f"{DEVICE_ID_ENV}={wanted} selects device {wanted}; logical_chip_id {logical_chip_id} ignored")
    return by_dev[dev]
