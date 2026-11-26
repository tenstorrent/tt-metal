# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pick the 1x1 submesh a per-chip DIDT test runs on.

The submesh index (galaxy_chip_N) follows the mesh layout, not the device id a full-mesh run logs (galaxy_chip_20
can be device 26). DIDT_DEVICE_ID=<device id> picks by device id instead, so a rerun lands on exactly the chip a
failed run named.
"""

import os

from loguru import logger

DEVICE_ID_ENV = "DIDT_DEVICE_ID"


def pick_submesh(submeshes, logical_chip_id):
    wanted = os.environ.get(DEVICE_ID_ENV)
    if not wanted:
        return submeshes[logical_chip_id]
    by_dev = {s.get_device_ids()[0]: s for s in submeshes}
    if int(wanted) not in by_dev:
        raise ValueError(f"{DEVICE_ID_ENV}={wanted} is not in this mesh, device ids {sorted(by_dev)}")
    logger.info(f"{DEVICE_ID_ENV}={wanted} selects device {wanted}; logical_chip_id {logical_chip_id} ignored")
    return by_dev[int(wanted)]
