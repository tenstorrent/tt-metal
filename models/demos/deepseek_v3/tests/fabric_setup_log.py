# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.tt_transformers.tt.ccl import get_num_links


def log_fabric_setup(mesh_device, requested_fabric, requested_num_links=None):
    """Log the fabric config and link count a test asked for next to what the device actually provides."""
    try:
        available_links = (
            f"axis0={get_num_links(mesh_device, cluster_axis=0)}, axis1={get_num_links(mesh_device, cluster_axis=1)}"
        )
    except Exception as e:  # a diagnostic must never be the reason a test fails
        available_links = f"unknown ({e})"

    try:
        fabric_cfg = ttnn.get_fabric_config()
    except Exception as e:  # a diagnostic must never be the reason a test fails
        fabric_cfg = f"unknown ({e})"

    logger.info(
        f"Fabric: requested={requested_fabric}, active={fabric_cfg}; "
        f"links: requested={requested_num_links}, available {available_links}"
    )
