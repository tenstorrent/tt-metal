# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# The tests and the demo pass fabric_config=True. Without this fixture the root conftest reads True as
# FabricConfig 1 (FABRIC_1D_NEIGHBOR_EXCHANGE), and the all-gathers on 8 devices hang.
from models.ttt_compat.conftest import device_params

__all__ = ["device_params"]
