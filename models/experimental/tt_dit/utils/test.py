# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

line_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
ring_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
