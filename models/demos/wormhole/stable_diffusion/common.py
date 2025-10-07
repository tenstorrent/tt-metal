# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.common.utility_functions import is_blackhole

# L1 Small Size Constants
SD_L1_SMALL_SIZE = 20928
# Trace Region Size Constants
SD_TRACE_REGION_SIZE = 814510080 if is_blackhole() else 789835776
