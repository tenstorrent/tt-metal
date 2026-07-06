# ============================================================================
# TODO: scaffold from `stable_diffusion` for `meituan-longcat/LongCat-Image` — adapt model-specific
#       wiring (loader, preprocess, head, output decode) before this passes.
#       Track adaptation points by searching for `SCAFFOLD-TODO` in this folder.
# ============================================================================
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.common.utility_functions import is_blackhole

# L1 Small Size Constants
SD_L1_SMALL_SIZE = 21760 if is_blackhole() else 20928
# Trace Region Size Constants
SD_TRACE_REGION_SIZE = 926000000 if is_blackhole() else 789835776
