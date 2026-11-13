# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os

# Hy4-Preview routed experts are bfp4 (contract D3-1: bfp4 mandatory, 8 experts/chip = 12.4 GiB/chip). The DSV4 package's __init__ defaults the same switch to
# bfp8 (setdefault), so this must run before the first DSV4 import: set it here, in the hy4 package that every hy4 module imports first. MOE_COMPUTE_BFP8_WEIGHTS=1 still overrides.
os.environ.setdefault("MOE_COMPUTE_BFP8_WEIGHTS", "0")
