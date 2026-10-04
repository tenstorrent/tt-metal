# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

# Validated configuration for every reported number: bfp8 routed-expert weights (bfp4 costs ~0.05-0.08 logits PCC and only ~3 ms/token, see PREFILL/BFP4 notes)
# and fp32 accumulation in moe_compute. Both are read from the environment at weight-prep / program-creation time (also by the C++ program factory), so they must
# be set before the first op: setdefault keeps an explicit override working (MOE_COMPUTE_BFP8_WEIGHTS=0 for bfp4 experiments).
os.environ.setdefault("MOE_COMPUTE_BFP8_WEIGHTS", "1")
os.environ.setdefault("MOE_COMPUTE_FP32_ACC", "1")
