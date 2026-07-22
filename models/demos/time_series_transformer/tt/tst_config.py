# tt/tst_config.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Model configuration shared across embedding, attention, and layer modules.

Single source of truth — do not redefine these values locally elsewhere.

HEAD_DIM_TRUE vs HEAD_DIM_PADDED: HF's true per-head width is 13 (26 / 2
heads). TTNN's tile-aligned kernels require each head padded to a full
32-wide tile. Padding affects tensor layout only — every attention scale
factor must use HEAD_DIM_TRUE. Scaling by HEAD_DIM_PADDED instead is a real
numeric error (ratio sqrt(13/32)), invisible to pre-softmax PCC checks since
PCC is scale-invariant.
"""

NUM_HEADS = 2
HEAD_DIM_TRUE = 13
HEAD_DIM_PADDED = 32
NEG_INF = -1e9
D_MODEL = 26  # true, unpadded feature width — never changes
CONTEXT_LENGTH = 24
LAGS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]
PADDED_WIDTH = NUM_HEADS * HEAD_DIM_PADDED  # 64
PREDICTION_LENGTH = 24
NUM_PARALLEL_SAMPLES = 100
