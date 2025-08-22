# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn
import pytest

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
random.seed(0)

parameters = {
    "nightly": {
        "expand_specs": [
            {"shape": [1, 1, 1, 16, 1], "size": [1, 1, 1, 16, 2]},
            {"shape": [1, 1, 1, 16], "size": [1, 12, 16, 16]},
            {"shape": [1, 1, 1, 19], "size": [1, 1, 19, 19]},
            {"shape": [1, 1, 1, 24], "size": [1, 1, 1, 24]},
            {"shape": [1, 1, 1, 24], "size": [1, 1, 24, 24]},
            {"shape": [1, 1, 1, 32], "size": [1, 1, 32, 32]},
            {"shape": [1, 1, 1, 45], "size": [1, 1, 45, 45]},
            {"shape": [1, 1, 1, 46], "size": [1, 1, 1, 46]},
            {"shape": [1, 1, 1, 59], "size": [1, 1, 59, 59]},
            {"shape": [1, 1, 1, 60], "size": [1, 1, 1, 60]},
            {"shape": [1, 1, 1, 920], "size": [-1, 8, -1, -1]},
            # {"shape": [1, 1, 1, "s10 + 1"], "size": [1, 1, 1, "s10 + 1"]},
            {"shape": [1, 1, 1024], "size": [1, -1, -1]},
            {"shape": [1, 1, 12, 16, 2], "size": [1, 1, -1, -1, -1]},
            {"shape": [1, 1, 1280], "size": [1, -1, -1]},
            {"shape": [1, 1, 16384, 256], "size": [1, 1, 16384, 256]},
            {"shape": [1, 1, 16384, 32], "size": [1, 1, 16384, 32]},
            {"shape": [1, 1, 19, 19], "size": [1, 1, 19, 19]},
            {"shape": [1, 1, 19200, 300], "size": [1, 1, 19200, 300]},
            {"shape": [1, 1, 19200, 64], "size": [1, 1, 19200, 64]},
            {"shape": [1, 1, 192], "size": [1, -1, -1]},
            {"shape": [1, 1, 256, 32], "size": [1, 1, 256, 32]},
            {"shape": [1, 1, 300, 64], "size": [1, 1, 300, 64]},
            {"shape": [1, 1, 32, 256], "size": [1, 1, 32, 256]},
            {"shape": [1, 1, 32, 32], "size": [1, 1, 32, 32]},
            {"shape": [1, 1, 38, 38], "size": [1, 512, 38, 38]},
            {"shape": [1, 1, 45, 45], "size": [1, 1, 45, 45]},
            {"shape": [1, 1, 45], "size": [1, 1, 45]},
            {"shape": [1, 1, 512], "size": [-1, 1, -1]},
            {"shape": [1, 1, 512], "size": [-1, -1, -1]},
            {"shape": [1, 1, 59, 59], "size": [1, 1, 59, 59]},
            {"shape": [1, 1, 64, 300], "size": [1, 1, 64, 300]},
            {"shape": [1, 1, 64, 7], "size": [1, 71, 64, 7]},
            {"shape": [1, 1, 7, 64], "size": [1, 71, 7, 64]},
            {"shape": [1, 1, 7, 7], "size": [2, 1, 7, 7]},
            {"shape": [1, 1, 768], "size": [1, -1, -1]},
            {"shape": [1, 100, 192], "size": [1, -1, -1]},
            {"shape": [1, 10], "size": [1, 10]},
            {"shape": [1, 10], "size": [10, 10]},
            {"shape": [1, 12, 1, 10], "size": [1, 12, 1, 10]},
            {"shape": [1, 12, 1, 1], "size": [1, 12, 1, 1]},
            {"shape": [1, 12, 1, 2], "size": [1, 12, 1, 2]},
            {"shape": [1, 12, 1, 46], "size": [1, 12, 1, 46]},
            {"shape": [1, 12, 1, 64], "size": [1, 12, 1, 64]},
            # {"shape": [1, 12, 1, "s0 + 1"], "size": [1, 12, 1, "s0 + 1"]},
            # {"shape": [1, 12, 1, "s10 + 1"], "size": [1, 12, 1, "s10 + 1"]},
            {"shape": [1, 12, 10, 10], "size": [1, 12, 10, 10]},
            {"shape": [1, 12, 10, 64], "size": [1, 12, 10, 64]},
            {"shape": [1, 12, 2, 64], "size": [1, 12, 2, 64]},
            {"shape": [1, 12, 201, 201], "size": [1, 12, 201, 201]},
            {"shape": [1, 12, 201, 64], "size": [1, 12, 201, 64]},
            {"shape": [1, 12, 25, 25], "size": [1, 12, 25, 25]},
            {"shape": [1, 12, 25, 64], "size": [1, 12, 25, 64]},
            {"shape": [1, 12, 45, 45], "size": [1, 12, 45, 45]},
            {"shape": [1, 12, 45, 64], "size": [1, 12, 45, 64]},
            {"shape": [1, 12, 46, 64], "size": [1, 12, 46, 64]},
            {"shape": [1, 12, 64, 10], "size": [1, 12, 64, 10]},
            {"shape": [1, 12, 64, 12], "size": [1, 12, 64, 12]},
            {"shape": [1, 12, 64, 14], "size": [1, 12, 64, 14]},
            {"shape": [1, 12, 64, 16], "size": [1, 12, 64, 16]},
            {"shape": [1, 12, 64, 197], "size": [1, 12, 64, 197]},
            {"shape": [1, 12, 64, 1], "size": [1, 12, 64, 1]},
            {"shape": [1, 12, 64, 201], "size": [1, 12, 64, 201]},
            {"shape": [1, 12, 64, 25], "size": [1, 12, 64, 25]},
            {"shape": [1, 12, 64, 2], "size": [1, 12, 64, 2]},
            {"shape": [1, 12, 64, 45], "size": [1, 12, 64, 45]},
            {"shape": [1, 12, 64, 46], "size": [1, 12, 64, 46]},
            {"shape": [1, 12, 64, 7], "size": [1, 12, 64, 7]},
            {"shape": [1, 12, 64, 9], "size": [1, 12, 64, 9]},
            # {"shape": [1, 12, 64, "s0 + 1"], "size": [1, 12, 64, "s0 + 1"]},
            # {"shape": [1, 12, 64, "s10 + 1"], "size": [1, 12, 64, "s10 + 1"]},
            {"shape": [1, 12, 7, 64], "size": [1, 12, 7, 64]},
            {"shape": [1, 12, 7, 7], "size": [1, 12, 7, 7]},
            {"shape": [1, 12, 9, 64], "size": [1, 12, 9, 64]},
            {"shape": [1, 12, 9, 9], "size": [1, 12, 9, 9]},
            # {"shape": [1, 12, "s0 + 1", 64], "size": [1, 12, "s0 + 1", 64]},
            # {"shape": [1, 12, "s10 + 1", 64], "size": [1, 12, "s10 + 1", 64]},
            {"shape": [1, 136], "size": [100, 136]},
            {"shape": [1, 16, 1, 10], "size": [1, 16, 1, 10]},
            {"shape": [1, 16, 1, 1], "size": [1, 16, 1, 1]},
            {"shape": [1, 16, 1, 2], "size": [1, 16, 1, 2]},
            {"shape": [1, 16, 1, 64], "size": [1, 16, 1, 64]},
            {"shape": [1, 16, 1, 6], "size": [1, 16, 1, 6]},
            # {"shape": [1, 16, 1, "s0 + 1"], "size": [1, 16, 1, "s0 + 1"]},
            # {"shape": [1, 16, 1, "s10 + 1"], "size": [1, 16, 1, "s10 + 1"]},
            {"shape": [1, 16, 10, 10], "size": [1, 16, 10, 10]},
            {"shape": [1, 16, 10, 64], "size": [1, 16, 10, 64]},
            {"shape": [1, 16, 128, 9], "size": [1, 16, 128, 9]},
            {"shape": [1, 16, 197, 197], "size": [1, 16, 197, 197]},
            {"shape": [1, 16, 197, 64], "size": [1, 16, 197, 64]},
            {"shape": [1, 16, 2, 64], "size": [1, 16, 2, 64]},
            {"shape": [1, 16, 256, 256], "size": [1, 16, 256, 256]},
            {"shape": [1, 16, 256, 64], "size": [1, 16, 256, 64]},
            {"shape": [1, 16, 5, 5], "size": [1, 16, 5, 5]},
            {"shape": [1, 16, 5, 64], "size": [1, 16, 5, 64]},
            {"shape": [1, 16, 6, 64], "size": [1, 16, 6, 64]},
            {"shape": [1, 16, 64, 10], "size": [1, 16, 64, 10]},
            {"shape": [1, 16, 64, 197], "size": [1, 16, 64, 197]},
            {"shape": [1, 16, 64, 1], "size": [1, 16, 64, 1]},
            {"shape": [1, 16, 64, 256], "size": [1, 16, 64, 256]},
            {"shape": [1, 16, 64, 2], "size": [1, 16, 64, 2]},
            {"shape": [1, 16, 64, 5], "size": [1, 16, 64, 5]},
            {"shape": [1, 16, 64, 6], "size": [1, 16, 64, 6]},
            {"shape": [1, 16, 64, 9], "size": [1, 16, 64, 9]},
            # {"shape": [1, 16, 64, "s0 + 1"], "size": [1, 16, 64, "s0 + 1"]},
            # {"shape": [1, 16, 64, "s10 + 1"], "size": [1, 16, 64, "s10 + 1"]},
            {"shape": [1, 16, 9, 128], "size": [1, 16, 9, 128]},
            {"shape": [1, 16, 9, 64], "size": [1, 16, 9, 64]},
            {"shape": [1, 16, 9, 9], "size": [1, 16, 9, 9]},
            # {"shape": [1, 16, "s0 + 1", 64], "size": [1, 16, "s0 + 1", 64]},
            # {"shape": [1, 16, "s10 + 1", 64], "size": [1, 16, "s10 + 1", 64]},
            {"shape": [1, 16], "size": [12, 16]},
            {"shape": [1, 17], "size": [13, 17]},
            {"shape": [1, 19], "size": [19, 19]},
            {"shape": [1, 1], "size": [1, 1]},
            {"shape": [1, 1], "size": [1, 512]},
            {"shape": [1, 2, 256, 32], "size": [1, 2, 256, 32]},
            {"shape": [1, 2, 300, 64], "size": [1, 2, 300, 64]},
            {"shape": [1, 2, 32, 256], "size": [1, 2, 32, 256]},
            {"shape": [1, 2, 4096, 256], "size": [1, 2, 4096, 256]},
            {"shape": [1, 2, 4096, 32], "size": [1, 2, 4096, 32]},
            {"shape": [1, 2, 4800, 300], "size": [1, 2, 4800, 300]},
            {"shape": [1, 2, 4800, 64], "size": [1, 2, 4800, 64]},
            {"shape": [1, 2, 64, 300], "size": [1, 2, 64, 300]},
            {"shape": [1, 20], "size": [20, 20]},
            {"shape": [1, 24, 32, 49], "size": [1, 24, 32, 49]},
            {"shape": [1, 24, 32, 64], "size": [1, 24, 32, 64]},
            {"shape": [1, 24, 49, 32], "size": [1, 24, 49, 32]},
            {"shape": [1, 24, 49, 49], "size": [1, 24, 49, 49]},
            {"shape": [1, 24, 64, 1], "size": [1, 24, 64, 32]},
            {"shape": [1, 24, 64, 32], "size": [1, 24, 64, 32]},
            {"shape": [1, 24, 64, 64], "size": [1, 24, 64, 64]},
            {"shape": [1, 25], "size": [1, 25]},
            {"shape": [1, 2], "size": [2, 2]},
            {"shape": [1, 3, 1445, 1445], "size": [1, 3, 1445, 1445]},
            {"shape": [1, 3, 1445, 64], "size": [1, 3, 1445, 64]},
            {"shape": [1, 3, 64, 1445], "size": [1, 3, 64, 1445]},
            {"shape": [1, 32, 32, 49], "size": [1, 32, 32, 49]},
            {"shape": [1, 32, 32, 64], "size": [1, 32, 32, 64]},
            {"shape": [1, 32, 49, 32], "size": [1, 32, 49, 32]},
            {"shape": [1, 32, 49, 49], "size": [1, 32, 49, 49]},
            {"shape": [1, 32, 64, 1], "size": [1, 32, 64, 32]},
            {"shape": [1, 32, 64, 32], "size": [1, 32, 64, 32]},
            {"shape": [1, 32, 64, 64], "size": [1, 32, 64, 64]},
            {"shape": [1, 34], "size": [25, 34]},
            {"shape": [1, 38], "size": [38, 38]},
            {"shape": [1, 3], "size": [3, 3]},
            {"shape": [1, 5, 1, 16, 1], "size": [1, 5, 1, 16, 2]},
            {"shape": [1, 5, 1024, 256], "size": [1, 5, 1024, 256]},
            {"shape": [1, 5, 1024, 32], "size": [1, 5, 1024, 32]},
            {"shape": [1, 5, 1200, 300], "size": [1, 5, 1200, 300]},
            {"shape": [1, 5, 1200, 64], "size": [1, 5, 1200, 64]},
            {"shape": [1, 5, 256, 32], "size": [1, 5, 256, 32]},
            {"shape": [1, 5, 300, 64], "size": [1, 5, 300, 64]},
            {"shape": [1, 5, 32, 256], "size": [1, 5, 32, 256]},
            {"shape": [1, 5, 64, 300], "size": [1, 5, 64, 300]},
            {"shape": [1, 5], "size": [5, 5]},
            {"shape": [1, 6, 1, 15], "size": [1, 6, 1, 15]},
            {"shape": [1, 6, 1, 17], "size": [1, 6, 1, 17]},
            {"shape": [1, 6, 1, 1], "size": [1, 6, 1, 1]},
            {"shape": [1, 6, 1, 2], "size": [1, 6, 1, 2]},
            {"shape": [1, 6, 1, 64], "size": [1, 6, 1, 64]},
            # {"shape": [1, 6, 1, "s0 + 1"], "size": [1, 6, 1, "s0 + 1"]},
            {"shape": [1, 6, 15, 15], "size": [1, 6, 15, 15]},
            {"shape": [1, 6, 15, 64], "size": [1, 6, 15, 64]},
            {"shape": [1, 6, 17, 64], "size": [1, 6, 17, 64]},
            {"shape": [1, 6, 2, 64], "size": [1, 6, 2, 64]},
            {"shape": [1, 6, 64, 15], "size": [1, 6, 64, 15]},
            {"shape": [1, 6, 64, 17], "size": [1, 6, 64, 17]},
            {"shape": [1, 6, 64, 1], "size": [1, 6, 64, 1]},
            {"shape": [1, 6, 64, 2], "size": [1, 6, 64, 2]},
            # {"shape": [1, 6, 64, "s0 + 1"], "size": [1, 6, 64, "s0 + 1"]},
            # {"shape": [1, 6, "s0 + 1", 64], "size": [1, 6, "s0 + 1", 64]},
            {"shape": [1, 64, 64, 9], "size": [1, 64, 64, 9]},
            {"shape": [1, 64, 9, 64], "size": [1, 64, 9, 64]},
            {"shape": [1, 64, 9, 9], "size": [1, 64, 9, 9]},
            {"shape": [1, 68], "size": [50, 68]},
            {"shape": [1, 71, 7, 64], "size": [1, 71, 7, 64]},
            {"shape": [1, 71, 7, 7], "size": [1, 71, 7, 7]},
            {"shape": [1, 8, 1, 10], "size": [1, 8, 1, 10]},
            {"shape": [1, 8, 1, 1], "size": [1, 8, 1, 1]},
            {"shape": [1, 8, 1, 2], "size": [1, 8, 1, 2]},
            {"shape": [1, 8, 1, 64], "size": [1, 8, 1, 64]},
            # {"shape": [1, 8, 1, "s0 + 1"], "size": [1, 8, 1, "s0 + 1"]},
            {"shape": [1, 8, 10, 10], "size": [1, 8, 10, 10]},
            {"shape": [1, 8, 10, 64], "size": [1, 8, 10, 64]},
            {"shape": [1, 8, 2, 64], "size": [1, 8, 2, 64]},
            {"shape": [1, 8, 2048, 160], "size": [1, 8, 2048, 160]},
            {"shape": [1, 8, 2048, 256], "size": [1, 8, 2048, 256]},
            {"shape": [1, 8, 2048, 32], "size": [1, 8, 2048, 32]},
            {"shape": [1, 8, 256, 160], "size": [1, 8, 256, 160]},
            {"shape": [1, 8, 256, 2048], "size": [1, 8, 256, 2048]},
            {"shape": [1, 8, 256, 256], "size": [1, 8, 256, 256]},
            {"shape": [1, 8, 256, 32], "size": [1, 8, 256, 32]},
            {"shape": [1, 8, 256, 96], "size": [1, 8, 256, 96]},
            {"shape": [1, 8, 300, 300], "size": [1, 8, 300, 300]},
            {"shape": [1, 8, 300, 64], "size": [1, 8, 300, 64]},
            {"shape": [1, 8, 32, 2048], "size": [1, 8, 32, 2048]},
            {"shape": [1, 8, 32, 256], "size": [1, 8, 32, 256]},
            {"shape": [1, 8, 64, 10], "size": [1, 8, 64, 10]},
            {"shape": [1, 8, 64, 1], "size": [1, 8, 64, 1]},
            {"shape": [1, 8, 64, 2], "size": [1, 8, 64, 2]},
            {"shape": [1, 8, 64, 300], "size": [1, 8, 64, 300]},
            # {"shape": [1, 8, 64, "s0 + 1"], "size": [1, 8, 64, "s0 + 1"]},
            # {"shape": [1, 8, "s0 + 1", 64], "size": [1, 8, "s0 + 1", 64]},
            {"shape": [1, 9], "size": [7, 9]},
            {"shape": [10, 1], "size": [10, 10]},
            {"shape": [100, 1], "size": [100, 136]},
            {"shape": [12, 1], "size": [12, 16]},
            {"shape": [13, 1], "size": [13, 17]},
            {"shape": [16, 6, 32, 49], "size": [16, 6, 32, 49]},
            {"shape": [16, 6, 32, 64], "size": [16, 6, 32, 64]},
            {"shape": [16, 6, 49, 32], "size": [16, 6, 49, 32]},
            {"shape": [16, 6, 49, 49], "size": [16, 6, 49, 49]},
            {"shape": [16, 6, 64, 1], "size": [16, 6, 64, 32]},
            {"shape": [16, 6, 64, 32], "size": [16, 6, 64, 32]},
            {"shape": [16, 6, 64, 64], "size": [16, 6, 64, 64]},
            {"shape": [16, 8, 32, 49], "size": [16, 8, 32, 49]},
            {"shape": [16, 8, 32, 64], "size": [16, 8, 32, 64]},
            {"shape": [16, 8, 49, 32], "size": [16, 8, 49, 32]},
            {"shape": [16, 8, 49, 49], "size": [16, 8, 49, 49]},
            {"shape": [16, 8, 64, 1], "size": [16, 8, 64, 32]},
            {"shape": [16, 8, 64, 32], "size": [16, 8, 64, 32]},
            {"shape": [16, 8, 64, 64], "size": [16, 8, 64, 64]},
            {"shape": [19, 1], "size": [19, 19]},
            {"shape": [1], "size": [1]},
            {"shape": [2, 1, 1, 7], "size": [2, 1, 7, 7]},
            {"shape": [2, 1], "size": [2, 2]},
            {"shape": [20, 1], "size": [20, 20]},
            {"shape": [2048, 768], "size": [1, -1, -1]},
            {"shape": [24, 12, 64], "size": [24, 12, 64]},
            {"shape": [24, 64, 24], "size": [24, 64, 24]},
            {"shape": [25, 1], "size": [25, 34]},
            {"shape": [256, 1280], "size": [1, -1, -1]},
            {"shape": [3, 1], "size": [3, 3]},
            {"shape": [38, 1], "size": [38, 38]},
            {"shape": [4, 12, 32, 49], "size": [4, 12, 32, 49]},
            {"shape": [4, 12, 32, 64], "size": [4, 12, 32, 64]},
            {"shape": [4, 12, 49, 32], "size": [4, 12, 49, 32]},
            {"shape": [4, 12, 49, 49], "size": [4, 12, 49, 49]},
            {"shape": [4, 12, 64, 1], "size": [4, 12, 64, 32]},
            {"shape": [4, 12, 64, 32], "size": [4, 12, 64, 32]},
            {"shape": [4, 12, 64, 64], "size": [4, 12, 64, 64]},
            {"shape": [4, 16, 32, 49], "size": [4, 16, 32, 49]},
            {"shape": [4, 16, 32, 64], "size": [4, 16, 32, 64]},
            {"shape": [4, 16, 49, 32], "size": [4, 16, 49, 32]},
            {"shape": [4, 16, 49, 49], "size": [4, 16, 49, 49]},
            {"shape": [4, 16, 64, 1], "size": [4, 16, 64, 32]},
            {"shape": [4, 16, 64, 32], "size": [4, 16, 64, 32]},
            {"shape": [4, 16, 64, 64], "size": [4, 16, 64, 64]},
            {"shape": [5, 1], "size": [5, 5]},
            {"shape": [50, 1], "size": [50, 68]},
            {"shape": [64, 3, 32, 49], "size": [64, 3, 32, 49]},
            {"shape": [64, 3, 32, 64], "size": [64, 3, 32, 64]},
            {"shape": [64, 3, 49, 32], "size": [64, 3, 49, 32]},
            {"shape": [64, 3, 49, 49], "size": [64, 3, 49, 49]},
            {"shape": [64, 3, 64, 32], "size": [64, 3, 64, 32]},
            {"shape": [64, 3, 64, 64], "size": [64, 3, 64, 64]},
            {"shape": [64, 4, 32, 49], "size": [64, 4, 32, 49]},
            {"shape": [64, 4, 32, 64], "size": [64, 4, 32, 64]},
            {"shape": [64, 4, 49, 32], "size": [64, 4, 49, 32]},
            {"shape": [64, 4, 49, 49], "size": [64, 4, 49, 49]},
            {"shape": [64, 4, 64, 1], "size": [64, 4, 64, 32]},
            {"shape": [64, 4, 64, 32], "size": [64, 4, 64, 32]},
            {"shape": [64, 4, 64, 64], "size": [64, 4, 64, 64]},
            {"shape": [7, 1], "size": [7, 9]},
            {"shape": [768], "size": [1, 1, -1]},
        ],
        "dtype": [ttnn.bfloat16, ttnn.int32],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    }
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"
    if test_vector["dtype"] == ttnn.bfloat8_b:
        if len(test_vector["slice_specs"]["dims"]) < 2:
            return True, "bfloat8_b not supported with dims  < 2"

    return False, None


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


def run(
    expand_specs,
    dtype,
    layout,
    *,
    device,
):
    torch_tensor = torch_random(expand_specs["shape"], -10, 10, dtype=torch.bfloat16)
    expanded_tensor = torch_tensor.expand(expand_specs["size"])

    ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=layout, dtype=dtype)

    start_time = start_measuring_time()
    expanded_ttnn_tensor = ttnn.expand(ttnn_tensor, expand_specs["size"])
    e2e_perf = stop_measuring_time(start_time)

    ttnn_output_tensor = ttnn.to_torch(expanded_ttnn_tensor)

    result = check_with_pcc(expanded_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
    # raise Exception("Expand is not supported, TODO: implement via recursive concat with itself")


@pytest.mark.parametrize("expand_specs", parameters["nightly"]["expand_specs"])
@pytest.mark.parametrize("dtype", parameters["nightly"]["dtype"])
@pytest.mark.parametrize("layout", parameters["nightly"]["layout"])
def test_run(
    expand_specs,
    dtype,
    layout,
    *,
    device,
):
    torch_tensor = random_torch_tensor(dtype, expand_specs["shape"])
    expanded_tensor = torch_tensor.expand(expand_specs["size"])

    ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=layout, dtype=dtype)

    start_time = start_measuring_time()
    expanded_ttnn_tensor = ttnn.expand(ttnn_tensor, expand_specs["size"])
    e2e_perf = stop_measuring_time(start_time)

    ttnn_output_tensor = ttnn.to_torch(expanded_ttnn_tensor)

    result = check_with_pcc(expanded_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
