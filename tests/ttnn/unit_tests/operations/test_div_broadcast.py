# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np

with ttnn.manage_device(device_id=0) as device:
    x = torch.ones((1, 2, 3, 4), dtype=torch.bfloat16)
    y = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16) * 5.0

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x_y_mult_tt = ttnn.multiply(x_tt, y_tt)
    print(ttnn.to_torch(x_y_mult_tt))

    x_y_divide_tt = ttnn.divide(x_tt, y_tt)
    print(ttnn.to_torch(x_y_divide_tt))
