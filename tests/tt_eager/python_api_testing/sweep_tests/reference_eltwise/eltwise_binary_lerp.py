# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np


def custom_lerp(start, end, weight):
    # LERP FORMULA
    # out = start + weight * (end - start)
    mid_1 = end - start
    mid_2 = mid_1 * weight
    return start + mid_2


# TERNERY INP1, INP2, INP3 all are tensors FOR BINARY WEIGHT IS SCALAR
x = np.linspace(1, 100, 10)
y = np.linspace(1, 50, 10)
start = torch.from_numpy(x)
end = torch.from_numpy(y)
weight = 10

# Computed result
lhs = torch.lerp(start, end, weight)
rhs = custom_lerp(start, end, weight)

plt.plot(x, lhs, "+r", label="torch lerp")
plt.plot(x, rhs, "--g", label="custom lerp")
plt.legend(loc="lower center")
plt.show()
