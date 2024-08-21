# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


def sqr_difference(x, y):
    # Initial approximation
    diff = x - y
    return diff * diff


x = np.linspace(1, 100, 100)
y = np.linspace(1, 50, 100)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
# LHS & RHS has similar implementation, there is no direct API in torch
lhs = sqr_difference(x, y)
rhs = sqr_difference(x, y)

plt.plot(x, lhs, "+r", label="sqr_difference")
plt.plot(x, rhs, "--g", label="sqr_difference")
plt.legend(loc="upper center")
plt.show()
