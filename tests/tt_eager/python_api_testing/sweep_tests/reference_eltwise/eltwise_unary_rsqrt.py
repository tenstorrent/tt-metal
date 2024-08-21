# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


def rsqrt(x, iterations):
    y = 1.0 / x
    condition = (x > 0) & (x < 1)
    y = torch.where(condition, torch.tensor(1.0), y)
    for _ in range(iterations):
        y = y * (1.5 - 0.5 * x * y * y)  # Newton-Raphson iteration
    return y


n = np.linspace(0, 10, 100)
n = torch.from_numpy(n)
lhs = torch.rsqrt(n)
rhs_approx = rsqrt(n, 10)
rhs_accurate = rsqrt(n, 25)

plt.plot(n, lhs, "-r", label="rsqrt")
plt.plot(n, rhs_accurate, "--g", label="custom rsqrt accurate")
plt.plot(n, rhs_approx, "+y", label="custom rsqrt approx")
plt.legend(loc="upper center")
plt.show()
