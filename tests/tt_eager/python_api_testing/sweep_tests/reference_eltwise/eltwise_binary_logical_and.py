# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


def logical_and(x, y):
    mul = torch.mul(x, y)
    return torch.where(mul != 0, True, False)


x = np.linspace(1, 10, 100)
x = torch.from_numpy(x)
y = np.linspace(1, 10, 100)
y = torch.from_numpy(y)
lhs = torch.logical_and(x, y)
rhs = logical_and(x, y)


plt.plot(x, lhs, "+r", label="logical_and")
plt.plot(x, rhs, "--g", label="custom logical_and")
plt.legend(loc="lower center")
plt.show()
