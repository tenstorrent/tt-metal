# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


def logical_xor(x, y):
    return torch.where(x == 0, y != 0, y == 0)


x = [0, 0, 1, 1]
y = [0, 1, 1, 0]
x = torch.as_tensor(x)
y = torch.as_tensor(y)
lhs = torch.logical_xor(x, y)
rhs = logical_xor(x, y)


plt.plot(x, lhs, "+r", label="logical_xor")
plt.plot(x, rhs, "--g", label="custom logical_xor")
plt.legend(loc="lower center")
plt.show()
