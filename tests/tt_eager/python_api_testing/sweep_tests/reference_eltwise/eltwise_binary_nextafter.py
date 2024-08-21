# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


def custom_nextafter(x, target):
    # Eps for GS 0.001953125, WH 1.19209e-07
    eps = torch.tensor(0.001953125)
    res = torch.where(x < target, x + eps, x)
    res = torch.where(x > target, x - eps, res)
    return res


x = np.linspace(-100, 100, 50)
x = torch.from_numpy(x)
target = np.linspace(-50, 50, 50)
target = torch.from_numpy(target)
lhs = torch.nextafter(x, target)
rhs = custom_nextafter(x, target)

plt.plot(x, lhs, "-b", label="torch custom_nextafter")
plt.plot(x, rhs, "+r", label="custom custom_nextafter")
plt.legend(loc="lower center")
plt.show()
