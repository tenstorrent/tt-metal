# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_implementation(in_x, in_y):
    if in_y == 0:
        return 0
    else:
        mul = torch.mul(in_x, in_y)
        return torch.where(mul != 0, True, False)


x = np.linspace(-100, 100, 60, dtype=np.float32)
x = torch.from_numpy(x)
y = torch.tensor([5])

t_out = torch.logical_and(x, y)
cust_out = custom_implementation(x, y)


plt.plot(x, t_out, "ob", label="torch result")
plt.plot(x, cust_out, "*r", label="custom result")
plt.legend(loc="upper center")
