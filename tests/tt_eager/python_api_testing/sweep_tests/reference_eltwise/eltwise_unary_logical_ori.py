# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_implementation(in_x, in_y):
    if in_y == 0:
        result = in_x != 0
    else:
        result = np.ones_like(in_x)

    return result


x = np.linspace(0, 100, 60, dtype=np.float32)
y = 5

t_x = torch.from_numpy(x)
t_out = torch.logical_or(t_x, torch.tensor(y))

cust_out = custom_implementation(x, y)


plt.plot(x, t_out, "ob", label="torch result")
plt.plot(x, cust_out, "*r", label="custom result")
plt.legend(loc="upper center")
