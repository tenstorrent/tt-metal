# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_implementation(in_x, in_y):
    r1 = in_x != 0
    r2 = in_y != 0
    add_r = r1 + r2
    result = add_r > 0

    return result


x = np.linspace(-100, 100, 60, dtype=np.float32)
y = np.linspace(-100, 100, 60, dtype=np.float32)
# y[5] = 0
# y[6] = 0
# x[5] = 0
# x[6] = 0

t_x = torch.from_numpy(x)
t_y = torch.from_numpy(y)
t_out = torch.logical_or(t_x, t_y)
cust_out = custom_implementation(x, y)


plt.plot(y, t_out, "ob", label="torch result")
plt.plot(y, cust_out, "*r", label="custom result")
plt.legend(loc="upper center")
