# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib.pyplot as plt


def function_logical_noti(input, immediate):
    input = torch.as_tensor(input)
    immediate = torch.full_like(input, immediate)
    l = torch.logical_not(immediate)
    return l


def custom_logical_noti(input, immediate):
    input = torch.as_tensor(input)
    immediate = torch.full_like(input, immediate)
    result = torch.where(immediate == 0, torch.tensor(True), torch.tensor(False))
    return result


x = np.linspace(-3, 3, 10)
y = 0
z = function_logical_noti(x, y)
z1 = custom_logical_noti(x, y)

plt.plot(x, z, "--g", label="logical_noti")
plt.plot(x, z1, "+r", label="custom_logical_noti")
plt.legend(loc="upper center")
plt.show()
