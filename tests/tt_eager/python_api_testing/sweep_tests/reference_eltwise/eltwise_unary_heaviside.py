# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib.pyplot as plt


def function_heaviside(input, other):
    input = torch.as_tensor(input)
    other = torch.as_tensor(other)
    heaviside = torch.heaviside(input, other)
    return heaviside


def custom_heaviside(input, value):
    return np.where(x < 0.0, 0.0, np.where(x == 0, value, 1.0))


x = np.linspace(-100, 100, 30)
y = np.linspace(5, 6, 30)
z = function_heaviside(x, y)
z1 = custom_heaviside(x, y)
plt.plot(x, z, "--g", label="heaviside")
plt.plot(x, z1, "+r", label="custom heaviside")
plt.legend(loc="upper center")
plt.show()
