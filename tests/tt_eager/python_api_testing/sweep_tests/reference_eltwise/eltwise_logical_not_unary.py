# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib.pyplot as plt


def function_logical_not(input):
    input = torch.as_tensor(input)
    l = torch.logical_not(input)
    return l


def custom_logical_not(input):
    x = torch.as_tensor(input)
    result = torch.where(x == 0.0, True, False)
    return result


x = np.linspace(-3, 3, 10)
z = function_logical_not(x)
z1 = custom_logical_not(x)
plt.plot(x, z, "--g", label="logical_not")
plt.plot(x, z1, "+r", label="custom logical_not")
plt.legend(loc="upper center")
plt.show()
