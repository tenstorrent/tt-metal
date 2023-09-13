# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import matplotlib.pyplot as plt
import torch


def function_tanhshrink(x):
    x = torch.as_tensor(x)
    result = torch.nn.functional.tanhshrink(x)
    return result


def custom_tanhshrink(x):
    x = torch.as_tensor(x)
    res_tanh = torch.tanh(x)
    result = x - res_tanh
    return result


x = np.linspace(-10, 10, 50)
z = function_tanhshrink(x)
z1 = custom_tanhshrink(x)
plt.plot(x, z, "--g", label="tanhshrink")
plt.plot(x, z1, "+r", label="custom tanhshrink")
plt.legend(loc="upper center")
plt.show()
