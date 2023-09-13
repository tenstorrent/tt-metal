# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def function_add_alpha(input, other, alpha):
    input = torch.as_tensor(input)
    other = torch.as_tensor(other)
    addalpha = torch.add(input, other, alpha=alpha)
    return addalpha


def custom_add_alpha(input, other, alpha):
    result = input + (alpha * other)
    return result


x = np.linspace(1, 100, 10)
y = np.linspace(1, 100, 10)
alpha = random.randint(1, 100)
z = function_add_alpha(x, y, alpha)
z1 = custom_add_alpha(x, y, alpha)
plt.plot(x, z, "--g", label="add alpha")
plt.plot(x, z1, "+r", label="custom add alpha")
plt.legend(loc="upper center")
plt.show()
