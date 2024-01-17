# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib.pyplot as plt


def function_xlogy(input, other):
    input = torch.as_tensor(input)
    other = torch.as_tensor(other)
    xlogy = torch.special.xlogy(input, other)
    return xlogy


def custom_xlogy(input, other):
    result = np.where(other == 0, 0, np.where(np.isnan(other), np.nan, x * np.log(other)))
    return result


x = np.linspace(1, 100, 20)
y = np.linspace(1, 100, 20)
z = function_xlogy(x, y)
z1 = custom_xlogy(x, y)
plt.plot(x, z, "--g", label="xlogy")
plt.plot(x, z1, "+r", label="custom xlogy")
plt.legend(loc="upper center")
plt.show()
