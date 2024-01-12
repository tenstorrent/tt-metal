# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import matplotlib.pyplot as plt


def function_multigammaln(x):
    result = torch.lgamma(x)
    result += torch.lgamma(x - 0.5)
    result += torch.lgamma(x - 1)
    result += torch.lgamma(x - 1.5)
    result += 3.434189657547
    return result


def pytorch_multigammaln(input):
    input = torch.as_tensor(input)
    result = torch.special.multigammaln(input, 4)
    return result


x = torch.tensor([[[[1.6, 1.7], [1.9, 1.8]]]])
z = function_multigammaln(x)
z1 = pytorch_multigammaln(x)
x_values = x.view(-1)
z_values = z.view(-1)
z1_values = z1.view(-1)

plt.plot(x_values, z_values, "--g", label="function_multigammaln")
plt.plot(x_values, z1_values, "+r", label="pytorch_multigammaln")
plt.legend(loc="upper center")
plt.xlabel("x")
plt.ylabel("Value")
plt.title("Comparison of multigammaln")
plt.grid(True)
plt.show()
