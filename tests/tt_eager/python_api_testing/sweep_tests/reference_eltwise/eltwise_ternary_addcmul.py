# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib.pyplot as plt

value = 0.5


def function_addcmul(input, nume, deno, value):
    input = torch.as_tensor(input)
    nume = torch.as_tensor(nume)
    deno = torch.as_tensor(deno)
    addcmul = torch.addcmul(input, nume, deno, value=value)
    return addcmul


def custom_addcmul(input, nume, deno, value):
    out_1 = (deno) * nume * value
    final_out = input + out_1
    return final_out


x = np.linspace(1, 100, 20)
nume = np.linspace(1, 100, 20)
deno = np.linspace(1, 100, 20)

z = function_addcmul(x, nume, deno, value)
z1 = custom_addcmul(x, nume, deno, value)
plt.plot(x, z, "--g", label="addcmul")
plt.plot(x, z1, "+r", label="custom addcmul")
plt.legend(loc="upper center")
plt.show()
