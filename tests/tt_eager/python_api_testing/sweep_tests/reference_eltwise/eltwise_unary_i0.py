# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np
import math

torch.manual_seed(2)

cons = [
    1,
    0.25,
    0.015625,
    0.0004340277778,
    0.000006781684028,
    6.78e-08,
    4.71e-10,
    2.40e-12,
    9.39e-15,
    2.90e-17,
    7.24e-20,
    1.50e-22,
]


def i0_accurate(x, iterations):
    y = 0
    x2 = x * x
    for i in range(iterations):
        y = y + (pow(x2, i) * cons[i])
    return y


n = np.linspace(-10, 10, 100)
rhs_approx = i0_accurate(n, 11)
n = torch.from_numpy(n)
lhs = torch.i0(n)

plt.plot(n, lhs, "-r", label="i0")
plt.plot(n, rhs_approx, "+b", label="custom i0 approx")
plt.legend(loc="upper center")
plt.show()
