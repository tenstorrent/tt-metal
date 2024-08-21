# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def polygamma_series(n, x, terms=10):
    term = 0.0
    pos_neg = 1 if n % 2 else -1

    for k in range(terms):
        term = term + 1 / (x + k) ** (n + 1)

    return term * pos_neg * math.factorial(n)


# Added the support upto the range(1, 10) & n[1, 10]
x = np.linspace(1, 10, 100)
x = torch.from_numpy(x)
n = 2

custom_poly = polygamma_series(n, x)
torch_poly = torch.polygamma(n, x)

plt.plot(x, torch_poly, "-b", label="torch polygamma")
plt.plot(x, custom_poly, "+r", label="custom polygamma formula")
plt.legend(loc="upper center")
plt.show()
