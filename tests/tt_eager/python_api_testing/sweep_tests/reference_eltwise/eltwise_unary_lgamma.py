# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import math
from cmath import exp
import matplotlib.pyplot as plt
import numpy as np

p = [
    1,
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
]


def custom_lgamma(z):
    z -= 1
    x = 1
    for i in range(1, 7):
        x += p[i] / (z + i)

    t = z + 5 + 0.5
    y_log = 0.918938531357171 + (z + 0.5) * math.log(t) - t + math.log(x)
    return y_log


x_values = np.linspace(1, 100, 50)
pytorch_lgamma_values = [math.lgamma(x) for x in x_values]
custom_gamma_values = [custom_lgamma(x) for x in x_values]

plt.plot(x_values, pytorch_lgamma_values, "-g", label="lgamma")
plt.plot(x_values, custom_gamma_values, "+r", label="custom lgamma")
plt.legend(loc="upper center")
plt.show()
