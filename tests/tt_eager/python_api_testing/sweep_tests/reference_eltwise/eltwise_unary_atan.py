# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib.pyplot as plt


def custom_atan(y):
    if abs(y) > 1:
        t3 = 1 / abs(y)
    else:
        t3 = abs(y)

    t4 = t3 * t3
    t0 = -float(0.013480470)
    t0 = t0 * t4 + float(0.057477314)
    t0 = t0 * t4 - float(0.121239071)
    t0 = t0 * t4 + float(0.195635925)
    t0 = t0 * t4 - float(0.332994597)
    t0 = t0 * t4 + float(0.999995630)
    t3 = t0 * t3

    if abs(y) > 1:
        t3 = 1.570796327 - t3
    else:
        t3 = t3

    if y < 0:
        t3 = -t3
    else:
        t3 = t3

    return t3


# Create a sample input tensor
x = torch.tensor(np.linspace(-100, 100, 100))

output = torch.empty_like(x)

result = torch.atan(x)

for i, (x_val) in enumerate(x):
    output[i] = custom_atan(x_val)

# Plot the results
plt.plot(x, result, "ob", label="torch_atan")
plt.plot(x, output, "+r", label="tt_atan")

plt.legend(loc="upper center")
plt.show()
