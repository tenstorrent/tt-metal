# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# tan(x) = x + (x^3)/3 + (2x^5)/15 + (17x^7)/315 + (62x^9)/2835 + (1382x^11)/155925 + (21844x^13)/6081075 + (929569x^15)/638512875 +
# (6404582x^17) / 10854718875 + (443861162x^19) / 1856156927625

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_implementation(ix):
    ix = ix % np.pi
    ix = np.where(ix > np.pi / 2, ix - np.pi, np.where(ix < -np.pi / 2, ix + np.pi, ix))

    c1 = 1 / 3.0
    c2 = 2 / 15.0
    c3 = 17 / 315.0
    c4 = 62 / 2835.0
    c5 = 1382 / 155925.0
    c6 = 21844 / 6081075.0

    result = (ix) + (c1 * np.power(ix, 3)) + (c2 * np.power(ix, 5))
    +(c3 * np.power(ix, 7)) + (c4 * np.power(ix, 9)) + (c5 * np.power(ix, 11))
    +(c6 * np.power(ix, 13))
    return result


x = np.linspace(-10, 10, 100)

t_x = torch.from_numpy(x)
t_out = torch.tan(t_x)
cust_out = custom_implementation(x)

plt.plot(x, t_out, "ob", label="torch result")
plt.plot(x, cust_out, "*r", label="custom result")
plt.legend(loc="upper center")
