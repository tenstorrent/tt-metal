# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_asin_mac(input):
    c1 = 1.0 / 6
    c2 = 3.0 / 40
    c3 = 5.0 / 112
    c4 = 35.0 / 1152
    c5 = 63.0 / 2816
    result = (
        (input)
        + (c1 * (np.power(input, 3)))
        + (c2 * (np.power(input, 5)))
        + (c3 * (np.power(input, 7)))
        + (c4 * (np.power(input, 9)))
        + (c5 * (np.power(input, 11)))
    )
    return result


def custom_acos_mac(input):
    result = ((math.pi / 2)) - custom_asin_mac(input)
    return result


x = np.linspace(-1, 1, 100)

t_in = torch.from_numpy(x)
t_out = torch.asin(t_in)
cust_out = custom_asin_mac(x)

t_cout = torch.acos(t_in)
cust_cout = custom_acos_mac(x)

figure, axis = plt.subplots(2)
axis[0].plot(x, t_out, "+b", label="torch asin")
axis[0].plot(x, cust_out, "*r", label="custom asin mac")
axis[0].set_title("ArcSine Function")
axis[0].legend(loc="upper center")

axis[1].plot(x, t_cout, "+b", label="torch acos")
axis[1].plot(x, cust_cout, "*r", label="custom acos mac")
axis[1].set_title("ArcCosine Function")
axis[1].legend(loc="upper center")
