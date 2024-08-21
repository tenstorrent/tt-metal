# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_acosh_formula(input):
    # log (x+sqrt(x^2 - 1))
    result = np.log(input + np.sqrt((input**2) - 1))
    return result


x = np.linspace(1, 100, 50)
t_in = torch.from_numpy(x)
t_out = torch.acosh(t_in)

cust_out_formula = custom_acosh_formula(x)

# print("t-in", x)
# print("t-out", t_out)
# print("cus-out-formula", cust_out_formula)

plt.plot(x, t_out, "-b", label="torch asinh")
plt.plot(x, cust_out_formula, "+r", label="custom acosh formula")
plt.legend(loc="upper center")
plt.show()
