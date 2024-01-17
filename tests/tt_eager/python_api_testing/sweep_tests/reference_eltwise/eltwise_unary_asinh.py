# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_asinh_formula(input):
    # log (x+sqrt(1+x^2))
    result = np.log(input + np.sqrt((input**2) + 1))
    return result


x = np.linspace(-100, 100, 1000)
t_in = torch.from_numpy(x)
t_out = torch.asinh(t_in)
cust_out_formula = custom_asinh_formula(x)

# print("t-in", x)
# print("t-out", t_out)
# print("cus-out-formula", cust_out_formula)

plt.plot(x, t_out, "--b", label="torch asinh")
plt.plot(x, cust_out_formula, "+r", label="custom asinh formula")
plt.legend(loc="upper center")
plt.show()
