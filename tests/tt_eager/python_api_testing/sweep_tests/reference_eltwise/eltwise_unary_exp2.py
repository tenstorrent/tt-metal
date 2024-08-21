# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_exp2(input):
    # exp2 = 2^input == >exp(log(2) * input)
    # log(2) = 0.6931471805599453
    result = np.exp(0.6931471805599453 * input)
    return result


x = np.linspace(-1, 1, 100)

t_in = torch.from_numpy(x)
t_out = torch.exp2(t_in)
cust_out = custom_exp2(x)


plt.plot(x, t_out, "*b", label="torch exp2")
plt.plot(x, cust_out, "-r", label="custom exp2")
plt.legend(loc="upper center")
