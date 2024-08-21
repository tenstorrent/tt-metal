# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
import torch


def custom_expm1(input):
    result = np.exp(input) - 1
    return result


x = np.linspace(-1, 1, 100)

t_in = torch.from_numpy(x)
t_out = torch.expm1(t_in)
cust_out = custom_expm1(x)


plt.plot(x, t_out, "*b", label="torch expm1")
plt.plot(x, cust_out, "-r", label="custom expm1")
plt.legend(loc="upper center")
