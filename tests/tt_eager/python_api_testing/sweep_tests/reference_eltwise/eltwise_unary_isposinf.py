# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import torch


def custom_isposinf(x):
    return torch.where(x == float("inf"), True, False)


x = torch.tensor([1.0, 2.0, float("inf"), float("-inf"), 5.0, 6.0, float("nan"), 8.0, 9.0, 10.0])
t_out = torch.isposinf(x)
cust_out_formula = custom_isposinf(x)

plt.plot(x, t_out, "-b", label="torch isposinf")
plt.plot(x, cust_out_formula, "+r", label="custom isposinf formula")
plt.legend(loc="upper center")
plt.show()
