# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import torch


def custom_isnan(x):
    return torch.where(x == float("nan"), True, False)


x = torch.tensor([1.0, 2.0, float("inf"), 5.0, float("-inf"), 6.0, float("nan"), 8.0, 9.0, 10.0])
t_out = torch.isnan(x)
cust_out_formula = custom_isnan(x)

plt.plot(x, t_out, "-b", label="torch isnan")
plt.plot(x, cust_out_formula, "+r", label="custom isnan formula")
plt.legend(loc="upper center")
plt.show()
