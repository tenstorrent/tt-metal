# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import torch


def custom_isfinite(x):
    isposinf = torch.where(x == float("inf"), False, True)
    isneginf = torch.where(x == float("-inf"), False, True)
    isnan = torch.where(torch.isnan(x), False, True)
    return torch.mul(isnan, torch.mul(isposinf, isneginf))


x = torch.tensor([1.0, 2.0, float("inf"), float("-inf"), 5.0, 6.0, float("nan"), 8.0, 9.0, float("nan")])
t_out = torch.isfinite(x)
cust_out_formula = custom_isfinite(x)

print(t_out, cust_out_formula)

plt.plot(x, t_out, "-b", label="torch isfinite")
plt.plot(x, cust_out_formula, "+r", label="custom isfinite formula")
plt.legend(loc="upper center")
plt.show()
