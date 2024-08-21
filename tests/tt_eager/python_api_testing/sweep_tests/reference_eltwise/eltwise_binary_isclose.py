# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


# ∣input−other∣≤atol+rtol×∣other∣
def custom_isclose(x, y, rtol, atol):
    sub_lhs = torch.abs(torch.sub(x, y))
    abs_rhs = torch.add(atol, torch.mul(rtol, torch.abs(y)))
    return torch.where(sub_lhs <= abs_rhs, True, False)


x = np.linspace(-10, 10, 100)
x = torch.from_numpy(x)
y = np.linspace(1, 10, 100)
y = torch.from_numpy(y)
atol = 1e-8
rtol = 1e-5
lhs = torch.isclose(x, y, rtol, atol)
rhs = custom_isclose(x, y, rtol, atol)

plt.plot(x, lhs, "+r", label="isclose")
plt.plot(x, rhs, "--g", label="custom_isclose")
plt.legend(loc="upper center")
plt.show()
