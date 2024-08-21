# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import matplotlib.pyplot as plt


def erfinv_approx(x):
    a_for_erf = 0.140012
    b = -4.5469 - (torch.log(1.0 - x**2) * 0.5)
    return torch.sign(x) * torch.sqrt(b + torch.sqrt(b**2 - torch.log(1.0 - x**2) * 7.1427))


x = torch.linspace(-1, 1, 200)
erfinv_result = erfinv_approx(x)
erf_result = torch.erf(erfinv_result)
erfinv_pytorch_result = torch.erfinv(x)

print(" x    erfinv_approx  erfinv_pytorch  erf_pytorch")
print("-" * 80)
for i in range(len(x)):
    print(f"{x[i]:.2f}     {erfinv_result[i]:.6f}       {erfinv_pytorch_result[i]:.6f}       {erf_result[i]:.6f}")
z = erfinv_result
z1 = erfinv_pytorch_result

plt.plot(x, z, "--g", label="custom_erfinv")
plt.plot(x, z1, "+r", label="erfinv")
plt.legend(loc="upper center")
plt.show()
