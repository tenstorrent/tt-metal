# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


#          c1       c2           c3           c4        c5             c6            c7               c8
# ln(z) - 1/(2z) - 1/(12z^2) + 1/(120z^4) - 1/(252z^6) + 1/(240z^8) - 1/(132z^10) + 691/(32760z^12) - 1/(12z^14)
# Asymptotic expansion
def digamma_Asymptotic(x):
    c1 = 1 / 2
    c2 = 1 / 12
    c3 = 1 / 120
    c4 = 1 / 252
    c5 = 1 / 240
    c6 = 1 / 132
    c7 = 691 / 32760
    c8 = 1 / 12
    recip = 1 / x
    result = (
        torch.log(x)
        - (c1 * recip)
        - (c2 * torch.pow(recip, 2))
        + (c3 * torch.pow(recip, 4))
        - (c4 * torch.pow(recip, 6))
        + (c5 * torch.pow(recip, 8))
        - (c6 * torch.pow(recip, 10))
        + (c7 * torch.pow(recip, 12))
        - (c8 * torch.pow(recip, 14))
    )
    return result


x = np.linspace(1, 1e32, 50)
x = torch.from_numpy(x)
lhs = torch.digamma(x)
rhs = digamma_Asymptotic(x)

plt.plot(x, lhs, "-b", label="torch digamma")
plt.plot(x, rhs, "+r", label="custom digamma")
plt.legend(loc="lower center")
plt.show()
