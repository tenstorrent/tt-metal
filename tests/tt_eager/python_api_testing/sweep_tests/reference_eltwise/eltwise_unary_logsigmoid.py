# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)


def log_sigmoid(x):
    sigmoid = torch.sigmoid(x)
    return torch.log(sigmoid)


x = np.linspace(1, 10, 100)
x = torch.from_numpy(x)
lhs = torch.nn.functional.logsigmoid(x)
rhs = log_sigmoid(x)


plt.plot(x, lhs, "+r", label="log_sigmoid")
plt.plot(x, rhs, "--g", label="custom log_sigmoid")
plt.legend(loc="lower center")
plt.show()
