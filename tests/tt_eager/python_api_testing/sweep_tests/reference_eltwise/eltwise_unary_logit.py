# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib.pyplot as plt

value = 1e-6


def function_logit(input, value):
    input = torch.as_tensor(input)
    logit = torch.special.logit(input, eps=value)
    return logit


def custom_logit(input, value):
    input = torch.as_tensor(input)
    value = torch.as_tensor(value)

    mask_condition1 = (input <= 1 - value) & (input >= value) | torch.isnan(value)
    mask_condition2 = input < value
    mask_condition3 = input > (1 - value)

    out1 = torch.where(
        mask_condition1,
        input / (1 - input),
        torch.where(mask_condition2, value / (1 - value), (1 - value) / (1 - (1 - value))),
    )
    out2 = torch.log(out1)

    return out2


x = np.linspace(-2, 1, 6)
z = function_logit(x, value)
z1 = custom_logit(x, value)
plt.plot(x, z, "--g", label="logit")
plt.plot(x, z1, "+r", label="custom logit")
plt.legend(loc="upper center")
plt.show()
