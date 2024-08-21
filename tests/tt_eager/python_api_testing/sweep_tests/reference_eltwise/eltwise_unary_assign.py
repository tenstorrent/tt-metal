# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch


def function_assign_unary(x):
    y = x.clone()
    return y


x = torch.tensor([5.0, 5.3, 9.0, 23])
print("Unary Assign : ")
z = function_assign_unary(x)
print(z)
