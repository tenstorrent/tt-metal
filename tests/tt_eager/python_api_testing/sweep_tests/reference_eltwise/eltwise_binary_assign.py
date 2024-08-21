# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch


def function_assign_binary(x, y):
    if x.shape != y.shape or x.is_contiguous() != y.is_contiguous():
        raise ValueError("Tensors x and y do not have the same shape and memory layout.")
    y.copy_(x)
    print(y)


x = torch.tensor([5.0, 5.3, 9.0, 23])
y = torch.tensor([5.0, 12, 9.0, 23])
print("Binary Assign : ")
function_assign_binary(x, y)
