# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import numpy as np

input = torch.randn(2, 3, 320, 64)
print(input, input.shape)
repeat = 5

print("TORCH OUTPUT")
LHS = torch.repeat_interleave(input=input, repeats=repeat, dim=2)
print(LHS, LHS.shape)

print("RESHAPE OUTPUT")
RESHAPE = torch.reshape(input, shape=(1, 1, input.size(0) * input.size(1) * input.size(2), input.size(3)))
print(RESHAPE, RESHAPE.shape)

print("CONCAT OUTPUT")
CONCAT = torch.cat([RESHAPE] * repeat, dim=1)
print(CONCAT, CONCAT.shape)

print("PERMUTE OUTPUT")
PERMUTE1 = torch.permute(CONCAT, dims=(0, 2, 1, 3))
print(PERMUTE1, PERMUTE1.shape)

print("RESHAPE OUTPUT")
RHS = torch.reshape(PERMUTE1, shape=(input.size(0), input.size(1), input.size(2) * repeat, input.size(3)))
print(RHS, RHS.shape)

assert pytest.approx(LHS) == RHS
