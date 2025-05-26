# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import types


# Dummy class to satisfy unpickler
class TorchTensor(torch.Tensor):
    pass


import sys

core = types.ModuleType("core")
core.TorchTensor = TorchTensor
sys.modules["core"] = core

# Now try loading
data1 = torch.load("not_working_encoder_hidden_states.pt", map_location="cpu")
data2 = torch.load("working_encoder_hidden_states.pt", map_location="cpu")

tensor1 = data1.as_subclass(torch.Tensor)
tensor2 = data2.as_subclass(torch.Tensor)

# Compare
print("hidden states equal?", torch.equal(tensor1, tensor2))

data1 = torch.load("not_working_kv_weights.pt", map_location="cpu")
data2 = torch.load("working_kv_weights.pt", map_location="cpu")

tensor1 = data1.as_subclass(torch.Tensor)
tensor2 = data2.as_subclass(torch.Tensor)

# Compare
print("kv_weights states equal?", torch.equal(tensor1, tensor2))

data1 = torch.load("not_working_kv_fused.pt", map_location="cpu")
data2 = torch.load("working_kv_fused.pt", map_location="cpu")

tensor1 = data1.as_subclass(torch.Tensor)
tensor2 = data2.as_subclass(torch.Tensor)

# Compare
print("kv output  equal?", torch.equal(tensor1, tensor2))
