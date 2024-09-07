# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import ttnn
from torch import nn


class GELUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = ttnn.gelu
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        return self.act(input, memory_config=self.out_mem_config_l1)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu = get_activation("gelu")
