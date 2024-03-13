# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import os
from pathlib import Path
from typing import Callable

import tt_lib

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.experimental.mamba.tt.residual_block import TtResidualBlock


class TtTensorLoader:
    def __init__(self, state_dict, device, tt_cache_path: str = ""):
        self.state_dict = state_dict
        self.tt_cache_path = tt_cache_path
        self.device = device

        if len(tt_cache_path) > 0 and not os.path.exists(self.tt_cache_path):
            os.makedirs(self.tt_cache_path)

    def get_tensor_loader(self, layer_num):
        def load_tt_tensor(
            name: str,
            tm_fn: Callable = lambda x: x,
            postfix: str = "",
            device: tt_lib.device = self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
            torch_tensor=None,
        ):
            tensor_name = f"layers.{layer_num}.{name}"

            tensor_cache_filepath = Path(self.tt_cache_path) / (tensor_name + postfix + ".bin")

            if tensor_cache_filepath.exists() and (len(self.tt_cache_path) > 0):
                tt_tensor = tt_lib.tensor.load_tensor(str(tensor_cache_filepath)).to(device, tt_memory_config)
            else:
                if torch_tensor is None:
                    torch_tensor = self.state_dict[tensor_name]
                torch_tensor = tm_fn(torch_tensor)
                tt_tensor = torch2tt_tensor(
                    torch_tensor,
                    device,
                    tt_layout=tt_layout,
                    tt_memory_config=tt_memory_config,
                    tt_dtype=tt_dtype,
                )
                if len(self.tt_cache_path) > 0:
                    tt_lib.tensor.dump_tensor(
                        str(tensor_cache_filepath),
                        tt_tensor.cpu(),
                    )
            return tt_tensor

        return load_tt_tensor


class MambaTT(torch.nn.Module):
    def __init__(self, reference_model, device: tt_lib.device, tt_cache_path: str = "", num_layers=None):
        super().__init__()
        self.embedding = reference_model.embedding
        self.args = reference_model.args
        self.device = device
        self.tt_cache_path = tt_cache_path

        if num_layers is None:
            self.num_layers = len(reference_model.layers)
        else:
            self.num_layers = num_layers
        print(f"Initalizing MambaTT with {self.num_layers} layers")

        loader = TtTensorLoader(reference_model.state_dict(), self.device, tt_cache_path=tt_cache_path)

        self.layers = [TtResidualBlock(self.args, device, loader.get_tensor_loader(i)) for i in range(self.num_layers)]

        self.norm_f = reference_model.norm_f

        self.lm_head = reference_model.lm_head

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # (BS, 1, 1, E)
        x = torch2tt_tensor(
            x,
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        for layer in self.layers:
            x = layer(x)

        x = tt2torch_tensor(x).squeeze(1).to(torch.float32)
        x = self.norm_f(x)
        x = self.lm_head(x)
        return x
