# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib as ttl

from loguru import logger

from pathlib import Path
from typing import Callable, Optional
from models.demos.mamba.tt.types import ModelMode
from models.demos.mamba.tt.residual_block import TtResidualBlock
from models.demos.mamba.reference.decode_model import RMSNorm


class TtTensorLoader:
    def __init__(self, state_dict, device, tt_cache_path: Optional[str] = None):
        self.state_dict = state_dict
        self.tt_cache_path = tt_cache_path
        self.device = device

    def get_tensor_loader(self, layer_num: Optional[int] = None):
        def load_tt_tensor(
            name: str,
            tm_fn: Callable = lambda x: x,
            postfix: str = "",
            device: ttnn.Device = self.device,
            tt_layout=ttnn.TILE_LAYOUT,
            tt_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tt_dtype=ttnn.bfloat16,
            torch_tensor=None,
            return_as_torch=False,
            return_layer_num=False,
        ):
            if layer_num == None:
                tensor_name = name
            else:
                tensor_name = f"layers.{layer_num}.{name}"
                if return_layer_num:
                    return layer_num

            if self.tt_cache_path is not None:
                tensor_cache_filepath = str(Path(self.tt_cache_path) / (tensor_name + postfix))
            else:
                tensor_cache_filepath = None

            if torch_tensor is None:
                torch_tensor = self.state_dict[tensor_name]
            torch_tensor = tm_fn(torch_tensor)

            if return_as_torch:
                return torch_tensor

            # All tensors need to be rank 4 because of op performance issues with rank 1/2 inputs in ttnn
            while len(torch_tensor.size()) < 4:
                torch_tensor = torch_tensor.unsqueeze(0)

            tt_tensor = ttnn.as_tensor(
                torch_tensor,
                device=device,
                layout=tt_layout,
                memory_config=tt_memory_config,
                dtype=tt_dtype,
                cache_file_name=tensor_cache_filepath,
            )
            return tt_tensor

        return load_tt_tensor


class MambaTT(torch.nn.Module):
    def __init__(
        self,
        reference_model,
        device: ttnn.Device,
        configs,
        tt_cache_path: Optional[str] = None,
        num_layers=None,
    ):
        super().__init__()
        self.args = reference_model.args
        self.device = device
        self.tt_cache_path = tt_cache_path
        self.configs = configs

        if num_layers is None:
            self.num_layers = len(reference_model.layers)
        else:
            self.num_layers = num_layers
        logger.info(f"Initalizing MambaTT with {self.num_layers} layers")

        self.embedding = reference_model.embedding

        loader = TtTensorLoader(reference_model.state_dict(), self.device, tt_cache_path=tt_cache_path)

        self.layers = [
            TtResidualBlock(self.args, device, configs, loader.get_tensor_loader(i)) for i in range(self.num_layers)
        ]

        self.use_torch_rms_norm = False

        load_fn = loader.get_tensor_loader()
        if not self.use_torch_rms_norm:
            self.norm_f_weights = load_fn(
                "norm_f.weight",
                tt_dtype=ttnn.bfloat8_b,
            )
        else:
            self.torch_norm_f_weights = load_fn("norm_f.weight", return_as_torch=True)
            self.torch_norm_f = RMSNorm(self.args.d_model)
            self.torch_norm_f.weight = torch.nn.Parameter(self.torch_norm_f_weights)

        self.lm_head_weights = load_fn(
            "lm_head.weight",
            lambda x: x.transpose(-1, -2),
            tt_dtype=ttnn.bfloat16,
        )
        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def forward(self, x):
        assert len(x.shape) == 2, f"Mamba expects inputs to be rank 2 (was {len(x.shape)})"

        x = self.embedding(x)  # (B, 1, E)
        if self.args.mode == ModelMode.PREFILL:
            x = x.unsqueeze(0)
        elif self.args.mode == ModelMode.DECODE:
            x = x.squeeze(1).unsqueeze(0).unsqueeze(0)  # (1, 1, B, E)

        assert len(x.shape) == 4, f"Expected embedding to be rank 4 (was {len(x.shape)})"

        x = ttnn.from_torch(
            x,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.configs["dtype"]["activations"],
        )

        for i, layer in enumerate(self.layers):
            # print(f"Running layer {i}")
            x = layer(x)

        if not self.use_torch_rms_norm:
            x = ttnn.experimental.tensor.interleaved_to_sharded(x, sharded_mem_config=self.configs["sharded_h"])
            x = ttnn.experimental.operations.primary.rmsnorm(
                x,
                self.args.eps,
                self.norm_f_weights,
                program_config=self.configs["SHARDED_NORM_PRGM_CFG"],
                output_mem_config=self.configs["sharded_h"],
            )
            x = ttnn.experimental.tensor.sharded_to_interleaved(x)
        else:
            x = ttnn.to_torch(x)
            x = self.torch_norm_f(x)
            x = ttnn.from_torch(
                x,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=self.configs["dtype"]["activations"],
            )
        x = ttnn.linear(
            x,
            self.lm_head_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            use_1d_systolic_array=True,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.configs["dtype"]["activations"],
        )

        x = ttnn.to_torch(x).to(torch.float32)  # (1, 1, B, E)
        x = x.squeeze(0).squeeze(0).unsqueeze(1)

        return x
