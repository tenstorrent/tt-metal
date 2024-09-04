# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from pathlib import Path
from typing import Callable, Optional

from models.demos.wormhole.mamba.tt.residual_block import TtResidualBlock
from models.demos.wormhole.mamba.reference.args import ModelMode


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
        ):
            if layer_num == None:
                tensor_name = name
            else:
                tensor_name = f"layers.{layer_num}.{name}"

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
        self.return_logits = True

        if num_layers is None:
            self.num_layers = len(reference_model.layers)
        else:
            self.num_layers = num_layers

        logger.info(f"Initalizing Mamba with {self.num_layers} layers")

        loader = TtTensorLoader(reference_model.state_dict(), self.device, tt_cache_path=tt_cache_path)
        self.layers = [
            TtResidualBlock(self.args, device, configs, loader.get_tensor_loader(i)) for i in range(self.num_layers)
        ]

        load_fn = loader.get_tensor_loader()

        self.norm_f_weights = load_fn(
            "norm_f.weight",
            tt_dtype=ttnn.bfloat8_b,
        )
        self.lm_head_weights = load_fn(
            "lm_head.weight",
            lambda x: x.transpose(-1, -2),
            tt_dtype=ttnn.bfloat8_b,
        )
        self.embedding_weights = load_fn("embedding.weight", tt_dtype=ttnn.bfloat16, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def to_prefill(self, prefill_config):
        self.configs = prefill_config
        self.return_logits = False
        for i in range(self.num_layers):
            self.layers[i].to_prefill(prefill_config)

    def to_decode(self, decode_config):
        self.configs = decode_config
        self.return_logits = True
        for i in range(self.num_layers):
            self.layers[i].to_decode(decode_config)

    def embedding(self, x):
        assert len(x.shape) == 2, f"Mamba expects inputs to be rank 2 (was {len(x.shape)})"
        x = ttnn.embedding(
            x, self.embedding_weights, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
        )  # ttnn.embedding always returns (B, L, E)
        return ttnn.reshape(x, [1, 1, self.configs["outer_dim"], x.shape[2]])

    def _forward(self, x):
        assert len(x.shape) == 2, f"Expected tensor to be rank 2 (shape was {x.shape})"
        assert (
            x.shape[-1] <= self.configs["max_seq_length"]
        ), f"Expected L to be less than or equal to max sequence length (was {x.shape[-1]}, expected <= {self.configs['max_seq_length']})"

        x = self.embedding(x)
        x = ttnn.typecast(ttnn.to_layout(x, ttnn.TILE_LAYOUT), self.configs["dtype"]["activations"])

        assert len(x.shape) == 4, f"Expected embedding output to be rank 4 (shape was {x.shape})"
        assert x.layout == ttnn.TILE_LAYOUT, f"Expected embedding to be tile layout (was {x.layout})"

        for i, layer in enumerate(self.layers):
            x = layer(x)

        if self.return_logits or self.configs["mode"] == ModelMode.DECODE:
            x = ttnn.interleaved_to_sharded(x, self.configs["sharded_h"])
            x = ttnn.rms_norm(
                x,
                epsilon=self.args.eps,
                weight=self.norm_f_weights,
                program_config=self.configs["SHARDED_NORM_PRGM_CFG"],
                memory_config=self.configs["sharded_h"],
            )
            x = ttnn.sharded_to_interleaved(x)
            x = ttnn.linear(
                x,
                self.lm_head_weights,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=x.device().core_grid,
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.configs["dtype"]["activations"],
            )

        return x

    def forward(self, x):
        assert len(x.shape) == 2, f"Mamba expects inputs to be rank 2 (was {len(x.shape)})"
        x = ttnn.from_torch(
            x,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.uint32,
        )
        x = self._forward(x)
        if self.return_logits or self.configs["mode"] == ModelMode.DECODE:
            x = ttnn.to_torch(x).to(torch.float32)  # (1, 1, B, E)
            x = x.view((self.configs["batch_size"], self.configs["seq_len"], -1))
            return x

    def reset(self):
        for layer in self.layers:
            layer.reset()
