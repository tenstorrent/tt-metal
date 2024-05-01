# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.demos.t3000.mixtral8x7b.tt.mixtral_decoder import TtTransformerBlock
from models.demos.t3000.mixtral8x7b.tt.mixtral_rms_norm import TtRMSNormSharded


class TtTransformer(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        args,
        dtype,
        layers,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.devices = devices
        self.model_config = args.get_model_config()
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    devices=devices,
                    state_dict=state_dict,
                    args=args,
                    dtype=dtype,
                    layer_num=i,
                )
                for i in layers
            ]
        )
        self.norm = [
            TtRMSNormSharded(
                device=dev,
                state_dict=state_dict,
                args=args,
                dtype=ttnn.bfloat16,
                layer_num=None,
                weight_key="norm",
            )
            for dev in self.devices
        ]
        self.state_dict = state_dict

        self.output_weight = [
            ttnn.as_tensor(
                self.state_dict["output.weight"].permute(1, 0),
                device=dev,
                layout=self.model_config["OUTPUT_W_LAYOUT_TILE"],
                dtype=dtype,
                memory_config=self.model_config["OUTPUT_WEIGHTS_MEMCFG"],
                cache_file_name=args.weight_cache_path(dtype) / "output.weight",
            )
            for dev in self.devices
        ]

        self.compute_kernel = self.args.get_compute_kernel_config()

    def forward(
        self,
        x,
        start_pos,
        current_pos,
        rot_mats,
    ):
        for i, layer in enumerate(self.layers):
            x = layer(x, start_pos, current_pos, rot_mats)

        outputs = []
        x_norm = []
        for i in range(len(self.devices)):
            x_norm.append(self.norm[i](x[i]))
            output_i = ttnn.linear(
                x_norm[i],
                self.output_weight[i],
                core_grid=self.args.max_grid_size,
                use_1d_systolic_array=True,
                memory_config=self.model_config["OUTPUT_MM_MEMCFG"],
                compute_kernel_config=self.compute_kernel,
            )
            outputs.append(output_i)

        return outputs
