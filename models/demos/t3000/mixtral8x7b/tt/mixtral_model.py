# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.demos.t3000.mixtral8x7b.tt.mixtral_decoder import TtTransformerBlock
from models.demos.t3000.mixtral8x7b.tt.mixtral_rms_norm import TtRMSNormSharded
from ttnn import ReplicateTensorToMesh


class TtTransformer(torch.nn.Module):
    def __init__(
        self,
        device_mesh,
        state_dict,
        args,
        dtype,
        layers,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.device_mesh = device_mesh
        self.model_config = args.get_model_config()
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    device_mesh=device_mesh,
                    state_dict=state_dict,
                    args=args,
                    dtype=dtype,
                    layer_num=i,
                )
                for i in layers
            ]
        )
        self.norm = TtRMSNormSharded(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=None,
            weight_key="norm",
        )

        self.state_dict = state_dict

        if args.dummy_weights:
            output_cache_name = None
        else:
            output_cache_name = (args.weight_cache_path(dtype) / "output_multidevice.weight",)

        self.output_weight = ttnn.as_tensor(
            self.state_dict["output.weight"].permute(1, 0),
            device=device_mesh,
            layout=self.model_config["OUTPUT_W_LAYOUT_TILE"],
            dtype=dtype,
            memory_config=self.model_config["OUTPUT_WEIGHTS_MEMCFG"],
            cache_file_name=output_cache_name,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        self.compute_kernel = self.args.get_compute_kernel_config()

    def forward(
        self,
        x,
        start_pos,
        current_pos,
        attn_masks,
        rot_mats,
    ):
        for i, layer in enumerate(self.layers):
            x = layer(x, start_pos, current_pos, attn_masks, rot_mats)
        attn_masks.deallocate(True)

        x_norm = self.norm(x)
        outputs = ttnn.linear(
            x_norm,
            self.output_weight,
            core_grid=self.args.max_grid_size,
            use_1d_systolic_array=True,
            memory_config=self.model_config["OUTPUT_MM_MEMCFG"],
            compute_kernel_config=self.compute_kernel,
        )

        return outputs
