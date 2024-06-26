# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.grok.tt.grok_decoder import TtTransformerBlock
from models.experimental.grok.tt.grok_rms_norm import TtRMSNormSharded, TtRMSNorm
from ttnn import ReplicateTensorToMesh
from models.experimental.grok.tt.grok_common import LightweightModule


class TtTransformer(LightweightModule):
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
        self.output_multiplier_scale = args.output_multiplier_scale
        assert self.vocab_size > 0

        self.layers = [
            TtTransformerBlock(
                device_mesh=device_mesh,
                state_dict=state_dict,
                args=args,
                dtype=dtype,
                layer_num=i,
            )
            for i in layers
        ]
        self.norm = TtRMSNorm(
            device_mesh=device_mesh,
            state_dict=state_dict,
            args=args,
            dtype=ttnn.bfloat16,
            layer_num=None,
            weight_key="model.norm",
        )

        self.state_dict = state_dict

        if args.dummy_weights:
            output_cache_name = None
        else:
            output_cache_name = args.weight_cache_path(dtype) / "output_multidevice_4d.weight"

        self.output_weight = ttnn.as_tensor(
            self.state_dict["lm_head.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0),
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
        outputs = ttnn.experimental.operations.primary.matmul(
            x_norm,
            self.output_weight,
            # compute_with_storage_grid_size=(8, 8), # FIXME: from Mixtral, presumably dI/dT workaround?
            program_config=self.model_config["OUTPUT_MM_PROGCFG"],
            output_mem_config=self.model_config["OUTPUT_MM_MEMCFG"],
            compute_kernel_config=self.compute_kernel,
        )

        outputs = outputs * self.output_multiplier_scale

        return outputs
