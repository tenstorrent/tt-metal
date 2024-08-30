# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.grok.tt.grok_decoder import TtTransformerBlock
from models.experimental.grok.tt.grok_rms_norm import TtRMSNormSharded, TtRMSNorm
from models.experimental.grok.tt.grok_common import LightweightModule
from models.experimental.grok.scripts.tlog import tlog, tlog_mesh_device


class TtTransformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        args,
        dtype,
        layers,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        tlog_mesh_device = mesh_device
        self.model_config = args.get_model_config()
        self.output_multiplier_scale = args.output_multiplier_scale
        assert self.vocab_size > 0

        self.layers = [
            TtTransformerBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                args=args,
                dtype=dtype,
                layer_num=i,
            )
            for i in layers
        ]
        self.norm = TtRMSNormSharded(
            mesh_device=mesh_device,
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
            device=mesh_device,
            layout=self.model_config["OUTPUT_W_LAYOUT_TILE"],
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["OUTPUT_WEIGHTS_MEMCFG"],
            cache_file_name=output_cache_name,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )

        self.compute_kernel = self.args.get_compute_kernel_output_config()

    def forward(
        self,
        x,
        current_pos,
        attn_masks,
        rot_mats,
    ):
        for i, layer in enumerate(self.layers):
            x = layer(x, current_pos, attn_masks, rot_mats)
        attn_masks.deallocate(True)

        x_norm = self.norm(x)
        # tlog('our_model_norm', x_norm)
        multidevice_outputs = ttnn.matmul(
            x_norm,
            self.output_weight,
            # compute_with_storage_grid_size=(8, 8), # TODO: can we re-enable this here and in Mixtral?
            program_config=self.model_config["OUTPUT_MM_PROGCFG"],
            memory_config=self.model_config["OUTPUT_MM_MEMCFG"],
            compute_kernel_config=self.compute_kernel,
        )
        # tlog('our_model_lm_head', multidevice_outputs, gather_dim=-1)

        assert not multidevice_outputs.is_sharded(), "#9773: sharded inputs not supported by mul"
        multidevice_outputs = multidevice_outputs * self.output_multiplier_scale
        # tlog('our_model_scale', multidevice_outputs, gather_dim=-1)

        return multidevice_outputs
