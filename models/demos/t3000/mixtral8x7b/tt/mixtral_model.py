# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import get_single_rot_mat_multi_pos, get_single_rot_mat_torch
from ttnn import ReplicateTensorToMesh
import torch


class TtTransformer(LightweightModule):
    def __init__(self, mesh_device, state_dict, args, dtype, layers, start_pos_ids, rotary_on_host=False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.model_config = args.get_model_config()
        self.rotary_on_host = rotary_on_host
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
        self.norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=None,
            weight_dtype=ttnn.bfloat16,
            weight_key="norm",
        )

        self.state_dict = state_dict

        if args.dummy_weights:
            output_cache_name = None
        else:
            output_cache_name = args.weight_cache_path(dtype) / "output_multidevice_4d.weight"

        self.output_weight = ttnn.as_tensor(
            self.state_dict["output.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            layout=self.model_config["OUTPUT_W_LAYOUT_TILE"],
            dtype=dtype,
            memory_config=self.model_config["OUTPUT_WEIGHTS_MEMCFG"],
            cache_file_name=output_cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        self.compute_kernel = self.args.get_compute_kernel_config()

        if self.rotary_on_host:
            self.current_rot_mat, self.rot_matrix = get_single_rot_mat_torch(self.args.head_dim, start_pos_ids)
        else:
            self.current_rot_mat, self.rot_matrix = get_single_rot_mat_multi_pos(
                self.args.head_dim, mesh_device, start_pos_ids
            )

    def forward(
        self,
        x,
        start_pos_ids,
        attn_masks=None,
        rot_mats=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        get_last_token=-1,
    ):
        for i, layer in enumerate(self.layers):
            if mode == "decode":
                if self.rotary_on_host:
                    rot_mats = ttnn.from_torch(
                        self.current_rot_mat,  # 1,1,head_dim,head_dim
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    )
                else:
                    rot_mats = self.current_rot_mat

            x = layer(x, start_pos_ids, attn_masks, rot_mats, transformation_mats, user_id, mode)
        if attn_masks is not None:
            attn_masks.deallocate(True)

        if mode == "prefill" and get_last_token == -1:
            return x

        # slicing for the last token
        if get_last_token != -1:
            x = ttnn.slice(
                x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, 4096)
            )  # [:, :, get_last_token:get_last_token+32, :]

        x_norm = self.norm(x)
        outputs = ttnn.matmul(
            x_norm,
            self.output_weight,
            # compute_with_storage_grid_size=(8, 8),
            program_config=self.model_config["OUTPUT_MM_PROGCFG"],
            memory_config=self.model_config["OUTPUT_MM_MEMCFG"],
            compute_kernel_config=self.compute_kernel,
        )
        if get_last_token != -1:
            return outputs

        if self.rotary_on_host:
            prev_rot_mat = self.current_rot_mat
            self.current_rot_mat = torch.matmul(self.rot_matrix, prev_rot_mat)
        else:
            if (start_pos_ids[0] + 1) % 32 == 0:
                # generate new rotmat to avoid numerical instability every 32 tokens
                self.current_rot_mat, self.rot_matrix = get_single_rot_mat_multi_pos(
                    self.args.head_dim, self.mesh_device, [pos + 1 for pos in start_pos_ids]
                )
            else:
                # assigning to a new variable to explictly deallocate since matmul creates a new buffer for the output
                prev_rot_mat = self.current_rot_mat
                self.current_rot_mat = ttnn.matmul(
                    self.rot_matrix,
                    prev_rot_mat,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    dtype=ttnn.bfloat16,
                )
                prev_rot_mat.deallocate(True)

        return outputs
