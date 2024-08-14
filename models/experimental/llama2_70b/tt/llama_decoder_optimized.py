# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import torch
from torch import nn
from ttnn import experimental as tt_lib
import ttnn
from ttnn import ReplicateTensorToMesh


from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.experimental.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized
from models.experimental.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized


class TtLlamaDecoder_optimized:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        transformation_mats,
        cache_path=None,
        read_cache=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        self.model_config = model_config
        self.read_cache = read_cache

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.padded_local_heads = 32
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.norm_eps = configuration.norm_eps
        self.rope_theta = configuration.rope_theta

        self.llama3 = configuration.vocab_size == 128256

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path

        self.attention = TtLlamaAttention_optimized(
            device_mesh,
            state_dict,
            base_url,
            layer_num,
            model_config,
            configuration,
            transformation_mats,
            cache_path=cache_path,
            read_cache=read_cache,
        )

        self.mlp = TtLlamaMLP_optimized(
            device_mesh,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
            cache_path=cache_path,
            read_cache=read_cache,
        )

        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config
        self.attention.set_model_config(model_config)
        self.mlp.set_model_config(model_config)

    def load_weights(self):
        """
        Loads weights that this layer is responsible for.
        Doesn't touch the weights of the submodules.
        """
        assert not hasattr(self, "attn_norm"), "attn_norm_list is already an attribute of this object"
        assert not hasattr(self, "ffn_norm"), "ffn_norm_list is already an attribute of this object"
        attn_norm_str = f"{self.layer_name}.attention_norm.weight"
        ffn_norm_str = f"{self.layer_name}.ffn_norm.weight"

        pt_attn_norm = None
        pt_ffn_norm = None
        if not self.read_cache:
            pt_attn_norm = self.state_dict[attn_norm_str].reshape([1, 1, -1, 32])
            pt_ffn_norm = self.state_dict[ffn_norm_str].reshape([1, 1, -1, 32])

        attn_norm_ttnn = ttnn.as_tensor(
            pt_attn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            cache_file_name=self.cache_path / attn_norm_str,
        )
        self.attn_norm = ttnn.to_device(attn_norm_ttnn, self.device_mesh)

        ffn_norm_ttnn = ttnn.as_tensor(
            pt_ffn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            cache_file_name=self.cache_path / ffn_norm_str,
        )
        self.ffn_norm = ttnn.to_device(ffn_norm_ttnn, self.device_mesh)

    def __call__(
        self,
        xs: List[tt_lib.tensor.Tensor],
        rot_mats: List[tt_lib.tensor.Tensor],
        start_pos: int,
        attn_masks: List[tt_lib.tensor.Tensor],
        user_id: int = 0,
    ) -> tt_lib.tensor.Tensor:
        if self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(xs, rot_mats, start_pos, attn_masks, user_id)
        elif self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def decode_forward(
        self,
        xs: List[tt_lib.tensor.Tensor],
        rot_mats: List[tt_lib.tensor.Tensor],
        start_pos: int,
        attn_masks: List[tt_lib.tensor.Tensor],
    ) -> List[tt_lib.tensor.Tensor]:
        ### xs (residual stream) is fractured on all chips
        xs_replicated = ttnn.all_gather(
            xs,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # In-place RMSNorm
        attn_norm_replicated = ttnn.rms_norm(
            xs_replicated,
            epsilon=self.norm_eps,
            weight=self.attn_norm,
            program_config=self.model_config["LN_ATTN_PROGCFG"],
            memory_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )
        # attn_norm_replicated is sharded

        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_replicated, rot_mats, start_pos, attn_masks)

        ### Fractured residual add
        # Add attn output to residiual first in place to save memory
        output = xs
        output = ttnn.add(
            output,
            attn_outs,
            memory_config=self.model_config["ATTN_ADD_OUTPUT_MEMCFG"],
        )
        attn_outs.deallocate(True)

        attn_resid_replicated = ttnn.all_gather(
            output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # In-place RMSNorm
        ffn_norm_replicated = ttnn.rms_norm(
            attn_resid_replicated,
            epsilon=self.norm_eps,
            weight=self.ffn_norm,
            program_config=self.model_config["LN_MLP_PROGCFG"],
            memory_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )
        # ffn_norm_replicated is sharded

        ffn_out = self.mlp(ffn_norm_replicated)

        ### residual in place
        output = ttnn.add(
            output,
            ffn_out,
            memory_config=self.model_config["MLP_ADD_OUTPUT_MEMCFG"],
        )
        ffn_out.deallocate(True)

        return output

    def sharded_rmsnorm(self, xs, eps, norm_list):
        # Do sharded RMS by partial sequence length of 128 or 512
        # Input xs[0] is [1, 1, seq_len, 8192]
        seq_len = xs.shape[2]
        slice_size = 512 if seq_len == 2048 else 128
        num_slices = seq_len // slice_size  # we do 128 per iteration (slice), then we concat the result.

        xs_output_cat = ttnn.as_tensor(
            torch.zeros([1, 1, seq_len, self.hidden_size]),
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
        )

        layernorm_num_cores_x, layernorm_num_cores_y = (
            self.model_config["layernorm_params"]["layernorm_num_cores_x"],
            self.model_config["layernorm_params"]["layernorm_num_cores_y"],
        )
        layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim = (
            self.model_config["layernorm_params"]["layernorm_shard_height_hidden_dim"],
            self.model_config["layernorm_params"]["layernorm_shard_width_hidden_dim"],
        )

        for slice_i in range(num_slices):
            xs_slice = ttnn.interleaved_to_sharded_partial(
                xs,
                (layernorm_num_cores_x, layernorm_num_cores_y),
                [layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim],
                num_slices,  # num_slices
                slice_i,  # slice_index
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            )

            xs_slice = ttnn.rms_norm(
                xs_slice,
                epsilon=eps,
                weight=norm_list,
                program_config=self.model_config["LN_ATTN_PROGCFG"],
                memory_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
            )

            ttnn.sharded_to_interleaved_partial(
                xs_slice,
                xs_output_cat,
                num_slices,
                slice_i,
                memory_config=self.model_config["DRAM_MEMCFG"],
            )
            xs_slice.deallocate(True)
        return xs_output_cat

    def prefill_forward(
        self,
        xs: List[tt_lib.tensor.Tensor],
        rot_mats: List[tt_lib.tensor.Tensor],
        start_pos: int,
        attn_masks: List[tt_lib.tensor.Tensor],
        user_id: int = 0,
    ) -> List[tt_lib.tensor.Tensor]:
        ### xs (residual stream) is fractured on all chips
        # TODO: Reenable when typcast supports multidevice
        # xs_replicated = []
        # for i in range(self.num_devices):
        #     xs_replicated.append(
        #         tt_lib.tensor.typecast(tt_lib.tensor.clone(xs[i]), dtype=tt_lib.tensor.DataType.BFLOAT8_B)
        #     )

        xs_replicated = ttnn.all_gather(
            xs,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )

        attn_norm_interleaved = self.sharded_rmsnorm(xs_replicated, self.norm_eps, self.attn_norm)

        xs_replicated.deallocate(True)

        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_interleaved, rot_mats, start_pos, attn_masks, user_id)

        ### Fractured residual add
        residual = xs
        output = ttnn.add(residual, attn_outs)
        attn_outs.deallocate(True)

        attn_resid_replicated = ttnn.all_gather(
            output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )

        ffn_norm_interleaved = self.sharded_rmsnorm(attn_resid_replicated, self.norm_eps, self.ffn_norm)

        attn_resid_replicated.deallocate(True)

        ffn_out = self.mlp(ffn_norm_interleaved)

        ### residual add
        output = ttnn.add(output, ffn_out)
        ffn_out.deallocate(True)
        return output
