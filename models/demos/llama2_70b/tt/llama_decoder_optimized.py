# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized
from models.demos.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized
from models.demos.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    generate_rot_emb,
    gather_rotary_emb,
    get_weight_cache_path,
    precompute_freqs,
    freqs_to_rotation_matrix,
    get_rotation_mat,
    get_rotation_mat_prefill,
)


class TtLlamaDecoder_optimized:
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        batch,
        emulated=False,
        load_weights=True,
        cache_path=None,
        kv_cache_dir=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.model_config = model_config
        self.emulated = emulated
        self.batched_attn = self.num_devices == 8
        self.padded_local_heads = 32
        self.llm_mode = model_config["LLM_MODE"]

        self.layer_name = f"{base_url}.{layer_num}"

        self.norm_eps = configuration.norm_eps
        self.cache_path = cache_path

        self.attention = TtLlamaAttention_optimized(
            devices,
            state_dict,
            base_url,
            layer_num,
            model_config,
            configuration,
            emulated=emulated,
            load_weights=load_weights,
            cache_path=cache_path,
            kv_cache_dir=kv_cache_dir,
        )

        self.mlp = TtLlamaMLP_optimized(
            devices,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
            emulated=emulated,
            load_weights=load_weights,
            cache_path=cache_path,
        )

        if load_weights:
            self.load_weights()

    def load_weights(self):
        """
        Loads weights that this layer is responsible for.
        Doesn't touch the weights of the submodules.
        """
        assert not hasattr(self, "attn_norm_list"), "attn_norm_list is already an attribute of this object"
        assert not hasattr(self, "ffn_norm_list"), "ffn_norm_list is already an attribute of this object"
        attn_norm_str = f"{self.layer_name}.attention_norm.weight"
        ffn_norm_str = f"{self.layer_name}.ffn_norm.weight"

        self.attn_norm_list = []
        self.ffn_norm_list = []

        # TODO: Weight caching!
        test_cache_path = get_weight_cache_path(self.cache_path, ffn_norm_str, self.num_devices - 1, self.num_devices)
        if test_cache_path.exists():
            for i in range(self.num_devices):
                tensor_cache_path = get_weight_cache_path(self.cache_path, attn_norm_str, i, self.num_devices)
                self.attn_norm_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )

                tensor_cache_path = get_weight_cache_path(self.cache_path, ffn_norm_str, i, self.num_devices)
                self.ffn_norm_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )
        else:
            for i in range(self.num_devices):
                attn_norm_host = tt_lib.tensor.Tensor(
                    # Expand to size of input since we decomped norm
                    self.state_dict[attn_norm_str].reshape([1, 1, -1, 32]),
                    self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
                )
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, attn_norm_str, i, self.num_devices)), attn_norm_host
                )
                self.attn_norm_list.append(attn_norm_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))

                ffn_norm_host = tt_lib.tensor.Tensor(
                    # Expand to size of input since we decomped norm
                    self.state_dict[ffn_norm_str].reshape([1, 1, -1, 32]),
                    self.model_config["LN_MLP_WEIGHTS_DTYPE"],
                )
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, ffn_norm_str, i, self.num_devices)), ffn_norm_host
                )
                self.ffn_norm_list.append(ffn_norm_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))

    def free_weights(self):
        for i in range(self.num_devices):
            self.attn_norm_list[i].deallocate()
            self.ffn_norm_list[i].deallocate()
        del self.attn_norm_list
        del self.ffn_norm_list

    def free_layer(self):
        # All epilogue logic
        self.attention.save_state()
        self.attention.free_weights()
        self.mlp.free_weights()

        # Free this layer's weights (RMS norms)
        self.free_weights()

    def load_layer(self):
        # All weight loading logic
        ## Load weights
        self.attention.load_weights()
        self.attention.load_state()
        self.mlp.load_weights()

        # Load this layer's weights (RMS norms)
        self.load_weights()

    def prepare_inputs(self, x, start_pos):
        # Only called by decoder tests
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        batch, seq_len, hidden_size = x.shape

        rot_mat = generate_rot_emb(self.head_dim, self.max_seq_len * 2)

        if self.llm_mode == "prefill":
            assert seq_len == 128, "prefill mode only supports 128 seq_len"
            assert batch == 1, "prefill mode only supports batch size 1"
            x = x.unsqueeze(1)  # [batch, 1, seq_len, hidden_dim]
            rot_mat = get_rotation_mat_prefill(rot_mat=rot_mat, start_pos=start_pos, seqlen=seq_len, batch=batch)
            attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            attn_mask = attn_mask.expand(batch, self.n_local_heads, -1, -1)

            as_tensor = lambda tensor, name, device_id: ttnn.as_tensor(
                tensor,
                dtype=ttnn.bfloat16,
                device=self.devices[device_id],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=get_weight_cache_path(self.cache_path, name, device_id, self.num_devices),
            )
            # expected shapes:
            # x: [batch, 1, seq_len, hidden_dim]
            # start_pos: int
            # rot_mat: [batch, seq_len, head_dim, head_dim]
            # attn_mask: [batch, self.n_heads, seq_len, seq_len]
            assert x.size() == (batch, 1, seq_len, self.hidden_size)
            assert rot_mat.size() == (batch, seq_len, self.head_dim, self.head_dim)
            assert attn_mask.size() == (batch, self.n_local_heads, seq_len, seq_len)

            x_fractured = torch.chunk(x, self.num_devices, dim=-1)
            xs, rot_mats, attn_masks = [], [], []
            for i in range(self.num_devices):
                xs.append(
                    ttnn.as_tensor(x_fractured[i], dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT)
                )
                rot_mats.append(as_tensor(rot_mat.clone(), "rot_mat_prefill", i))
                attn_masks.append(as_tensor(attn_mask.clone(), "attn_mask_prefill", i))

            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = seq_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
            for i in range(self.num_devices):
                # xs[i] = tt_lib.tensor.interleaved_to_sharded(
                #     xs[i], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
                # )
                attn_masks[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_masks[i], sharded_mem_config=attention_mask_memconfig
                )
        elif self.llm_mode == "decode":
            assert seq_len == 1, "Only supporting decode mode"
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]

            rot_mat = get_rotation_mat(
                rot_mat=rot_mat,
                start_pos=start_pos,
                seqlen=seq_len,
                batch=1,  # use batch=1 because we assume all users use same rot_mat
            )

            padded_layer_past_len = nearest_32(start_pos + 1)
            if self.batched_attn:
                attn_mask_shape = (batch, seq_len, self.padded_local_heads, padded_layer_past_len)
            else:
                attn_mask_shape = (seq_len, self.n_local_heads, batch, padded_layer_past_len)
            attn_mask = torch.zeros(*attn_mask_shape)
            attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min

            # expected shapes:
            # x: (seq_len, 1, batch, hidden_dim)
            # start_pos: int
            # rot_mat: [1, 1, head_dim, head_dim]
            # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
            assert x.size() == (seq_len, 1, batch, self.hidden_size)
            assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
            assert attn_mask.size() == attn_mask_shape

            x_fractured = torch.chunk(x, self.num_devices, dim=-1)
            xs, rot_mats, attn_masks = [], [], []
            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = padded_layer_past_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
            for i in range(self.num_devices):
                device = self.devices[i]
                xs.append(
                    torch2tt_tensor(
                        x_fractured[i],
                        device,
                        tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                    )
                )
                rot_mats.append(
                    torch2tt_tensor(
                        rot_mat.clone(),
                        device,
                        tt_memory_config=self.model_config["ROT_MAT_MEMCFG"],  # TODO: Put on L1 instead of DRAM
                        tt_dtype=self.model_config["ROT_MAT_DTYPE"],
                    )
                )
                attn_masks.append(
                    torch2tt_tensor(
                        attn_mask.clone(),
                        device,
                        tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
                    )
                )
            for i in range(self.num_devices):
                xs[i] = tt_lib.tensor.interleaved_to_sharded(
                    xs[i], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
                )
                attn_masks[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_masks[i], sharded_mem_config=attention_mask_memconfig
                )

        if self.emulated:
            # save l1 space by sharing the same input across "devices"
            for i in range(1, self.num_devices):
                # x_multichip[i].deallocate(True)
                rot_mats[i].deallocate(True)
                attn_masks[i].deallocate(True)
                # x_multichip[i] = x_multichip[0]
                rot_mats[i] = rot_mats[0]
                attn_masks[i] = attn_masks[0]

        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def __call__(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        if self.llm_mode == "prefill":
            return self.prefill_forward(xs, rot_mats, start_pos, attn_masks)
        elif self.llm_mode == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        else:
            raise ValueError(f"Unknown llm_mode: {self.llm_mode}")

    def decode_forward(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        ### xs (residual stream) is fractured on all chips
        xs_replicated = []
        # Put xs back on DRAM and do allgather
        for i in range(self.num_devices):
            xs_replicated.append(
                tt_lib.tensor.sharded_to_interleaved(xs[i], output_mem_config=self.model_config["L1_MEMCFG"])
            )
        ### Duplicate inputs for layernorm
        if self.emulated:
            xs_replicated = tt_all_gather_torch(xs_replicated, dim=-1)
        else:
            xs_replicated = tt_lib.tensor.all_gather(
                xs_replicated,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["L1_MEMCFG"],
            )

        for i in range(self.num_devices):
            # RMSNorm must execute on sharded input
            xs_replicated[i] = tt_lib.tensor.interleaved_to_sharded(
                xs_replicated[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        attn_norm_replicated = []
        for i in range(self.num_devices):
            attn_norm_replicated.append(
                tt_lib.operations.primary.rmsnorm(
                    xs_replicated[i],
                    self.norm_eps,
                    self.attn_norm_list[i],
                    program_config=self.model_config["LN_ATTN_PROGCFG"],
                    output_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                )
            )  # attn_norm_replicated is sharded
            # xs_replicated[i].deallocate(True) # !!! Cant deallocate here, will drop PCC
        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_replicated, rot_mats, start_pos, attn_masks)

        ### Fractured residual add
        # Add attn output to residiual first in place to save memory
        output = []
        residual = xs
        for i in range(self.num_devices):
            output.append(
                tt_lib.operations.primary.add(
                    residual[i],
                    attn_outs[i],
                    output_mem_config=self.model_config["ATTN_ADD_OUTPUT_MEMCFG"],
                    in_place=True,
                )
            )
            attn_outs[i].deallocate(True)

        attn_resid_replicated = []
        for i in range(self.num_devices):
            # Put attn_resid back on DRAM
            attn_resid_replicated.append(
                tt_lib.tensor.sharded_to_interleaved(output[i], output_mem_config=self.model_config["L1_MEMCFG"])
            )

        ### Duplicate attention residual on all chips
        if self.emulated:
            attn_resid_replicated = tt_all_gather_torch(attn_resid_replicated, dim=-1)
        else:
            attn_resid_replicated = tt_lib.tensor.all_gather(
                attn_resid_replicated,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["L1_MEMCFG"],
            )

        for i in range(self.num_devices):
            # RMSNorm must execute on sharded input
            attn_resid_replicated[i] = tt_lib.tensor.interleaved_to_sharded(
                attn_resid_replicated[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        ### Duplicate FFN layernorm
        ffn_norm_replicated = []
        for i in range(self.num_devices):
            ffn_norm_replicated.append(
                tt_lib.operations.primary.rmsnorm(
                    attn_resid_replicated[i],
                    self.norm_eps,
                    self.ffn_norm_list[i],
                    program_config=self.model_config["LN_MLP_PROGCFG"],
                    output_mem_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"],
                )
            )  # ffn_norm_replicated is sharded
        # attn_resid_replicated[i].deallocate(True) # !!! Cant deallocate here, will drop PCC

        ffn_out = self.mlp(ffn_norm_replicated)

        ### residual in place
        for i in range(self.num_devices):
            output[i] = tt_lib.operations.primary.add(
                output[i],
                ffn_out[i],
                output_mem_config=self.model_config["MLP_ADD_OUTPUT_MEMCFG"],
                in_place=True,
            )
            ffn_out[i].deallocate(True)

        # FOR BRINGUP! Outputs are sharded. Interleave them
        # for i in range(self.num_devices):
        #     output[i] = tt_lib.tensor.sharded_to_interleaved(
        #         output[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
        #     )
        return output

    def prefill_forward(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        ### xs (residual stream) is fractured on all chips
        xs_replicated = []
        for i in range(self.num_devices):
            xs_replicated.append(tt_lib.tensor.clone(xs[i]))

        ### Duplicate inputs for layernorm
        if self.emulated:
            xs_replicated = tt_all_gather_torch(xs_replicated, dim=-1)
        else:
            xs_replicated = tt_lib.tensor.all_gather(
                xs_replicated,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["DRAM_MEMCFG"],
            )

        for i in range(self.num_devices):
            # RMSNorm must execute on sharded input
            xs_replicated[i] = tt_lib.tensor.interleaved_to_sharded(
                xs_replicated[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )

        attn_norm_interleaved = []
        for i in range(self.num_devices):
            xs_replicated[i] = tt_lib.operations.primary.rmsnorm(
                xs_replicated[i],
                self.norm_eps,
                self.attn_norm_list[i],
                program_config=self.model_config["LN_ATTN_PROGCFG"],
                output_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
            )

            # xs_replicated[i] is sharded, turn it into interleaved
            attn_norm_interleaved.append(
                tt_lib.tensor.sharded_to_interleaved(
                    xs_replicated[i], output_mem_config=self.model_config["DRAM_MEMCFG"]
                )
            )
            xs_replicated[i].deallocate(True)
        # attn_outs is fractured
        attn_outs = self.attention(attn_norm_interleaved, rot_mats, start_pos, attn_masks)

        ### Fractured residual add
        # Add attn output to residiual first in place to save memory
        output = []
        residual = xs
        for i in range(self.num_devices):
            output.append(
                ttnn.add(
                    residual[i],
                    attn_outs[i],
                )
            )
            attn_outs[i].deallocate(True)

        attn_resid_replicated = []
        for i in range(self.num_devices):
            attn_resid_replicated.append(tt_lib.tensor.clone(output[i]))

        ### Duplicate attention residual on all chips
        if self.emulated:
            attn_resid_replicated = tt_all_gather_torch(attn_resid_replicated, dim=-1)
        else:
            attn_resid_replicated = tt_lib.tensor.all_gather(
                attn_resid_replicated,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["L1_MEMCFG"],
            )

        for i in range(self.num_devices):
            # RMSNorm must execute on sharded input
            attn_resid_replicated[i] = tt_lib.tensor.interleaved_to_sharded(
                attn_resid_replicated[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        ### Duplicate FFN layernorm
        ffn_norm_interleaved = []
        for i in range(self.num_devices):
            attn_resid_replicated[i] = tt_lib.operations.primary.rmsnorm(
                attn_resid_replicated[i],
                self.norm_eps,
                self.ffn_norm_list[i],
                program_config=self.model_config["LN_MLP_PROGCFG"],
                output_mem_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"],
            )

            ffn_norm_interleaved.append(
                tt_lib.tensor.sharded_to_interleaved(
                    attn_resid_replicated[i], output_mem_config=self.model_config["DRAM_MEMCFG"]
                )
            )
            attn_resid_replicated[i].deallocate(True)

        # breakpoint()
        ffn_out = self.mlp(ffn_norm_interleaved)

        ### residual in place
        for i in range(self.num_devices):
            output[i] = ttnn.add(
                output[i],
                ffn_out[i],
            )
            ffn_out[i].deallocate(True)
        return output
