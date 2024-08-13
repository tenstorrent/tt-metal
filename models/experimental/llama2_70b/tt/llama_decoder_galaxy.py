# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from torch import nn
import ttnn.deprecated
import ttnn
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.experimental.llama2_70b.tt.llama_attention_galaxy import TtLlamaAttention_galaxy
from models.experimental.llama2_70b.tt.llama_mlp_galaxy import TtLlamaMLP_galaxy
from models.experimental.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    generate_rot_emb,
    gather_rotary_emb,
    get_weight_cache_path,
)


class TtLlamaDecoder_galaxy:
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        batch,
        transformation_mats,
        emulated=False,
        cache_path=None,
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
        self.padded_local_heads = 32
        self.model_config = model_config
        self.emulated = emulated

        assert emulated, "Only emulated mode is supported for galaxy now."

        # Each device should have 1 kv heads, so n_kv_heads forms a group
        # num_device_groups = num_devices // n_kv_heads
        # parallize the batch dimension across the device groups
        assert self.num_devices == 32, "Only 32 devices supported for galaxy"
        assert self.num_devices % configuration.n_kv_heads == 0, "num_devices must be divisible by n_kv_heads"
        self.num_device_groups = self.num_devices // configuration.n_kv_heads
        self.num_devices_per_group = configuration.n_kv_heads
        self.device_groups = [
            devices[i * self.num_devices_per_group : (i + 1) * self.num_devices_per_group]
            for i in range(self.num_device_groups)
        ]
        self.max_batch_size = configuration.max_batch_size
        self.batch_size_per_device_group = self.max_batch_size // self.num_device_groups
        # [[0, 1, 2, 3, 4, 5, 6, 7],
        # [8, 9, 10, 11, 12, 13, 14, 15],
        # [16, 17, 18, 19, 20, 21, 22, 23],
        # [24, 25, 26, 27, 28, 29, 30, 31]]

        self.layer_name = f"{base_url}.{layer_num}"
        self.norm_eps = configuration.norm_eps
        self.cache_path = cache_path

        self.batched_attn = self.num_devices == 8 or self.num_devices == 32

        self.attention = TtLlamaAttention_galaxy(
            devices,
            state_dict,
            base_url,
            layer_num,
            model_config,
            configuration,
            transformation_mats,
            emulated=emulated,
            cache_path=cache_path,
        )

        self.mlp = TtLlamaMLP_galaxy(
            devices,
            state_dict,
            base_url,
            layer_num,
            self.hidden_size,
            model_config,
            emulated=emulated,
            cache_path=cache_path,
        )
        self.rot_emb = generate_rot_emb(self.head_dim, self.max_seq_len * 2)

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

        test_cache_path = get_weight_cache_path(self.cache_path, ffn_norm_str, self.num_devices - 1, self.num_devices)
        if test_cache_path.exists():
            for i in range(self.num_devices):
                tensor_cache_path = get_weight_cache_path(self.cache_path, attn_norm_str, i, self.num_devices)
                self.attn_norm_list.append(
                    ttnn.experimental.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )

                tensor_cache_path = get_weight_cache_path(self.cache_path, ffn_norm_str, i, self.num_devices)
                self.ffn_norm_list.append(
                    ttnn.experimental.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )
        else:
            for i in range(self.num_devices):
                attn_norm_host = ttnn.experimental.tensor.Tensor(
                    # Expand to size of input since we decomped norm
                    self.state_dict[attn_norm_str].reshape([1, 1, -1, 32]),
                    self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
                )
                ttnn.experimental.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, attn_norm_str, i, self.num_devices)), attn_norm_host
                )
                self.attn_norm_list.append(attn_norm_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))

                ffn_norm_host = ttnn.experimental.tensor.Tensor(
                    # Expand to size of input since we decomped norm
                    self.state_dict[ffn_norm_str].reshape([1, 1, -1, 32]),
                    self.model_config["LN_MLP_WEIGHTS_DTYPE"],
                )
                ttnn.experimental.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, ffn_norm_str, i, self.num_devices)), ffn_norm_host
                )
                self.ffn_norm_list.append(ffn_norm_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))

    def prepare_inputs(self, x, start_pos):
        # Only called by decoder tests
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)

        x_multichip = []
        for i in range(self.num_devices):
            x_multichip.append(
                torch2tt_tensor(
                    x.clone(),
                    self.devices[i],
                    tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                )
            )
        for i in range(self.num_devices):
            x_multichip[i] = ttnn.experimental.tensor.interleaved_to_sharded(
                x_multichip[i], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
            )

        attn_batch = batch // 4

        position_ids = torch.ones(seq_len, attn_batch, dtype=torch.long) * start_pos
        rot_mat = gather_rotary_emb(self.rot_emb, position_ids)[:, :1]
        assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
        rot_mats = []
        for i in range(self.num_devices):
            rot_mats.append(
                torch2tt_tensor(
                    rot_mat.clone(),
                    self.devices[i],
                    tt_memory_config=self.model_config["ROT_MAT_MEMCFG"],  # TODO: Put on L1 instead of DRAM
                    tt_dtype=self.model_config["ROT_MAT_DTYPE"],
                )
            )

        padded_layer_past_len = nearest_32(start_pos + 1)
        if self.batched_attn:
            attn_mask_shape = (1, seq_len, self.padded_local_heads, padded_layer_past_len)
        else:
            attn_mask_shape = (seq_len, 1, attn_batch, padded_layer_past_len)
        attn_mask = torch.zeros(*attn_mask_shape)
        attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
        assert attn_mask.size() == attn_mask_shape
        attn_masks = []
        for i in range(self.num_devices):
            attn_masks.append(
                torch2tt_tensor(
                    attn_mask.clone(),
                    self.devices[i],
                    tt_dtype=self.model_config["ATTN_MASK_DTYPE"],  # BFLOAT16_DTYPE currently pushes faster
                )
            )
        if self.batched_attn:
            repeat_shape = (attn_batch, 1, 1, 1)
        else:
            repeat_shape = (1, self.n_local_heads, 1, 1)

        for i in range(self.num_devices):
            attn_masks[i] = ttnn.repeat(
                attn_masks[i], ttnn.Shape(repeat_shape), memory_config=self.model_config["DRAM_MEMCFG"]
            )
        # Put attn_mask on the device with the sharded config
        attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = padded_layer_past_len
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        for i in range(self.num_devices):
            attn_masks[i] = ttnn.experimental.tensor.interleaved_to_sharded(
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
            x_multichip,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def prepare_inputs_attention(self, x_multichip):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq, hidden_dim)
        start_pos: int
        """

        assert len(x_multichip) == 32, "Only 32 devices supported for galaxy"

        if self.emulated:
            # save l1 space by sharing the same input across "devices"
            for i in range(1, self.num_devices):
                x_multichip[i].deallocate(True)
                x_multichip[i] = x_multichip[0]

        return x_multichip

    def __call__(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> ttnn.experimental.tensor.Tensor:
        ### xs (residual stream) is full activation on all chips

        ## FOR BRINGUP!!! Put on DRAM because two activations makes L1 full on a single chip
        xs_replicated = []
        for i in range(self.num_devices):
            xs_replicated.append(
                ttnn.experimental.tensor.sharded_to_interleaved(
                    xs[i], output_mem_config=self.model_config["DRAM_MEMCFG"]
                )
            )

        attn_norm_replicated = []
        for i in range(self.num_devices):
            # TODO: Not Inplace RMSNorm because we need to keep the residual
            attn_norm_replicated.append(
                ttnn.rms_norm(
                    xs[i],
                    epsilon=self.norm_eps,
                    weight=self.attn_norm_list[i],
                    program_config=self.model_config["LN_ATTN_PROGCFG"],
                    memory_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                )
            )  # attn_norm_replicated is sharded on 32 cores on all 32 chips

        # TODO: Remove this when we support using 32 batch in attention galaxy
        attn_norm_replicated = self.prepare_inputs_attention(attn_norm_replicated)

        attn_outs = self.attention(attn_norm_replicated, rot_mats, start_pos, attn_masks)

        ## FOR BRINGUP!!! Only shard on L1 after attention deallocated inputs
        for i in range(self.num_devices):
            xs_replicated[i] = ttnn.experimental.tensor.interleaved_to_sharded(
                xs_replicated[i], sharded_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"]
            )

        ### Fractured residual add
        # Add attn output to residiual first in place to save memory
        output = []
        residual = xs_replicated  #  ## FOR BRINGUP!!! residual = xs
        for i in range(self.num_devices):
            output.append(
                ttnn.add(
                    residual[i],
                    attn_outs[i],
                    memory_config=self.model_config["ATTN_ADD_OUTPUT_MEMCFG"],
                    output_tensor=residual[i],
                )
            )
            attn_outs[i].deallocate(True)

        ### Duplicate FFN layernorm
        ffn_norm_replicated = []
        for i in range(self.num_devices):
            # TODO: Not Inplace RMSNorm because we need to keep the residual
            ffn_norm_replicated.append(
                ttnn.rms_norm(
                    output[i],
                    epsilon=self.norm_eps,
                    weight=self.ffn_norm_list[i],
                    program_config=self.model_config["LN_MLP_PROGCFG"],
                    memory_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"],
                )
            )  # ffn_norm_replicated is sharded on 32 cores on all 32 chips

        ffn_out = self.mlp(ffn_norm_replicated)

        ### residual in place add
        for i in range(self.num_devices):
            output[i] = ttnn.add(
                output[i],
                ffn_out[i],
                memory_config=self.model_config["MLP_ADD_OUTPUT_MEMCFG"],
                output_tensor=output[i],
            )
            ffn_out[i].deallocate(True)

        return output
