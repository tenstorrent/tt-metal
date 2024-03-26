# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import math
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, nearest_32, pad_by_zero
from models.demos.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb,
    get_weight_cache_path,
    generate_rot_emb,
)
from models.demos.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized


class TtLlamaAttention_galaxy(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        emulated=False,
        load_weights=True,
        cache_path=None,
        kv_cache_dir=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
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
        self.n_kv_heads = configuration.n_kv_heads
        self.max_batch_size = configuration.max_batch_size
        self.batch_size_per_device_group = self.max_batch_size // self.num_device_groups

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.scale = 1 / math.sqrt(self.head_dim)

        # initialize num_device_groups attention modules.
        # num_devices_per_group devices should be used for each attention module.
        self.attentions = [
            TtLlamaAttention_optimized(
                self.device_groups[i],
                state_dict,
                base_url,
                layer_num,
                model_config,
                configuration,
                emulated,
                load_weights,
                cache_path,
                kv_cache_dir,
                batch_size=self.batch_size_per_device_group,
            )
            for i in range(self.num_device_groups)
        ]

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
            x_multichip[i] = tt_lib.tensor.interleaved_to_sharded(
                x_multichip[i], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
            )

        attn_batch = batch // 4

        position_ids = torch.ones(seq_len, attn_batch, dtype=torch.long) * start_pos
        rot_emb = generate_rot_emb(self.head_dim, self.max_seq_len * 2)
        rot_mat = gather_rotary_emb(rot_emb, position_ids)[:, :1]
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
        attn_mask_shape = (1, seq_len, 32, padded_layer_past_len)
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
        repeat_shape = (attn_batch, 1, 1, 1)

        for i in range(self.num_devices):
            attn_masks[i] = tt_lib.tensor.repeat(
                attn_masks[i], repeat_shape, output_mem_config=self.model_config["DRAM_MEMCFG"]
            )
        # Put attn_mask on the device with the sharded config
        attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = padded_layer_past_len
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        for i in range(self.num_devices):
            attn_masks[i] = tt_lib.tensor.interleaved_to_sharded(
                attn_masks[i], sharded_mem_config=attention_mask_memconfig
            )

        if self.emulated:
            # save l1 space by sharing the same input across "devices"
            for i in range(1, self.num_devices):
                x_multichip[i].deallocate(True)
                rot_mats[i].deallocate(True)
                attn_masks[i].deallocate(True)
                x_multichip[i] = x_multichip[0]
                rot_mats[i] = rot_mats[0]
                attn_masks[i] = attn_masks[0]

        return (
            x_multichip,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def forward(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        # forward pass for each attention module
        all_attn_outputs = []
        # First calculate attention for each device group
        for group_id in range(self.num_device_groups):
            # fetch input per group
            devices = self.device_groups[group_id]
            xs_group = xs[group_id * self.num_devices_per_group : (group_id + 1) * self.num_devices_per_group]
            rot_mats_group = rot_mats[
                group_id * self.num_devices_per_group : (group_id + 1) * self.num_devices_per_group
            ]
            attn_masks_group = attn_masks[
                group_id * self.num_devices_per_group : (group_id + 1) * self.num_devices_per_group
            ]

            # QKV projection
            query_layer, key_layer, value_layer = self.attentions[group_id].attn_qkv(xs_group, rot_mats_group)

            # Each attention group is responsible for different batch ids
            batch_offset = self.batch_size_per_device_group * group_id
            for i in range(len(query_layer)):
                # Unpad the batch dimension to original batch if the original batch is not 32
                query_layer[i] = tt_lib.tensor.unpad(
                    query_layer[i],
                    [batch_offset, 0, 0, 0],
                    [
                        batch_offset + self.batch_size_per_device_group - 1,
                        0,
                        31,
                        self.head_dim - 1,
                    ],
                    output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                )

            # Attention
            outputs_group = self.attentions[group_id].attn_mqa(
                query_layer, key_layer, value_layer, start_pos, attn_masks_group, batch_offset=batch_offset
            )
            all_attn_outputs.append(outputs_group)
        # Do all gather across device groups before selfout
        for mqa_id in range(self.num_devices_per_group):
            output_per_mqa = [outputs_group[mqa_id] for outputs_group in all_attn_outputs]
            output_per_mqa = tt_all_gather_torch(output_per_mqa, dim=0)  # gather on batch dimension
            # update all_attn_outputs
            for group_id in range(self.num_device_groups):
                all_attn_outputs[group_id][mqa_id] = output_per_mqa[group_id]
        # Do selfout
        for group_id in range(self.num_device_groups):
            all_attn_outputs[group_id] = self.attentions[group_id].attn_selfout(all_attn_outputs[group_id])

        if self.emulated:
            for group_id in range(self.num_device_groups):
                all_attn_outputs[group_id] = tt_all_gather_torch(all_attn_outputs[group_id], dim=-1)

        # flatten the all_attn_outputs and return
        all_attn_outputs = [output for outputs_group in all_attn_outputs for output in outputs_group]

        if self.emulated:
            # FOR BRINGUP! Outputs are Interaved, Shard them
            for i in range(len(all_attn_outputs)):
                all_attn_outputs[i] = tt_lib.tensor.interleaved_to_sharded(
                    all_attn_outputs[i], sharded_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"]
                )

        return all_attn_outputs
