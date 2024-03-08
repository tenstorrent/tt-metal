# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
from loguru import logger

import math
import tt_lib
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, nearest_32, pad_by_zero
from models.demos.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    precompute_freqs as tt_precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb as tt_gather_rotary_emb,
    get_weight_cache_path,
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

    def prepare_inputs(self, x_all, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq, hidden_dim)
        start_pos: int
        """
        assert x_all.size(2) == self.hidden_size
        assert len(x_all.size()) == 3
        batch_all = x_all.size(0)
        seq_len = x_all.size(1)
        assert seq_len == 1, "Only supporting decode mode"

        all_xs, all_rot_mats, all_attn_masks = [], [], []

        for i in range(self.num_device_groups):
            # index out batch for each device groups
            devices = self.device_groups[i]
            x = x_all[i * self.batch_size_per_device_group : (i + 1) * self.batch_size_per_device_group]
            batch = x.size(0)

            # prepare inputs for each attention module
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
            rot_mat = self.attentions[0].get_rotation_mat(
                dhead=self.head_dim, end=self.max_seq_len * 2, start_pos=start_pos, seqlen=seq_len, batch=batch
            )
            rot_mat = rot_mat[:, :1]

            padded_layer_past_len = nearest_32(start_pos + 1)
            attn_mask_shape = (batch, seq_len, self.attentions[0].padded_local_heads, padded_layer_past_len)
            attn_mask = torch.zeros(*attn_mask_shape)
            attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min

            # expected shapes:
            # x: (seq_len, 1, batch, hidden_dim)
            # start_pos: int
            # rot_mat: [1, 1, head_dim, head_dim]
            # attn_mask: [batch, seq_len, n_heads, padded_layer_past_len]
            assert x.size() == (seq_len, 1, batch, self.hidden_size)
            assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
            assert attn_mask.size() == attn_mask_shape
            xs, rot_mats, attn_masks = [], [], []
            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = padded_layer_past_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
            for i in range(self.num_devices_per_group):
                device = devices[i]
                xs.append(
                    pad_by_zero(  # padded x_tt to seq_len, 1, 32, hidden_dim
                        x.clone(),
                        device,
                        tt_dtype=self.model_config["LN_ATTN_OUTPUT_DTYPE"],
                    )[0]
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
            for i in range(self.num_devices_per_group):
                xs[i] = tt_lib.tensor.interleaved_to_sharded(
                    xs[i], sharded_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"]
                )
                # attn_masks[i] is [8, 1, 32, 128]
                attn_masks[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_masks[i], sharded_mem_config=attention_mask_memconfig
                )

            # if emulated, use only the first copy of xs per group
            if self.emulated:
                first_x = xs[0]
                [x.deallocate(True) for x in xs[1:]]
                xs = [first_x for _ in range(self.num_devices_per_group)]

            # extend to the all_xs, all_rot_mats, all_attn_masks
            all_xs.extend(xs)
            all_rot_mats.extend(rot_mats)
            all_attn_masks.extend(attn_masks)

        # if emulated, use only the first copy rot_mats and attn_masks
        if self.emulated:
            first_rot_mat, first_attn_mask = all_rot_mats[0], all_attn_masks[0]
            [rot_mat.deallocate(True) for rot_mat in all_rot_mats[1:]]
            [attn_mask.deallocate(True) for attn_mask in all_attn_masks[1:]]
            all_rot_mats = [first_rot_mat for _ in range(self.num_devices)]
            all_attn_masks = [first_attn_mask for _ in range(self.num_devices)]

        return (
            all_xs,
            start_pos,
            all_rot_mats,
            all_attn_masks,
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
            devices = self.device_groups[group_id]
            xs_group = xs[group_id * self.num_devices_per_group : (group_id + 1) * self.num_devices_per_group]
            rot_mats_group = rot_mats[
                group_id * self.num_devices_per_group : (group_id + 1) * self.num_devices_per_group
            ]
            attn_masks_group = attn_masks[
                group_id * self.num_devices_per_group : (group_id + 1) * self.num_devices_per_group
            ]
            outputs_group = self.attentions[group_id].attn_qkv_mqa(
                xs_group, rot_mats_group, start_pos, attn_masks_group
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

        # flatten the all_attn_outputs and return
        return [output for outputs_group in all_attn_outputs for output in outputs_group]
