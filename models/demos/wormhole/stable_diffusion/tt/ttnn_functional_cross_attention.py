# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import os
from ttnn import squeeze, unsqueeze_to_4D
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    is_tile_dim_alligned,
    round_up_to_tile_dim,
    dealloc_input,
    determine_blocking,
    reshard_to,
)


def compare(tensor, name, transpose=False, unpad=False):
    return
    from models.utility_functions import comp_pcc

    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    tensor = ttnn.from_device(tensor)
    tensor = ttnn.to_torch(tensor)

    golden = torch.load(name)
    if transpose:
        tensor = tensor.transpose(-1, -2)
    unpad = tensor.shape[-1] != golden.shape[-1] or unpad
    if unpad:
        tensor = tensor[:, :, : golden.shape[-2], : golden.shape[-1]]

    while len(tensor.shape) > len(golden.shape):
        golden = golden.unsqueeze(0)
    while len(golden.shape) > len(tensor.shape):
        tensor = tensor.unsqueeze(0)

    passed, message = comp_pcc(tensor, golden, 0.95)
    print(f"Maches on {name}: {passed} with message {message}, tensor shape: {tensor.shape}")


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def pad_heads(tensor, num_heads=8, dim=-1):
    device = tensor.device()
    memory_config = ttnn.get_memory_config(tensor)

    padding_needed = not is_tile_dim_alligned(tensor.shape[dim] // num_heads)
    if padding_needed:
        tensor = ttnn_to_torch(tensor)
        unpadded_len = tensor.shape[-1] // num_heads
        padding_needed = round_up_to_tile_dim(unpadded_len) - unpadded_len
        unpadded_tensors = torch.split(tensor, tensor.shape[dim] // num_heads, dim=dim)
        if len(tensor.shape) == 2:
            padding = (
                torch.zeros((tensor.shape[-2], padding_needed))
                if dim == -1
                else torch.zeros((padding_needed, tensor.shape[-1]))
            )
        else:
            padding = (
                torch.zeros((1, 1, tensor.shape[-2], padding_needed))
                if dim == -1
                else torch.zeros((1, 1, padding_needed, tensor.shape[-1]))
            )
        padded_tensor = torch.Tensor()
        for unpadded_tensor in unpadded_tensors:
            padded_tensor = torch.cat([padded_tensor, unpadded_tensor, padding], dim=dim)

        while len(padded_tensor.shape) < 4:
            padded_tensor = padded_tensor.unsqueeze(0)

        padded_tensor = ttnn.from_torch(padded_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        padded_tensor = ttnn.to_device(padded_tensor, device, memory_config=memory_config)
        tensor = padded_tensor

    return tensor


def concatenate_qkv(q, k, v):
    dim = -1
    device = k.device()
    memory_config = ttnn.get_memory_config(k)

    if q is not None:
        q = ttnn_to_torch(q)
        assert is_tile_dim_alligned(q.shape[dim])

    k = ttnn_to_torch(k)
    v = ttnn_to_torch(v)

    assert is_tile_dim_alligned(k.shape[dim])
    assert is_tile_dim_alligned(v.shape[dim])

    num_heads = 8
    head_size = k.shape[dim] // num_heads
    if q is not None:
        # time_sharded attention uses interleaved_to_sharded_partial, so leave results interleaved
        interleaved_output = q.shape[-2] == 320 or q.shape[-2] == 640
        if interleaved_output:
            qkv = torch.cat([q, k, v], dim=dim)
        else:
            for i in range(num_heads):
                qkv_partial = torch.cat(
                    [
                        q[:, :, :, head_size * i : head_size * (i + 1)],
                        k[:, :, :, head_size * i : head_size * (i + 1)],
                        v[:, :, :, head_size * i : head_size * (i + 1)],
                    ],
                    dim=dim,
                )
                if i == 0:
                    qkv = qkv_partial
                else:
                    qkv = torch.cat([qkv, qkv_partial], dim=dim)

    else:
        for i in range(num_heads):
            qkv_partial = torch.cat(
                [k[:, :, :, head_size * i : head_size * (i + 1)], v[:, :, :, head_size * i : head_size * (i + 1)]],
                dim=dim,
            )
            if i == 0:
                qkv = qkv_partial
            else:
                qkv = torch.cat([qkv, qkv_partial], dim=dim)

    qkv = ttnn.from_torch(qkv, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    qkv = ttnn.to_device(qkv, device, memory_config=memory_config)
    return qkv


def weight_to_bfp8(weight):
    device = weight.device()
    memory_config = ttnn.get_memory_config(weight)
    weight = ttnn_to_torch(weight)
    weight = ttnn.from_torch(weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device, memory_config=memory_config)
    return weight


class cross_attention:
    def __init__(self, device, parameters, seq_len):
        self.program_configs = {}
        self.seq_len = seq_len
        self.fused_qkv = parameters.to_q.weight.shape[-2] == parameters.to_k.weight.shape[-2]
        for key in ["to_q", "to_k", "to_v"]:
            parameters[key].weight = pad_heads(parameters[key].weight, 8)
            assert "bias" not in parameters[key]
        parameters.to_out[0].weight = pad_heads(parameters.to_out[0].weight, 8, dim=-2)

        if self.fused_qkv:
            parameters["qkv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.qkv["weight"] = concatenate_qkv(
                parameters.to_q.weight, parameters.to_k.weight, parameters.to_v.weight
            )
            parameters.qkv.weight = ttnn.unsqueeze_to_4D(parameters.qkv.weight)
            self.qkv_len = parameters.qkv.weight.shape[-1]
        else:
            parameters["kv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.kv["weight"] = concatenate_qkv(None, parameters.to_k.weight, parameters.to_v.weight)
            parameters.to_q.weight = weight_to_bfp8(parameters.to_q.weight)
            self.q_len = parameters.to_q.weight.shape[-1]
            self.kv_len = parameters.kv.weight.shape[-1]

            parameters.kv.weight = ttnn.unsqueeze_to_4D(parameters.kv.weight)
            parameters.to_q.weight = ttnn.unsqueeze_to_4D(parameters.to_q.weight)

        parameters.to_out[0].weight = ttnn.unsqueeze_to_4D(parameters.to_out[0].weight)
        parameters.to_out[0].bias = ttnn.unsqueeze_to_4D(parameters.to_out[0].bias)
        self.parameters = parameters
        self.device = device

        if seq_len == 4096:
            self.dim_head = 40
        elif seq_len == 1024:
            self.dim_head = 80
        else:
            self.dim_head = 160

        scale = torch.ones((1, 1, 1, 1)) * 1 / math.sqrt(self.dim_head)
        self.scale = ttnn.from_torch(scale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

        attention_mask_96 = torch.ones((1, 1, seq_len, 96)) * -1e9
        attention_mask_96[:, :, :, :77] = 0
        attention_mask_96 = ttnn.from_torch(
            attention_mask_96, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        attention_mask_full = torch.zeros((1, 1, seq_len, seq_len))
        attention_mask_full = ttnn.from_torch(
            attention_mask_full, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.attention_masks = {"cross": attention_mask_96, "self": attention_mask_full}

        self.parameters.to_out[0].weight = weight_to_bfp8(self.parameters.to_out[0].weight)
        self.parameters.to_out[0].bias = weight_to_bfp8(self.parameters.to_out[0].bias)

        self.dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.l1_interleaved_memory_config = ttnn.L1_MEMORY_CONFIG
        self.block_sharded_memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        self.height_sharded_memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        self.width_sharded_memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.encoder_hidden_states_shape = (2, 96, 768)
        if seq_len == 4096:
            hidden_dim = 320
        elif seq_len == 1024:
            hidden_dim = 640
        else:
            hidden_dim = 1280
        self.hidden_states_shape = (1, 1, 2 * seq_len, hidden_dim)
        self.grid_sizes = {4096: (5, 8), 1024: (5, 8), 256: (8, 8), 64: (8, 4)}
        if self.fused_qkv:
            grid_size = self.grid_sizes[seq_len]
            M = seq_len * 2
            K, N = self.parameters.qkv.weight.shape[-2], self.parameters.qkv.weight.shape[-1]
            in0_block_h, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = determine_blocking(
                M, K, N, grid_size
            )
            self.program_configs["qkv"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=out_block_h,
                per_core_N=out_block_w,
                transpose_mcast=False,
                fused_activation=None,
            )
        else:
            grid_size = self.grid_sizes[seq_len]
            M = seq_len * 2
            K, N = self.parameters.to_q.weight.shape[-2], self.parameters.to_q.weight.shape[-1]
            in0_block_h, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = determine_blocking(
                M, K, N, grid_size
            )
            self.program_configs["q"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=out_block_h,
                per_core_N=out_block_w,
                transpose_mcast=False,
                fused_activation=None,
            )

            M, K, N = (
                self.encoder_hidden_states_shape[-2] * self.encoder_hidden_states_shape[-3],
                self.encoder_hidden_states_shape[-1],
                self.parameters.kv.weight.shape[-1],
            )
            grid_sizes = {4096: (8, 2), 1024: (8, 2), 256: (8, 2), 64: (4, 2)}
            grid_size = grid_sizes[seq_len]
            in0_block_h, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = determine_blocking(
                M, K, N, grid_size
            )
            self.program_configs["kv"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=out_block_h,
                per_core_N=out_block_w,
                transpose_mcast=False,
                fused_activation=None,
            )

        have_time_sharded_attention = seq_len == 4096 or seq_len == 1024
        if have_time_sharded_attention:
            self.key_len = 64 if seq_len == 4096 else 96
            self.num_slices = 16
            self.tsa_grid_size = (8, 8)
            num_cores = self.tsa_grid_size[0] * self.tsa_grid_size[1]
            seq_len = self.seq_len
            tiles_per_shard = math.ceil((((self.num_slices * seq_len) / num_cores) / self.num_slices) / 32)
            self.tsa_mm_activations_height_shard_spec = [tiles_per_shard * 32, self.key_len]
            mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

            out_subblock_h = 1
            out_subblock_w = 8
            slow_mm = os.environ.get("SLOW_MATMULS", "0") == "1"
            if slow_mm:
                out_subblock_h = 1
                out_subblock_w = 1
            self.program_configs["tsa_qkt"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=self.tsa_grid_size,
                in0_block_w=self.key_len // 32,
                per_core_M=tiles_per_shard,
                per_core_N=seq_len // 32,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )

            self.program_configs["tsa_softmax"] = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=self.tsa_grid_size,
                subblock_w=1,
                block_h=mm_output_height_shard_spec[0] // 32,
                block_w=mm_output_height_shard_spec[1] // 32,
            )
            out_subblock_h = tiles_per_shard
            out_subblock_w = self.key_len // 32
            if slow_mm:
                out_subblock_h = 1
                out_subblock_w = 1
            self.program_configs["tsa_v"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=self.tsa_grid_size,
                in0_block_w=seq_len // 32,
                per_core_M=tiles_per_shard,
                per_core_N=self.key_len // 32,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )
            self.tsa_output_shape = (2, 8, seq_len, self.key_len)
            output_tensor = torch.zeros(self.tsa_output_shape)
            output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            self.output_tensor = ttnn.to_device(output_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            output_cores = 16
            self.tsa_output_shard_shape = [
                self.output_tensor.volume() // self.output_tensor.shape[-1] // output_cores,
                self.output_tensor.shape[-1],
            ]

    def time_sharded_attention(self, query, t_key, value, head_size, attn_type):
        attention_mask = self.attention_masks[attn_type]
        for j in range(2):
            for i in range(self.num_slices // 2):
                slice = ttnn.interleaved_to_sharded_partial(
                    query,
                    self.tsa_grid_size,
                    self.tsa_mm_activations_height_shard_spec,
                    self.num_slices,
                    j * self.num_slices // 2 + i,
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.ShardOrientation.ROW_MAJOR,
                )

                k_slice = ttnn.slice(
                    t_key,
                    (j, i, 0, 0),
                    (j, i, self.key_len - 1, self.seq_len - 1),
                    memory_config=self.l1_interleaved_memory_config,
                )

                mm_slice = ttnn.matmul(
                    slice,
                    k_slice,
                    program_config=self.program_configs["tsa_qkt"],
                    memory_config=self.height_sharded_memory_config,
                    dtype=ttnn.bfloat8_b,
                    compute_kernel_config=self.compute_kernel_config,
                )
                k_slice.deallocate()
                slice.deallocate()

                use_mask = False
                if use_mask:
                    mm_slice = ttnn.scale_mask_softmax_in_place(
                        mm_slice,
                        1 / math.sqrt(head_size),
                        attention_mask,
                        program_config=self.program_configs["tsa_softmax"],
                        is_causal_mask=False,
                    )
                else:
                    # this needs to be in-place, when optional output tensors are available
                    # this code should be updated
                    mm_slice = ttnn.multiply(
                        mm_slice,
                        self.scale,
                        memory_config=self.height_sharded_memory_config,
                    )
                    mm_slice = ttnn.softmax_in_place(
                        mm_slice,
                        program_config=self.program_configs["tsa_softmax"],
                    )

                v_slice = ttnn.slice(
                    value,
                    (j, i, 0, 0),
                    (j, i, self.seq_len - 1, self.key_len - 1),
                    memory_config=self.l1_interleaved_memory_config,
                )
                mm_slice = ttnn.matmul(
                    mm_slice,
                    v_slice,
                    program_config=self.program_configs["tsa_v"],
                    memory_config=self.height_sharded_memory_config,
                    dtype=ttnn.bfloat8_b,
                    compute_kernel_config=self.compute_kernel_config,
                )
                v_slice.deallocate()

                ttnn.sharded_to_interleaved_partial(
                    mm_slice,
                    self.output_tensor,
                    self.num_slices,
                    j * self.num_slices // 2 + i,
                    memory_config=self.dram_interleaved_memory_config,
                )
        output = ttnn.interleaved_to_sharded(
            self.output_tensor,
            (8, 2),
            self.tsa_output_shard_shape,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return output

    def sharded_attention(self, query, key, value, head_size, attn_type):
        grid_size = (2, 8)
        num_cores = grid_size[0] * grid_size[1]
        num_heads = 16
        seq_len = query.shape[-2]
        inner = query.shape[-1]
        key_len = key.shape[-1]
        attention_mask = self.attention_masks[attn_type]

        query, key, value = [
            ttnn.reshape(tensor, (1, tensor.shape[-3] * 2, tensor.shape[-2], tensor.shape[-1]))
            for tensor in [query, key, value]
        ]

        if not query.is_sharded():
            q_sharded = ttnn.interleaved_to_sharded(
                query,
                grid_size,
                [num_heads * seq_len // num_cores, inner],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            query.deallocate()
        else:
            q_sharded = query
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=inner // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=num_heads * seq_len // num_cores // 32,
            per_core_N=key_len // 32,
        )
        attention_scores = dealloc_input(
            ttnn.matmul,
            q_sharded,
            key,
            program_config=program_config,
            memory_config=self.height_sharded_memory_config,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
        )

        num_cores_softmax = num_heads * seq_len // 32
        if num_cores_softmax > 64:
            num_cores_softmax = 64
            end_grid = ttnn.CoreCoord(7, 7)
            compute_with_storage_grid_size = (8, 8)
        elif num_cores_softmax == 48:
            end_grid = ttnn.CoreCoord(7, 5)
            compute_with_storage_grid_size = (8, 6)
        elif num_cores_softmax == 32:
            end_grid = ttnn.CoreCoord(7, 3)
            compute_with_storage_grid_size = (8, 4)

        height_per_core = num_heads * seq_len // num_cores_softmax
        orig_mem_config = attention_scores.memory_config()

        output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end_grid)})
        output_shard_spec = ttnn.ShardSpec(
            output_shard_grid,
            [height_per_core, key_len],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            output_shard_spec,
        )
        attention_scores = ttnn.reshard(
            attention_scores,
            output_mem_config,
        )
        # attention_scores = ttnn.experimental.tensor.move_sharded(attention_scores)
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=compute_with_storage_grid_size,
            subblock_w=1,
            block_h=height_per_core // 32,
            block_w=key_len // 32,
        )
        use_mask = attn_type == "cross"
        if use_mask:
            attention_scores = ttnn.scale_mask_softmax_in_place(
                attention_scores,
                1 / math.sqrt(head_size),
                attention_mask,
                program_config=softmax_program_config,
                is_causal_mask=False,
            )
        else:
            # This needs to be updated when optional output tensors are available
            attention_scores_temp = attention_scores
            attention_scores = ttnn.multiply(
                attention_scores_temp,
                self.scale,
                memory_config=attention_scores.memory_config(),
            )
            attention_scores_temp.deallocate()
            attention_scores = ttnn.softmax_in_place(
                attention_scores,
                program_config=softmax_program_config,
            )
        attention_scores = dealloc_input(ttnn.reshard, attention_scores, orig_mem_config)

        if not value.is_sharded():
            v_sharded = ttnn.interleaved_to_sharded(
                value,
                grid_size,
                [num_heads * key_len // num_cores, inner],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            value.deallocate()
        else:
            v_sharded = value

        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=key_len // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=num_heads * seq_len // num_cores // 32,
            per_core_N=inner // 32,
        )
        attention_scores = ttnn.matmul(
            attention_scores,
            v_sharded,
            program_config=program_config,
            memory_config=self.height_sharded_memory_config,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
        )
        attention_scores = reshard_to(attention_scores, (8, 2), ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
        v_sharded.deallocate()
        return ttnn.reshape(attention_scores, (2, 8, attention_scores.shape[-2], attention_scores.shape[-1]))

    def get_attention_scores_opt(self, query, t_key, value, head_size, attn_type):
        if (query.shape[-2] == 4096 and t_key.shape[-1] == 4096) or (
            query.shape[-2] == 1024 and t_key.shape[-1] == 1024
        ):
            return self.time_sharded_attention(query, t_key, value, head_size, attn_type)
        else:
            return self.sharded_attention(query, t_key, value, head_size, attn_type)

    def out(self, hidden_states):
        size = hidden_states.shape[-2] // 2  # 2 is the batch size

        grid_sizes = {4096: (4, 8), 1024: (4, 8), 256: (5, 8), 64: (8, 4)}
        shard_directions = {
            4096: ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            1024: ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            256: ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            64: ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        }
        out_subblock_hs = {256: 8, 64: 4}

        grid_size = grid_sizes[size]
        num_cores = grid_size[0] * grid_size[1]
        B, M, K, N = 1, hidden_states.shape[-2], hidden_states.shape[-1], self.parameters.to_out[0].weight.shape[-1]

        hs = shard_directions[size] == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if hs:
            hidden_states = reshard_to(hidden_states, grid_size, ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
            output_mem_config = self.height_sharded_memory_config
            program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=K // 32 if hs else 1,
                per_core_M=B * M // num_cores // 32 if hs else B * M // 32,
                per_core_N=N // 32 if hs else N // num_cores // 32,
                out_subblock_h=1 if hs else out_subblock_hs[size],
                out_subblock_w=2 if hs else 1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False if hs else True,
            )
        else:
            hidden_states = reshard_to(hidden_states, grid_size, ttnn.TensorMemoryLayout.BLOCK_SHARDED)
            output_mem_config = self.block_sharded_memory_config
            in0_block_h, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = determine_blocking(
                M, K, N, grid_size
            )
            program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=out_block_h,
                per_core_N=out_block_w,
                transpose_mcast=False,
                fused_activation=None,
            )

        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters.to_out[0].weight,
            bias=self.parameters.to_out[0].bias,
            program_config=program_config,
            memory_config=output_mem_config,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
        )

        return hidden_states

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        query_dim: int = None,
        cross_attention_dim=None,
        heads: int = 8,
        dim_head: int = 64,
        attention_mask=None,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_kwargs={},
        index=-1,
    ):
        assert dim_head == self.dim_head

        if not hidden_states.is_sharded():
            grid_size = self.grid_sizes[self.seq_len]
            hidden_states = ttnn.interleaved_to_sharded(
                hidden_states,
                grid_size,
                [self.hidden_states_shape[-2] // grid_size[1], self.hidden_states_shape[-1] // grid_size[0]],
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        if encoder_hidden_states:
            encoder_hidden_states = ttnn.reshape(
                encoder_hidden_states,
                (1, 1, self.encoder_hidden_states_shape[-2] * 2, self.encoder_hidden_states_shape[-1]),
            )

        if self.fused_qkv:
            program_config = self.program_configs["qkv"]
            # TODO: Output sharded once https://github.com/tenstorrent/tt-metal/issues/6775 is fixed
            interleaved_out = self.seq_len == 4096 or self.seq_len == 1024
            qkv_out = ttnn.matmul(
                hidden_states,
                self.parameters.qkv.weight,
                program_config=program_config,
                memory_config=(
                    self.l1_interleaved_memory_config if interleaved_out else self.block_sharded_memory_config
                ),
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(hidden_states)
            qkv_out = ttnn.reshape(qkv_out, (2, self.seq_len, self.qkv_len))
            if interleaved_out:
                query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                    qkv_out,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG if self.seq_len == 4096 else ttnn.L1_MEMORY_CONFIG,
                    num_heads=heads,
                )
            else:
                qkv_out = reshard_to(qkv_out, (8, 2), ttnn.TensorMemoryLayout.BLOCK_SHARDED)
                query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                    qkv_out,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    num_heads=heads,
                )
                query = reshard_to(query, (2, 8), ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
                key = reshard_to(key, (2, 8), ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
                value = reshard_to(value, (2, 8), ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
            ttnn.deallocate(qkv_out)
        else:
            if self.seq_len == 4096:
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

            program_config = self.program_configs["q"]
            q_proj = ttnn.matmul(
                hidden_states,
                self.parameters.to_q.weight,
                program_config=program_config,
                memory_config=self.block_sharded_memory_config,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(hidden_states)

            program_config = self.program_configs["kv"]
            kv_proj = ttnn.matmul(
                encoder_hidden_states,
                self.parameters.kv.weight,
                program_config=program_config,
                memory_config=self.block_sharded_memory_config,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_kernel_config,
            )
            end_core = ttnn.CoreCoord(7, 1) if self.seq_len != 64 else ttnn.CoreCoord(3, 1)
            grid_size = (8, 2) if self.seq_len != 64 else (4, 2)
            q_proj = reshard_to(q_proj, grid_size, ttnn.TensorMemoryLayout.BLOCK_SHARDED)
            output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end_core)})
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                [self.seq_len, self.q_len // grid_size[0]],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            )
            output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                output_shard_spec,
            )
            q_proj = ttnn.reshard(
                q_proj,
                output_mem_config,
            )

            output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end_core)})
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                [96, self.kv_len // grid_size[0]],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            )
            output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                output_shard_spec,
            )
            kv_proj = ttnn.reshard(
                kv_proj,
                output_mem_config,
            )
            q_proj = ttnn.reshape(q_proj, (2, 1, self.seq_len, self.q_len))
            kv_proj = ttnn.reshape(kv_proj, (2, 1, 96, self.kv_len))
            query, key, value = ttnn.experimental.create_qkv_heads_from_separate_tensors(
                q_proj,
                kv_proj,
                num_heads=8,
                num_kv_heads=8,
                transpose_k_heads=True,
                memory_config=self.height_sharded_memory_config,
            )
            ttnn.deallocate(kv_proj)
            ttnn.deallocate(q_proj)
            query = reshard_to(query, (2, 8), ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
            key = reshard_to(key, (2, 8), ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
            value = reshard_to(value, (2, 8), ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

        hidden_states = self.get_attention_scores_opt(
            query, key, value, dim_head, attn_type="cross" if encoder_hidden_states else "self"
        )

        hidden_states = ttnn.transformer.concatenate_heads(
            hidden_states,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        )

        if hidden_states.shape.with_tile_padding()[-1] != hidden_states.shape[-1]:
            assert False
            hidden_states = hidden_states[:, :, : hidden_states.shape[-1]]

        B, M, K, N = 1, hidden_states.shape[-2], hidden_states.shape[-1], self.parameters.to_out[0].weight.shape[-1]
        hidden_states = ttnn.reshape(hidden_states, (1, 1, 2 * hidden_states.shape[-2], hidden_states.shape[-1]))
        hidden_states = self.out(hidden_states)

        if len(hidden_states.shape) == 3:
            hidden_states = unsqueeze_to_4D(hidden_states)

        return hidden_states
