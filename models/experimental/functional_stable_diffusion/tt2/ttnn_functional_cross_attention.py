# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import tt_lib as ttl
from ttnn.operations.core import squeeze, unsqueeze_to_4D
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    is_tile_dim_alligned,
    round_up_to_tile_dim,
)


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
        padding = (
            torch.zeros((tensor.shape[-4], tensor.shape[-3], tensor.shape[-2], padding_needed))
            if dim == -1
            else torch.zeros((tensor.shape[-4], tensor.shape[-3], padding_needed, tensor.shape[-1]))
        )
        padded_tensor = torch.Tensor()
        for unpadded_tensor in unpadded_tensors:
            padded_tensor = torch.cat([padded_tensor, unpadded_tensor, padding], dim=dim)

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

    if q is not None:
        qkv = torch.cat([q, k, v], dim=dim)
    else:
        qkv = torch.cat([k, v], dim=dim)

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
    def __init__(self, device, parameters):
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
        else:
            parameters["kv"] = ttnn.model_preprocessing.ParameterDict()
            parameters.kv["weight"] = concatenate_qkv(None, parameters.to_k.weight, parameters.to_v.weight)
            parameters.to_q.weight = weight_to_bfp8(parameters.to_q.weight)

        self.parameters = parameters
        self.device = device

        scale = torch.ones((1, 1)) * 40**-0.5
        self.scale_40 = ttnn.from_torch(scale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

        scale = torch.ones((1, 1)) * 80**-0.5
        self.scale_80 = ttnn.from_torch(scale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

        scale = torch.ones((1, 1)) * 160**-0.5
        self.scale_160 = ttnn.from_torch(scale, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)

        self.scales = {40: self.scale_40, 80: self.scale_80, 160: self.scale_160}

        attention_mask_96 = torch.ones((2, 1, 1024, 96)) * -1e9
        attention_mask_96[:, :, :, :64] = 0
        attention_mask_96 = ttnn.from_torch(
            attention_mask_96, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_96_160 = torch.ones((2, 1, 1, 96)) * -1e9
        attention_mask_96_160[:, :, :, :64] = 0
        attention_mask_96_160 = ttnn.from_torch(
            attention_mask_96_160, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_64 = torch.zeros((2, 1, 1, 64))
        attention_mask_64 = ttnn.from_torch(
            attention_mask_64, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_256 = torch.zeros((2, 1, 1, 256))
        attention_mask_256 = ttnn.from_torch(
            attention_mask_256, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_1024 = torch.zeros((2, 1, 1, 1024))
        attention_mask_1024 = ttnn.from_torch(
            attention_mask_1024, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        attention_mask_4096 = torch.zeros((2, 1, 1, 4096))
        attention_mask_4096 = ttnn.from_torch(
            attention_mask_4096, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        attention_mask_time_sharded = torch.zeros((1, 1, 4096, 4096))
        attention_mask_time_sharded = ttnn.from_torch(
            attention_mask_time_sharded, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.attention_mask_time_sharded = attention_mask_time_sharded

        self.attention_masks = {
            64: attention_mask_64,
            96: attention_mask_96,
            96160: attention_mask_96_160,
            256: attention_mask_256,
            1024: attention_mask_1024,
            4096: attention_mask_4096,
        }

        padding_shapes = [[2, 4000, 1024], [2, 928, 1536], [2, 160, 2560], [2, 32, 1280]]
        self.padded_tensors = {}
        for shape in padding_shapes:
            padding = torch.zeros(shape)
            padding = ttnn.from_torch(padding, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device)
            self.padded_tensors[shape[-2]] = padding
        self.parameters.to_out[0].weight = weight_to_bfp8(self.parameters.to_out[0].weight)
        self.parameters.to_out[0].bias = weight_to_bfp8(self.parameters.to_out[0].bias)

        self.grid_sizes = {8192: (8, 5), 2048: (8, 5), 512: (8, 8), 128: (4, 8)}
        self.out_subblock_hs = {8192: 8, 2048: 8, 512: 2, 128: 1}
        self.dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.DRAM,
        )
        self.l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttl.tensor.BufferType.L1,
        )
        self.block_sharded_memory_config = ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
        )
        self.height_sharded_memory_config = ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
        )

        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        output_tensor = torch.zeros(2, 8, 4096, 64)
        self.output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.output_tensor = ttnn.to_device(self.output_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def time_sharded_attention(self, query, t_key, value, head_size):
        attention_mask = self.attention_mask_time_sharded
        num_slices = 16
        grid_size = (8, 8)
        num_cores = grid_size[0] * grid_size[1]
        seq_len = 4096
        tiles_per_shard = math.ceil((((num_slices * seq_len) / num_cores) / num_slices) / 32)
        mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
        mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]
        for j in range(2):
            for i in range(num_slices // 2):
                slice = ttl.tensor.interleaved_to_sharded_partial(
                    query,
                    grid_size,
                    mm_activations_height_shard_spec,
                    num_slices,
                    j * num_slices // 2 + i,
                    ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                )
                program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=2,
                    per_core_M=tiles_per_shard,
                    per_core_N=seq_len // 32,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=False,
                )
                k_slice = ttl.tensor.unpad(
                    t_key,
                    (j, i, 0, 0),
                    (j, i, 63, seq_len - 1),
                    output_mem_config=self.l1_interleaved_memory_config,
                )

                height_sharded_memory_config = ttl.tensor.MemoryConfig(
                    memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
                )
                mm_slice = ttl.operations.primary.matmul(
                    slice,
                    k_slice,
                    program_config=program_config,
                    output_mem_config=height_sharded_memory_config,
                    output_dtype=ttl.tensor.DataType.BFLOAT8_B,
                    compute_kernel_config=self.compute_kernel_config,
                )
                k_slice.deallocate()
                slice.deallocate()

                softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    subblock_w=1,
                    block_h=mm_output_height_shard_spec[0] // 32,
                    block_w=mm_output_height_shard_spec[1] // 32,
                )

                mm_slice = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
                    mm_slice,
                    1 / math.sqrt(head_size),
                    attention_mask,
                    program_config=softmax_program_config,
                    is_causal_mask=False,
                )

                program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=seq_len // 32,
                    per_core_M=tiles_per_shard,
                    per_core_N=2,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=False,
                )
                v_slice = ttl.tensor.unpad(
                    value,
                    (j, i, 0, 0),
                    (j, i, seq_len - 1, 63),
                    output_mem_config=self.l1_interleaved_memory_config,
                )
                mm_slice = ttl.operations.primary.matmul(
                    mm_slice,
                    v_slice,
                    program_config=program_config,
                    output_mem_config=height_sharded_memory_config,
                    output_dtype=ttl.tensor.DataType.BFLOAT8_B,
                    compute_kernel_config=self.compute_kernel_config,
                )
                v_slice.deallocate()

                ttl.tensor.sharded_to_interleaved_partial(
                    mm_slice,
                    self.output_tensor,
                    num_slices,
                    j * num_slices // 2 + i,
                    self.dram_interleaved_memory_config,
                )
        return self.output_tensor

    def sharded_attention(self, query, key, value, original_seq_len, head_size, attention_mask):
        grid_size = (8, 2)
        num_cores = grid_size[0] * grid_size[1]
        num_heads = 16
        seq_len = query.shape[-2]
        inner = query.shape[-1]
        key_len = key.shape[-1]

        query, key, value = [
            ttnn.reshape(tensor, (1, tensor.shape[-3] * 2, tensor.shape[-2], tensor.shape[-1]))
            for tensor in [query, key, value]
        ]

        program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=inner // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=num_heads * seq_len // num_cores // 32,
            per_core_N=key_len // 32,
        )

        q_sharded = ttl.tensor.interleaved_to_sharded(
            query,
            grid_size,
            [num_heads * seq_len // num_cores, inner],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )
        query.deallocate()
        attention_scores = ttl.operations.primary.matmul(
            q_sharded,
            key,
            program_config=program_config,
            output_mem_config=self.height_sharded_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=self.compute_kernel_config,
        )
        q_sharded.deallocate()

        # attention_scores = ttl.tensor.move_sharded(attention_scores)
        if seq_len == 64 and key_len == 64:
            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 2),
                subblock_w=1,
                block_h=num_heads * seq_len // 16 // 32,
                block_w=key_len // 32,
            )
            attention_scores = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
                attention_scores,
                1 / math.sqrt(head_size),
                attention_mask,
                program_config=softmax_program_config,
                is_causal_mask=False,
            )
        else:
            height_per_core = num_heads * seq_len // 64
            orig_mem_config = attention_scores.memory_config()
            output_shard_grid = ttl.tensor.CoreRangeSet(
                {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(7, 7))}
            )
            output_shard_spec = ttl.tensor.ShardSpec(
                output_shard_grid, [height_per_core, key_len], ttl.tensor.ShardOrientation.COL_MAJOR, False
            )
            output_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec
            )
            attention_scores = ttl.tensor.reshard(
                attention_scores,
                output_mem_config,
            )
            # attention_scores = ttl.tensor.move_sharded(attention_scores)
            print(attention_scores.shape)
            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                subblock_w=1,
                block_h=height_per_core // 32,
                block_w=key_len // 32,
            )
            attention_scores = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
                attention_scores,
                1 / math.sqrt(head_size),
                attention_mask,
                program_config=softmax_program_config,
                is_causal_mask=False,
            )
            attention_scores = ttl.tensor.reshard(attention_scores, orig_mem_config)

        if attention_scores.shape[-2] > original_seq_len:
            attention_scores = attention_scores[:, :, :original_seq_len, :]

        v_sharded = ttl.tensor.interleaved_to_sharded(
            value,
            grid_size,
            [num_heads * key_len // num_cores, inner],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=key_len // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=num_heads * seq_len // num_cores // 32,
            per_core_N=inner // 32,
        )
        attention_scores = ttl.operations.primary.matmul(
            attention_scores,
            v_sharded,
            program_config=program_config,
            output_mem_config=self.l1_interleaved_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=compute_kernel_config,
        )
        v_sharded.deallocate()
        return ttnn.reshape(attention_scores, (2, 8, attention_scores.shape[-2], attention_scores.shape[-1]))

    def get_attention_scores_opt(self, query, t_key, value, original_seq_len, head_size, attention_mask=None):
        if query.shape[-2] == 4096 and t_key.shape[-1] == 4096:
            return self.time_sharded_attention(query, t_key, value, head_size)
        elif not (query.shape[-2] == 1024 and t_key.shape[-1] == 1024) and not (
            query.shape[-2] == 96 and t_key.shape[-1] == 96
        ):
            return self.sharded_attention(query, t_key, value, original_seq_len, head_size, attention_mask)

        print("Legacy path")
        if query.shape[-2] == 96:
            attention_mask = self.attention_masks[96160]
        attention_scores = ttnn.matmul(
            query,
            t_key,
        )
        ttnn.deallocate(query)
        ttnn.deallocate(t_key)
        attention_scores = ttnn.transformer.attention_softmax(
            attention_scores, attention_mask=attention_mask, head_size=head_size
        )

        if attention_scores.shape[-2] > original_seq_len:
            attention_scores = attention_scores[:, :, :original_seq_len, :]
        attention_scores = ttnn.matmul(
            attention_scores,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        return attention_scores

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
    ):
        assert dim_head in self.scales
        original_seq_len = hidden_states.shape[-2] // 2  # 2 is the batch size

        if encoder_hidden_states and len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = squeeze(encoder_hidden_states, 0)

        if self.fused_qkv:
            # TODO: Move into init
            grid_size = self.grid_sizes[hidden_states.shape[-2]]
            M, K = hidden_states.shape[-2], hidden_states.shape[-1]
            N = self.parameters.qkv.weight.shape[-1]
            Nt = N // 32
            per_core_N = (Nt - 1) // (grid_size[1] - 1) if Nt != 16 else 4
            program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=K // grid_size[1] // 32,
                out_subblock_h=self.out_subblock_hs[hidden_states.shape[-2]],
                out_subblock_w=1,
                per_core_M=M // grid_size[0] // 32,
                per_core_N=per_core_N,
                fused_activation=None,
                transpose_mcast=True,
            )
            qkv_out = ttl.operations.primary.matmul(
                hidden_states,
                self.parameters.qkv.weight,
                program_config=program_config,
                output_mem_config=self.l1_interleaved_memory_config,
                output_dtype=ttl.tensor.DataType.BFLOAT8_B,
                compute_kernel_config=self.compute_kernel_config,
            )
            qkv_out = ttnn.reshape(qkv_out, (2, qkv_out.shape[-2] // 2, qkv_out.shape[-1]))
            ttnn.deallocate(hidden_states)

            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv_out,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if hidden_states.shape[-2] == 8192 else ttnn.L1_MEMORY_CONFIG,
                num_heads=heads,
            )
            ttnn.deallocate(qkv_out)
            attention_mask = self.attention_masks[key.shape[-1]]
        else:
            if hidden_states.shape[-2] == 8092:
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

            grid_size = self.grid_sizes[hidden_states.shape[-2]]
            M, K = hidden_states.shape[-2], hidden_states.shape[-1]
            N = self.parameters.to_q.weight.shape[-1]
            Nt = N // 32
            per_core_N = (Nt - 1) // (grid_size[1] - 1) if Nt != 16 else 4
            program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=K // grid_size[1] // 32,
                out_subblock_h=self.out_subblock_hs[hidden_states.shape[-2]],
                out_subblock_w=1,
                per_core_M=M // grid_size[0] // 32,
                per_core_N=per_core_N,
                fused_activation=None,
                transpose_mcast=True,
            )
            q_proj = ttl.operations.primary.matmul(
                hidden_states,
                self.parameters.to_q.weight,
                program_config=program_config,
                output_mem_config=self.l1_interleaved_memory_config,
                output_dtype=ttl.tensor.DataType.BFLOAT8_B,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(hidden_states)
            q_proj = ttnn.reshape(q_proj, (2, q_proj.shape[-2] // 2, q_proj.shape[-1]))

            hidden_seq_len = q_proj.shape.with_tile_padding()[-2]
            encoder_hidden_seq_len = encoder_hidden_states.shape.with_tile_padding()[-2]

            if encoder_hidden_seq_len > hidden_seq_len:
                padding_needed = encoder_hidden_seq_len - hidden_seq_len
                q_proj = ttnn.concat([q_proj, self.padded_tensors[padding_needed]], dim=1)
                hidden_states_padded = True

            kv_proj = ttnn.linear(
                encoder_hidden_states,
                self.parameters.kv.weight,
                core_grid=encoder_hidden_states.device().core_grid,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            # Can't deallocate, mnaybe we just move to dram if it causes issues.
            # ttnn.deallocate(encoder_hidden_states)
            if hidden_seq_len > encoder_hidden_seq_len:
                padding_needed = hidden_seq_len - encoder_hidden_seq_len
                kv_proj = ttnn.concat([kv_proj, self.padded_tensors[padding_needed]], dim=1)

            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                q_proj,
                kv_proj,
                memory_config=ttnn.L1_MEMORY_CONFIG,  # ttnn.DRAM_MEMORY_CONFIG if hidden_states.shape[-2] == 4096 else ttnn.L1_MEMORY_CONFIG,
                num_heads=heads,
            )
            ttnn.deallocate(kv_proj)
            ttnn.deallocate(q_proj)

            if key.shape[-1] > 96:
                key = key[:, :, :, :96]
            if value.shape[-2] > 96:
                value = value[:, :, :96, :]
            assert key.shape[-1] in self.attention_masks
            attention_mask = self.attention_masks[key.shape[-1]]

        hidden_states = self.get_attention_scores_opt(
            query,
            key,
            value,
            original_seq_len,
            dim_head,
            attention_mask,
        )

        hidden_states = ttnn.transformer.concatenate_heads(
            hidden_states, memory_config=self.l1_interleaved_memory_config
        )
        if hidden_states.shape.with_tile_padding()[-1] != hidden_states.shape[-1]:
            hidden_states = hidden_states[:, :, : hidden_states.shape[-1]]

        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters.to_out[0].weight,
            bias=self.parameters.to_out[0].bias,
            core_grid=hidden_states.device().core_grid,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        if len(hidden_states.shape) == 3:
            hidden_states = unsqueeze_to_4D(hidden_states)

        hidden_states = ttnn.reshape(hidden_states, (1, 1, 2 * hidden_states.shape[-2], hidden_states.shape[-1]))
        return hidden_states
