# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.mlp import matmul_1d_config, matmul_2d_config


def convert_to_layout(tensor, input_memory_layout, output_memory_layout, clone=False):
    if input_memory_layout == output_memory_layout:
        return tensor
    else:  # Convert layout
        if isinstance(
            tensor, list
        ):  # if input is a list of tensors call convert_to_layout for each tensor individually
            return [convert_to_layout(t, input_memory_layout, output_memory_layout, clone=clone) for t in tensor]
        else:
            if input_memory_layout.is_sharded() and not output_memory_layout.is_sharded():  # sharded_to_interleaved
                tensor = ttnn.experimental.tensor.sharded_to_interleaved(tensor, output_mem_config=output_memory_layout)
            elif not input_memory_layout.is_sharded() and output_memory_layout.is_sharded():  # interleaved_to_sharded
                tensor = ttnn.experimental.tensor.interleaved_to_sharded(
                    tensor, sharded_mem_config=output_memory_layout
                )
            elif (
                not input_memory_layout.is_sharded() and not output_memory_layout.is_sharded()
            ):  # interleaved to interleaved with different memory location
                if clone:
                    tensor = ttnn.experimental.tensor.clone(tensor, output_mem_config=output_memory_layout)
                else:
                    tensor = ttnn.experimental.tensor.move(tensor, output_mem_config=output_memory_layout)
            else:  # reshard
                tensor = ttnn.experimental.tensor.sharded_to_interleaved(
                    tensor,
                    output_mem_config=ttnn.experimental.tensor.MemoryConfig(
                        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
                    ),
                )
                tensor = ttnn.experimental.tensor.interleaved_to_sharded(
                    tensor, sharded_mem_config=output_memory_layout
                )
            return tensor


def memcfg_1d_width_sharded_from_tensor_shape(shape, grid=ttnn.CoreGrid(x=8, y=8)):
    start_core_coord = ttnn.experimental.tensor.CoreCoord(0, 0)
    end_core_coord = ttnn.experimental.tensor.CoreCoord(grid.x - 1, grid.y - 1)
    assert shape[3] % (grid.x * grid.y) == 0, f"Tensor width must be divisible by the number of cores"
    shard_width = int(shape[3] / (grid.x * grid.y))
    shard_height = int(shape[0] * shape[1] * shape[2])
    return ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.experimental.tensor.BufferType.L1,
        ttnn.experimental.tensor.ShardSpec(
            ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        start_core_coord,
                        end_core_coord,
                    ),
                }
            ),
            [
                shard_height,
                shard_width,
            ],
            ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )


def get_dram_memcfg():
    return ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
    )


def falcon_prefill_matmul(
    in0,
    in1,
    compute_kernel_config,
    output_mem_config=get_dram_memcfg(),
    output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
    grid=ttnn.CoreGrid(x=8, y=8),
    act=None,
    transpose_mcast=False,
    overwrite_per_core_k=None,
    overwrite_subblock_w=None,
    overwrite_subblock_h=None,
    fuse_batch_mm2d=True,
):
    in0_shape = in0.shape
    in1_shape = in1.shape
    m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
    if not fuse_batch_mm2d:
        m = in0_shape[2]

    use_2d_mm = m >= 512  # select 2d matmul for S >= 512, otherwise fall back to matmul 1d

    is_fp32_accumulate = compute_kernel_config.fp32_dest_acc_en

    if use_2d_mm:
        # print("Selecting MM 2d")
        matmul_pgmcfg = matmul_2d_config(
            m,
            k,
            n,
            grid,
            act,
            is_fp32_accumulate,
            transpose_mcast,
            override_per_core_k=overwrite_per_core_k,
            override_subblock_w=overwrite_subblock_w,
            override_subblock_h=overwrite_subblock_h,
            fuse_batch=fuse_batch_mm2d,
        )
        # print(f"Program config: {matmul_pgmcfg}")
        return ttnn.matmul(
            in0,
            in1,
            program_config=matmul_pgmcfg,
            memory_config=output_mem_config,
            dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        # print("Selecting MM 1d")
        matmul_pgmcfg = matmul_1d_config(
            m,
            k,
            n,
            grid,
            act,
            is_fp32_accumulate,
            override_per_core_k=overwrite_per_core_k,
            override_subblock_w=overwrite_subblock_w,
            override_subblock_h=overwrite_subblock_h,
        )
        # print(f"Program config: {matmul_pgmcfg}")
        return ttnn.matmul(
            in0,
            in1,
            program_config=matmul_pgmcfg,
            memory_config=output_mem_config,
            dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )


def partial_layernorm(
    xs,
    ln_gamma,
    ln_beta,
    ln_eps,
    layernorm_params,
    memconfig,
    pgmconfig,
    dtype,
    ln_output_tensors_dict,
):
    # Do partial layernorm by partial sequence length of 128
    # Input xs[0] is [1, 1, seq_len, 8192]
    seq_len = xs.shape[2]

    slice_size = layernorm_params["slice_size"]

    layernorm_num_cores_x, layernorm_num_cores_y = (
        layernorm_params["layernorm_num_cores_x"],
        layernorm_params["layernorm_num_cores_y"],
    )
    layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim = (
        layernorm_params["layernorm_shard_height_hidden_dim"],
        layernorm_params["layernorm_shard_width_hidden_dim"],
    )

    if seq_len > slice_size:
        assert seq_len % slice_size == 0, "Sequence length must be divisible by layernorm slice size {slice_size}"
        num_slices = seq_len // slice_size  # we do 128 per iteration (slice), then we concat the result.

        xs_output_cat = ln_output_tensors_dict[seq_len]

        for slice_i in range(num_slices):
            xs_slice = ttnn.experimental.tensor.interleaved_to_sharded_partial(
                xs,
                (layernorm_num_cores_x, layernorm_num_cores_y),
                [layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim],
                num_slices,  # num_slices
                slice_i,  # slice_index
                ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            )
            xs_slice = ttnn.layer_norm(
                xs_slice,
                epsilon=ln_eps,
                weight=ln_gamma,
                bias=ln_beta,
                memory_config=memconfig,
                program_config=pgmconfig,
            )
            ttnn.experimental.tensor.sharded_to_interleaved_partial(
                xs_slice,
                xs_output_cat,
                num_slices,
                slice_i,
                get_dram_memcfg(),
            )
        xs_slice.deallocate(True)
    else:
        xs = convert_to_layout(xs, get_dram_memcfg(), memconfig)
        xs = ttnn.layer_norm(
            xs,
            epsilon=ln_eps,
            weight=ln_gamma,
            bias=ln_beta,
            memory_config=memconfig,
            program_config=pgmconfig,
        )
        xs = convert_to_layout(xs, memconfig, get_dram_memcfg())
        xs_output_cat = ttnn.experimental.tensor.typecast(xs, dtype)

    return xs_output_cat


def determine_tensor_deallocation(layernorm_slice_size, seq_len):
    """
    Tensors will be reused for seq_lens > 512 and deallocated for seq_lens <= 512
    All tensors that satisfy above condition used in layernorm processing will be later deallocated in:
    - falcon_mlp
    - falcon_attention
    - falcon_causallm
    Args:
        layernorm_slice_size (int): The slice size used for layer normalization.
        seq_len (int): The sequence length.

    Returns:
        bool: True if tensors should be deallocated, False otherwise.
    """
    return seq_len <= layernorm_slice_size


def generate_layernorm_persistent_tensors(seq_len, slice_size, ln_output_tensors_dict, device_mesh, hidden_size, dtype):
    if seq_len <= slice_size:
        return

    tensor = torch.zeros(1, 1, seq_len, hidden_size)
    for name in ["final_layernorm", "mlp_layernorm", "attn_layernorm"]:
        output_tensor = ttnn.as_tensor(
            tensor=tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device_mesh),
        )
        if name in ln_output_tensors_dict and ln_output_tensors_dict[name] is not None:
            ln_output_tensors_dict[name].update({seq_len: output_tensor})
        else:
            ln_output_tensors_dict[name] = {seq_len: output_tensor}


def fused_partial_layernorm(
    xs,
    ln_gamma_1,
    ln_beta_1,
    ln_gamma_2,
    ln_beta_2,
    ln_eps,
    layernorm_params,
    memconfig,
    pgmconfig,
    out_tensor_dict_1,
    out_tensor_dict_2,
):
    # Do partial layernorm by partial sequence length of 128
    # Input xs is [1, 1, seq_len, 8192]
    seq_len = xs.shape[2]

    slice_size = layernorm_params["slice_size"]

    layernorm_num_cores_x, layernorm_num_cores_y = (
        layernorm_params["layernorm_num_cores_x"],
        layernorm_params["layernorm_num_cores_y"],
    )
    layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim = (
        layernorm_params["layernorm_shard_height_hidden_dim"],
        layernorm_params["layernorm_shard_width_hidden_dim"],
    )

    dram_memcfg = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
    )
    interleaved_l1_memcfg = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
    )

    if seq_len > slice_size:
        assert seq_len % slice_size == 0, "Sequence length must be divisible by layernorm slice size {slice_size}"
        num_slices = seq_len // slice_size  # we do 128 per iteration (slice), then we concat the result.

        out_tensor_1 = out_tensor_dict_1[seq_len]
        out_tensor_2 = out_tensor_dict_2[seq_len]

        for slice_i in range(num_slices):
            xs_slice = ttnn.experimental.tensor.interleaved_to_sharded_partial(
                xs,
                (layernorm_num_cores_x, layernorm_num_cores_y),
                [layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim],
                num_slices,  # num_slices
                slice_i,  # slice_index
                ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            )

            xs_slice = ttnn.layer_norm(
                xs_slice,
                epsilon=ln_eps,
                weight=None,
                bias=None,
                memory_config=memconfig,
                program_config=pgmconfig,
            )

            # Apply first layernorm gamma+beta
            xs_output_slice_1 = ttnn.experimental.tensor.bcast(
                xs_slice,
                ln_gamma_1,
                math_op=ttnn.experimental.tensor.BcastOpMath.MUL,
                dim=ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=memconfig,
            )
            xs_output_slice_1 = ttnn.experimental.tensor.bcast(
                xs_output_slice_1,
                ln_beta_1,
                math_op=ttnn.experimental.tensor.BcastOpMath.ADD,
                dim=ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=memconfig,
            )

            ttnn.experimental.tensor.sharded_to_interleaved_partial(
                xs_output_slice_1,
                out_tensor_1,
                num_slices,
                slice_i,
                dram_memcfg,
            )
            xs_output_slice_1.deallocate(True)

            # Apply second layernorm gamma+beta inplace
            xs_slice = ttnn.experimental.tensor.bcast(
                xs_slice,
                ln_gamma_2,
                math_op=ttnn.experimental.tensor.BcastOpMath.MUL,
                dim=ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=memconfig,
            )
            xs_slice = ttnn.experimental.tensor.bcast(
                xs_slice,
                ln_beta_2,
                math_op=ttnn.experimental.tensor.BcastOpMath.ADD,
                dim=ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=memconfig,
            )

            ttnn.experimental.tensor.sharded_to_interleaved_partial(
                xs_slice,
                out_tensor_2,
                num_slices,
                slice_i,
                dram_memcfg,
            )
            xs_slice.deallocate(True)

            output1 = out_tensor_1
            output2 = out_tensor_2
    else:
        xs_output2 = ttnn.experimental.tensor.interleaved_to_sharded(xs, sharded_mem_config=memconfig)
        xs_output2 = ttnn.layer_norm(
            xs_output2,
            epsilon=ln_eps,
            weight=None,
            bias=None,
            memory_config=memconfig,
            program_config=pgmconfig,
        )

        # Apply first layernorm gamma+beta
        xs_output1 = ttnn.experimental.tensor.bcast(
            xs_output2,
            ln_gamma_1,
            math_op=ttnn.experimental.tensor.BcastOpMath.MUL,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=memconfig,
        )
        xs_output1 = ttnn.experimental.tensor.bcast(
            xs_output1,
            ln_beta_1,
            math_op=ttnn.experimental.tensor.BcastOpMath.ADD,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=memconfig,
        )
        xs_output1 = ttnn.experimental.tensor.sharded_to_interleaved(xs_output1, output_mem_config=dram_memcfg)

        # Apply second layernorm gamma+beta
        xs_output2 = ttnn.experimental.tensor.bcast(
            xs_output2,
            ln_gamma_2,
            math_op=ttnn.experimental.tensor.BcastOpMath.MUL,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=memconfig,
        )
        xs_output2 = ttnn.experimental.tensor.bcast(
            xs_output2,
            ln_beta_2,
            math_op=ttnn.experimental.tensor.BcastOpMath.ADD,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=memconfig,
        )
        xs_output2 = ttnn.experimental.tensor.sharded_to_interleaved(xs_output2, output_mem_config=dram_memcfg)

        output1 = xs_output1
        output2 = xs_output2

    return output1, output2
