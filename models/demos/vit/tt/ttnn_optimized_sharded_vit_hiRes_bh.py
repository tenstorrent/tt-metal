# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import inspect

import torch
import transformers
from ttnn.dot_access import DotAccessDict
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.experimental.tt_dit.utils.matmul import get_matmul_config


def update_model_config(config, batch_size, sequence_size):
    should_reallocate_in_attention = True

    if batch_size == 1:
        grid_y = 8
        grid_x = 8
    else:
        grid_y = 10
        grid_x = 12
        should_reallocate_in_attention = True

    core_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
    core_grid_8x8 = ttnn.CoreGrid(y=8, x=8)
    core_grid_12x10 = ttnn.CoreGrid(y=10, x=12)
    core_grid_13x10 = ttnn.CoreGrid(y=10, x=13)

    # INPUTS
    TILE_HEIGHT = 32
    seqL = sequence_size  # 1024, 2048, 3072
    dim_t = config.hidden_size // TILE_HEIGHT  # 24, 36, 48, 72
    head_num = config.num_attention_heads  # 16

    # BLOCK SHARDED
    # x = 8 for now to avoid padding the hidden_size;
    # y = 8 for now to avoid padding the SeqL;
    # core_grid_BLOCK_SHARDED = ttnn.CoreGrid(y=8, x=8)
    if config.hidden_size <= 1024:
        core_grid_BLOCK_SHARDED = ttnn.CoreGrid(y=8, x=8)
    else:
        core_grid_BLOCK_SHARDED = ttnn.CoreGrid(y=8, x=12)

    # SPLIT HEADS SHARDED
    # ttnn.transformer.split_query_key_value_and_split_heads
    # assert -> Batch size 1 must be equal to num cores 8
    # x = 8 so 2 heads per core
    if config.hidden_size <= 1024:
        core_grid_SPLIT_HEADS_SHARDED = ttnn.CoreGrid(y=batch_size, x=8)
    else:
        core_grid_SPLIT_HEADS_SHARDED = ttnn.CoreGrid(y=batch_size, x=8)

    # HEIGHT SHARDED
    H_x = 8
    H_y = min(batch_size * head_num // H_x, 8)
    core_grid_HEIGHT_SHARDED = ttnn.CoreGrid(y=H_y, x=H_x)
    # core_grid_HEIGHT_SHARDED = ttnn.CoreRangeSet(
    #     {
    #         ttnn.CoreRange(ttnn.CoreCoord(0, 0), (12, 8)),
    #         ttnn.CoreRange(ttnn.CoreCoord(0, 9), (10,9)),
    #     }
    # )

    seqL_t = seqL // TILE_HEIGHT  # 32, 64, 96
    seqL_t__y = (batch_size * seqL_t) // core_grid_BLOCK_SHARDED.y  # 4, 8, 12
    dim_t__x = dim_t // core_grid_BLOCK_SHARDED.x  # 2, 3, 4, 6
    dim_t__x_full_grid = dim_t // core_grid_12x10.x  # 3

    head_seqL_t__x = (batch_size * head_num * seqL_t) // (
        core_grid_HEIGHT_SHARDED.x * core_grid_HEIGHT_SHARDED.y
    )  # 64, 128, 192
    head_size_t = dim_t // head_num  # 1, 2, 3, 4
    # 1000 classes padded to 1152
    class__x = (1152 // TILE_HEIGHT) // core_grid.x  #   3
    class_subb_w = class__x
    if class_subb_w > 8:  # max ratio of sub_block_w / sub_block_h = 8
        if class_subb_w % 3 == 0:
            class_subb_w = class__x // 3
        elif class_subb_w % 2 == 0:
            class_subb_w = class__x // 2
        else:
            class_subb_w = 1

    # sharding configs
    program_configs = {
        "layernorm_before_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_BLOCK_SHARDED.x, core_grid_BLOCK_SHARDED.y),
            # shard_shape_is = [seqL_t, dim_t__x_full_grid], in tiles
            subblock_w=dim_t__x,  #
            block_h=seqL_t__y,  #
            block_w=dim_t__x,  #
            inplace=False,
        ),
        # shard_spec = [224, 96]
        "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_BLOCK_SHARDED.x, core_grid_BLOCK_SHARDED.y),
            # shard_shape_is = [seqL_t__y, dim_t__x], in tiles
            in0_block_w=dim_t__x,  #
            out_subblock_h=1,
            out_subblock_w=dim_t__x,  #
            per_core_M=seqL_t__y,  #
            per_core_N=3 * dim_t__x,  #
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid_HEIGHT_SHARDED.x, core_grid_HEIGHT_SHARDED.y),
            in0_block_w=head_size_t,  # 2,
            out_subblock_h=1,
            out_subblock_w=8,  # 7,
            per_core_M=head_seqL_t__x,  ##int((head_num//) * seqL_t),  # 14,
            per_core_N=seqL_t,  # 7,
        ),
        "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_HEIGHT_SHARDED.x, core_grid_HEIGHT_SHARDED.y),
            subblock_w=seqL_t__y,  # 7,
            block_h=head_seqL_t__x,  # 14,
            block_w=seqL_t__y,  # 7,
        ),
        "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid_HEIGHT_SHARDED.x, core_grid_HEIGHT_SHARDED.y),
            in0_block_w=head_seqL_t__x,  # 7,
            out_subblock_h=1,
            out_subblock_w=head_size_t,  # 2,
            per_core_M=head_seqL_t__x,  # 14,
            per_core_N=head_size_t,  # 2,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_BLOCK_SHARDED.x, core_grid_BLOCK_SHARDED.y),
            # shard_shape_is = [seqL_t, dim_t__x_full_grid], in tiles
            in0_block_w=dim_t__x,  # 3
            out_subblock_h=1,
            out_subblock_w=dim_t__x,  # 3
            per_core_M=seqL_t__y,  # 7,
            per_core_N=dim_t__x,  # 3,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_BLOCK_SHARDED.x, core_grid_BLOCK_SHARDED.y),
            subblock_w=dim_t__x,  #
            block_h=seqL_t__y,  #
            block_w=dim_t__x,  #
            inplace=False,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_BLOCK_SHARDED.x, core_grid_BLOCK_SHARDED.y),
            # shard_shape_is = [seqL_t, dim_t__x_full_grid], in tiles
            in0_block_w=dim_t__x // 2,  # 96 == 3 tiles,
            out_subblock_h=1,
            out_subblock_w=min(4, (dim_t__x * 4) // 2),  # 6,
            per_core_M=seqL_t__y,  # 7,
            per_core_N=dim_t__x * 4,  # 12,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_BLOCK_SHARDED.x, core_grid_BLOCK_SHARDED.y),
            # shard shape is [seqL_t, dim_t__x_full_grid * 4], in tiles
            in0_block_w=dim_t__x * 2,  # 12
            out_subblock_h=1,
            out_subblock_w=dim_t__x,
            per_core_M=seqL_t__y,  # 7,
            per_core_N=dim_t__x,  # 3
            transpose_mcast=False,
            fused_activation=None,
        ),
        "classifer_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t__x // 2,  # 1,
            out_subblock_h=1,
            out_subblock_w=class_subb_w,
            per_core_M=seqL_t,  # 7,
            per_core_N=class__x,  # 6,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "ln_compute_config": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    }

    # properties are not in the output of config.to_dict() but can be used later in the model
    # e.g. https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/configuration_utils.py#L368-L378
    property_names = [name for name, value in inspect.getmembers(config.__class__) if isinstance(value, property)]
    properties = {name: getattr(config, name) for name in property_names}

    return DotAccessDict(
        dict(
            **(config.to_dict() | properties),
            core_grid=core_grid,
            core_grid_8x8=core_grid_8x8,
            core_grid_12x10=core_grid_12x10,
            core_grid_13x10=core_grid_13x10,
            core_grid_HEIGHT_SHARDED=core_grid_HEIGHT_SHARDED,
            core_grid_BLOCK_SHARDED=core_grid_BLOCK_SHARDED,
            core_grid_SPLIT_HEADS_SHARDED=core_grid_SPLIT_HEADS_SHARDED,
            should_reallocate_in_attention=should_reallocate_in_attention,
            program_configs=program_configs,
            seqL=seqL,
        )
    )


def run_minimal_matmul(
    device,
    tt_input,
    tt_weight,
    tt_bias,
    activation=None,
    memory_config=None,
):
    B, M, K, N = (
        tt_input.padded_shape[0],
        tt_input.padded_shape[-2],
        tt_input.padded_shape[-1],
        tt_weight.padded_shape[-1],
    )
    core_grid = device.compute_with_storage_grid_size()
    matmul_config = get_matmul_config(M * B, K, N, core_grid)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    tt_output = ttnn.experimental.minimal_matmul(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        config=matmul_config,
        fused_activation=activation,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    return tt_output


# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = config.patch_size
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    folded_pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)  # 1568, 1024
    ttnn.deallocate(pixel_values)
    folded_pixel_values = ttnn.to_memory_config(folded_pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG)
    # Convert back to interleaved or otherwise to_layout will fail
    folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn_linear(
        folded_pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=config.core_grid,
    )

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, patch_size_sq_trpl))

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    *,
    parameters,
):
    parameters = parameters.vit.embeddings

    l1_memory_config = ttnn.L1_MEMORY_CONFIG

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    embedding_output = ttnn.concat([cls_token, patch_embeddings], -2, memory_config=l1_memory_config)
    embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)
    embedding_output = ttnn.add(
        embedding_output, position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    return embedding_output


def ttnn_linear(*args, **kwargs):
    if True:
        batch, M, K, N = (
            args[0].padded_shape[0],
            args[0].padded_shape[-2],
            args[0].padded_shape[-1],
            args[1].padded_shape[-1],
        )
        core_grid = ttnn.CoreGrid(y=8, x=8)
        sharded_memory_config = ttnn.create_sharded_memory_config(
            [M * batch, N],
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        output = run_minimal_matmul(
            device=args[0].device(),
            tt_input=args[0],
            tt_weight=args[1],
            tt_bias=kwargs.get("bias", args[2] if len(args) > 2 else None),
            activation=None,
            memory_config=sharded_memory_config,
        )
    else:
        output = ttnn.linear(*args, **kwargs)
    return output


def vit_attention(
    config,
    hidden_states,
    parameters,
):
    num_heads = config.num_attention_heads  # num_heads = 16
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn_linear(
        hidden_states,
        parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_key_value_matmul_program_config"],
    )
    query_key_value = ttnn.to_memory_config(query_key_value, ttnn.L1_MEMORY_CONFIG)
    """
    #reshard to (batch_size*num_heads) cores
    block_sharded_config_SPLIT_HEADS_SHARDED = ttnn.create_sharded_memory_config(
        query_key_value.padded_shape,
        core_grid=config.core_grid_SPLIT_HEADS_SHARDED,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    query_key_value = ttnn.reshard(query_key_value, block_sharded_config_SPLIT_HEADS_SHARDED)

    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        num_heads=num_heads,
        transpose_key=False,
    )
    """
    query, key, value = ttnn.experimental.nlp_create_qkv_heads(
        query_key_value,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        transpose_k_heads=False,
    )

    ttnn.deallocate(query_key_value)
    ttnn.deallocate(hidden_states)
    if config.should_reallocate_in_attention:
        value = ttnn.reallocate(value)

    # SDPA code
    query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
    key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
    value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

    if config.seqL == 1024:
        q_chunk_size = 128
        k_chunk_size = 256
        if head_size < 128:
            k_chunk_size = 512
    elif config.seqL == 2048:
        q_chunk_size = 256
        k_chunk_size = 512
        if head_size < 128:
            k_chunk_size = 1024
    elif config.seqL == 3072:
        q_chunk_size = 256
        k_chunk_size = 512
        if head_size < 128:
            k_chunk_size = 1024
        elif head_size < 96:
            q_chunk_size = 512
    else:
        q_chunk_size = 128
        k_chunk_size = 128

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(config.core_grid_13x10.x, config.core_grid_13x10.y),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,  # NOTE: False is more correct
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    context_layer = ttnn.transformer.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=head_size**-0.5,
        is_causal=False,
        program_config=program_config,
        # compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # ttnn.deallocate(query)
    # ttnn.deallocate(key)
    # ttnn.deallocate(value)

    """

    print("query.memory_config()", query.memory_config())
    B, H, S, Z = query.padded_shape
    if B < 8:
        #reshard to (batch_size*num_heads) cores

        query_reshaped = ttnn.reshape(query, (B*H, S, Z))
        key_reshaped = ttnn.reshape(key, (B*H, Z, S))
        value_reshaped = ttnn.reshape(value, (B*H, S, Z))

        print("query_reshaped.padded_shape", query_reshaped.padded_shape)
        qv_sharded_config_HEIGHT_SHARDED = ttnn.create_sharded_memory_config(
            query_reshaped.padded_shape,
            core_grid=config.core_grid_HEIGHT_SHARDED,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape = False,
        )
        k_sharded_config_HEIGHT_SHARDED = ttnn.create_sharded_memory_config(
            key_reshaped.padded_shape,
            core_grid=config.core_grid_HEIGHT_SHARDED,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape = True,
        )

        query = ttnn.reshard(query_reshaped, qv_sharded_config_HEIGHT_SHARDED)
        key = ttnn.reshard(key_reshaped, k_sharded_config_HEIGHT_SHARDED)
        value = ttnn.reshard(value_reshaped, qv_sharded_config_HEIGHT_SHARDED)
        ttnn.deallocate(query_reshaped)
        ttnn.deallocate(key_reshaped)
        ttnn.deallocate(value_reshaped)

        print("query.memory_config()", query.memory_config())


    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_by_key_matmul_program_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    scale = 1.0 / (head_size**0.5)
    attention_scores = ttnn.mul_(
        attention_scores,
        scale,
    )

    attention_probs = ttnn.softmax_in_place(
        attention_scores,
        program_config=config.program_configs["softmax_program_config"],
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.reshape(context_layer, (B, H, S, Z))
    print("context_layer.padded_shape", context_layer.padded_shape)
    print("context_layer.memory_config()", context_layer.memory_config())

    #reshard to (batch_size*num_heads) cores
    height_sharded_config_CONCAT_HEADS_SHARDED = ttnn.create_sharded_memory_config(
        context_layer.padded_shape,
        core_grid=config.core_grid_SPLIT_HEADS_SHARDED,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    context_layer = ttnn.reshard(context_layer, height_sharded_config_CONCAT_HEADS_SHARDED)

    """

    # print("context_layer.shape()", context_layer.shape)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        # memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    )
    # print("context_layer.padded_shape", context_layer.padded_shape, context_layer.shape)

    block_sharded_config_64_cores = ttnn.create_sharded_memory_config(
        context_layer.padded_shape,
        core_grid=config.core_grid_BLOCK_SHARDED,  # 64 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # reshard back to 64 cores
    # cant use reshard as it's not working here, so use s2i followed by i2s
    # workaround for issue #22640, once fixed first call can be removed
    context_layer = ttnn.to_memory_config(context_layer, ttnn.DRAM_MEMORY_CONFIG)
    context_layer = ttnn.to_memory_config(context_layer, block_sharded_config_64_cores)
    self_output = ttnn_linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)
    if config.should_reallocate_in_attention:
        self_output = ttnn.reallocate(self_output)

    return self_output


def vit_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    output = ttnn_linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    output = ttnn_linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)
    if residual.memory_config().shard_spec != output.memory_config().shard_spec:
        residual = ttnn.to_memory_config(residual, output.memory_config())
    output = ttnn.add(output, residual, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(residual)

    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = vit_intermediate(config, hidden_states, parameters=parameters.intermediate)
    hidden_states = vit_output(config, intermediate, attention_output, parameters=parameters.output)
    return hidden_states


def vit_layer(
    config,
    hidden_states,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_before_program_config"],
        compute_kernel_config=config.program_configs["ln_compute_config"],
    )

    multi_head_attention_output = vit_attention(
        config,
        layernorm_before_output,
        parameters=parameters.attention,
    )

    multi_head_attention_output = ttnn.unsqueeze(multi_head_attention_output, 1)
    if multi_head_attention_output.memory_config().shard_spec != hidden_states.memory_config().shard_spec:
        multi_head_attention_output = ttnn.to_memory_config(multi_head_attention_output, hidden_states.memory_config())
    multi_head_attention_output = ttnn.add(
        multi_head_attention_output,
        hidden_states,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(hidden_states)
    multi_head_attention_output = ttnn.reallocate(multi_head_attention_output)

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_after_output_program_config"],
        compute_kernel_config=config.program_configs["ln_compute_config"],
    )

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    embeddings,
    parameters,
):
    TILE_HEIGHT = 32
    emb_N, emb_S, emb_D = embeddings.shape
    emb_S = (((emb_S - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT
    encoder_input = ttnn.to_memory_config(
        embeddings,
        memory_config=ttnn.create_sharded_memory_config(
            [emb_N, emb_S, emb_D],
            core_grid=config.core_grid_12x10,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(embeddings)

    for index, encoder_parameters in enumerate(parameters.layer):
        encoder_output = vit_layer(
            config,
            encoder_input,
            encoder_parameters,
        )
        encoder_input = encoder_output

    return encoder_output


def vit(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, cls_token, position_embeddings, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_before_program_config"],
        compute_kernel_config=config.program_configs["ln_compute_config"],
    )

    # reshard back to 48 cores as we are losing a bit of precision if this is 64 cores
    block_sharded_config_48_cores = ttnn.create_sharded_memory_config(
        output.padded_shape,
        core_grid=config.core_grid,  # 48
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    output = ttnn.reshard(output, block_sharded_config_48_cores)

    # Classifier
    classifier_output = ttnn_linear(
        output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["classifer_matmul_program_config"],
    )
    return classifier_output


def preprocess_inputs(
    input_ids,
    token_type_ids,
    device,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    return input_ids, token_type_ids


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.vit.modeling_vit.ViTEmbeddings):
        weight = torch_model.patch_embeddings.projection.weight
        bias = torch_model.patch_embeddings.projection.bias

        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (int(three_times_hidden_size * (4 / c)), three_times_hidden_size)
        )

        parameters = {"patch_embeddings": {}}
        parameters["patch_embeddings"] = {"projection": {}}
        parameters["patch_embeddings"]["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        parameters["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

        parameters["cls_token"] = ttnn.from_torch(torch_model.cls_token, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters["position_embeddings"] = ttnn.from_torch(
            torch_model.position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        num_heads = 16
        hidden_size = torch_model.query.weight.shape[0]
        head_size = hidden_size // num_heads
        hidden_size = num_heads * head_size * 3

        qkv_weight = torch.cat(
            [
                torch_model.query.weight.reshape([num_heads, head_size, -1]),
                torch_model.key.weight.reshape([num_heads, head_size, -1]),
                torch_model.value.weight.reshape([num_heads, head_size, -1]),
            ],
            dim=1,
        ).reshape([hidden_size, -1])
        qkv_bias = torch.cat(
            [
                torch_model.query.bias.reshape([num_heads, head_size]),
                torch_model.key.bias.reshape([num_heads, head_size]),
                torch_model.value.bias.reshape([num_heads, head_size]),
            ],
            dim=1,
        ).reshape([hidden_size])

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)

    elif isinstance(torch_model, torch.nn.Linear):
        # TODO: better way of detection for the classify linear weights
        if torch_model.weight.shape[0] == 1000:
            preprocessed_weight = torch.nn.functional.pad(torch_model.weight, (0, 0, 0, int(1152 - 1000)))
            preprocessed_bias = torch.nn.functional.pad(torch_model.bias, (0, int(1152 - 1000)))
            parameters["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat8_b)
        else:
            parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)

    return parameters
