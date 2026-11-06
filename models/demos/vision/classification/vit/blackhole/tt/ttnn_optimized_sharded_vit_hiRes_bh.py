# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import inspect

import torch
import transformers
from ttnn.dot_access import DotAccessDict
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.demos.vision.classification.vit.blackhole.tt.ttnn_optimized_sharded_vit_bh import largest_divisor_at_most

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(header):
        pass


def block_sharded_grid_width(dim_t, limit):
    """Widest column count that splits the hidden dim so ff1's in0_block_w (half a core's width) divides it."""
    for width in range(min(dim_t, limit), 0, -1):
        width_t = dim_t // width
        if dim_t % width == 0 and width_t >= 2 and width_t % (width_t // 2) == 0:
            return width
    raise ValueError(f"No block-sharded grid width up to {limit} fits a hidden dim of {dim_t} tiles")


def update_model_config(config, batch_size, sequence_size, device):
    # In case of < 6 cores per batch, we need to do move in attention to remove defragmentation
    should_reallocate_in_attention = True
    device_grid = device.compute_with_storage_grid_size()

    # INPUTS
    TILE_HEIGHT = 32
    seqL = sequence_size  # 1024, 2048, 3072
    dim_t = config.hidden_size // TILE_HEIGHT  # 24, 36, 48, 72
    head_num = config.num_attention_heads  # 16
    class_t = 1152 // TILE_HEIGHT  # 1000 classes padded to 1152

    if batch_size == 1:
        core_grid = ttnn.CoreGrid(y=min(8, device_grid.y), x=min(8, device_grid.x))
    else:
        core_grid = ttnn.CoreGrid(y=min(10, device_grid.y), x=largest_divisor_at_most(class_t, device_grid.x))

    # BLOCK SHARDED
    # Up to a 1024 hidden size 8 columns avoid padding the hidden dim; 8 rows avoid padding the sequence.
    block_width_limit = 8 if config.hidden_size <= 1024 else device_grid.x
    core_grid_BLOCK_SHARDED = ttnn.CoreGrid(
        y=largest_divisor_at_most(batch_size * (seqL // TILE_HEIGHT), min(8, device_grid.y)),
        x=block_sharded_grid_width(dim_t, min(block_width_limit, device_grid.x)),
    )

    # SPLIT HEADS SHARDED
    # ttnn.transformer.split_query_key_value_and_split_heads
    # assert -> Batch size 1 must be equal to num cores 8
    # x = 8 so 2 heads per core
    core_grid_SPLIT_HEADS_SHARDED = ttnn.CoreGrid(y=batch_size, x=8)

    # HEIGHT SHARDED
    H_x = 8
    H_y = min(batch_size * head_num // H_x, 8)
    core_grid_HEIGHT_SHARDED = ttnn.CoreGrid(y=H_y, x=H_x)

    seqL_t = seqL // TILE_HEIGHT  # 32, 64, 96
    seqL_t__y = (batch_size * seqL_t) // core_grid_BLOCK_SHARDED.y  # 4, 8, 12
    dim_t__x = dim_t // core_grid_BLOCK_SHARDED.x  # 2, 3, 4, 6

    # Each sharded projection keeps its block of the hidden dim, its output and the residual in a core's L1;
    # the widest is the MLP's 4x fc1 output. Past a quarter of a core's L1 they no longer fit, so the qkv,
    # output and MLP projections run through DRAM instead.
    BFLOAT8_B_TILE_BYTES = 1088
    fc1_output_bytes_per_core = seqL_t__y * dim_t__x * 4 * BFLOAT8_B_TILE_BYTES
    l1_bytes_per_core = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    projections_in_dram = fc1_output_bytes_per_core > l1_bytes_per_core // 4

    head_seqL_t__x = (batch_size * head_num * seqL_t) // (
        core_grid_HEIGHT_SHARDED.x * core_grid_HEIGHT_SHARDED.y
    )  # 64, 128, 192
    head_size_t = dim_t // head_num  # 1, 2, 3, 4
    # 1000 classes padded to 1152
    class__x = class_t // core_grid.x
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
            in0_block_w=dim_t__x,
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
            in0_block_w=dim_t__x,
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
    properties = {}
    for _prop_name in property_names:
        try:
            properties[_prop_name] = getattr(config, _prop_name)
        except AttributeError:
            # transformers 5.x adds properties (e.g. rope_scaling -> rope_parameters) that
            # raise on configs without rope (e.g. ViTConfig); skip them.
            pass

    return DotAccessDict(
        dict(
            **(config.to_dict() | properties),
            core_grid=core_grid,
            device_grid=ttnn.CoreGrid(y=device_grid.y, x=device_grid.x),
            core_grid_HEIGHT_SHARDED=core_grid_HEIGHT_SHARDED,
            core_grid_BLOCK_SHARDED=core_grid_BLOCK_SHARDED,
            core_grid_SPLIT_HEADS_SHARDED=core_grid_SPLIT_HEADS_SHARDED,
            should_reallocate_in_attention=should_reallocate_in_attention,
            projections_in_dram=projections_in_dram,
            program_configs=program_configs,
        )
    )


# https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = config.patch_size
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    folded_pixel_values = ttnn.fold(pixel_values, stride_h, stride_w, collapse_output=True)  # 1568, 1024
    ttnn.deallocate(pixel_values)
    folded_pixel_values = ttnn.to_memory_config(folded_pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG)
    # Convert back to interleaved or otherwise to_layout will fail
    folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.linear(
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


def to_block_sharded(config, tensor):
    block_sharded_config = ttnn.create_sharded_memory_config(
        tensor.padded_shape,
        core_grid=config.core_grid_BLOCK_SHARDED,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.to_memory_config(tensor, block_sharded_config)


def vit_attention(
    config,
    hidden_states,
    parameters,
):
    num_heads = config.num_attention_heads  # num_heads = 16

    if config.projections_in_dram:
        query_key_value = ttnn.linear(
            ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG),
            parameters.query_key_value.weight,
            bias=parameters.query_key_value.bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
    else:
        query_key_value = ttnn.linear(
            hidden_states,
            parameters.query_key_value.weight,
            bias=parameters.query_key_value.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=config.program_configs["query_key_value_matmul_program_config"],
        )
    query_key_value = ttnn.to_memory_config(query_key_value, ttnn.L1_MEMORY_CONFIG)

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

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(config.device_grid.x, config.device_grid.y),
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=False,  # NOTE: False is more correct
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    context_layer = ttnn.transformer.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
    )

    context_layer = ttnn.to_memory_config(context_layer, ttnn.DRAM_MEMORY_CONFIG)
    if config.projections_in_dram:
        self_output = ttnn.linear(
            context_layer,
            parameters.o_proj.weight,
            bias=parameters.o_proj.bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        self_output = to_block_sharded(config, self_output)
    else:
        context_layer = to_block_sharded(config, context_layer)
        self_output = ttnn.linear(
            context_layer,
            parameters.o_proj.weight,
            bias=parameters.o_proj.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=config.program_configs["self_output_matmul_program_config"],
        )
    ttnn.deallocate(context_layer)
    if config.should_reallocate_in_attention:
        self_output = ttnn.reallocate(self_output)

    return self_output


def vit_mlp_dram(
    config,
    hidden_states,
    *,
    parameters,
):
    hidden_states_dram = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(hidden_states)

    intermediate = ttnn.linear(
        hidden_states_dram,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        activation="gelu",
    )
    ttnn.deallocate(hidden_states_dram)

    output = ttnn.linear(
        intermediate,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(intermediate)

    # Back on the block-sharded grid the residual add and the next layer expect.
    return to_block_sharded(config, output)


def vit_mlp(
    config,
    hidden_states,
    *,
    parameters,
):
    if config.projections_in_dram:
        return vit_mlp_dram(config, hidden_states, parameters=parameters)

    intermediate = ttnn.linear(
        hidden_states,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.linear(
        intermediate,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(intermediate)

    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    output = vit_mlp(config, hidden_states, parameters=parameters.mlp)
    output = ttnn.add(output, attention_output, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(attention_output)
    return output


def vit_layer(
    config,
    hidden_states,
    parameters,
):
    signpost(header="ViT Layer Start")

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
            core_grid=config.core_grid_BLOCK_SHARDED,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(embeddings)

    for encoder_parameters in parameters:
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
        parameters=parameters.vit.layers,
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
    classifier_output = ttnn.linear(
        output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["classifer_matmul_program_config"],
    )
    return classifier_output


def fuse_query_key_value_by_projection(attention):
    """One linear for q, k and v, all of q's columns first, then k's, then v's.

    nlp_create_qkv_heads reads the fused output in that order and splits each part into heads itself,
    unlike the standard model's split, which wants the columns grouped head by head.
    """
    qkv_weight = torch.cat([attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight], dim=0)
    qkv_bias = torch.cat([attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias], dim=0)
    return {
        "weight": preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b),
        "bias": preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b),
    }


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

    if isinstance(torch_model, transformers.models.vit.modeling_vit.ViTAttention):
        # Returning parameters for a module stops ttnn from visiting its children, so o_proj goes here too.
        parameters = {
            "query_key_value": fuse_query_key_value_by_projection(torch_model),
            "o_proj": {
                "weight": preprocess_linear_weight(torch_model.o_proj.weight, dtype=ttnn.bfloat8_b),
                "bias": preprocess_linear_bias(torch_model.o_proj.bias, dtype=ttnn.bfloat8_b),
            },
        }

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
