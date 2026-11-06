# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import inspect
import math

import torch
import transformers
from ttnn.dot_access import DotAccessDict
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn


def largest_divisor_at_most(value, limit):
    return max(divisor for divisor in range(1, min(value, limit) + 1) if value % divisor == 0)


def ff1_exceeds_l1_budget(device, rows_t, cols_t):
    """Whether a core's ff1 output, a block of rows_t x (4 * cols_t) bfloat8_b tiles, is too big for its L1.

    ff1's output is 4x wider than the hidden block it comes from and is the largest activation a core holds,
    alongside that block, the residual and the matmul buffers. The limit is a quarter of a core's L1.
    """
    ff1_output_bytes = rows_t * cols_t * 4 * ttnn.tile_size(ttnn.bfloat8_b)
    l1_bytes_per_core = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    return ff1_output_bytes > l1_bytes_per_core // 4


def update_model_config(config, batch_size, device):
    """TTNN settings for ViT-Base on the device's grid, merged into the Hugging Face config.

    The grid width is the largest divisor of the hidden tiles, heads and padded classes that fits the device,
    and its height the largest divisor of the batch. Activations stay sharded in L1 with the program configs
    below unless ff1 does not fit (ff1_exceeds_l1_budget); then the whole model runs from DRAM and the ops
    pick their own program configs. Besides the Hugging Face fields the result carries core_grid, in_dram,
    memory_configs, should_reallocate_in_attention and program_configs.
    """
    TILE_HEIGHT = 32
    device_grid = device.compute_with_storage_grid_size()

    patch_count = config.image_size // config.patch_size  # 224/16=14
    seqL = patch_count * patch_count + 1  # 197 with the class token
    seqL_padded = (((seqL - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT  # 224
    seqL_t = seqL_padded // TILE_HEIGHT  # 7
    dim_t = config.hidden_size // TILE_HEIGHT  # 768 / 32 = 24
    head_num = config.num_attention_heads  # 12
    # 1000 classes padded to 1152
    class_t = 1152 // TILE_HEIGHT  # 36

    # Each core column holds a slice of the hidden dim, whole heads and a slice of the classes, so the
    # grid width must divide all of them. Each core row holds whole images.
    grid_x = largest_divisor_at_most(math.gcd(dim_t, head_num, class_t), device_grid.x)
    grid_y = largest_divisor_at_most(batch_size, device_grid.y)
    batch_per_y = batch_size // grid_y
    core_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
    # With more than one image per core row, attention reallocates after the head split to undo L1 fragmentation.
    should_reallocate_in_attention = batch_per_y > 1

    seqL_t__y = batch_per_y * seqL_t  # 7 for one image per row
    dim_t__x = dim_t // grid_x  # 2 on a 12-wide grid
    head_seqL_t__x = (batch_per_y * head_num * seqL_t) // grid_x  # 7 on a 12-wide grid
    head_size_t = dim_t // head_num  # 2
    class__x = class_t // grid_x  # 3 on a 12-wide grid

    # Placement by role: embeddings interleaved, encoder activations block-sharded over core_grid, attention
    # heads height-sharded. On a grid whose L1 cannot hold ff1, every role goes to DRAM.
    in_dram = ff1_exceeds_l1_budget(device, seqL_t__y, dim_t__x)
    if in_dram:
        memory_configs = {
            "embeddings": ttnn.DRAM_MEMORY_CONFIG,
            "encoder": ttnn.DRAM_MEMORY_CONFIG,
            "attention_heads": ttnn.DRAM_MEMORY_CONFIG,
        }
    else:
        memory_configs = {
            "embeddings": ttnn.L1_MEMORY_CONFIG,
            "encoder": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            "attention_heads": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        }
    class_subb_w = class__x
    if class_subb_w > 8:  # an output subblock holds at most 8 tiles
        if class_subb_w % 3 == 0:
            class_subb_w = class__x // 3
        elif class_subb_w % 2 == 0:
            class_subb_w = class__x // 2
        else:
            class_subb_w = 1

    # Every config below shards an activation of [batch * seqL, ...] tiles over core_grid: a core row
    # holds seqL_t__y tile rows (whole images) and a core column a slice of the hidden dim, heads or classes.
    grid_size = (core_grid.x, core_grid.y)
    program_configs = {
        "layernorm_before_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=dim_t__x,
            block_h=seqL_t__y,
            block_w=dim_t__x,
            inplace=False,
        ),
        "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=dim_t__x,
            out_subblock_h=1,
            out_subblock_w=dim_t__x,
            per_core_M=seqL_t__y,
            per_core_N=3 * dim_t__x,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=head_size_t,
            out_subblock_h=1,
            out_subblock_w=seqL_t,
            per_core_M=head_seqL_t__x,
            per_core_N=seqL_t,
        ),
        "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=seqL_t,
            block_h=head_seqL_t__x,
            block_w=seqL_t,
        ),
        "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seqL_t,
            out_subblock_h=1,
            out_subblock_w=head_size_t,
            per_core_M=head_seqL_t__x,
            per_core_N=head_size_t,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=dim_t__x,
            out_subblock_h=1,
            out_subblock_w=dim_t__x,
            per_core_M=seqL_t__y,
            per_core_N=dim_t__x,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=dim_t__x,
            block_h=seqL_t__y,
            block_w=dim_t__x,
            inplace=False,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=dim_t__x,
            out_subblock_h=1,
            out_subblock_w=(dim_t__x * 4) // 2,
            per_core_M=seqL_t__y,
            per_core_N=dim_t__x * 4,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=dim_t__x * 4,
            out_subblock_h=1,
            out_subblock_w=dim_t__x,
            per_core_M=seqL_t__y,
            per_core_N=dim_t__x,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "classifer_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=dim_t__x // 2,
            out_subblock_h=1,
            out_subblock_w=class_subb_w,
            per_core_M=seqL_t__y,
            per_core_N=class__x,
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
    if in_dram:
        program_configs = {
            name: None if name != "ln_compute_config" else program_config
            for name, program_config in program_configs.items()
        }
        program_configs["softmax_program_config"] = ttnn.SoftmaxDefaultProgramConfig()

    # properties are not in the output of config.to_dict() but can be used later in the model
    # e.g. https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/configuration_utils.py
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
            in_dram=in_dram,
            memory_configs=memory_configs,
            should_reallocate_in_attention=should_reallocate_in_attention,
            program_configs=program_configs,
        )
    )


# https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/vit/modeling_vit.py


def activation_memory_config(config, shape):
    """Where an encoder activation of this shape lives: block-sharded over core_grid, or DRAM."""
    if config.in_dram:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.create_sharded_memory_config(
        shape,
        core_grid=config.core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def pixel_values_memory_config(config, device, shape):
    """Memory config for the NHWC pixels that vit_patch_embeddings folds.

    Height-sharded in L1, two cores per image so that every shard holds whole groups of patch rows as the
    fold needs; interleaved in DRAM when the model runs from DRAM.
    """
    if config.in_dram:
        return ttnn.DRAM_MEMORY_CONFIG
    batch_size, height, width, channels = shape
    num_cores = 2 * batch_size
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    shard_spec = ttnn.ShardSpec(
        shard_grid, [batch_size * height * width // num_cores, channels], ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


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
    folded_pixel_values = ttnn.to_memory_config(folded_pixel_values, memory_config=config.memory_configs["embeddings"])
    # Convert back to interleaved or otherwise to_layout will fail
    folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.linear(
        folded_pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=config.memory_configs["embeddings"],
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

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    embedding_output = ttnn.concat([cls_token, patch_embeddings], -2, memory_config=config.memory_configs["embeddings"])
    embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)
    embedding_output = ttnn.add(
        embedding_output, position_embeddings, memory_config=config.memory_configs["embeddings"], dtype=ttnn.bfloat8_b
    )

    return embedding_output


def split_heads_interleaved(query_key_value, num_heads, head_size):
    """Query, key (transposed) and value per head from the fused projection.

    The fused weight is laid out head by head ([q, k, v] for each head), the order the sharded
    split_query_key_value_and_split_heads expects; the interleaved op expects all of q, then k, then v,
    so the heads are split here instead.
    """
    batch_size, sequence_size, _ = query_key_value.shape
    heads = ttnn.reshape(query_key_value, (batch_size, sequence_size, num_heads, 3 * head_size))
    heads = ttnn.permute(heads, (0, 2, 1, 3))
    query = ttnn.slice(heads, (0, 0, 0, 0), (batch_size, num_heads, sequence_size, head_size))
    key = ttnn.slice(heads, (0, 0, 0, head_size), (batch_size, num_heads, sequence_size, 2 * head_size))
    value = ttnn.slice(heads, (0, 0, 0, 2 * head_size), (batch_size, num_heads, sequence_size, 3 * head_size))
    ttnn.deallocate(heads)
    return query, ttnn.permute(key, (0, 1, 3, 2)), value


def vit_attention(
    config,
    hidden_states,
    parameters,
):
    num_heads = config.num_attention_heads  # num_heads = 12
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        memory_config=config.memory_configs["encoder"],
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_key_value_matmul_program_config"],
    )

    if config.in_dram:
        query, key, value = split_heads_interleaved(query_key_value, num_heads, head_size)
    else:
        (
            query,
            key,
            value,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            query_key_value,
            memory_config=config.memory_configs["attention_heads"],
            num_heads=num_heads,
        )
    ttnn.deallocate(query_key_value)
    ttnn.deallocate(hidden_states)
    if config.should_reallocate_in_attention:
        value = ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=config.memory_configs["attention_heads"],
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
        memory_config=config.memory_configs["attention_heads"],
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=config.memory_configs["encoder"],
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.o_proj.weight,
        bias=parameters.o_proj.bias,
        memory_config=config.memory_configs["encoder"],
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)
    if config.should_reallocate_in_attention:
        self_output = ttnn.reallocate(self_output)

    return self_output


def vit_mlp(
    config,
    hidden_states,
    *,
    parameters,
):
    intermediate = ttnn.linear(
        hidden_states,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=config.memory_configs["encoder"],
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
        # The sharded ff1 program config fuses GELU; without it the activation is passed explicitly.
        activation="gelu" if config.in_dram else None,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.linear(
        intermediate,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=config.memory_configs["encoder"],
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
    output = ttnn.add(output, attention_output, memory_config=config.memory_configs["encoder"], dtype=ttnn.bfloat8_b)
    ttnn.deallocate(attention_output)
    return output


def vit_layer(
    config,
    hidden_states,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        memory_config=config.memory_configs["encoder"],
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
        memory_config=config.memory_configs["encoder"],
        dtype=ttnn.bfloat8_b,
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        memory_config=config.memory_configs["encoder"],
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
    if config.in_dram:
        # The embeddings are already bfloat8_b in DRAM. to_memory_config would hand back this same tensor,
        # so it must not be freed.
        encoder_input = embeddings
    else:
        TILE_HEIGHT = 32
        emb_N, emb_S, emb_D = embeddings.shape
        emb_S = (((emb_S - 1) // TILE_HEIGHT) + 1) * TILE_HEIGHT
        encoder_input = ttnn.to_memory_config(
            embeddings, memory_config=activation_memory_config(config, [emb_N, emb_S, emb_D]), dtype=ttnn.bfloat8_b
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
        memory_config=config.memory_configs["encoder"],
        program_config=config.program_configs["layernorm_before_program_config"],
    )

    # Classifier
    classifier_output = ttnn.linear(
        output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
        memory_config=config.memory_configs["encoder"],
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["classifer_matmul_program_config"],
    )
    return classifier_output


def fuse_query_key_value(attention):
    """One linear for q, k and v, laid out head by head as split_query_key_value_and_split_heads expects."""
    num_heads = attention.num_attention_heads
    head_dim = attention.head_dim
    qkv_weight = torch.cat(
        [
            attention.q_proj.weight.reshape([num_heads, head_dim, -1]),
            attention.k_proj.weight.reshape([num_heads, head_dim, -1]),
            attention.v_proj.weight.reshape([num_heads, head_dim, -1]),
        ],
        dim=1,
    ).reshape([3 * num_heads * head_dim, -1])
    qkv_bias = torch.cat(
        [
            attention.q_proj.bias.reshape([num_heads, head_dim]),
            attention.k_proj.bias.reshape([num_heads, head_dim]),
            attention.v_proj.bias.reshape([num_heads, head_dim]),
        ],
        dim=1,
    ).reshape([3 * num_heads * head_dim])
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
            "query_key_value": fuse_query_key_value(torch_model),
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
