# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import torch
from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
)

import ttnn

from ttnn.dot_access import DotAccessDict


def update_model_config(config, batch_size):
    core_grid = ttnn.CoreGrid(y=8, x=12)

    program_configs = {
        "fold_output_program_config": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(12, 7),
                        ),
                    }
                ),
                [
                    224,
                    192,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        ),
        "embedding_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=3,
            out_subblock_h=1,
            out_subblock_w=6,
            per_core_M=7,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            per_core_M=7,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=7,
            per_core_M=7,
            per_core_N=7,
        ),
        "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=7,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=7,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=7,
            per_core_N=8,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=8,
            out_subblock_h=7,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "classifer_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            per_core_M=7,
            per_core_N=3,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "layernorm_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=2,
            block_h=7,
            block_w=2,
            # math_fidelity=ttnn.MathFidelity.HiFi4,
            # im_data_format=ttnn.bfloat16,
            # out_data_format=ttnn.bfloat8_b,
            inplace=True,
        ),
        "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=2,
            block_h=7,
            block_w=2,
            # math_fidelity=ttnn.MathFidelity.HiFi4,
            # im_data_format=ttnn.bfloat16,
            # out_data_format=ttnn.bfloat8_b,
            inplace=False,
        ),
        "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=7,
            block_h=7,
            block_w=7,
            # math_fidelity=ttnn.MathFidelity.HiFi4,
            # im_data_format=ttnn.bfloat16,
        ),
    }

    return DotAccessDict(dict(**config.to_dict(), core_grid=core_grid, program_configs=program_configs))


# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings(
    config,
    pixel_values,
    *,
    parameters,
):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 16
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    fold_h_padded = (batch_size * img_h * patch_count_all) + 224
    fold_w_padded = (4 * patch_size * patch_size) + 128

    # pixel_values = ttnn.reshape_on_device(pixel_values, batch_size, img_h, img_w // patch_size, 4 * patch_size)
    folded_pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)  # 1568, 1024
    ttnn.deallocate(pixel_values)
    x = ttnn.reallocate(folded_pixel_values)
    folded_pixel_values = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    #### Exp 1 of resharding after Fold and before Matmul
    # pixel_values = ttnn.pad(pixel_values, ((0, 0), (0, 0), (0, 224), (0, 128)), 0)
    # output_sharded_memory_config_args = dict(core_grid=ttnn.CoreGrid(y=8, x=12), strategy=ttnn.ShardStrategy.BLOCK)
    # input_shape = [fold_h_padded, fold_w_padded]
    # output_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **output_sharded_memory_config_args)
    # resharded_pixel_values = ttnn.to_memory_config(pixel_values, output_shard_memory_config)

    #### Exp 2 of resharding after Fold and before Matmul
    # pixel_values = ttnn.pad(pixel_values, ((0, 0), (0, 0), (0, 224), (0, 128)), 0)
    # post_fold_config = ttnn.MemoryConfig(
    #     ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    #     ttnn.BufferType.L1,
    #     ttnn.ShardSpec(
    #         ttnn.CoreRangeSet(
    #             {ttnn.CoreRange(
    #                     ttnn.CoreCoord(0, 0),
    #                     ttnn.CoreCoord(11, 7),
    #             ),},
    #         ),
    #         [224,192],
    #         ttnn.ShardOrientation.ROW_MAJOR,
    #         False,
    #     ),
    # )
    # resharded_pixel_values = ttnn.reshard(pixel_values, post_fold_config)

    # return resharded_pixel_values

    ## Needed only when running the standalone module pytest test_vit_patch_embeddings
    ## Please comment out when running the pytest on parent module like test_vit_embeddings or test_vit
    # parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.linear(
        folded_pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=8, x=12),
        # program_config=config.program_configs["embedding_matmul_program_config"],
    )
    # ttnn.deallocate(pixel_values)

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape_on_device(
        patch_embedding_output, batch_size, patch_count_all, patch_size_sq_trpl
    )

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
    # cls_token = parameters.cls_token
    # position_embeddings = parameters.position_embeddings

    l1_memory_config = ttnn.L1_MEMORY_CONFIG

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    # print("clcs", cls_token.shape)
    # print("patch", patch_embeddings.shape)
    # patch_embeddings = ttnn.pad(patch_embeddings, padding=((0, 0), (1, 27), (0, 0)), value=0)
    # embedding_output = ttnn.to_layout(patch_embeddings, layout=ttnn.TILE_LAYOUT)

    embedding_output = ttnn.concat([cls_token, patch_embeddings], -2, memory_config=l1_memory_config)
    embedding_output = ttnn.pad(embedding_output, padding=((0, 0), (0, 27), (0, 0)), value=0)
    # print("out", embedding_output.shape)
    embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)
    # print("outTilized", embedding_output.shape)
    embedding_output = ttnn.add(
        embedding_output, position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    # embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)
    # Needed to improve PCC in an older commit
    # embedding_output = ttnn.pad(embedding_output, ((0, 0), (0, 27), (0, 0)), 0)

    return embedding_output


def vit_layernorm_before(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

    return attention_output


def vit_layernorm_after(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

    return attention_output


def vit_attention(
    config,
    hidden_states,
    attention_mask,
    parameters,
):
    num_heads = config.num_attention_heads
    num_heads = 12
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # encoder_input = ttnn.to_memory_config(
    #     hidden_states,
    #     memory_config=ttnn.create_sharded_memory_config(
    #         hidden_states.shape,
    #         core_grid=config.core_grid,
    #         strategy=ttnn.ShardStrategy.BLOCK,
    #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #         #orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
    #     ),
    #     dtype=ttnn.bfloat8_b,
    # )
    # ttnn.deallocate(hidden_states)

    encoder_input = hidden_states

    query_key_value = ttnn.linear(
        encoder_input,
        parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        # core_grid=ttnn.CoreGrid(y=8, x=8),
        program_config=config.program_configs["query_key_value_matmul_program_config"],
    )

    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        # core_grid=ttnn.CoreGrid(y=8, x=8),
        program_config=config.program_configs["query_by_key_matmul_program_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.transformer.attention_softmax_(
        attention_scores,
        attention_mask=attention_mask,
        head_size=head_size,
        program_config=config.program_configs["softmax_program_config"],
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        # core_grid=ttnn.CoreGrid(y=8, x=8),
        program_config=config.program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        # core_grid=ttnn.CoreGrid(y=8, x=8),
        program_config=config.program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    return self_output


def vit_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
        # core_grid=ttnn.CoreGrid(y=8, x=8),
        # activation="gelu",
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
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        # core_grid=ttnn.CoreGrid(y=8, x=8),
        program_config=config.program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    # residual_sh = ttnn.to_memory_config(
    #     residual,
    #     memory_config=ttnn.create_sharded_memory_config(
    #         residual.shape,
    #         core_grid=config.core_grid,
    #         strategy=ttnn.ShardStrategy.BLOCK,
    #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #         # orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
    #     ),
    #     dtype=ttnn.bfloat8_b,
    # )
    # ttnn.deallocate(residual)

    residual_sh = residual

    output = ttnn.add(output, residual_sh, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(residual_sh)

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
    attention_mask,
    parameters,
):
    # encoder_input = ttnn.to_memory_config(
    #     hidden_states,
    #     memory_config=ttnn.create_sharded_memory_config(
    #         hidden_states.shape,
    #         core_grid=config.core_grid,
    #         strategy=ttnn.ShardStrategy.BLOCK,
    #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #         # orientation=ttnn.ShardOrientation.COLUMN_MAJOR,
    #     ),
    #     dtype=ttnn.bfloat8_b,
    # )
    # ttnn.deallocate(hidden_states)

    encoder_input = hidden_states

    layernorm_before_output = ttnn.layer_norm(
        encoder_input,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

    multi_head_attention_output = vit_attention(
        config,
        layernorm_before_output,
        attention_mask=attention_mask,
        parameters=parameters.attention,
    )

    residual = ttnn.add(
        multi_head_attention_output,
        encoder_input,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(multi_head_attention_output)

    layernorm_after_output = ttnn.layer_norm(
        residual,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_after_output_program_config"],
    )

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        residual,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    embeddings,
    head_masks,
    parameters,
):
    encoder_input = ttnn.to_memory_config(
        embeddings,
        memory_config=ttnn.create_sharded_memory_config(
            [8, 224, 768],  # embeddings.shape
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )

    # ttnn.deallocate(embeddings)
    # encoder_input = embeddings

    encoder_output = None
    for index, encoder_parameters in enumerate(parameters.layer):
        encoder_output = vit_layer(
            config,
            encoder_input,
            head_masks[index],
            encoder_parameters,
        )
        encoder_input = encoder_output

    return encoder_output


def vit(
    config,
    pixel_values,
    attention_mask,
    cls_token,
    position_embeddings,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, cls_token, position_embeddings, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        attention_mask,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_program_config"],
    )

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


def preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    device,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, 0, 0, 0, 0, batch_size - 1))
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return input_ids, token_type_ids, attention_mask


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
        qkv_weight = torch.cat(
            [
                torch_model.query.weight,
                torch_model.key.weight,
                torch_model.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.query.bias, torch_model.key.bias, torch_model.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)

    elif isinstance(torch_model, torch.nn.Linear):
        # print(torch_model.weight.shape)
        if torch_model.weight.shape[0] == 1000:
            preprocessed_weight = torch.nn.functional.pad(torch_model.weight, (0, 0, 0, int(1152 - 1000)))
            preprocessed_bias = torch.nn.functional.pad(torch_model.bias, (0, int(1152 - 1000)))
            parameters["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat8_b)
        else:
            parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)

    return parameters
