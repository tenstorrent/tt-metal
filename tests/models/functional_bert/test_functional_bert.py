# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import dataclasses
import time
from typing import Optional

from loguru import logger
import torch
import torch.nn.functional as F
import transformers

import tt_lib as ttl
from tt_lib.tensor import MemoryConfig, BufferStorage, DataType

from tests.models.functional_bert.torch_functional_bert import (
    torch_bert_for_question_answering,
)
from tests.models.functional_bert.parameters import (
    preprocess_model_parameters,
    ParametersConfig,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)


@dataclasses.dataclass
class MatmulConfig:
    program_config: ttl.operations.primary.MatmulProgramConfig = (
        ttl.operations.primary.MatmulDefaultProgramConfig()
    )
    output_mem_config: MemoryConfig = MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.DRAM)
    output_dtype: Optional[DataType] = None


@dataclasses.dataclass
class MultiHeadAttentionConfig:
    fused_qkv_matmul_config: MatmulConfig = MatmulConfig()
    queue_by_key_matmul_config: MatmulConfig = MatmulConfig()
    attention_probs_by_value_matmul_config: MatmulConfig = MatmulConfig()
    self_output_matmul_config: MatmulConfig = MatmulConfig()

    split_fused_qkv_and_split_heads_output_mem_config: MemoryConfig = MemoryConfig(
        True, BufferStorage.DRAM
    )
    concatenate_heads_output_mem_config: MemoryConfig = MemoryConfig(
        True, BufferStorage.DRAM
    )


def tt_multi_head_attention(
    hidden_states,
    attention_mask,
    fused_qkv_weight,
    fused_qkv_bias,
    self_output_weight,
    self_output_bias,
    *,
    head_size,
    config: MultiHeadAttentionConfig,
):
    batch_size, *_ = hidden_states.shape()

    fused_qkv_output = ttl.operations.primary.matmul(
        hidden_states,
        fused_qkv_weight,
        bias=fused_qkv_bias,
        program_config=config.fused_qkv_matmul_config.program_config,
        output_mem_config=config.fused_qkv_matmul_config.output_mem_config,
        output_dtype=config.fused_qkv_matmul_config.output_dtype,
    )

    (
        query,
        key,
        value,
    ) = ttl.operations.primary.transformers.split_fused_qkv_and_split_heads(
        fused_qkv_output,
        ttl.tensor.CoreCoord(12, batch_size),
        config.split_fused_qkv_and_split_heads_output_mem_config,
    )
    fused_qkv_output.deallocate()

    attention_scores = ttl.operations.primary.matmul(
        query,
        key,
        program_config=config.queue_by_key_matmul_config.program_config,
        output_mem_config=config.queue_by_key_matmul_config.output_mem_config,
        output_dtype=config.queue_by_key_matmul_config.output_dtype,
    )
    query.deallocate()
    key.deallocate()

    attention_probs = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
        attention_scores, 1 / (head_size**0.5), attention_mask
    )

    context_layer = ttl.operations.primary.matmul(
        attention_probs,
        value,
        program_config=config.attention_probs_by_value_matmul_config.program_config,
        output_mem_config=config.attention_probs_by_value_matmul_config.output_mem_config,
        output_dtype=config.attention_probs_by_value_matmul_config.output_dtype,
    )
    attention_probs.deallocate()

    context_layer = ttl.operations.primary.transformers.concatenate_heads(
        context_layer,
        ttl.tensor.CoreCoord(12, batch_size),
        config.concatenate_heads_output_mem_config,
    )

    self_output = ttl.operations.primary.matmul(
        context_layer,
        self_output_weight,
        bias=self_output_bias,
        program_config=config.self_output_matmul_config.program_config,
        output_mem_config=config.self_output_matmul_config.output_mem_config,
        output_dtype=config.self_output_matmul_config.output_dtype,
    )
    context_layer.deallocate()

    return self_output


@dataclasses.dataclass
class FeedforwardConfig:
    ff1_matmul_config: MatmulConfig = MatmulConfig()
    ff2_matmul_config: MatmulConfig = MatmulConfig()


def tt_feedforward(
    hidden_states,
    intermediate_weight,
    intermediate_bias,
    output_weight,
    output_bias,
    *,
    config: FeedforwardConfig,
):
    ff1_output = ttl.operations.primary.matmul(
        hidden_states,
        intermediate_weight,
        bias=intermediate_bias,
        program_config=config.ff1_matmul_config.program_config,
        output_mem_config=config.ff1_matmul_config.output_mem_config,
        output_dtype=config.ff1_matmul_config.output_dtype,
    )

    ff2_output = ttl.operations.primary.matmul(
        ff1_output,
        output_weight,
        bias=output_bias,
        program_config=config.ff2_matmul_config.program_config,
        output_mem_config=config.ff2_matmul_config.output_mem_config,
        output_dtype=config.ff2_matmul_config.output_dtype,
    )
    ff1_output.deallocate()

    return ff2_output


def tt_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    encoder_index,
    head_size,
    **kwargs,
):
    multi_head_attention_output = tt_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters[
            f"bert.encoder.layer.{encoder_index}.attention.self.fused_qkv.dense.weight"
        ],
        parameters[
            f"bert.encoder.layer.{encoder_index}.attention.self.fused_qkv.dense.bias"
        ],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"],
        head_size=head_size,
        config=kwargs["multi_head_attention_config"],
    )

    multi_head_attention_add_and_layer_norm_output = (
        ttl.operations.primary.add_layernorm(
            hidden_states,
            multi_head_attention_output,
            1e-12,
            parameters[
                f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"
            ],
            parameters[
                f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"
            ],
            output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
        )
    )
    hidden_states.deallocate()

    feedforward_output = tt_feedforward(
        multi_head_attention_add_and_layer_norm_output,
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.dense.bias"],
        config=kwargs["feedforward_config"],
    )

    feedforward_add_and_layer_norm_output = ttl.operations.primary.add_layernorm(
        multi_head_attention_add_and_layer_norm_output,
        feedforward_output,
        1e-12,
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias"],
        output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
    )
    multi_head_attention_add_and_layer_norm_output.deallocate()

    return feedforward_add_and_layer_norm_output


def tt_bert_preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    **kwargs,
):
    word_embeddings = F.embedding(
        input_ids, parameters["bert.embeddings.word_embeddings.weight"]
    )
    token_type_embeddings = F.embedding(
        token_type_ids, parameters["bert.embeddings.token_type_embeddings.weight"]
    )
    embeddings = word_embeddings + token_type_embeddings

    embeddings = (
        ttl.tensor.Tensor(embeddings.unsqueeze(1), DataType.BFLOAT16)
        .to(ttl.tensor.Layout.TILE)
        .to(kwargs["device"], MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1))
    )
    encoder_input = ttl.operations.primary.layernorm(
        embeddings,
        1e-12,
        parameters["bert.embeddings.LayerNorm.weight"],
        parameters["bert.embeddings.LayerNorm.bias"],
        output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
    )
    embeddings.deallocate()
    encoder_input = ttl.tensor.move(encoder_input)

    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    attention_mask = F.pad(attention_mask, (0, 0, 0, 31))
    attention_mask = (
        ttl.tensor.Tensor(attention_mask, DataType.BFLOAT16)
        .to(ttl.tensor.Layout.TILE)
        .to(kwargs["device"], MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1))
    )

    return encoder_input, attention_mask


def tt_bert(
    encoder_input,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
    **kwargs,
):
    encoder_output = None
    for encoder_index in range(num_encoders):
        encoder_output = tt_encoder(
            encoder_input,
            attention_mask,
            parameters,
            encoder_index=encoder_index,
            head_size=head_size,
            **kwargs,
        )
        encoder_output = ttl.tensor.move(encoder_output)
        encoder_input = encoder_output
    return encoder_output


def tt_bert_for_question_answering(
    encoder_input,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
    **kwargs,
):
    bert_output = tt_bert(
        encoder_input,
        attention_mask,
        parameters,
        num_encoders=num_encoders,
        head_size=head_size,
        **kwargs,
    )

    qa_outputs = ttl.operations.primary.matmul(
        bert_output,
        parameters["qa_outputs.weight"],
        bias=parameters["qa_outputs.bias"],
        output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
    )
    return qa_outputs


def custom_preprocessor(parameters_config, torch_model, full_name, **kwargs):
    parameters = {}
    if isinstance(
        torch_model, transformers.models.bert.modeling_bert.BertSelfAttention
    ):
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
        parameters[f"{full_name}fused_qkv.dense.weight"] = preprocess_linear_weight(
            parameters_config, qkv_weight, **kwargs
        )
        parameters[f"{full_name}fused_qkv.dense.bias"] = preprocess_linear_bias(
            parameters_config, qkv_bias, **kwargs
        )
    return parameters


def is_to_be_converted(torch_model, full_name):
    return "dense" in full_name or "LayerNorm" in full_name or "qa_outputs" in full_name


def create_bert_multi_head_attention_config(batch_size):
    return MultiHeadAttentionConfig(
        fused_qkv_matmul_config=MatmulConfig(
            program_config=ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(12, batch_size),
                in0_block_w=4,
                out_subblock_h=4,
                out_subblock_w=2,
                per_core_M=12,
                per_core_N=8,
                transpose_mcast=False,
                fused_activation=None,
            ),
            output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
            output_dtype=DataType.BFLOAT8_B,
        ),
        queue_by_key_matmul_config=MatmulConfig(
            program_config=(
                ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(12, batch_size),
                    in0_block_w=1,
                    out_subblock_h=4,
                    out_subblock_w=2,
                    per_core_M=12,
                    per_core_N=12,
                )
            ),
            output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
            output_dtype=DataType.BFLOAT16,
        ),
        attention_probs_by_value_matmul_config=MatmulConfig(
            program_config=ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(12, batch_size),
                in0_block_w=2,
                out_subblock_h=4,
                out_subblock_w=2,
                per_core_M=12,
                per_core_N=2,
            ),
            output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
            output_dtype=DataType.BFLOAT8_B,
        ),
        self_output_matmul_config=MatmulConfig(
            program_config=(
                ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(12, batch_size),
                    in0_block_w=4,
                    out_subblock_h=6,
                    out_subblock_w=1,
                    per_core_M=12,
                    per_core_N=3,
                    transpose_mcast=False,
                    fused_activation=None,
                )
            ),
            output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
            output_dtype=DataType.BFLOAT16,
        ),
        split_fused_qkv_and_split_heads_output_mem_config=MemoryConfig(
            True, BufferStorage.L1
        ),
        concatenate_heads_output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
    )


def create_bert_feedforward_config(batch_size):
    return FeedforwardConfig(
        ff1_matmul_config=MatmulConfig(
            program_config=(
                ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(12, batch_size),
                    in0_block_w=4,
                    out_subblock_h=6,
                    out_subblock_w=1,
                    per_core_M=12,
                    per_core_N=11,
                    transpose_mcast=False,
                    fused_activation=(ttl.tensor.FusibleActivation.GELU, True),
                )
            ),
            output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
            output_dtype=DataType.BFLOAT8_B,
        ),
        ff2_matmul_config=MatmulConfig(
            program_config=(
                ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(12, batch_size),
                    in0_block_w=4,
                    out_subblock_h=6,
                    out_subblock_w=1,
                    per_core_M=12,
                    per_core_N=3,
                    transpose_mcast=False,
                    fused_activation=None,
                )
            ),
            output_mem_config=MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1),
            output_dtype=DataType.BFLOAT16,
        ),
    )


def run_bert_question_and_answering_inference(
    model_name, batch_size, sequence_size, device
):
    enable_persistent_kernel_cache()

    torch.manual_seed(1234)

    torch_model = transformers.BertForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    ).eval()
    config = torch_model.config

    num_encoders = config.num_hidden_layers
    head_size = config.hidden_size // config.num_attention_heads

    torch_bert_input = torch.randint(
        0, torch_model.config.vocab_size, (batch_size, sequence_size)
    )
    torch_token_type_ids = torch.randint(0, 1, (1, sequence_size))
    torch_attention_mask = torch.zeros(1, sequence_size)

    torch_output = torch_bert_for_question_answering(
        torch_bert_input,
        torch_token_type_ids,
        torch_attention_mask,
        parameters=torch_model.state_dict(),
        num_encoders=num_encoders,
        head_size=head_size,
    )

    # Run TT Model
    parameters_config = ParametersConfig(
        linear_weight_dtype=DataType.BFLOAT8_B,
        linear_bias_dtype=DataType.BFLOAT8_B,
        layernorm_parameter_dtype=DataType.BFLOAT16,
    )
    parameters = preprocess_model_parameters(
        parameters_config,
        torch_model,
        is_to_be_converted=is_to_be_converted,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    import time

    num_iterations = 3
    for index in range(num_iterations):
        tt_bert_inputs = tt_bert_preprocess_inputs(
            torch_bert_input,
            torch_token_type_ids,
            torch_attention_mask,
            parameters=parameters,
            device=device,
        )

        start = time.time()
        tt_output = tt_bert_for_question_answering(
            *tt_bert_inputs,
            parameters=parameters,
            num_encoders=num_encoders,
            head_size=head_size,
            multi_head_attention_config=create_bert_multi_head_attention_config(
                batch_size
            ),
            feedforward_config=create_bert_feedforward_config(batch_size),
            device=device,
        )
        ttl.device.Synchronize()
        end = time.time()
        duration = end - start
        logger.info(f"{index}: Duration: {duration}")
        logger.info(f"{index}: Samples per second: {1 / duration * batch_size}")
    tt_output = tt_output.cpu().to_torch().squeeze(1).to(torch.float32)[:, :, :2]

    # assert torch.allclose(torch_output, tt_output)

    disable_persistent_kernel_cache()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert(use_program_cache, batch_size, sequence_size, device):
    model_name = "phiyodr/bert-large-finetuned-squad2"

    run_bert_question_and_answering_inference(
        model_name, batch_size, sequence_size, device
    )
