# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from transformers import BloomForCausalLM, BloomTokenizerFast
import torch
from models import (
    generation_utils,
)
import math
from typing import Tuple
import time

from tests.ttnn.utils_for_testing import update_process_id

import torch.utils.checkpoint
from torch.nn import functional as F
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    ParametersConfig,
)


def bloom_gelu_forward(x: ttnn.Tensor) -> ttnn.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`ttnn.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + ttnn.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# From transformers/models/bloom/modeling_bloom.py
def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


# Based on transformers/models/bloom/modeling_bloom.py
def _split_heads(fused_qkv: torch.Tensor, num_heads, head_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    fused_qkv = ttnn.reshape(fused_qkv, shape=(batch_size, seq_length, num_heads, 3, head_size))
    return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]


# Based on transformers/models/bloom/modeling_bloom.py
def _merge_heads(x: ttnn.Tensor, num_heads, head_size) -> ttnn.Tensor:
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size_and_num_heads, _, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    # First view to decompose the batch size
    # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
    x = ttnn.reshape(x, shape=(batch_size, num_heads, seq_length, head_size))

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = ttnn.permute(x, (0, 2, 1, 3))

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return ttnn.reshape(x, shape=(batch_size, seq_length, num_heads * head_size))


def layer_normalization(hidden_states, weight, bias):
    return ttnn.experimental.layer_norm(
        hidden_states,
        weight=weight,
        bias=bias,
    )


def create_query_key_value(device, hidden_states, weight, bias, num_heads, head_size):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    fused_qkv = hidden_states @ weight
    fused_qkv = fused_qkv + bias
    fused_qkv = ttnn.reshape(fused_qkv, (batch_size, sequence_size, 3 * hidden_size))
    (query_layer, key_layer, value_layer) = _split_heads(fused_qkv, num_heads, head_size)
    batch_size, q_length, _, _ = query_layer.shape
    query_layer = ttnn.to_device(query_layer, device)
    key_layer = ttnn.to_device(key_layer, device)
    value_layer = ttnn.to_device(value_layer, device)
    query_layer = ttnn.permute(query_layer, (0, 2, 1, 3))
    query_layer = ttnn.reshape(query_layer, (batch_size * num_heads, 1, q_length, head_size))
    key_layer = ttnn.permute(key_layer, (0, 2, 3, 1))
    key_layer = ttnn.reshape(key_layer, (batch_size * num_heads, 1, head_size, q_length))
    value_layer = ttnn.permute(value_layer, (0, 2, 1, 3))
    value_layer = ttnn.reshape(value_layer, (batch_size * num_heads, 1, q_length, head_size))
    return query_layer, key_layer, value_layer


def compute_attention_scores(query_layer, key_layer, alibi, head_size, batch_size):
    num_heads_and_batch, _, q_length, _ = query_layer.shape
    _, _, _, kv_length = key_layer.shape
    beta = 1.0
    inv_norm_factor = inv_norm_factor = 1.0 / math.sqrt(head_size)
    matmul_result = beta * alibi + inv_norm_factor * (query_layer @ key_layer)
    num_heads = num_heads_and_batch // batch_size
    return ttnn.reshape(matmul_result, shape=(batch_size, num_heads, q_length, kv_length))


def compute_attention_probs(attention_scores, attention_mask):
    batch_size, num_heads, q_length, kv_length = attention_scores.shape
    fill_value = -3.3895313892515355e38
    attn_weights = attention_scores * (1 + (attention_mask * -1)) + attention_mask * fill_value
    attention_probs = ttnn.softmax(attn_weights, dim=-1)
    return ttnn.reshape(attention_probs, shape=(batch_size * num_heads, 1, q_length, kv_length))


def compute_context_layer(attention_probs, value_layer, num_heads, head_size):
    context_layer = attention_probs @ value_layer
    return _merge_heads(context_layer, num_heads, head_size)


def finalize_output(context_layer, output_weight, output_bias, hidden_states):
    output_tensor = context_layer @ output_weight
    output_tensor = output_tensor + output_bias
    return hidden_states + output_tensor


def bloom_multi_head_attention(
    device,
    hidden_states,
    attention_mask,
    input_layernorm_weight,
    input_layernorm_bias,
    query_key_value_weight,
    query_key_value_bias,
    output_weight,
    output_bias,
    alibi,
    *,
    head_size,
):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    num_heads = hidden_size // head_size
    layernorm_output = layer_normalization(hidden_states, input_layernorm_weight, input_layernorm_bias)
    query_layer, key_layer, value_layer = create_query_key_value(
        device, layernorm_output, query_key_value_weight, query_key_value_bias, num_heads, head_size
    )
    attention_scores = compute_attention_scores(query_layer, key_layer, alibi, head_size, batch_size)
    attention_probs = compute_attention_probs(attention_scores, attention_mask)
    context_layer = compute_context_layer(attention_probs, value_layer, num_heads, head_size)
    output_tensor = finalize_output(context_layer, output_weight, output_bias, hidden_states)
    present = (key_layer, value_layer)
    attn_outputs = (output_tensor, present)
    return attn_outputs


def mlp(attention_output, layernorm_output, h_to_4h_weight, h_to_4h_bias, dense_4h_to_h_weight, dense_4h_to_h_bias):
    dense_h_to_4h = layernorm_output @ h_to_4h_weight
    dense_h_to_4h = dense_h_to_4h + h_to_4h_bias
    hidden_states = bloom_gelu_forward(dense_h_to_4h)
    dense_4h_to_h = hidden_states @ dense_4h_to_h_weight
    dense_4h_to_h = dense_4h_to_h + dense_4h_to_h_bias
    # output_tensor = F.dropout(dense_4h_to_h, p=0.0, training=False)
    output_tensor = attention_output + dense_4h_to_h
    return output_tensor


def ttnn_bloom(
    input_ids,
    causal_mask,
    alibi,
    num_heads,
    hidden_layers,
    parameters,
    device,
):
    inputs_embeds = ttnn.embedding(input_ids, parameters["transformer.word_embeddings.weight"])
    embed_dim = inputs_embeds.shape[2]
    hidden_size = embed_dim
    head_size = hidden_size // num_heads

    hidden_states = ttnn.experimental.layer_norm(
        inputs_embeds,
        weight=parameters[f"transformer.word_embeddings_layernorm.weight"],
        bias=parameters[f"transformer.word_embeddings_layernorm.bias"],
    )

    batch_size, seq_length = input_ids.shape

    presents = ()

    for i in range(0, hidden_layers):
        attn_outputs = bloom_multi_head_attention(
            device,
            hidden_states,
            causal_mask,
            parameters[f"transformer.h.{i}.input_layernorm.weight"],
            parameters[f"transformer.h.{i}.input_layernorm.bias"],
            parameters[f"transformer.h.{i}.self_attention.query_key_value.weight"],
            parameters[f"transformer.h.{i}.self_attention.query_key_value.bias"],
            parameters[f"transformer.h.{i}.self_attention.dense.weight"],
            parameters[f"transformer.h.{i}.self_attention.dense.bias"],
            alibi,
            head_size=head_size,
        )
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        layernorm_output = ttnn.experimental.layer_norm(
            attention_output,
            weight=parameters[f"transformer.h.{i}.post_attention_layernorm.weight"],
            bias=parameters[f"transformer.h.{i}.post_attention_layernorm.bias"],
        )

        # MLP.
        output_tensor = mlp(
            attention_output,
            layernorm_output,
            parameters[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"],
            parameters[f"transformer.h.{i}.mlp.dense_h_to_4h.bias"],
            parameters[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"],
            parameters[f"transformer.h.{i}.mlp.dense_4h_to_h.bias"],
        )
        outputs = (output_tensor,) + outputs
        hidden_states = outputs[0]
        presents = presents + (outputs[1],)

    hidden_states = ttnn.experimental.layer_norm(
        hidden_states,
        weight=parameters[f"transformer.ln_f.weight"],
        bias=parameters[f"transformer.ln_f.bias"],
    )
    return hidden_states


def bloom_for_causal_lm_forward(input_ids, causal_mask, alibi, num_heads, hidden_layers, parameters, device):
    # start
    start = time.time()
    hidden_states = ttnn_bloom(input_ids, causal_mask, alibi, num_heads, hidden_layers, parameters, device)
    # Unfortuntely we do not have the ability to handle large tensors yet.  So this is a workaround.
    hidden_states = ttnn.from_device(hidden_states)
    # stop
    end = time.time()

    batch_size, _ = input_ids.shape
    duration = end - start
    logger.info(f"Duration: {duration}")
    logger.info(f"Samples per second: {1 / duration * batch_size}")

    hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.to_torch(hidden_states)
    # returning the logits
    return hidden_states @ parameters[f"lm_head.weight"]


def ttnn_bloom_preprocess_inputs(
    input_ids,
    attention_mask,
    **kwargs,
):
    num_heads = kwargs["num_heads"]
    current_size = attention_mask.shape[-1]
    batch_size, seq_length = attention_mask.shape

    alibi = build_alibi_tensor(attention_mask, num_heads, dtype=torch.float)
    alibi = alibi.unsqueeze(1)
    alibi = alibi.expand(batch_size * num_heads, 1, seq_length, seq_length)
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16)
    alibi = ttnn.to_device(alibi, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    mask = torch.empty((seq_length, seq_length), dtype=torch.bool)
    seq_ids = torch.arange(seq_length)
    mask[:, 0:] = seq_ids[:, None] < seq_ids[None, :]
    past_key_values_length = 0
    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    causal_mask = mask[None, None, :, :].expand(1, batch_size * num_heads, seq_length, seq_length)
    causal_mask = ttnn.from_torch(causal_mask.float(), dtype=ttnn.bfloat16)
    causal_mask = ttnn.to_device(causal_mask, kwargs["device"])

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)
    input_ids = ttnn.to_device(input_ids, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    return input_ids, causal_mask, alibi


def generate_next_token(
    input_ids, attention_mask, num_heads, hidden_layers, parameters, logits_processor, token_length, device
):
    input_ids, causal_mask, alibi = ttnn_bloom_preprocess_inputs(
        input_ids, attention_mask, device=device, num_heads=num_heads
    )
    logits = bloom_for_causal_lm_forward(input_ids, causal_mask, alibi, num_heads, hidden_layers, parameters, device)
    next_token_logits = logits[:, token_length - 1, :]  # Get the logits for the last token
    processed_logits = logits_processor(input_ids, next_token_logits)
    next_token = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)
    return next_token


def is_to_be_converted(torch_model, full_name):
    return True


def generate_text_with_functional_approach(text, torch_model, tokenizer, device, max_length=64):
    num_heads = torch_model.config.n_head
    hidden_layers = torch_model.config.n_layer
    torch_model_config = torch_model.config

    parameters_config = ParametersConfig(
        linear_weight_dtype=ttnn.bfloat16,
        linear_bias_dtype=ttnn.bfloat16,
        layernorm_parameter_dtype=ttnn.bfloat16,
    )
    parameters = preprocess_model_parameters(
        f"ttnn-functional-bloom",
        parameters_config,
        torch_model,
        is_to_be_converted=is_to_be_converted,
        device=device,
        use_model_cache=True,
    )
    # Using torch when multipling the weights because we do not support large tensors yet
    parameters[f"lm_head.weight"] = torch_model.state_dict()[f"lm_head.weight"].T.type(torch.bfloat16)

    # Tokenize the input text and get initial input_ids
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Initialize logits processor based on the model's configuration
    logits_processor = generation_utils.get_logits_processor(input_ids, torch_model_config)

    generated = input_ids
    attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long)
    token_length = generated.shape[-1]
    padding_needed = (32 - (token_length % 32)) % 32
    generated_with_padding = F.pad(generated, (0, padding_needed, 0, 0))
    attention_mask = F.pad(attention_mask, (0, padding_needed, 0, 0))
    while token_length < max_length:
        next_token = generate_next_token(
            generated_with_padding,
            attention_mask,
            num_heads,
            hidden_layers,
            parameters,
            logits_processor,
            token_length,
            device,
        )

        # Check if the next token is the end-of-sequence token
        if next_token == tokenizer.eos_token_id:
            break

        generated = torch.cat((generated, next_token), dim=1)
        logger.debug(f"Building : `{tokenizer.decode(generated.squeeze(), skip_special_tokens=True)}`")
        token_length = token_length + 1
        padding_needed = (32 - (token_length % 32)) % 32
        generated_with_padding = F.pad(generated, (0, padding_needed, 0, 0))

    return tokenizer.decode(generated.squeeze(), skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    torch_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

    input_text = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    expected_generated_text = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information. You can also add a few more sentences to the summary. The summary is a great way to get a quick"

    device_id = 0
    device = ttnn.open(device_id)
    update_process_id()
    generated_text = generate_text_with_functional_approach(input_text, torch_model, tokenizer, device)
    ttnn.close(device)

    logger.info(generated_text)
