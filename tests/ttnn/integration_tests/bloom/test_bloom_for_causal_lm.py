# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from transformers import BloomConfig, BloomForCausalLM, BloomTokenizerFast


from models import generation_utils
from models.experimental.functional_bloom.reference import torch_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import skip_for_wormhole_b0

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    ParametersConfig,
)


def generate_next_token(model, input_ids, parameters, num_heads, hidden_layers, logits_processor, max_length, **kwargs):
    num_tokens = input_ids.shape[-1]
    padded_input_ids, alibi, causal_mask = model.preprocess_inputs(
        input_ids=input_ids,
        num_heads=num_heads,
        max_length=max_length,
        attention_mask=None,
        **kwargs,
    )

    logits = model.bloom_for_causal_lm(padded_input_ids, alibi, causal_mask, parameters, num_heads, hidden_layers)
    next_token_logits = logits[:, num_tokens - 1, :]  # Get the logits for the last token
    processed_logits = logits_processor(input_ids, next_token_logits)
    next_token = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)
    return next_token


def generate_text(
    model,
    input_ids,
    parameters,
    tokenizer,
    logits_processor,
    num_heads,
    hidden_layers,
    num_tokens_to_decode,
    max_length=384,
    **kwargs,
):
    # Tokenize the input text and get initial input_ids

    for _ in range(num_tokens_to_decode):
        next_token = generate_next_token(
            model,
            input_ids,
            parameters,
            num_heads,
            hidden_layers,
            logits_processor,
            max_length,
            **kwargs,
        )

        # Check if the next token is the end-of-sequence token
        if torch.all(next_token == tokenizer.eos_token_id):
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)
        logger.debug(f"Building : {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# Verify that the torch functional model matches exactly the default model.
def test_torch_bloom_for_causal_lm():
    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    model = BloomForCausalLM.from_pretrained(model_name).eval()

    input_text = "Hello, my dog is cute"
    expected_generated_text = "Hello, my dog is cute. He is a little shy, but he loves"

    # Initialize logits processor based on the model's configuration
    num_heads = config.n_head
    hidden_layers = config.n_layer

    parameters = torch_functional_bloom.preprocess_parameters(model.state_dict(), num_heads)

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    logits_processor = generation_utils.get_logits_processor(input_ids, config)

    generated_text = generate_text(
        torch_functional_bloom,
        input_ids,
        parameters,
        tokenizer,
        logits_processor,
        num_heads,
        hidden_layers,
        num_tokens_to_decode=10,
    )
    assert expected_generated_text == generated_text


@skip_for_wormhole_b0()
def test_ttnn_bloom_for_causal_lm(device, batch_size=8):
    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    model = BloomForCausalLM.from_pretrained(model_name).eval()

    input_text = "Hello, my dog is cute"
    expected_generated_text = "Hello, my dog is cute and sweet. He loves to play with me and"

    num_heads = config.n_head
    hidden_layers = config.n_layer

    parameters_config = ParametersConfig(
        linear_weight_dtype=ttnn.bfloat16,
        linear_bias_dtype=ttnn.bfloat16,
        layernorm_parameter_dtype=ttnn.bfloat16,
    )
    parameters = preprocess_model_parameters(
        f"ttnn-functional-bloom-for-causal-lm",
        "version_0",
        parameters_config,
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
    )
    parameters[f"lm_head.weight"] = model.state_dict()[f"lm_head.weight"].T.to(torch.float32)

    # Initialize logits processor based on the model's configuration
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.expand((batch_size, input_ids.shape[-1]))
    logits_processor = generation_utils.get_logits_processor(input_ids, config)

    generated_text = generate_text(
        ttnn_optimized_functional_bloom,
        input_ids,
        parameters,
        tokenizer,
        logits_processor,
        num_heads,
        hidden_layers,
        num_tokens_to_decode=10,
        device=device,
    )
    assert expected_generated_text == generated_text
