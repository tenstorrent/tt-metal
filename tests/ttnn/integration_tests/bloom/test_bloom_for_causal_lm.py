# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import BloomConfig, BloomForCausalLM, BloomTokenizerFast


from models import generation_utils
from models.demos.grayskull.functional_bloom.reference import torch_functional_bloom
from models.demos.grayskull.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import is_wormhole_b0, skip_for_grayskull, is_blackhole

from ttnn.model_preprocessing import preprocess_model_parameters


def generate_next_token(config, model, input_ids, parameters, logits_processor, max_length, **kwargs):
    num_tokens = input_ids.shape[-1]
    padded_input_ids, alibi, causal_mask = model.preprocess_inputs(
        input_ids=input_ids,
        num_heads=config.n_head,
        max_length=max_length,
        attention_mask=None,
        **kwargs,
    )

    logits = model.bloom_for_causal_lm(config, padded_input_ids, alibi, causal_mask, parameters=parameters)
    next_token_logits = logits[:, num_tokens - 1, :]  # Get the logits for the last token
    processed_logits = logits_processor(input_ids, next_token_logits)
    next_token = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)
    return next_token


def generate_text(
    config,
    model,
    input_ids,
    parameters,
    tokenizer,
    logits_processor,
    num_tokens_to_decode,
    max_length=384,
    **kwargs,
):
    # Tokenize the input text and get initial input_ids

    for _ in range(num_tokens_to_decode):
        next_token = generate_next_token(
            config,
            model,
            input_ids,
            parameters,
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


def test_torch_bloom_for_causal_lm():
    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    input_text = "Hello, my dog is cute"
    expected_generated_text = "Hello, my dog is cute. He is a little shy, but he loves"

    parameters = preprocess_model_parameters(
        model_name="torch_functional_bloom_for_causal_lm",
        initialize_model=lambda: BloomForCausalLM.from_pretrained(model_name).eval(),
        custom_preprocessor=torch_functional_bloom.custom_preprocessor,
        convert_to_ttnn=lambda *_: False,
    )

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Initialize logits processor based on the model's configuration
    logits_processor = generation_utils.get_logits_processor(input_ids, config)

    generated_text = generate_text(
        config,
        torch_functional_bloom,
        input_ids,
        parameters,
        tokenizer,
        logits_processor,
        num_tokens_to_decode=10,
    )
    assert expected_generated_text == generated_text


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@skip_for_grayskull(reason_str="#10797: OOM")
def test_ttnn_bloom_for_causal_lm(device, batch_size=8):
    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    input_text = "Hello, my dog is cute"
    expected_generated_text = "Hello, my dog is cute and sweet. He loves to play with me and"

    parameters = preprocess_model_parameters(
        model_name="ttnn_functional_bloom_for_causal_lm",
        initialize_model=lambda: BloomForCausalLM.from_pretrained(model_name).eval(),
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )

    # Initialize logits processor based on the model's configuration
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.expand((batch_size, input_ids.shape[-1]))
    logits_processor = generation_utils.get_logits_processor(input_ids, config)

    generated_text = generate_text(
        config,
        ttnn_optimized_functional_bloom,
        input_ids,
        parameters,
        tokenizer,
        logits_processor,
        num_tokens_to_decode=10,
        device=device,
    )
    assert expected_generated_text == generated_text
