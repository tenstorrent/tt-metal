# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
import pytest
import torch
from loguru import logger

from transformers import AutoTokenizer

from transformers import FalconForCausalLM
import time

falcon1b = "tiiuae/falcon-rw-1b"
MODEL_VERSION = "tiiuae/falcon-7b-instruct"


def post_process(logits):
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens[:, None]
    return ids


def generate_next_id(causalLMModel, post_processor, input_ids, kv_cache=None, use_cache=None):
    outputs = causalLMModel(input_ids, past_key_values=kv_cache, use_cache=use_cache)
    return (
        post_processor(logits=outputs.logits),
        outputs.past_key_values,
    )


@pytest.mark.parametrize(
    "batch_size",
    ([1, 32]),
)
def test_cpu_demo_no_kv(batch_size):
    post_processor = partial(post_process)

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
    num_tokens = 128
    prompt_text = ["Write a poem about Valencia"] * batch_size

    logger.info("Tokenizing inputs")
    tokenized_inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"]

    logger.info("Initializing CausalLM Model")
    causalLM = FalconForCausalLM.from_pretrained(MODEL_VERSION, device_map="auto", offload_folder="offload")
    causalLM.eval()

    generator = partial(generate_next_id, causalLMModel=causalLM, post_processor=post_processor)

    logger.info("Generating new ids")
    ids = input_ids
    for i in range(num_tokens):
        start_ = time.time()
        logger.info(f"generating token {i}")
        ids, kv_cache = generator(input_ids=ids)
        logger.info(f"iteration {i} duration {time.time() - start_}")

    text = tokenizer.batch_decode(ids)

    for input_text, output_text in zip(prompt_text, text):
        logger.info(f"input: {input_text}")
        logger.info(f"output: {output_text}")


@pytest.mark.parametrize(
    "batch_size",
    ([1, 32]),
)
def test_cpu_demo_kv(batch_size):
    torch.manual_seed(0)

    post_processor = partial(post_process)

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)

    num_tokens = 128

    prompt_text = ["Write a poem about Valencia"] * batch_size

    logger.info("Tokenizing inputs")
    tokenized_inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"]

    logger.info("Initializing CausalLM Model")
    causalLM = FalconForCausalLM.from_pretrained(MODEL_VERSION, device_map="auto", offload_folder="offload")
    causalLM.eval()

    generator = partial(generate_next_id, causalLMModel=causalLM, post_processor=post_processor)

    logger.info("Generating new ids")
    ids = input_ids
    generated_ids = torch.tensor(ids)

    # input 10 tokens
    ids, kv_cache = generator(input_ids=ids, use_cache=True)

    # [batch x 1]
    generated_ids = torch.concat((generated_ids, ids), dim=1)
    print("OUTPUT OF PREFILL", generated_ids)

    for i in range(num_tokens):
        start_ = time.time()
        logger.info(f"generating token {i}")
        # input:
        # kv cache of len = 10
        # ids= len 1 - new generated token
        ids, kv_cache = generator(input_ids=ids, kv_cache=kv_cache, use_cache=True)

        generated_ids = torch.concat((generated_ids, ids), dim=1)
        logger.info(f"token {i} generated in {time.time() - start_} secs")

        output_prompts = tokenizer.batch_decode(generated_ids.tolist())
        for output_prompt in output_prompts:
            logger.info(f"output::: {output_prompt}")

    generated_ids = generated_ids.tolist()
    text = tokenizer.batch_decode(generated_ids)

    for input_text, output_text in zip(prompt_text, text):
        logger.info(f"input: {input_text}")
        logger.info(f"output: {output_text}")


@pytest.mark.parametrize(
    "batch_size",
    ([1, 3, 8]),
)
def test_cpu_demo_with_kv_split(batch_size):
    post_processor = partial(post_process)

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
    num_tokens = 128
    prompts = ["Write a poem about Valencia"] * batch_size

    logger.info("Tokenizing inputs")
    input_ids = []
    for i in range(batch_size):
        tokenized_inputs = tokenizer(prompts[i], padding=False, add_special_tokens=False, return_tensors="pt")
        input_ids.append(tokenized_inputs["input_ids"])

    logger.info("Initializing CausalLM Model")
    generators = []
    for i in range(batch_size):
        causalLM = FalconForCausalLM.from_pretrained(MODEL_VERSION, device_map="auto", offload_folder="offload")
        causalLM.eval()
        generator = partial(generate_next_id, causalLMModel=causalLM, post_processor=post_processor)
        generators.append(generator)

    assert len(generators) == len(prompts)

    logger.info("prefill stage")

    prefill_cache = []
    prefill_ids = []
    for ids, generator in zip(input_ids, generators):
        gen_ids, kv_cache = generator(input_ids=ids, use_cache=True)
        prefill_cache.append(kv_cache)
        prefill_ids.append(gen_ids)

    generated_ids = torch.concat(prefill_ids, dim=0)
    ids = generated_ids[:, -1].unsqueeze(1)

    def _concat_kv_cache(prefill_cache):
        batch_size, decoder_size = len(prefill_cache), len(prefill_cache[0])
        logger.info(f"decoder size: {decoder_size}")
        results = []
        for d in range(decoder_size):
            results.append([])

            for i in [0, 1]:  # key, query
                T = []
                for batch_id in range(batch_size):
                    T.append(prefill_cache[batch_id][d][i])

                results[-1].append(torch.concat(T, dim=0))

        return results

    logger.info("concat prefilled caches")
    kv_cache = _concat_kv_cache(prefill_cache)  # 32=batch_size, 10=seq_len, xxxx)

    # tuple(tuple(tensors))
    # tuple -> num_decodersx2
    # tensor: [batch x 32 x seq_len x 64]
    logger.info("Generate tokens batched")
    generator = generators[0]
    for i in range(num_tokens):
        # iterations should be about the same length
        # each iterations is less than 2 sec (machine dependents)
        logger.info(f"generating token {i}")
        start_ = time.time()
        # ids= (32x1)
        # kv cache= 32xseq_lenx....)
        ids, kv_cache = generator(input_ids=ids, kv_cache=kv_cache, use_cache=True)
        # output
        # ids -> (32, 2) - (last otken, new token)
        # kv cache -> (32, seq_len+1, xxxx)
        ids = ids[:, -1].unsqueeze(1)
        generated_ids = torch.concat((generated_ids, ids), dim=1)
        logger.info(f"token {i} generated in {time.time() - start_} secs")

    generated_ids = generated_ids.tolist()
    text = tokenizer.batch_decode(generated_ids)

    for input_text, output_text in zip(prompts, text):
        logger.info(f"input: {input_text}")
        logger.info(f"output: {output_text}")
