from functools import partial
import numpy as np
import torch
from loguru import logger

from transformers.generation.logits_process import LogitsProcessorList

from tests.python_api_testing.models.falcon.falcon_common import MODEL_VERSION

from tests.python_api_testing.models.falcon.reference.hf_falcon_model import (
    RWForCausalLM,
)

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM


def post_process(logits, input_ids, logits_processor):
    next_token_logits = logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    return ids


def generate_next_id(causalLMModel, post_processor, input_ids):
    outputs = causalLMModel(input_ids)
    return post_processor(logits=outputs.logits, input_ids=input_ids)


def test_cpu_demo():
    logits_processor = LogitsProcessorList()
    post_processor = partial(post_process, logits_processor=logits_processor)

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
    prompt_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"

    logger.info("Tokenizing inputs")
    tokenized_inputs = tokenizer(
        prompt_text, padding=False, add_special_tokens=False, return_tensors="pt"
    )
    input_ids = tokenized_inputs["input_ids"]

    logger.info("Initializing CausalLM Model")
    causalLM = RWForCausalLM.from_pretrained(MODEL_VERSION)
    causalLM.eval()

    generator = partial(
        generate_next_id, causalLMModel=causalLM, post_processor=post_processor
    )

    logger.info("Generating new ids")
    ids = input_ids
    for i in range(50):
        logger.info(f"generating token {i}")
        ids = generator(input_ids=ids)

    logger.info(f"Input Prompt: {prompt_text}")

    logger.info("decoding to text")
    text = tokenizer.decode(ids[0])
    logger.info("Total output (including input): ")
    logger.info(text)
