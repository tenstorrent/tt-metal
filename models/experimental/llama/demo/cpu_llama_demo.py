# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.experimental.llama.llama_utils import get_logits_processor

from loguru import logger
import pytest

prompt = """Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis.
They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
Mention the large language model based product mentioned in the paragraph above:"""


@pytest.mark.parametrize(
    "prompt, num_words",
    ((prompt, 30),),
)
def test_cpu_demo(prompt, num_words):
    # set parameters =================================================================
    _tokenizer_name = "huggyllama/llama-7b"
    _llama_model_name = "huggyllama/llama-7b"

    # load llama pytorch model =======================================================
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(_llama_model_name)

    hugging_face_reference_model.eval()

    # generate real input ============================================================
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.tokenize(prompt)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    seq_length = input_ids.shape[1]

    logger.info(f"Initial prompt: {prompt}")
    logger.info(f"Initial prompt ids: {input_ids}")
    logger.info(f"Initial prompt tokens: {tokens}")

    # generate Pytorch output of num_words with generate function ====================
    logits_processor = get_logits_processor(input_ids, hugging_face_reference_model.config)

    generate_ids = hugging_face_reference_model.generate(
        input_ids, logits_processor=logits_processor, max_length=seq_length + num_words
    )

    # decode output ids
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    tokens = tokenizer.tokenize(output)

    # print pytorch generated reponse ================================================
    logger.info(f"CPU's generated tokens: {tokens}")
    logger.info(f"CPU's predicted Output: {output}")
