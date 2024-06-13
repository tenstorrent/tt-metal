# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from typing import Optional
from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.reference.prefill_model import MambaPrefill


def generate_through_prefill(
    model, tokenizer, prompt: str, n_tokens_to_gen: int = 30, sample: bool = False, top_k: Optional[int] = None
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, n_tokens_to_gen, sample=sample, top_k=top_k)
    return [tokenizer.decode(out.tolist()) for out in output][0]


def generate_through_decode(model, tokenizer, prompt: str, n_tokens_to_gen: int = 51):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, n_tokens_to_gen)
    return tokenizer.batch_decode(output)[0]


@pytest.mark.parametrize(
    "model_version, batch, genlen",
    (("state-spaces/mamba-370m", 1, 32),),
)
def test_cpu_reference_model_decode_vs_prefill(
    model_version: MambaPretrainedModelName,
    batch: int,
    genlen: int,
):
    prompt = "Mamba is the"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    prefill_model = MambaPrefill.from_pretrained(model_version)
    decode_model = MambaDecode.from_pretrained(model_version)

    prefill_output = generate_through_prefill(prefill_model, tokenizer, prompt, genlen)
    decode_output = generate_through_decode(decode_model, tokenizer, prompt, genlen)

    logger.debug(f"Prefill output: >> '{prefill_output}'")
    logger.debug(f"Decode output: >> '{decode_output}'")

    assert len(prefill_output) == len(decode_output), "Model outputs should be the same length"
    assert prefill_output == decode_output, "Model outputs should match"
