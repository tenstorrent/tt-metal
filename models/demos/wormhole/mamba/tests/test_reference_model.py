# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from typing import Optional
from transformers import AutoTokenizer

from models.demos.wormhole.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.wormhole.mamba.reference.prefill_decode_model import Mamba as MambaPrefillDecode
from models.demos.wormhole.mamba.reference.model import Mamba


def generate_through_selective_scan(
    model, tokenizer, prompt: str, n_tokens_to_gen: int = 51, sample: bool = False, top_k: Optional[int] = None
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, n_tokens_to_gen)
    return tokenizer.batch_decode(output)[0]


def generate_through_decode(model, tokenizer, prompt: str, n_tokens_to_gen: int = 51):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, n_tokens_to_gen)
    return tokenizer.batch_decode(output)[0]


def generate_through_prefill_decode(model, tokenizer, prompt: str, n_tokens_to_gen: int = 51):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, n_tokens_to_gen)
    return tokenizer.batch_decode(output)[0]


@pytest.mark.parametrize(
    "model_version, batch, genlen",
    (
        ("state-spaces/mamba-130m", 1, 32),
        ("state-spaces/mamba-370m", 1, 32),
    ),
)
def test_cpu_reference_model_decode_vs_selective_scan(
    model_version: MambaPretrainedModelName,
    batch: int,
    genlen: int,
):
    prompt = "What's better, a CPU or a GPU?"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    selective_scan_model = Mamba.from_pretrained(model_version)
    decode_model = MambaDecode.from_pretrained(model_version)
    prefill_decode_model = MambaPrefillDecode.from_pretrained(model_version)

    prefill_decode_output = generate_through_prefill_decode(prefill_decode_model, tokenizer, prompt, genlen)
    selective_scan_output = generate_through_selective_scan(selective_scan_model, tokenizer, prompt, genlen)
    decode_output = generate_through_decode(decode_model, tokenizer, prompt, genlen)

    assert selective_scan_output == decode_output == prefill_decode_output, "Model outputs should match"
