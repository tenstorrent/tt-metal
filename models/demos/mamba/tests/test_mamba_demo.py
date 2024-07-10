# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.mamba.demo.demo import run_mamba_demo
from models.demos.mamba.tests.test_reference_model import (
    generate_through_selective_scan,
    generate_through_prefill_decode,
)
from models.demos.mamba.reference.prefill_decode_model import Mamba as MambaPrefillDecode
from models.demos.mamba.reference.model import Mamba
import pytest
from transformers import AutoTokenizer


@pytest.mark.parametrize(
    "user_input, model_version, max_gen_len",
    ((["Hello World"], "state-spaces/mamba-2.8b-slimpj", 2),),
)
def test_tt_demo(user_input, model_version, device, use_program_cache, get_tt_cache_path, max_gen_len):
    return run_mamba_demo(
        prompts=user_input,
        model_version=model_version,
        device=device,
        generated_sequence_length=max_gen_len,
        display=False,
        cache_dir=get_tt_cache_path(model_version),
    )


@pytest.mark.parametrize(
    "user_input, model_version, batch_size, max_gen_len",
    ((["Hello World"], "state-spaces/mamba-2.8b-slimpj", 32, 2),),
)
def test_reference_demo(user_input, model_version, batch_size, max_gen_len):
    if len(user_input) != batch_size:
        user_input = [user_input[0]] * batch_size

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    selective_scan_model = Mamba.from_pretrained(model_version)
    prefill_decode_model = MambaPrefillDecode.from_pretrained(model_version, batch_size)

    prefill_decode_outputs = generate_through_prefill_decode(prefill_decode_model, tokenizer, user_input, max_gen_len)
    selective_scan_outputs = generate_through_selective_scan(selective_scan_model, tokenizer, user_input, max_gen_len)

    for user_idx in range(batch_size):
        assert (
            selective_scan_outputs[user_idx] == prefill_decode_outputs[user_idx]
        ), f"Model outputs should match for user {user_idx}"
