# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.mamba.demo.demo import run_mamba_demo
import pytest


@pytest.mark.parametrize(
    "user_input, max_gen_len",
    ((["Hello World"], 2),),
)
@pytest.mark.skip(reason="#8146 ND Hang")
def test_demo(user_input, device, use_program_cache, max_gen_len):
    return run_mamba_demo(prompts=user_input, device=device, generated_sequence_length=max_gen_len, display=False)
