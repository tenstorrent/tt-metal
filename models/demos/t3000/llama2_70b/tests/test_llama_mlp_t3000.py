# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.utility_functions import skip_for_grayskull
from models.demos.t3000.llama2_70b.tests.test_llama_mlp import run_test_LlamaMLP_inference
from models.demos.t3000.llama2_70b.tt.llama_common import setup_llama_env, check_mesh_device


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "llama_version",
    (("llama3"),),
)
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9994), (1, 128, 0.9996), (1, 2048, 0.9996), (1, 8192, 0.9996)),
    ids=("decode", "prefill_128", "prefill_2k", "prefill_8k"),
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    (
        (32, 2048),
        # (16, 8192),
    ),
    ids=(
        "short_context",
        # "long_context",
    ),
)
def test_LlamaMLP_inference_t3000(
    batch,
    seq_len,
    pcc,
    t3k_mesh_device,
    max_batch_size,
    max_context_len,
    llama_version,
    use_program_cache,
):
    if batch > max_batch_size:
        pytest.skip(f"Decode with {batch} users is not supported with large context")

    if batch == 1 and seq_len > max_context_len:
        pytest.skip(f"Prefill with {seq_len=} is not supported with short context")

    if llama_version == "llama2" and seq_len > 2048:
        pytest.skip(f"Llama2 with {seq_len=} is not supported (max 2048)")

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        batch=batch,
        seq_len=seq_len,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    check_mesh_device(t3k_mesh_device, model_config)
    run_test_LlamaMLP_inference(
        t3k_mesh_device,
        batch,
        seq_len,
        pcc,
        model_config,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )
