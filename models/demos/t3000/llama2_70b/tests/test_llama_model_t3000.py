# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.t3000.llama2_70b.tests.test_llama_model import run_test_LlamaModel_inference
from models.demos.t3000.llama2_70b.tt.llama_common import check_mesh_device, setup_llama_env
from models.utility_functions import skip_for_grayskull

N_LAYERS_TO_PCC = {
    1: 0.99,
}


@skip_for_grayskull("Requires eth connected devices to run")
# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="See GH Issue #10317")
@pytest.mark.parametrize(
    "llama_version",
    (
        ("llama2"),
        ("llama3"),
    ),
)
@pytest.mark.parametrize("n_layers", (1,), ids=("1L",))
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1), (1, 128), (1, 2048), (1, 8192)),
    ids=("decode", "prefill_128", "prefill_2k", "prefill_8k"),
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    (
        (32, 4096),
        (16, 8192),
    ),
    ids=(
        "short_context",
        "long_context",
    ),
)
def test_LlamaModel_inference(
    batch,
    seq_len,
    n_layers,
    t3k_mesh_device,
    max_batch_size,
    max_context_len,
    llama_version,
    use_program_cache,
):
    if seq_len == 1 and batch != max_batch_size:
        pytest.skip(f"Input batch size should match max_batch_size")

    if batch == 1 and seq_len > max_context_len:
        pytest.skip(f"Prefill with seq_len={seq_len} is not supported with short context")

    if llama_version == "llama2" and seq_len > 2048:
        pytest.skip(f"Llama2 with seq_len={seq_len} is not supported (max 2048)")

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    check_mesh_device(t3k_mesh_device, model_config)

    run_test_LlamaModel_inference(
        t3k_mesh_device,
        batch,
        seq_len,
        max_batch_size,
        max_context_len,
        N_LAYERS_TO_PCC[n_layers],
        model_config,
        n_layers,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )
