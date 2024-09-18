# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.utility_functions import skip_for_grayskull
from models.demos.t3000.llama2_70b.tt.llama_common import setup_llama_env, check_mesh_device
from models.demos.tg.llama3_70b.tests.test_llama_model_galaxy import run_test_LlamaModel_inference


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "cluster_shape, mesh_device", [pytest.param((4, 8), (8, 4), id="4x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "llama_version",
    (("llama3-tg"),),
)
@pytest.mark.parametrize(
    "pcc, n_layers",
    ((0.99, 1),),
    ids=("1L",),
)
@pytest.mark.parametrize(
    "batch, seq_len",
    [
        (32, 1),
        #  (1, 256)
    ],
    ids=[
        "decode",
        #  "prefill"
    ],
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    ((32, 2048),),
    ids=("short_context",),
)
def test_LlamaModel_inference(
    batch,
    seq_len,
    pcc,
    n_layers,
    mesh_device,
    max_batch_size,
    max_context_len,
    llama_version,
    cluster_shape,
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

    check_mesh_device(mesh_device, model_config)

    run_test_LlamaModel_inference(
        mesh_device,
        cluster_shape,
        batch,
        seq_len,
        pcc,
        model_config,
        n_layers,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )
