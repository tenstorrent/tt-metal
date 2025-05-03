# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from tests.tt_eager.python_api_testing.unit_testing.misc.test_rotary_embedding_llama import (
    run_test_rotary_embedding_llama,
)


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (32, 1),
        (16, 1),
        (8, 1),
        (1, 1),
    ),
    ids=(
        "decode_32",
        "decode_16",
        "decode_8",
        "decode_1",
    ),
)
@pytest.mark.parametrize(
    "n_heads, n_kv_heads, head_dim",
    (
        (8, 1, 128),
        (71, 32, 64),
        (8, 1, 256),
    ),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_rotary_embedding_llama_fused_qk(
    batch,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    datatype,
    pcc,
    device,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    run_test_rotary_embedding_llama(
        device, batch, seq_len, pcc, n_heads, n_kv_heads, head_dim, 1, datatype, fuse_qk=True
    )

    # shift input/output tensor by creating very small tensor between loop
    inp = torch.randn(1, 1, 32, 32)
    test_tensor = (
        ttnn.Tensor(
            inp.reshape(-1).tolist(),
            inp.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (32, 1),
        (16, 1),
        (8, 1),
        (1, 1),
    ),
    ids=(
        "decode_32",
        "decode_16",
        "decode_8",
        "decode_1",
    ),
)
@pytest.mark.parametrize(
    "n_heads, n_kv_heads, head_dim",
    ((8, 1, 128),),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_rotary_embedding_llama_fused_qk_with_program_cache(
    batch,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    datatype,
    pcc,
    device,
    use_program_cache,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    mode = "decode" if seq_len == 1 else "prefill"

    cache_tensors = []
    for _ in range(3):
        run_test_rotary_embedding_llama(
            device, batch, seq_len, pcc, n_heads, n_kv_heads, head_dim, 1, datatype, fuse_qk=True
        )

        # shift input/output tensor by creating very small tensor between loop
        inp = torch.randn(1, 1, 32, 32)
        test_tensor = (
            ttnn.Tensor(
                inp.reshape(-1).tolist(),
                inp.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

        cache_tensors.append(test_tensor)

    num_ops = 5  # untilize cos/sin + embedding + fused_qk_rope + transpose + interleaved_to_sharded

    if (batch * 2) % ttnn.TILE_SIZE != 0:
        num_ops += 1  # slice

    assert device.num_program_cache_entries() == num_ops
