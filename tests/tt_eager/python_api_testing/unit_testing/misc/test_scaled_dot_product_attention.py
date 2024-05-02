# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import tt_lib
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0


def run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    torch.manual_seed(1234)

    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"attn_mask: {attn_mask.shape}")

    tt_Q = tt_lib.tensor.Tensor(Q, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_K = tt_lib.tensor.Tensor(K, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_V = tt_lib.tensor.Tensor(V, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    tt_attn_mask = tt_lib.tensor.Tensor(attn_mask, dtype).to(tt_lib.tensor.Layout.TILE).to(device)

    tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, tt_attn_mask, is_causal=True, program_config=program_config
    )
    tt_back = tt_back.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype", [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16], ids=["bfp8", "bf16"]
)
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 2048, 64],  # Falcon-40B
        [1, 71, 1, 2048, 64],  # Falcon-7B
        [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype", [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16], ids=["bfp8", "bf16"]
)
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 2048, 64],  # Falcon-40B
        [1, 71, 1, 2048, 64],  # Falcon-7B
    ),
)
def test_sdpa_tt_with_program_cache(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_program_cache):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")

    for _ in range(2):
        run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)

    assert device.num_program_cache_entries() == 1
