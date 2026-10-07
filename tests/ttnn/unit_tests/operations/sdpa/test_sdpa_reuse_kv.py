# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``scaled_dot_product_attention(..., reuse_kv=True)`` keeps K/V in a core's CBs across its consecutive Q chunks of
the same (batch, KV head) instead of re-reading them, and builds no K/V chains. Only where K/V come from changes, so
the output must be bit-identical to ``reuse_kv=False``. This is a subset that fits Wormhole and Blackhole grids; the
full sweep (including 12x10) is tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_reuse_kv_sweep.py."""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.reuse_kv_test_utils import check_reuse_kv


# (b, nh, nkv, s, d): GQA 32/8 at batch 4 and a group of 8 at a smaller head dim
@pytest.mark.parametrize("b, nh, nkv, s, d", [(4, 32, 8, 512, 128), (2, 16, 2, 256, 64)])
# 8x7: a core's Q chunks span two KV heads; 8x4: whole KV heads per core
@pytest.mark.parametrize("grid", [(8, 7), (8, 4)])
@pytest.mark.parametrize("q_chunk", [256, 128])
@pytest.mark.parametrize("concat, pack", [(False, False), (True, True)], ids=["head_major", "packed_concat"])
@pytest.mark.timeout(600)
def test_sdpa_reuse_kv(device, b, nh, nkv, s, d, grid, q_chunk, concat, pack):
    check_reuse_kv(device, b, nh, nkv, s, d, grid, q_chunk, concat, pack, torch_reference=not concat and not pack)


def test_sdpa_reuse_kv_rejects_unsupported(device, expect_error):
    q, k, v = (
        ttnn.from_torch(torch.randn(1, n, 256, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for n in (8, 2, 2)
    )
    with expect_error(RuntimeError, "reuse_kv supports non-causal"):
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, reuse_kv=True)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=128, k_chunk_size=128
    )
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False)
    with expect_error(RuntimeError, "reuse_kv needs a single K chunk"):
        ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, program_config=pc, compute_kernel_config=ck, reuse_kv=True
        )
