# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

L = ttnn.experimental.llama31_8b_spec
K = 14336
SHARD = 224


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


def ff13_out_mem_config(device):
    cores = ttnn.num_cores_to_corerangeset(K // SHARD, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (32, SHARD), ttnn.ShardOrientation.ROW_MAJOR),
    )


def pair(device, seed, scale, mem):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, 1, 32, K, generator=g) * scale
    b = torch.randn(1, 1, 32, K, generator=g) * scale
    put = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    return put(a), put(b)


def bits(t):
    return ttnn.to_torch(t).float().view(torch.int32)


def test_swiglu_decode_matches_mul_silu(device):
    mem = ff13_out_mem_config(device)
    entries = None
    for seed, scale in ((3, 2.0), (3, 12.0), (7, 2.0), (11, 12.0)):
        a, b = pair(device, seed, scale, mem)
        assert L.swiglu_decode_supported(a, b)
        ref = ttnn.mul(
            a,
            b,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=a.memory_config(),
        )
        out = L.swiglu_decode(a, b)
        if entries is None:
            entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries
        assert out.spec == ref.spec
        assert torch.equal(bits(out), bits(ref))
        for t in (a, b, ref, out):
            ttnn.deallocate(t)


def test_swiglu_decode_rejects_other_layouts(device, expect_error):
    a, b = pair(device, 3, 2.0, ttnn.L1_MEMORY_CONFIG)
    assert not L.swiglu_decode_supported(a, b)
    with expect_error(RuntimeError, "llama31_8b_spec.swiglu_decode"):
        L.swiglu_decode(a, b)
    mem = ff13_out_mem_config(device)
    c, d = pair(device, 3, 2.0, mem)
    assert not L.swiglu_decode_supported(c, b)
