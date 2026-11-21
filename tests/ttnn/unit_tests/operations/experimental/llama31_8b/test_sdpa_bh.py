# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.device_utils import is_blackhole

SHAPES = [(1024, 64, 64)]


def _require_p100(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Blackhole P100 only")


def _configs(device, q_chunk, k_chunk):
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        exp_approx_mode=False,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    return program_config, compute_kernel_config


def _inputs(device, seq, seed):
    torch.manual_seed(seed)
    tensors = []
    for heads in (32, 8, 8):
        t = torch.randn(1, heads, seq, 128)
        tensors.append(
            ttnn.from_torch(
                t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        )
    return tensors


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.parametrize("seq, q_chunk, k_chunk", SHAPES)
def test_sdpa_bh_matches_generic(device, seq, q_chunk, k_chunk):
    _require_p100(device)
    if device.compute_with_storage_grid_size().x < 11 or device.compute_with_storage_grid_size().y < 10:
        pytest.skip("needs an 11x10 grid")
    program_config, compute_kernel_config = _configs(device, q_chunk, k_chunk)
    for i, seed in enumerate((0, 1)):
        q, k, v = _inputs(device, seq, seed)
        assert ttnn.experimental.llama31_8b.sdpa_bh_supported(
            q, k, v, program_config, compute_kernel_config=compute_kernel_config
        )
        entries = device.num_program_cache_entries()
        actual = ttnn.experimental.llama31_8b.sdpa_bh(
            q, k, v, program_config, compute_kernel_config=compute_kernel_config
        )
        if i:
            assert device.num_program_cache_entries() == entries
        expected = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
        )
        assert torch.equal(ttnn.to_torch(actual), ttnn.to_torch(expected))


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.parametrize("seq, q_chunk, k_chunk", SHAPES)
def test_sdpa_bh_concat_heads_matches_generic(device, seq, q_chunk, k_chunk):
    if device.compute_with_storage_grid_size().x < 11 or device.compute_with_storage_grid_size().y < 10:
        pytest.skip("needs an 11x10 grid")
    program_config, compute_kernel_config = _configs(device, q_chunk, k_chunk)
    for i, seed in enumerate((0, 1)):
        q, k, v = _inputs(device, seq, seed)
        entries = device.num_program_cache_entries()
        actual = ttnn.experimental.llama31_8b.sdpa_bh(
            q, k, v, program_config, compute_kernel_config=compute_kernel_config, output_concat_heads=True
        )
        if i:
            assert device.num_program_cache_entries() == entries
        expected = ttnn.experimental.nlp_concat_heads(
            ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert list(actual.shape) == [1, 1, seq, 32 * 128]
        assert torch.equal(ttnn.to_torch(actual), ttnn.to_torch(expected))


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
def test_sdpa_bh_rejects_other_shapes(device, expect_error):
    _require_p100(device)
    program_config, compute_kernel_config = _configs(device, 64, 64)
    torch.manual_seed(0)
    q = ttnn.from_torch(torch.randn(1, 16, 1024, 128), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(torch.randn(1, 8, 1024, 128), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    assert not ttnn.experimental.llama31_8b.sdpa_bh_supported(
        q, k, k, program_config, compute_kernel_config=compute_kernel_config
    )
    with expect_error(RuntimeError, "llama31_8b.sdpa_bh"):
        ttnn.experimental.llama31_8b.sdpa_bh(q, k, k, program_config, compute_kernel_config=compute_kernel_config)


def _matmul_inputs(device):
    torch.manual_seed(7)
    a = ttnn.from_torch(torch.randn(1, 1, 1024, 4096), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    return a, b


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.parametrize("own_first", [True, False])
def test_sdpa_bh_repeated_between_other_ops(device, own_first):
    _require_p100(device)
    program_config, compute_kernel_config = _configs(device, 64, 64)
    q, k, v = _inputs(device, 1024, 2)
    q2, k2, v2 = _inputs(device, 1024, 3)
    a, b = _matmul_inputs(device)

    def own():
        return ttnn.to_torch(
            ttnn.experimental.llama31_8b.sdpa_bh(q, k, v, program_config, compute_kernel_config=compute_kernel_config)
        )

    def generic(x, y, z):
        return ttnn.to_torch(
            ttnn.transformer.scaled_dot_product_attention(
                x, y, z, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
            )
        )

    first = own() if own_first else None
    expected = generic(q, k, v)
    if first is not None:
        assert torch.equal(first, expected)
    for _ in range(6):
        ttnn.matmul(a, b)
        assert torch.equal(own(), expected)
        generic(q2, k2, v2)
        assert torch.equal(own(), expected)
