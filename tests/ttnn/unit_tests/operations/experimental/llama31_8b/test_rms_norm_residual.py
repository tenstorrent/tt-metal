# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


DIM = 4096
EPS = 1e-5


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole program")
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("scale", [1.0, 8.0])
@pytest.mark.parametrize("residual_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_rms_norm_residual_matches_add_then_rms_norm(device, seq_len, scale, residual_dtype):
    torch.manual_seed(seq_len + int(scale))
    x = torch.randn(1, 1, seq_len, DIM) * scale
    r = torch.randn(1, 1, seq_len, DIM) * scale
    w = 1.0 + 0.1 * torch.randn(DIM)

    dram = dict(layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, **dram)
    tr = ttnn.from_torch(r, dtype=residual_dtype, **dram)
    tw = ttnn.from_torch(
        w.reshape(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    h_ref = ttnn.add(tx, tr, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_ref = ttnn.rms_norm(h_ref, epsilon=EPS, weight=tw, compute_kernel_config=config)
    out, h = ttnn.experimental.llama31_8b.rms_norm_residual(tx, tr, tw, EPS, compute_kernel_config=config)

    assert torch.equal(ttnn.to_torch(h), ttnn.to_torch(h_ref))
    assert torch.equal(ttnn.to_torch(out), ttnn.to_torch(out_ref))


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole program")
def test_rms_norm_residual_program_cache(device):
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )
    dram = dict(layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tw = ttnn.from_torch(
        torch.ones(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    entries = None
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        tx = ttnn.from_torch(torch.randn(1, 1, 1024, DIM), dtype=ttnn.bfloat16, **dram)
        tr = ttnn.from_torch(torch.randn(1, 1, 1024, DIM), dtype=ttnn.bfloat8_b, **dram)
        h_ref = ttnn.add(tx, tr, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_ref = ttnn.rms_norm(h_ref, epsilon=EPS, weight=tw, compute_kernel_config=config)
        out, h = ttnn.experimental.llama31_8b.rms_norm_residual(tx, tr, tw, EPS, compute_kernel_config=config)
        if entries is None:
            entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries
        assert torch.equal(ttnn.to_torch(h), ttnn.to_torch(h_ref))
        assert torch.equal(ttnn.to_torch(out), ttnn.to_torch(out_ref))


def _add_then_norm(tx, tr, tw, config):
    if ttnn.experimental.llama31_8b.rms_norm_residual_supported(tx, tr):
        return ttnn.experimental.llama31_8b.rms_norm_residual(tx, tr, tw, EPS, compute_kernel_config=config)
    h = ttnn.add(tx, tr, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.rms_norm(h, epsilon=EPS, weight=tw, compute_kernel_config=config), h


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole program")
@pytest.mark.parametrize("residual_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_rms_norm_residual_falls_back_when_l1_is_occupied(device, residual_dtype, expect_error):
    torch.manual_seed(7)
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )
    dram = dict(layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tx = ttnn.from_torch(torch.randn(1, 1, 1024, DIM), dtype=ttnn.bfloat16, **dram)
    tr = ttnn.from_torch(torch.randn(1, 1, 1024, DIM), dtype=residual_dtype, **dram)
    tw = ttnn.from_torch(
        torch.ones(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert ttnn.experimental.llama31_8b.rms_norm_residual_supported(tx, tr)
    held = ttnn.from_torch(
        torch.zeros(1, 1, 4096, DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    assert not ttnn.experimental.llama31_8b.rms_norm_residual_supported(tx, tr)
    with expect_error(RuntimeError, "not enough free L1"):
        ttnn.experimental.llama31_8b.rms_norm_residual(tx, tr, tw, EPS, compute_kernel_config=config)
    h_ref = ttnn.add(tx, tr, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_ref = ttnn.rms_norm(h_ref, epsilon=EPS, weight=tw, compute_kernel_config=config)
    out, h = _add_then_norm(tx, tr, tw, config)
    assert torch.equal(ttnn.to_torch(h), ttnn.to_torch(h_ref))
    assert torch.equal(ttnn.to_torch(out), ttnn.to_torch(out_ref))
    assert torch.equal(ttnn.to_torch(held), torch.zeros(1, 1, 4096, DIM, dtype=torch.bfloat16))
    ttnn.deallocate(held)
    assert ttnn.experimental.llama31_8b.rms_norm_residual_supported(tx, tr)


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole program")
def test_rms_norm_residual_rejects_shapes_outside_the_table(device, expect_error):
    dram = dict(layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tx = ttnn.from_torch(torch.randn(1, 1, 128, DIM), dtype=ttnn.bfloat16, **dram)
    tr = ttnn.from_torch(torch.randn(1, 1, 128, DIM), dtype=ttnn.bfloat8_b, **dram)
    tw = ttnn.from_torch(
        torch.ones(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "llama31_8b.rms_norm_residual"):
        ttnn.experimental.llama31_8b.rms_norm_residual(tx, tr, tw, EPS)
