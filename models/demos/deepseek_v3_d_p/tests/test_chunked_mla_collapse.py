# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

HEADS = 8
Q_LEN = 640
KVPE_DIM = 576
KV_LORA_RANK = 512
NUM_BLOCKS = 4000
BLOCK_SIZE = 64
CHUNK_START = 254080
SCALE = 0.14467962580268923
BANKS = 8
PAD_PAGE_BYTES = 64
MAX_ABS = 1e6


@pytest.mark.timeout(0)
def test_chunked_mla_collapse():
    chip = int(os.environ.get("REPRO_CHIP", 26))
    k_addr = int(os.environ.get("REPRO_K_ADDR", "0x4003880"), 0)
    iters = int(os.environ.get("REPRO_ITERS", 1000))
    device = ttnn.open_device(device_id=chip)
    hit = None
    try:
        torch.manual_seed(0)
        tt_q = ttnn.from_torch(
            torch.randn(1, HEADS, Q_LEN, KVPE_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = ttnn.from_torch(
            torch.zeros(1, NUM_BLOCKS, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        probe = ttnn.from_torch(
            torch.zeros(1, BANKS * PAD_PAGE_BYTES // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        free_addr = probe.buffer_address()
        ttnn.deallocate(probe)
        pad = None
        if k_addr > free_addr:
            pad = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, (k_addr - free_addr) // PAD_PAGE_BYTES * BANKS, PAD_PAGE_BYTES // 4]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_k = ttnn.from_torch(
            torch.randn(1, 1, BLOCK_SIZE, KVPE_DIM),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.info(f"chip {chip}: single K block at {hex(tt_k.buffer_address())}")
        assert pad is None or tt_k.buffer_address() == k_addr, f"K landed at {hex(tt_k.buffer_address())}"
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(3, 1),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        t0 = time.perf_counter()
        for it in range(iters):
            out = ttnn.transformer.chunked_flash_mla_prefill(
                tt_q,
                tt_k,
                KV_LORA_RANK,
                tt_page_table,
                chunk_start_idx=CHUNK_START,
                scale=SCALE,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            rows = ttnn.to_torch(out).float().abs().reshape(-1, KV_LORA_RANK).amax(dim=-1)
            ttnn.deallocate(out)
            bad = (rows > MAX_ABS).nonzero().flatten()
            if bad.numel():
                hit = it
                logger.error(
                    f"HIT iter {it} at {time.perf_counter() - t0:.0f}s: {bad.numel()} rows, "
                    f"max {rows.max().item():.4g}, rows {bad[0].item()}..{bad[-1].item()}"
                )
                break
            if (it + 1) % 200 == 0:
                logger.info(f"{it + 1}/{iters} iters clean at {time.perf_counter() - t0:.0f}s")
        if pad is not None:
            ttnn.deallocate(pad)
    finally:
        ttnn.close_device(device)

    assert hit is None, f"chunked_flash_mla_prefill output exceeded {MAX_ABS:g} at iteration {hit}"
