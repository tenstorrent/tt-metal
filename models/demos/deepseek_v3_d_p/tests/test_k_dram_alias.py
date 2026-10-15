# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import glob
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.test_k_dram_hammer import (
    BANKS,
    BLOCK_SIZE,
    BUF_BYTES,
    BUF_WORDS,
    KVPE_DIM,
    NUM_TILES,
    PAD_PAGE_BYTES,
    REPO,
    TILE_BYTES,
)

DELTAS = [int(v, 0) for v in os.environ.get("ALIAS_DELTAS", "0,0x4000000,0x2000000,0x1000000").split(",")]

# Reads the K tiles once at their own address and at K minus each delta, with the same per-bank
# NoC address the K reads use, so a bad K read can be matched against what sits at the candidate alias.
KERNEL = f"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {{
    const uint32_t k_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_deltas = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto k_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto k = TensorAccessor(k_args, k_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t buf = get_write_ptr(cb);

    for (uint32_t d = 0; d < num_deltas; ++d) {{
        const uint32_t delta = get_arg_val<uint32_t>(3 + d);
        for (uint32_t t = 0; t < {NUM_TILES}; ++t) {{
            noc_async_read(k.get_noc_addr(t) - delta, buf + t * {TILE_BYTES}, {TILE_BYTES});
        }}
        noc_async_read_barrier();
        noc_async_write(buf, out.get_noc_addr(d), {BUF_BYTES});
        noc_async_write_barrier();
    }}
}}
"""


def test_k_dram_alias():
    chip = int(os.environ.get("REPRO_CHIP", 26))
    k_addr = int(os.environ.get("REPRO_K_ADDR", "0x4003880"), 0)
    hits = os.environ.get("ALIAS_HITS", str(REPO / "repro_logs" / "hammer" / "**" / "*.pt"))

    device = ttnn.open_device(device_id=chip)
    try:
        torch.manual_seed(0)
        probe = ttnn.from_torch(
            torch.zeros(1, BANKS * PAD_PAGE_BYTES // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        free_addr = probe.buffer_address()
        ttnn.deallocate(probe)
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
        assert tt_k.buffer_address() == k_addr, f"K landed at {hex(tt_k.buffer_address())}"
        tt_out = ttnn.from_torch(
            torch.zeros(1, 1, len(DELTAS), BUF_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [tt_k.buffer_address(), tt_out.buffer_address(), len(DELTAS)] + DELTAS
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,
            compile_time_args=[0]
            + ttnn.TensorAccessorArgs(tt_k).get_compile_time_args()
            + ttnn.TensorAccessorArgs(tt_out).get_compile_time_args(),
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        )
        cb = ttnn.CBDescriptor(
            total_size=BUF_BYTES,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=BUF_BYTES)],
        )
        ttnn.generic_op([tt_k, tt_out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
        reads = ttnn.to_torch(tt_out).reshape(len(DELTAS), NUM_TILES, -1).to(torch.int64) & 0xFFFFFFFF
        ttnn.deallocate(pad)
    finally:
        ttnn.close_device(device)

    torch.save({"deltas": DELTAS, "reads": reads}, REPO / "repro_logs" / "hammer" / f"alias_chip{chip}.pt")
    lane3 = lambda x: (x >> 24) & 0xFF
    for f in sorted(glob.glob(hits, recursive=True)):
        d = torch.load(f)
        bad = (d["bad"].to(torch.int64) & 0xFFFFFFFF).reshape(NUM_TILES, -1)
        gold = (d["gold"].to(torch.int64) & 0xFFFFFFFF).reshape(NUM_TILES, -1)
        for t in [t for t in range(NUM_TILES) if not torch.equal(bad[t], gold[t])]:
            match = {
                hex(delta): f"{int((lane3(reads[i, t]) == lane3(bad[t])).sum())}/{bad.shape[1]}"
                for i, delta in enumerate(DELTAS)
            }
            logger.info(f"{Path(f).name} tile {t} (bank {t % BANKS}): lane-3 bytes matching K-delta read {match}")
