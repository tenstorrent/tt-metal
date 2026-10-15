# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

KVPE_DIM = 576
BLOCK_SIZE = 64
BANKS = 8
PAD_PAGE_BYTES = 64
TILE_BYTES = 1088  # bfp8_b tile
NUM_TILES = (BLOCK_SIZE // 32) * (KVPE_DIM // 32)
BUF_BYTES = NUM_TILES * TILE_BYTES
BUF_WORDS = BUF_BYTES // 4
HDR_WORDS = 16
ROW_WORDS = HDR_WORDS + 2 * BUF_WORDS  # header, first bad copy, golden
NONE = 0xFFFFFFFF

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(os.environ.get("REPRO_OUT_DIR", REPO / "repro_logs" / "hammer"))

# Rereads the same K tiles the SDPA reader rereads, in the same tile order, and checks every word in L1
# against the first read. The first read is the golden; the host checks it is identical on every RISC.
KERNEL = f"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {{
    const uint32_t k_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t iters = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto k_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto k = TensorAccessor(k_args, k_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t base = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint32_t* bad = hdr + {HDR_WORDS};
    volatile tt_l1_ptr uint32_t* gold = bad + {BUF_WORDS};
    volatile tt_l1_ptr uint32_t* work = gold + {BUF_WORDS};
    const uint32_t gold_addr = base + {HDR_WORDS * 4 + BUF_BYTES};
    const uint32_t work_addr = gold_addr + {BUF_BYTES};

    for (uint32_t t = 0; t < {NUM_TILES}; ++t) {{
        noc_async_read(k.get_noc_addr(t), gold_addr + t * {TILE_BYTES}, {TILE_BYTES});
    }}
    noc_async_read_barrier();

    uint32_t bad_iters = 0, first_bad = {NONE}, first_words = 0, max_words = 0, last_bad = {NONE};
    for (uint32_t it = 0; it < iters; ++it) {{
        for (uint32_t t = 0; t < {NUM_TILES}; ++t) {{
            noc_async_read(k.get_noc_addr(t), work_addr + t * {TILE_BYTES}, {TILE_BYTES});
        }}
        noc_async_read_barrier();
        uint32_t n = 0;
        for (uint32_t w = 0; w < {BUF_WORDS}; ++w) {{
            n += work[w] != gold[w];
        }}
        if (n) {{
            if (first_bad == {NONE}) {{
                first_bad = it;
                first_words = n;
                for (uint32_t w = 0; w < {BUF_WORDS}; ++w) {{
                    bad[w] = work[w];
                }}
            }}
            ++bad_iters;
            last_bad = it;
            max_words = n > max_words ? n : max_words;
        }}
    }}
    hdr[0] = iters;
    hdr[1] = bad_iters;
    hdr[2] = first_bad;
    hdr[3] = first_words;
    hdr[4] = max_words;
    hdr[5] = last_bad;
    hdr[6] = 0xC0FFEE;
    noc_async_write(base, out.get_noc_addr(slot), {ROW_WORDS * 4});
    noc_async_write_barrier();
}}
"""


@pytest.mark.timeout(0)
def test_k_dram_hammer():
    chip = int(os.environ.get("REPRO_CHIP", 26))
    k_addr = int(os.environ.get("REPRO_K_ADDR", "0x4003880"), 0)
    launches = int(os.environ.get("HAMMER_LAUNCHES", 100))
    iters = int(os.environ.get("HAMMER_ITERS", 20000))
    gx, gy = (int(v) for v in os.environ.get("HAMMER_GRID", "3x1").split("x"))
    # One NoC per data-movement RISC (RISCV_1, RISCV_0). The SDPA reader reads K on NOC0 from RISCV_1.
    nocs = [int(v) for v in os.environ.get("HAMMER_NOCS", "0,1").split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        assert pad is None or tt_k.buffer_address() == k_addr, f"K landed at {hex(tt_k.buffer_address())}"
        cores = [(x, y) for y in range(gy) for x in range(gx)]
        slots = 2 * len(cores)
        tt_out = ttnn.from_torch(
            torch.zeros(1, 1, slots, ROW_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.info(
            f"chip {chip}: K at {hex(tt_k.buffer_address())}, {len(cores)} cores x 2 RISCs (RISCV_1 noc{nocs[0]}, RISCV_0 noc{nocs[1]}), "
            f"{iters} rereads of {NUM_TILES} tiles per launch"
        )

        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        cb_bytes = ROW_WORDS * 4 + BUF_BYTES
        cbs = [
            ttnn.CBDescriptor(
                total_size=cb_bytes,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=cb_bytes)
                ],
            )
            for i in (0, 1)
        ]
        accessor_args = (
            ttnn.TensorAccessorArgs(tt_k).get_compile_time_args()
            + ttnn.TensorAccessorArgs(tt_out).get_compile_time_args()
        )
        kernels = []
        processors = (ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0)
        for risc, (processor, noc) in enumerate(zip(processors, nocs)):
            cb = risc
            config = ttnn.DataMovementConfigDescriptor(processor=processor, noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[noc])
            rt = ttnn.RuntimeArgs()
            for i, (x, y) in enumerate(cores):
                rt[x][y] = [tt_k.buffer_address(), tt_out.buffer_address(), 2 * i + risc, iters]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=grid,
                    compile_time_args=[cb] + accessor_args,
                    runtime_args=rt,
                    config=config,
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

        golden = None
        total_bad = 0
        t0 = time.perf_counter()
        for launch in range(launches):
            ttnn.generic_op([tt_k, tt_out], program)
            rows = ttnn.to_torch(tt_out).reshape(slots, ROW_WORDS).to(torch.int64) & 0xFFFFFFFF
            assert (rows[:, 6] == 0xC0FFEE).all(), "a RISC did not report"
            gold = rows[:, HDR_WORDS + BUF_WORDS :]
            if golden is None:
                golden = gold[0].clone()
            gold_diff = [s for s in range(slots) if not torch.equal(gold[s], golden)]
            if gold_diff:
                logger.error(f"launch {launch}: first read differs from launch-0 golden on slots {gold_diff}")
            for s in range(slots):
                hdr = rows[s, :HDR_WORDS]
                if hdr[1]:
                    total_bad += int(hdr[1])
                    x, y = cores[s // 2]
                    bad = rows[s, HDR_WORDS : HDR_WORDS + BUF_WORDS]
                    diff = (bad != gold[s]).reshape(NUM_TILES, -1)
                    tiles = {t: int(diff[t].sum()) for t in range(NUM_TILES) if diff[t].any()}
                    xor = (bad ^ gold[s]).reshape(-1)
                    lanes = [int(((xor >> (8 * b)) & 0xFF).ne(0).sum()) for b in range(4)]
                    logger.error(
                        f"launch {launch} core ({x},{y}) {('RISCV_1', 'RISCV_0')[s % 2]} noc{nocs[s % 2]}: "
                        f"{int(hdr[1])} bad rereads, first at {int(hdr[2])}, max {int(hdr[4])} words; "
                        f"tile:words {tiles} (bank = tile % {BANKS}), bad bytes per lane {lanes}"
                    )
                    torch.save(
                        {"bad": bad, "gold": gold[s], "hdr": hdr, "core": (x, y), "risc": s % 2, "noc": nocs[s % 2]},
                        OUT_DIR / f"hammer_chip{chip}_launch{launch}_slot{s}.pt",
                    )
            if (launch + 1) % 10 == 0 or gold_diff:
                logger.info(
                    f"{launch + 1}/{launches} launches, {(launch + 1) * iters * slots} rereads, "
                    f"{total_bad} bad, {time.perf_counter() - t0:.0f}s"
                )
        if pad is not None:
            ttnn.deallocate(pad)
    finally:
        ttnn.close_device(device)
    assert total_bad == 0, f"{total_bad} K rereads differed from the first read"
