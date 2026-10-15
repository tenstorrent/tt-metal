# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

BANKS = 8
PAD_PAGE_BYTES = 64
PAGE_BYTES = 2048
PAGE_WORDS = PAGE_BYTES // 4
ALIAS_DELTA = 0x4000000
HDR_WORDS = 16
ROW_WORDS = HDR_WORDS + 2 * PAGE_WORDS  # header, first bad page, its partner page
NONE = 0xFFFFFFFF

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(os.environ.get("REPRO_OUT_DIR", REPO / "repro_logs" / "stress"))

# Every word of the target buffer holds its own per-bank byte address, so a bad read is found without a golden
# copy. On a bad page the kernel reads the partner page at H - 64MB and counts a bad word as an alias when every
# wrong byte equals the partner's byte. The partner may sit in reserved DRAM, so its contents are read, not assumed.
KERNEL = f"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {{
    const uint32_t hi_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t iters = get_arg_val<uint32_t>(3);
    const uint32_t num_pages = get_arg_val<uint32_t>(4);
    const uint32_t bank_mask = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr auto hi_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<hi_args.next_compile_time_args_offset()>();
    const auto hi = TensorAccessor(hi_args, hi_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t base = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint32_t* bad = hdr + {HDR_WORDS};
    volatile tt_l1_ptr uint32_t* first_partner = bad + {PAGE_WORDS};
    const uint32_t partner_addr = base + {ROW_WORDS * 4};
    volatile tt_l1_ptr uint32_t* partner = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(partner_addr);
    const uint32_t work_addr = partner_addr + {PAGE_BYTES};
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(work_addr);

    uint32_t bad_pages = 0, bad_words = 0, alias_words = 0, other_words = 0;
    uint32_t lane[4] = {{0, 0, 0, 0}};
    uint32_t first_iter = {NONE}, first_page = {NONE}, first_word = 0, first_got = 0, first_exp = 0;
    uint32_t pages[chunk];

    for (uint32_t it = 0; it < iters; ++it) {{
        uint32_t p = 0;
        while (p < num_pages) {{
            uint32_t n = 0;
            for (; p < num_pages && n < chunk; ++p) {{
                if ((bank_mask >> (p % {BANKS})) & 1) {{
                    noc_async_read(hi.get_noc_addr(p), work_addr + n * {PAGE_BYTES}, {PAGE_BYTES});
                    pages[n++] = p;
                }}
            }}
            noc_async_read_barrier();
            for (uint32_t i = 0; i < n; ++i) {{
                const uint32_t exp0 = hi_addr + (pages[i] / {BANKS}) * {PAGE_BYTES};
                volatile tt_l1_ptr uint32_t* w = work + i * {PAGE_WORDS};
                uint32_t page_bad = 0;
                for (uint32_t j = 0; j < {PAGE_WORDS}; ++j) {{
                    const uint32_t exp = exp0 + 4 * j;
                    const uint32_t got = w[j];
                    if (got != exp) {{
                        if (page_bad == 0) {{
                            noc_async_read(hi.get_noc_addr(pages[i]) - {ALIAS_DELTA}, partner_addr, {PAGE_BYTES});
                            noc_async_read_barrier();
                        }}
                        const uint32_t x = got ^ exp;
                        uint32_t m = 0;
                        for (uint32_t b = 0; b < 4; ++b) {{
                            const uint32_t byte_mask = 0xFFu << (8 * b);
                            if (x & byte_mask) {{
                                ++lane[b];
                                m |= byte_mask;
                            }}
                        }}
                        if (((got ^ partner[j]) & m) == 0) {{
                            ++alias_words;
                        }} else {{
                            ++other_words;
                        }}
                        if (first_iter == {NONE}) {{
                            first_iter = it;
                            first_page = pages[i];
                            first_word = j;
                            first_got = got;
                            first_exp = exp;
                            for (uint32_t k = 0; k < {PAGE_WORDS}; ++k) {{
                                bad[k] = w[k];
                                first_partner[k] = partner[k];
                            }}
                        }}
                        ++page_bad;
                    }}
                }}
                bad_words += page_bad;
                bad_pages += page_bad != 0;
            }}
        }}
    }}
    hdr[0] = iters;
    hdr[1] = bad_pages;
    hdr[2] = bad_words;
    hdr[3] = alias_words;
    hdr[4] = other_words;
    hdr[5] = first_iter;
    hdr[6] = first_page;
    hdr[7] = first_word;
    hdr[8] = first_got;
    hdr[9] = first_exp;
    for (uint32_t b = 0; b < 4; ++b) {{
        hdr[10 + b] = lane[b];
    }}
    hdr[15] = 0xC0FFEE;
    noc_async_write(base, out.get_noc_addr(slot), {ROW_WORDS * 4});
    noc_async_write_barrier();
}}
"""


def address_pattern(base, rows):
    r = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    w = torch.arange(PAGE_WORDS, dtype=torch.int64).unsqueeze(0)
    v = base + (r // BANKS) * PAGE_BYTES + 4 * w
    return v.to(torch.int32).reshape(1, 1, rows, PAGE_WORDS)


def pad_to(device, cur, target):
    assert target >= cur and (target - cur) % PAD_PAGE_BYTES == 0, f"cannot pad {hex(cur)} -> {hex(target)}"
    if target == cur:
        return None
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, (target - cur) // PAD_PAGE_BYTES * BANKS, PAD_PAGE_BYTES // 4]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def put(device, data):
    return ttnn.from_torch(
        data, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.timeout(0)
def test_dram_bank_alias_stress():
    chip = int(os.environ.get("REPRO_CHIP", 26))
    hi_base = int(os.environ.get("STRESS_HI", "0x4600000"), 0)
    bank_bytes = int(os.environ.get("STRESS_BANK_BYTES", "0x10000"), 0)
    banks = [int(b) for b in os.environ.get("STRESS_BANKS", "0,1,2,3,4,5,6,7").split(",")]
    launches = int(os.environ.get("STRESS_LAUNCHES", 100))
    iters = int(os.environ.get("STRESS_ITERS", 1500))
    chunk = int(os.environ.get("STRESS_CHUNK", 16))
    gx, gy = (int(v) for v in os.environ.get("STRESS_GRID", "12x10").split("x"))
    # The same NoC on both data-movement RISCs hangs this kernel.
    nocs = [int(v) for v in os.environ.get("STRESS_NOCS", "0,1").split(",")]
    assert hi_base & ALIAS_DELTA, "STRESS_HI must have bit 26 set so its partner is H - 64MB"
    assert bank_bytes % PAGE_BYTES == 0
    lo_base = hi_base - ALIAS_DELTA
    rows = bank_bytes // PAGE_BYTES * BANKS
    bank_mask = sum(1 << b for b in banks)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_device(device_id=chip)
    try:
        probe = put(device, torch.zeros(1, BANKS * PAD_PAGE_BYTES // 4, dtype=torch.int32))
        free_addr = probe.buffer_address()
        ttnn.deallocate(probe)
        # A partner below free DRAM is reserved, so it is left as-is and only read.
        partner_written = lo_base >= free_addr
        pad_lo = tt_lo = None
        if partner_written:
            pad_lo = pad_to(device, free_addr, lo_base)
            tt_lo = put(device, address_pattern(lo_base, rows))
            assert tt_lo.buffer_address() == lo_base, f"lo landed at {hex(tt_lo.buffer_address())}"
        pad_hi = pad_to(device, lo_base + bank_bytes if partner_written else free_addr, hi_base)
        tt_hi = put(device, address_pattern(hi_base, rows))
        assert tt_hi.buffer_address() == hi_base, f"hi landed at {hex(tt_hi.buffer_address())}"
        for t, b in ((tt_lo, lo_base), (tt_hi, hi_base)):
            if t is not None:
                got = ttnn.to_torch(t).to(torch.int64) & 0xFFFFFFFF
                assert torch.equal(got, address_pattern(b, rows).to(torch.int64)), f"readback at {hex(b)} is wrong"

        cores = [(x, y) for y in range(gy) for x in range(gx)]
        slots = 2 * len(cores)
        tt_out = put(device, torch.zeros(1, 1, slots, ROW_WORDS, dtype=torch.int32))
        reads_per_iter = sum(1 for p in range(rows) if (bank_mask >> (p % BANKS)) & 1)
        logger.info(
            f"chip {chip}: hi {hex(hi_base)}..{hex(hi_base + bank_bytes)}, partner {hex(lo_base)} "
            f"({'address-as-data' if partner_written else 'reserved, read as-is'}), banks {banks}, "
            f"{len(cores)} cores x 2 RISCs (RISCV_1 noc{nocs[0]}, RISCV_0 noc{nocs[1]}), "
            f"{iters} x {reads_per_iter} page reads of {PAGE_BYTES} B per RISC per launch"
        )

        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        cb_bytes = ROW_WORDS * 4 + (chunk + 1) * PAGE_BYTES
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
            ttnn.TensorAccessorArgs(tt_hi).get_compile_time_args()
            + ttnn.TensorAccessorArgs(tt_out).get_compile_time_args()
        )
        kernels = []
        processors = (ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0)
        for risc, (processor, noc) in enumerate(zip(processors, nocs)):
            rt = ttnn.RuntimeArgs()
            for i, (x, y) in enumerate(cores):
                rt[x][y] = [hi_base, tt_out.buffer_address(), 2 * i + risc, iters, rows, bank_mask]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=grid,
                    compile_time_args=[risc, chunk] + accessor_args,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=processor, noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[noc]
                    ),
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

        # Self-test: plant H-64MB's address-as-data in one page; every RISC must flag it, and decode it as an
        # alias when the partner really holds that data.
        inj = next(p for p in range(rows) if (bank_mask >> (p % BANKS)) & 1)
        planted = address_pattern(hi_base, rows)
        planted[0, 0, inj] -= ALIAS_DELTA
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(planted, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), tt_hi
        )
        ttnn.generic_op([tt_hi, tt_out], program)
        hdr = (ttnn.to_torch(tt_out).reshape(slots, ROW_WORDS).to(torch.int64) & 0xFFFFFFFF)[:, :HDR_WORDS]
        cols = [1, 2, 5, 6] + ([3, 4] if partner_written else [])
        want = [iters, iters * PAGE_WORDS, 0, inj] + ([iters * PAGE_WORDS, 0] if partner_written else [])
        assert (
            hdr[:, cols] == torch.tensor(want)
        ).all(), f"self-test: planted alias not decoded, got {hdr[0, cols].tolist()}"
        assert (hdr[:, 10:13] == 0).all() and (hdr[:, 13] == iters * PAGE_WORDS).all(), "self-test: lane counts wrong"
        logger.info(f"self-test: planted alias on page {inj} (bank {inj % BANKS}) decoded by all {slots} RISCs")
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(address_pattern(hi_base, rows), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), tt_hi
        )
        assert torch.equal(
            ttnn.to_torch(tt_hi).to(torch.int64) & 0xFFFFFFFF, address_pattern(hi_base, rows).to(torch.int64)
        )

        totals = {"pages": 0, "alias": 0, "other": 0}
        t0 = time.perf_counter()
        for launch in range(launches):
            ttnn.generic_op([tt_hi, tt_out], program)
            out = ttnn.to_torch(tt_out).reshape(slots, ROW_WORDS).to(torch.int64) & 0xFFFFFFFF
            assert (out[:, 15] == 0xC0FFEE).all(), "a RISC did not report"
            for s in range(slots):
                hdr = out[s, :HDR_WORDS]
                if not hdr[1]:
                    continue
                totals["pages"] += int(hdr[1])
                totals["alias"] += int(hdr[3])
                totals["other"] += int(hdr[4])
                x, y = cores[s // 2]
                page = int(hdr[6])
                logger.error(
                    f"launch {launch} core ({x},{y}) {('RISCV_1', 'RISCV_0')[s % 2]} noc{nocs[s % 2]}: "
                    f"{int(hdr[1])} bad pages, {int(hdr[2])} bad words ({int(hdr[3])} match H-64MB, {int(hdr[4])} other), "
                    f"bad bytes per lane {[int(v) for v in hdr[10:14]]}; first: iter {int(hdr[5])} page {page} "
                    f"(bank {page % BANKS}, addr {hex(hi_base + page // BANKS * PAGE_BYTES)}) word {int(hdr[7])} "
                    f"got {int(hdr[8]):#010x} exp {int(hdr[9]):#010x}"
                )
                torch.save(
                    {
                        "hdr": hdr,
                        "bad": out[s, HDR_WORDS : HDR_WORDS + PAGE_WORDS],
                        "partner": out[s, HDR_WORDS + PAGE_WORDS :],
                        "core": (x, y),
                        "risc": s % 2,
                        "noc": nocs[s % 2],
                    },
                    OUT_DIR / f"stress_chip{chip}_{hex(hi_base)}_launch{launch}_slot{s}.pt",
                )
            if (launch + 1) % 10 == 0:
                logger.info(
                    f"{launch + 1}/{launches} launches, {(launch + 1) * iters * reads_per_iter * slots} page reads, "
                    f"bad pages {totals['pages']} (words match H-64MB {totals['alias']}, other {totals['other']}), "
                    f"{time.perf_counter() - t0:.0f}s"
                )
        for t in (pad_lo, pad_hi):
            if t is not None:
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
    assert totals["pages"] == 0, f"{totals['pages']} page reads differed from address-as-data: {totals}"
