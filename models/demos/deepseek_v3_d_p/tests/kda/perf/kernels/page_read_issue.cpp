// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Issue cost of consecutive interleaved page reads (test_page_read_issue.py). For each variant, REPS rounds each issue
// PAGES consecutive page reads of the source tensor into one buffer; per round it records the cycles to issue them and
// the cycles until the read barrier. Results (uint32): out[(variant * REPS + rep) * 2 + {0, 1}] = {issue, complete}.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"

namespace {
constexpr uint32_t PAGES = get_compile_time_arg_val(0);
constexpr uint32_t REPS = get_compile_time_arg_val(1);
constexpr uint32_t VARIANTS = 8;

FORCE_INLINE uint32_t now() { return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t total_pages = get_arg_val<uint32_t>(2);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    DataflowBuffer buffer(0);
    DataflowBuffer results(1);
    Noc noc;
    buffer.reserve_back(PAGES);
    results.reserve_back(1);
    auto* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results.get_write_ptr());
    const uint32_t entry = buffer.get_entry_size();
    const uint32_t dst_base = buffer.get_write_ptr();
    constexpr bool is_dram = decltype(src)::DSpec::is_dram;
    constexpr uint32_t num_banks = is_dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    const uint32_t aligned = src.get_aligned_page_size();
    const uint8_t noc_id = noc.get_noc_id();

    for (uint32_t variant = 0; variant < VARIANTS; ++variant) {
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            const uint32_t base = (rep * PAGES * 3) % (total_pages - PAGES);
            const uint32_t t0 = now();
            if (variant == 0) {
                // Current: pages() iterator (a page id divide per page), default (any-length) read.
                uint32_t tile = 0;
                for (const auto& page : src.pages(base, base + PAGES)) {
                    noc.async_read(page, buffer, entry, {}, {.offset_bytes = tile * entry});
                    ++tile;
                }
            } else if (variant == 1) {
                // Accessor + page id per read (stage_value_slice style).
                for (uint32_t tile = 0; tile < PAGES; ++tile) {
                    noc.async_read(src, buffer, entry, {.page_id = base + tile}, {.offset_bytes = tile * entry});
                }
            } else if (variant == 2) {
                // pages() iterator, one-packet read.
                uint32_t tile = 0;
                for (const auto& page : src.pages(base, base + PAGES)) {
                    noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                        page, buffer, entry, {}, {.offset_bytes = tile * entry});
                    ++tile;
                }
            } else if (variant == 3) {
                // Incremental (bank, offset) walk, one-packet raw read.
                uint32_t row = interleaved_addr_gen::get_bank_offset_index<is_dram>(base);
                uint32_t bank = base - row * num_banks;
                uint32_t offset = src_addr + row * aligned;
                uint32_t dst = dst_base;
                for (uint32_t tile = 0; tile < PAGES; ++tile) {
                    const uint64_t a = noc_address_backend::bank_address<is_dram>(
                        bank, offset + interleaved_addr_gen::get_bank_offset<is_dram>(bank), noc_id);
                    noc_async_read_one_packet(a, dst, entry, noc_id);
                    dst += entry;
                    if (++bank == num_banks) {
                        bank = 0;
                        offset += aligned;
                    }
                }
            } else if (variant == 4) {
                // Incremental walk + stateful issue: control and length set once, per read the two addresses,
                // the coordinate and the send.
                uint32_t row = interleaved_addr_gen::get_bank_offset_index<is_dram>(base);
                uint32_t bank = base - row * num_banks;
                uint32_t offset = src_addr + row * aligned;
                uint32_t dst = dst_base;
                noc_async_read_one_packet_set_state(noc_address_backend::bank_address<is_dram>(0, 0, noc_id), entry);
                for (uint32_t tile = 0; tile < PAGES; ++tile) {
                    const uint64_t a = noc_address_backend::bank_address<is_dram>(
                        bank, offset + interleaved_addr_gen::get_bank_offset<is_dram>(bank), noc_id);
                    while (!noc_cmd_buf_ready(noc_id, read_cmd_buf));
                    NOC_CMD_BUF_WRITE_REG(noc_id, read_cmd_buf, NOC_RET_ADDR_LO, dst);
                    NOC_CMD_BUF_WRITE_REG(noc_id, read_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)a);
                    NOC_CMD_BUF_WRITE_REG(
                        noc_id,
                        read_cmd_buf,
                        NOC_TARG_ADDR_COORDINATE,
                        (uint32_t)(a >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
                    NOC_CMD_BUF_WRITE_REG(noc_id, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
                    noc_reads_num_issued[noc_id] += 1;
                    dst += entry;
                    if (++bank == num_banks) {
                        bank = 0;
                        offset += aligned;
                    }
                }
            } else if (variant == 5) {
                // Floor: one page, same core every time, with_state (two addresses + send).
                const uint64_t a = src.get_noc_addr(base);
                noc_async_read_one_packet_set_state(a, entry);
                uint32_t dst = dst_base;
                for (uint32_t tile = 0; tile < PAGES; ++tile) {
                    noc_async_read_one_packet_with_state<true>((uint32_t)a, dst);
                    dst += entry;
                }
            } else if (variant == 6) {
                // Address computation only (no NoC): pages() iterator.
                uint32_t acc = 0;
                for (const auto& page : src.pages(base, base + PAGES)) {
                    acc += (uint32_t)page.noc_addr() ^ (uint32_t)(page.noc_addr() >> 32);
                }
                res[1023] = acc;
            } else {
                // Address computation only: incremental walk.
                uint32_t row = interleaved_addr_gen::get_bank_offset_index<is_dram>(base);
                uint32_t bank = base - row * num_banks;
                uint32_t offset = src_addr + row * aligned;
                uint32_t acc = 0;
                for (uint32_t tile = 0; tile < PAGES; ++tile) {
                    const uint64_t a = noc_address_backend::bank_address<is_dram>(
                        bank, offset + interleaved_addr_gen::get_bank_offset<is_dram>(bank), noc_id);
                    acc += (uint32_t)a ^ (uint32_t)(a >> 32);
                    if (++bank == num_banks) {
                        bank = 0;
                        offset += aligned;
                    }
                }
                res[1023] = acc;
            }
            const uint32_t t1 = now();
            noc.async_read_barrier();
            const uint32_t t2 = now();
            res[(variant * REPS + rep) * 2] = t1 - t0;
            res[(variant * REPS + rep) * 2 + 1] = t2 - t0;
        }
    }
    // The iterator's stepped addresses against a lookup per page, for strides up to past twice the bank count and
    // starts across a bank row: out[1022] = mismatches, out[1021] = pages checked.
    uint32_t mismatches = 0;
    uint32_t checked = 0;
    for (uint32_t stride = 1; stride <= 2 * num_banks + 3; ++stride) {
        for (uint32_t start = 0; start < num_banks + 2; start += 3) {
            for (const auto& page : tensor_accessor::Pages(src, start, total_pages, stride, noc_id)) {
                mismatches += page.noc_addr() != src.get_noc_addr(page.page_id(), 0, noc_id);
                ++checked;
            }
        }
    }
    {
        auto it = src.pages(5, total_pages).begin();
        it += num_banks + 7;
        mismatches += it->noc_addr() != src.get_noc_addr(num_banks + 12, 0, noc_id);
        ++it;
        mismatches += it->noc_addr() != src.get_noc_addr(num_banks + 13, 0, noc_id);
        checked += 2;
    }
    res[1022] = mismatches;
    res[1021] = checked;
    noc.async_write(results, out, 1024 * sizeof(uint32_t), {}, {.page_id = 0});
    noc.async_write_barrier();
}
