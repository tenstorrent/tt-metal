// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One-core data-movement kernel for the doorbell and coherent-alias device tests: a Tensix
// writer or reader of an L2CPU tile and of a DRAM bank, with results in a DRAM page.
//
// The coherent alias of the tile's local DRAM, X280_LOCAL_DDR_TILE_BASE + offset, is a 47-bit
// local address. Metal's NoC helpers carry 36 local bits and leave NOC_TARG_ADDR_MID /
// NOC_RET_ADDR_MID at 0 (or mask them to the PCIe bit), so the raw operations here program the
// command buffer themselves, with the whole upper word in MID, and put MID back to 0 once the
// command buffer has taken the request.
//
// Runtime args:
//   0 op      kOpWriteRaw, kOpReadRaw, kOpWriteBank, kOpReadBank, kOpDoorbell
//   1 x       target NOC coordinate (translated), or the DRAM bank id for the bank ops
//   2 y
//   3 addr_hi target address, upper 32 bits (raw ops)
//   4 addr_lo target address, lower 32 bits (raw ops; the bank address for the bank ops)
//   5 lines   64-byte lines from the target address
//   6 first   pattern line index of the first line
//   7 seed    pattern seed (writes)
//   8 l1      64-byte aligned L1 scratch, lines * 64 bytes and at least 64
//   9 out     address of a one-page DRAM buffer (bank 0) receiving the results
//  10 value   doorbell: the u32 pushed into the MSI catcher at (x, y, addr_lo)
//  11 poll    doorbell: 64-byte aligned LIM address of the line holding the wake counter
//  12 offset  doorbell: byte offset of the counter's low word in that line
//  13 base    doorbell: counter value before the doorbell
//  14 budget  doorbell: poll budget in wall-clock ticks
//
// Results (u32 words in the out page): reads copy the lines; the doorbell writes
// {status, ticks, counter, polls}, status 1 when the counter moved and 2 at the budget.

#include "api/dataflow/dataflow_api.h"

#include "../../../src/x280_hw_probe.h"

#include <cstdint>

namespace {

constexpr uint32_t kOpWriteRaw = 0;
constexpr uint32_t kOpReadRaw = 1;
constexpr uint32_t kOpWriteBank = 2;
constexpr uint32_t kOpReadBank = 3;
constexpr uint32_t kOpDoorbell = 4;

void raw_write(uint32_t src_l1, uint32_t x, uint32_t y, uint64_t addr, uint32_t len) {
    const uint32_t noc = noc_index;
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_l1);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, static_cast<uint32_t>(addr));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, static_cast<uint32_t>(addr >> 32));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, NOC_XY_ENCODING(x, y));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, len);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, 0);
    noc_async_write_barrier();
}

void raw_read(uint32_t dst_l1, uint32_t x, uint32_t y, uint64_t addr, uint32_t len) {
    const uint32_t noc = noc_index;
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_l1);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, static_cast<uint32_t>(addr));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, static_cast<uint32_t>(addr >> 32));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, NOC_XY_ENCODING(x, y));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, len);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, 0);
    noc_async_read_barrier();
}

void fill_pattern(uint32_t l1, uint32_t lines, uint32_t first, uint32_t seed) {
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t line = 0; line < lines; line++) {
        for (uint32_t w = 0; w < X280_PROBE_LINE_BYTES / 4; w++) {
            words[line * (X280_PROBE_LINE_BYTES / 4) + w] = x280_probe_pattern(seed, first + line, w);
        }
    }
}

void store_out(uint32_t l1, uint32_t out, uint32_t bytes) {
    noc_async_write(l1, get_noc_addr_from_bank_id<true>(0, out), bytes);
    noc_async_write_barrier();
}

}  // namespace

void kernel_main() {
    const uint32_t op = get_arg_val<uint32_t>(0);
    const uint32_t x = get_arg_val<uint32_t>(1);
    const uint32_t y = get_arg_val<uint32_t>(2);
    const uint64_t addr = (static_cast<uint64_t>(get_arg_val<uint32_t>(3)) << 32) | get_arg_val<uint32_t>(4);
    const uint32_t lines = get_arg_val<uint32_t>(5);
    const uint32_t first = get_arg_val<uint32_t>(6);
    const uint32_t seed = get_arg_val<uint32_t>(7);
    const uint32_t l1 = get_arg_val<uint32_t>(8);
    const uint32_t out = get_arg_val<uint32_t>(9);
    const uint32_t bytes = lines * X280_PROBE_LINE_BYTES;

    if (op == kOpWriteRaw) {
        fill_pattern(l1, lines, first, seed);
        for (uint32_t i = 0; i < lines; i++) {
            raw_write(l1 + i * X280_PROBE_LINE_BYTES, x, y, addr + i * X280_PROBE_LINE_BYTES, X280_PROBE_LINE_BYTES);
        }
    } else if (op == kOpReadRaw) {
        for (uint32_t i = 0; i < lines; i++) {
            raw_read(l1 + i * X280_PROBE_LINE_BYTES, x, y, addr + i * X280_PROBE_LINE_BYTES, X280_PROBE_LINE_BYTES);
        }
        store_out(l1, out, bytes);
    } else if (op == kOpWriteBank) {
        fill_pattern(l1, lines, first, seed);
        noc_async_write(l1, get_noc_addr_from_bank_id<true>(x, static_cast<uint32_t>(addr)), bytes);
        noc_async_write_barrier();
    } else if (op == kOpReadBank) {
        noc_async_read(get_noc_addr_from_bank_id<true>(x, static_cast<uint32_t>(addr)), l1, bytes);
        noc_async_read_barrier();
        store_out(l1, out, bytes);
    } else if (op == kOpDoorbell) {
        const uint32_t value = get_arg_val<uint32_t>(10);
        const uint32_t poll = get_arg_val<uint32_t>(11);
        const uint32_t offset = get_arg_val<uint32_t>(12);
        const uint32_t base = get_arg_val<uint32_t>(13);
        const uint32_t budget = get_arg_val<uint32_t>(14);
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
        const uint32_t line = l1 + X280_PROBE_LINE_BYTES;
        auto* seen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(line + offset);
        words[0] = value;

        const uint64_t t0 = get_timestamp();
        noc_async_write(l1, get_noc_addr(x, y, static_cast<uint32_t>(addr)), sizeof(uint32_t));
        noc_async_write_barrier();
        uint32_t polls = 0;
        uint32_t status = 2;
        uint64_t elapsed = 0;
        do {
            noc_async_read(get_noc_addr(x, y, poll), line, X280_PROBE_LINE_BYTES);
            noc_async_read_barrier();
            polls++;
            elapsed = get_timestamp() - t0;
            if (*seen != base) {
                status = 1;
                break;
            }
        } while (elapsed < budget);

        words[0] = status;
        words[1] = static_cast<uint32_t>(elapsed);
        words[2] = *seen;
        words[3] = polls;
        store_out(l1, out, X280_PROBE_LINE_BYTES);
    }
}
