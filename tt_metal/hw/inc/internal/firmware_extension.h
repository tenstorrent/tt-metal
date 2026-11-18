// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What a BRISC firmware extension (TT_METAL_FW_SRC_BRISC) may use. brisc.cc calls the extension's
//
//     bool reload_next_stage(internal::fw_ext::Launch& launch, uint32_t& stage, uint32_t& round);
//
// once every RISC has finished the current image, before go-DONE. Returning true runs the launch
// body again on whatever install() put in place; false ends the launch.
//
// At the call: every RISC has returned and trigger_sync_register_init() has run. The NoC is not
// idle: call quiesce_core_noc() before overwriting L1. On true, brisc.cc redoes the NCRISC load
// trigger, firmware_config_init, the i$ invalidate, the NoC init for the installed config, CB setup
// and the start of every RISC. It does not redo the go wait, launch_msg_rd_ptr, the profiler zone,
// or DONE and the dispatcher notify, which happen once per launch.
//
// Order within a pass: quiesce_core_noc() -> issue -> flush() -> install().

#pragma once

#include <cstddef>
#include <cstdint>

#include "api/debug/assert.h"
#include "hostdev/dev_msgs.h"
#include "internal/debug/sanitize.h"
#include "noc_nonblocking_api.h"

#if !defined(ARCH_BLACKHOLE)
#error "firmware_extension.h is Blackhole only"
#endif
static_assert(NUM_NOCS == 2, "fw_ext assumes two NoCs");

extern uint8_t noc_index;
extern uint8_t noc_mode;

namespace internal::fw_ext {

// Virtual coordinates in NoC-0 orientation.
struct Rect {
    uint8_t start_x, start_y, end_x, end_y;
};

// Every field added to kernel_config_msg_t must be classified in install(): a new field either grows
// the struct or shrinks a pad, and both fail here.
static_assert(
    sizeof(kernel_config_msg_t::pad_sem_offset) == (ProgrammableCoreType::COUNT % 2) * 2,
    "classify the new field in fw_ext::Launch::install");
static_assert(sizeof(kernel_config_msg_t::pad2) == 1, "classify the new field in fw_ext::Launch::install");
static_assert(
    sizeof(kernel_config_msg_t::pad4) == (MaxProcessorsPerCoreType % 2) * 4,
    "classify the new field in fw_ext::Launch::install");
static_assert(
    sizeof(kernel_config_msg_t::pad3) == 9 - (MaxProcessorsPerCoreType % 2) * 2,
    "classify the new field in fw_ext::Launch::install");
static_assert(sizeof(kernel_config_msg_t) % sizeof(uint32_t) == 0);

class Launch {
public:
    explicit Launch(launch_msg_t* msg) :
        msg_(msg),
        kernel_config_base_(msg->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX]),
        extension_table_(msg->kernel_config.reload_table_addr),
        mode_(msg->kernel_config.mode),
        preload_(msg->kernel_config.preload),
        sub_device_origin_x_(msg->kernel_config.sub_device_origin_x),
        sub_device_origin_y_(msg->kernel_config.sub_device_origin_y) {}

    uint32_t extension_table() const { return extension_table_; }
    // 16 B (L1) aligned only. A DRAM source must be congruent with it mod 64.
    uint32_t kernel_config_base() const { return kernel_config_base_; }
    uint32_t noc() const { return msg_->kernel_config.brisc_noc_id; }

    // Copies a captured kernel_config_msg_t from local L1 over the live one, then restores the fields
    // this launch owns. False, with nothing changed, if captured_bytes is not the struct's size.
    bool install(uint32_t captured_l1_addr, uint32_t captured_bytes) {
        if (captured_bytes != sizeof(kernel_config_msg_t)) {
            return false;
        }
        invalidate_l1_cache();
        auto* dst = reinterpret_cast<volatile uint32_t*>(
            reinterpret_cast<uintptr_t>(msg_) + offsetof(launch_msg_t, kernel_config));
        auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(captured_l1_addr);
        for (uint32_t i = 0; i < sizeof(kernel_config_msg_t) / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
        auto& kc = msg_->kernel_config;
        kc.kernel_config_base[ProgrammableCoreType::TENSIX] = kernel_config_base_;
        kc.reload_table_addr = extension_table_;
        kc.mode = mode_;
        kc.preload = preload_;
        kc.sub_device_origin_x = sub_device_origin_x_;
        kc.sub_device_origin_y = sub_device_origin_y_;
        return true;
    }
    void set_enables(uint32_t enables) { msg_->kernel_config.enables = enables; }
    void disable() { set_enables(0); }

private:
    launch_msg_t* msg_;
    uint32_t kernel_config_base_;
    uint32_t extension_table_;
    uint8_t mode_;
    uint8_t preload_;
    uint8_t sub_device_origin_x_;
    uint8_t sub_device_origin_y_;
};

inline uint64_t noc_xy_addr(uint32_t noc, uint32_t x, uint32_t y, uint32_t addr) {
    return NOC_XY_ADDR(DYNAMIC_NOC_X(noc, x), DYNAMIC_NOC_Y(noc, y), addr);
}

// In dedicated mode BRISC's counters cover only noc_index (NCRISC's traffic on the other NoC is in its own
// counters), so every issue below must use it.
inline void check_noc(uint32_t noc) { ASSERT(noc_mode == DM_DYNAMIC_NOC || noc == noc_index); }

// Non-blocking DRAM/L1 -> local L1 read. The source must be congruent with dst_l1 mod 64 for DRAM.
__attribute__((noinline)) inline void read(uint32_t noc, uint64_t src, uint32_t dst_l1, uint32_t bytes, uint32_t vc) {
    check_noc(noc);
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src, dst_l1, bytes);
    if (noc_mode == DM_DYNAMIC_NOC) {
        ncrisc_noc_fast_read_any_len<DM_DYNAMIC_NOC>(noc, DYNAMIC_NOC_BRISC_RD_CMD_BUF, src, dst_l1, bytes, vc);
    } else {
        ncrisc_noc_fast_read_any_len(noc, BRISC_RD_CMD_BUF, src, dst_l1, bytes, vc);
    }
}

namespace detail {
// One copy for both modes and both write kinds: the stock any_len helper expands per mode.
__attribute__((noinline)) inline void write(
    uint32_t noc, uint32_t src_l1, uint64_t dst, uint32_t bytes, bool mcast, uint32_t num_dests) {
    check_noc(noc);
    const bool dynamic = (noc_mode == DM_DYNAMIC_NOC);
    const uint32_t cmd_buf = dynamic ? (mcast ? DYNAMIC_NOC_BRISC_WR_REG_CMD_BUF : DYNAMIC_NOC_BRISC_WR_CMD_BUF)
                                     : (mcast ? BRISC_WR_REG_CMD_BUF : BRISC_WR_CMD_BUF);
    const uint32_t cmd =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_RESP_MARKED |
        (mcast ? (NOC_CMD_STATIC_VC(NOC_MULTICAST_WRITE_VC) | NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET)
               : NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC));
    do {
        const uint32_t n = bytes > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : bytes;
        while (!noc_cmd_buf_ready(noc, cmd_buf)) {
        }
        if (dynamic) {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests);
        }
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, cmd);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_l1);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst >> 32) & NOC_PCIE_MASK);
        NOC_CMD_BUF_WRITE_REG(
            noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, n);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        if (!dynamic) {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += num_dests;
        }
        src_l1 += n;
        dst += n;
        bytes -= n;
    } while (bytes != 0);
}
}  // namespace detail

// Non-posted unicast write from local L1.
inline void write(uint32_t noc, uint32_t src_l1, uint64_t dst, uint32_t bytes) {
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst, src_l1, bytes);
    detail::write(noc, src_l1, dst, bytes, false, 1);
}

// Non-posted multicast write from local L1 to dst_addr on every core in dst, excluding this one.
// num_dests counts the cores that receive it.
inline void write_multicast(uint32_t noc, uint32_t src_l1, Rect dst, uint32_t dst_addr, uint32_t bytes, uint32_t num_dests) {
    const uint32_t sx = DYNAMIC_NOC_X(noc, dst.start_x);
    const uint32_t sy = DYNAMIC_NOC_Y(noc, dst.start_y);
    const uint32_t ex = DYNAMIC_NOC_X(noc, dst.end_x);
    const uint32_t ey = DYNAMIC_NOC_Y(noc, dst.end_y);
    // NoC 1 multicasts bottom-left to top-right, so its corners swap.
    const uint64_t addr =
        noc == 0 ? NOC_MULTICAST_ADDR(sx, sy, ex, ey, dst_addr) : NOC_MULTICAST_ADDR(ex, ey, sx, sy, dst_addr);
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, addr, src_l1, bytes);
    detail::write(noc, src_l1, addr, bytes, true, num_dests);
}

__attribute__((noinline)) inline void atomic_inc(uint32_t noc, uint64_t addr, uint32_t incr) {
    check_noc(noc);
    DEBUG_SANITIZE_NOC_ADDR(noc, addr, 4);
    if (noc_mode == DM_DYNAMIC_NOC) {
        noc_fast_atomic_increment<DM_DYNAMIC_NOC>(
            noc, DYNAMIC_NOC_BRISC_AT_CMD_BUF, addr, NOC_UNICAST_WRITE_VC, incr, 31, false);
    } else {
        noc_fast_atomic_increment(noc, BRISC_AT_CMD_BUF, addr, NOC_UNICAST_WRITE_VC, incr, 31, false);
    }
}

// Reads landed, nonposted writes and atomics acked, posted writes sent: in dynamic mode on both NoCs,
// counting both DM RISCs; in dedicated mode on BRISC's noc_index.
__attribute__((noinline)) inline void flush() {
    if (noc_mode == DM_DYNAMIC_NOC) {
        for (uint32_t noc = 0; noc < NUM_NOCS; ++noc) {
            do {
                invalidate_l1_cache();
            } while (!ncrisc_dynamic_noc_reads_flushed(noc) || !ncrisc_dynamic_noc_nonposted_writes_flushed(noc) ||
                     !ncrisc_dynamic_noc_nonposted_atomics_flushed(noc) || !ncrisc_dynamic_noc_posted_writes_sent(noc));
        }
    } else {
        while (!ncrisc_noc_reads_flushed(noc_index) || !ncrisc_noc_nonposted_writes_flushed(noc_index) ||
               !ncrisc_noc_nonposted_atomics_flushed(noc_index) || !ncrisc_noc_posted_writes_sent(noc_index)) {
        }
    }
}

// Waits until the kernels' traffic no longer touches this core's L1: no request still leaving and no
// read response still to land on either NoC, then flush() for what the counters can see. The counters
// are kept, not resynced from the hardware: acks still in flight would be lost and a later flush() hang.
// In dedicated mode NCRISC's write acks on the other NoC are not waited for: the kernel-end contract
// there is writes sent, not acked.
__attribute__((noinline)) inline void quiesce_core_noc() {
    for (uint32_t noc = 0; noc < NUM_NOCS; ++noc) {
        while (NOC_STATUS_READ_REG(noc, NIU_MST_RD_REQ_STARTED) != NOC_STATUS_READ_REG(noc, NIU_MST_RD_REQ_SENT)) {
        }
        while (NOC_STATUS_READ_REG(noc, NIU_MST_RD_REQ_SENT) != NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED)) {
        }
        while (NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_STARTED) !=
               NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT)) {
        }
        while (NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_ATOMIC_STARTED) !=
               NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_ATOMIC_SENT)) {
        }
        while (NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_STARTED) !=
               NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT)) {
        }
    }
    flush();
}

// A word in this core's L1 that other cores change over the NoC.
inline uint32_t load_fresh(uint32_t l1_addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
}

// Spins until the word is >= min and returns the value seen.
__attribute__((noinline)) inline uint32_t wait_local_ge(uint32_t l1_addr, uint32_t min) {
    uint32_t value;
    do {
        value = load_fresh(l1_addr);
    } while (value < min);
    return value;
}

}  // namespace internal::fw_ext
