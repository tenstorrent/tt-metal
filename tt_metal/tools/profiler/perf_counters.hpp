// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint16_t PERF_COUNTER_PROFILER_ID = 9090;

enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1_0, L1_1, INSTRN, L1_2, L1_3, L1_4 };
enum PerfCounterType : uint16_t {
    UNDEF = 0,
    // FPU Group (3 counters)
    FPU_COUNTER,
    SFPU_COUNTER,
    MATH_COUNTER,
    // TDMA_UNPACK Group (11 req + 11 grant = 22 counters)
    MATH_SRC_DATA_READY,        // Req 0: math instrn valid & src_data_ready
    DATA_HAZARD_STALLS_MOVD2A,  // Req 1: math instrn not stalled by D2A
    FIDELITY_PHASE_STALLS,      // Req 2: fidelity (HiFi) phase stalls
    MATH_INSTRN_STARTED,
    MATH_INSTRN_AVAILABLE,
    SRCB_WRITE_AVAILABLE,
    SRCA_WRITE_AVAILABLE,
    UNPACK0_BUSY_THREAD0,
    UNPACK1_BUSY_THREAD0,
    UNPACK0_BUSY_THREAD1,
    UNPACK1_BUSY_THREAD1,
    SRCB_WRITE,
    SRCA_WRITE,
    // TDMA_PACK Group (3 counters)
    PACKER_DEST_READ_AVAILABLE,
    PACKER_BUSY,
    AVAILABLE_MATH,
    // INSTRN_THREAD Group (61 counters)
    CFG_INSTRN_AVAILABLE_0,
    CFG_INSTRN_AVAILABLE_1,
    CFG_INSTRN_AVAILABLE_2,
    SYNC_INSTRN_AVAILABLE_0,
    SYNC_INSTRN_AVAILABLE_1,
    SYNC_INSTRN_AVAILABLE_2,
    THCON_INSTRN_AVAILABLE_0,
    THCON_INSTRN_AVAILABLE_1,
    THCON_INSTRN_AVAILABLE_2,
    XSEARCH_INSTRN_AVAILABLE_0,
    XSEARCH_INSTRN_AVAILABLE_1,
    XSEARCH_INSTRN_AVAILABLE_2,
    MOVE_INSTRN_AVAILABLE_0,
    MOVE_INSTRN_AVAILABLE_1,
    MOVE_INSTRN_AVAILABLE_2,
    FPU_INSTRN_AVAILABLE_0,
    FPU_INSTRN_AVAILABLE_1,
    FPU_INSTRN_AVAILABLE_2,
    UNPACK_INSTRN_AVAILABLE_0,
    UNPACK_INSTRN_AVAILABLE_1,
    UNPACK_INSTRN_AVAILABLE_2,
    PACK_INSTRN_AVAILABLE_0,
    PACK_INSTRN_AVAILABLE_1,
    PACK_INSTRN_AVAILABLE_2,
    THREAD_STALLS_0,
    THREAD_STALLS_1,
    THREAD_STALLS_2,
    WAITING_FOR_SRCA_CLEAR,
    WAITING_FOR_SRCB_CLEAR,
    WAITING_FOR_SRCA_VALID,
    WAITING_FOR_SRCB_VALID,
    WAITING_FOR_THCON_IDLE_0,
    WAITING_FOR_THCON_IDLE_1,
    WAITING_FOR_THCON_IDLE_2,
    WAITING_FOR_UNPACK_IDLE_0,
    WAITING_FOR_UNPACK_IDLE_1,
    WAITING_FOR_UNPACK_IDLE_2,
    WAITING_FOR_PACK_IDLE_0,
    WAITING_FOR_PACK_IDLE_1,
    WAITING_FOR_PACK_IDLE_2,
    WAITING_FOR_MATH_IDLE_0,
    WAITING_FOR_MATH_IDLE_1,
    WAITING_FOR_MATH_IDLE_2,
    WAITING_FOR_NONZERO_SEM_0,
    WAITING_FOR_NONZERO_SEM_1,
    WAITING_FOR_NONZERO_SEM_2,
    WAITING_FOR_NONFULL_SEM_0,
    WAITING_FOR_NONFULL_SEM_1,
    WAITING_FOR_NONFULL_SEM_2,
    WAITING_FOR_MOVE_IDLE_0,
    WAITING_FOR_MOVE_IDLE_1,
    WAITING_FOR_MOVE_IDLE_2,
    WAITING_FOR_MMIO_IDLE_0,
    WAITING_FOR_MMIO_IDLE_1,
    WAITING_FOR_MMIO_IDLE_2,
    WAITING_FOR_SFPU_IDLE_0,
    WAITING_FOR_SFPU_IDLE_1,
    WAITING_FOR_SFPU_IDLE_2,
    THREAD_INSTRUCTIONS_0,
    THREAD_INSTRUCTIONS_1,
    THREAD_INSTRUCTIONS_2,
    // L1 Bank 0 (MUX_CTRL bit 4 = 0, monitors L1 ports 0-7)
    L1_0_UNPACKER_0,            // Port 0: Unpacker #0
    L1_0_UNPACKER_1_ECC_PACK1,  // Port 1: Unpacker #1 / ECC / Packer #1
    L1_0_TDMA_BUNDLE_0_RISC,    // Port 2: TDMA Bundle 0 / RISC / TRISC0
    L1_0_TDMA_BUNDLE_1_TRISC,   // Port 3: TDMA Bundle 1 / TRISC1 / TRISC2
    L1_0_NOC_RING0_OUTGOING_0,  // Port 4: NOC Ring 0 Outgoing channel 0
    L1_0_NOC_RING0_OUTGOING_1,  // Port 5: NOC Ring 0 Outgoing channel 1
    L1_0_NOC_RING0_INCOMING_0,  // Port 6: NOC Ring 0 Incoming channel 0
    L1_0_NOC_RING0_INCOMING_1,  // Port 7: NOC Ring 0 Incoming channel 1
    // L1 Bank 1 (MUX_CTRL bit 4 = 1, monitors L1 ports 8-15)
    L1_1_TDMA_PACKER_2,         // Port 8: TDMA Packer 2 write
    L1_1_EXT_UNPACKER_1,        // Port 9: Extended Unpacker interface 1
    L1_1_EXT_UNPACKER_2,        // Port 10: Extended Unpacker interface 2
    L1_1_EXT_UNPACKER_3,        // Port 11: Extended Unpacker interface 3
    L1_1_NOC_RING1_OUTGOING_0,  // Port 12: NOC Ring 1 Outgoing channel 0
    L1_1_NOC_RING1_OUTGOING_1,  // Port 13: NOC Ring 1 Outgoing channel 1
    L1_1_NOC_RING1_INCOMING_0,  // Port 14: NOC Ring 1 Incoming channel 0
    L1_1_NOC_RING1_INCOMING_1,  // Port 15: NOC Ring 1 Incoming channel 1
    // Blackhole-specific L1 ports (differ from Wormhole at ports 1 and 8)
    L1_0_UNIFIED_PACKER,  // BH Port 1, mux 0: Unified Packer (WH has Unpacker#1/ECC/Pack1)
    L1_1_RISC_CORE,       // BH Port 8, mux 1: RISC Core L1 access (WH has TDMA Packer 2)
    // L1 grant counters (reqif_ready — cycles L1 interface was ready to accept)
    // Back-pressure = req - grant. Same port order as req counters above.
    // L1 Bank 0 grants
    L1_0_UNPACKER_0_GRANT,
    L1_0_PORT1_GRANT,  // WH: Unpacker#1/ECC/Pack1, BH: Unified Packer
    L1_0_TDMA_BUNDLE_0_GRANT,
    L1_0_TDMA_BUNDLE_1_GRANT,
    L1_0_NOC_RING0_OUTGOING_0_GRANT,
    L1_0_NOC_RING0_OUTGOING_1_GRANT,
    L1_0_NOC_RING0_INCOMING_0_GRANT,
    L1_0_NOC_RING0_INCOMING_1_GRANT,
    // L1 Bank 1 grants
    L1_1_PORT8_GRANT,  // WH: TDMA Packer 2, BH: RISC Core
    L1_1_EXT_UNPACKER_1_GRANT,
    L1_1_EXT_UNPACKER_2_GRANT,
    L1_1_EXT_UNPACKER_3_GRANT,
    L1_1_NOC_RING1_OUTGOING_0_GRANT,
    L1_1_NOC_RING1_OUTGOING_1_GRANT,
    L1_1_NOC_RING1_INCOMING_0_GRANT,
    L1_1_NOC_RING1_INCOMING_1_GRANT,
    // === Grant counters (accessed via out_fmt bit 16 = 1) ===
    // INSTRN_THREAD grant counters: actual instruction issue counts (8 types x 3 threads = 24)
    CFG_INSTRN_ISSUED_0,
    CFG_INSTRN_ISSUED_1,
    CFG_INSTRN_ISSUED_2,
    SYNC_INSTRN_ISSUED_0,
    SYNC_INSTRN_ISSUED_1,
    SYNC_INSTRN_ISSUED_2,
    THCON_INSTRN_ISSUED_0,
    THCON_INSTRN_ISSUED_1,
    THCON_INSTRN_ISSUED_2,
    XSEARCH_INSTRN_ISSUED_0,
    XSEARCH_INSTRN_ISSUED_1,
    XSEARCH_INSTRN_ISSUED_2,
    MOVE_INSTRN_ISSUED_0,
    MOVE_INSTRN_ISSUED_1,
    MOVE_INSTRN_ISSUED_2,
    FPU_INSTRN_ISSUED_0,
    FPU_INSTRN_ISSUED_1,
    FPU_INSTRN_ISSUED_2,
    UNPACK_INSTRN_ISSUED_0,
    UNPACK_INSTRN_ISSUED_1,
    UNPACK_INSTRN_ISSUED_2,
    PACK_INSTRN_ISSUED_0,
    PACK_INSTRN_ISSUED_1,
    PACK_INSTRN_ISSUED_2,
    // TDMA_UNPACK grant counters (11): detailed write/HF info
    INSTRN_2_HF_CYCLES,           // Math instrns that took 2 HF cycles (grant 257)
    INSTRN_1_HF_CYCLE,            // Math instrns that took 1 HF cycle (grant 258)
    SRCB_WRITE_ACTUAL,            // srcB writes not blocked by overwrite (grant 259)
    SRCA_WRITE_NOT_BLOCKED_OVR,   // srcA writes not blocked by overwrite (grant 260)
    SRCA_WRITE_ACTUAL,            // srcA writes not blocked by port (grant 261)
    SRCB_WRITE_NOT_BLOCKED_PORT,  // srcB writes not blocked by port (grant 262)
    SRCA_WRITE_THREAD0,           // srcA writes from thread 0 (grant 263)
    SRCB_WRITE_THREAD0,           // srcB writes from thread 0 (grant 264)
    SRCA_WRITE_THREAD1,           // srcA writes from thread 1 (grant 265)
    SRCB_WRITE_THREAD1,           // srcB writes from thread 1 (grant 266)
    MATH_INSTRN_NOT_BLOCKED_SRC,  // BH: Math not blocked by src_data_ready (grant 256)
    // TDMA_PACK additional req counters (WH only, BH has these tied to 0)
    PACKER_DEST_READ_1,  // Dest accumulator register 1 read request (req 12)
    PACKER_DEST_READ_2,  // Dest accumulator register 2 read request (req 13)
    PACKER_DEST_READ_3,  // Dest accumulator register 3 read request (req 14)
    PACKER_BUSY_0,       // Packer engine 0 busy (req 15)
    PACKER_BUSY_1,       // Packer engine 1 busy (req 16)
    PACKER_BUSY_2,       // Packer engine 2 busy (req 17)
    // TDMA_PACK grant counters: dest read grants and scoreboard stalls
    DEST_READ_GRANTED_0,           // Dest register 0 read granted (grant 267)
    DEST_READ_GRANTED_1,           // Dest register 1 read granted (grant 268)
    DEST_READ_GRANTED_2,           // Dest register 2 read granted (grant 269)
    DEST_READ_GRANTED_3,           // Dest register 3 read granted (grant 270)
    MATH_NOT_STALLED_DEST_WR_PORT,  // Math not stalled by dest write port (grant 271)
    // Note: AVAILABLE_MATH (existing, counter_sel 272) = math not stalled by scoreboard (grant 272)
    PACK_BANK6_GRANT,   // PACK bank 6 grant (counter_sel 273) — tied to 0 in RTL (2'b00)
    PACK_BANK7_GRANT,   // PACK bank 7 grant (counter_sel 274) — tied to 0 in RTL (2'b00)
    // L1 Bank 4 (BH only, MUX_CTRL[6:4] = 4, monitors misc ports 32-39)
    L1_4_MISC_PORT_0,  // Port 32: misc interface 0
    L1_4_MISC_PORT_1,  // Port 33: misc interface 1
    L1_4_MISC_PORT_2,  // Port 34: misc interface 2
    L1_4_MISC_PORT_3,  // Port 35: misc interface 3
    L1_4_MISC_PORT_4,  // Port 36: misc interface 4
    L1_4_MISC_PORT_5,  // Port 37: misc interface 5
    L1_4_MISC_PORT_6,  // Port 38: misc interface 6
    L1_4_MISC_PORT_7,  // Port 39: misc interface 7
    // L1 Bank 4 grant counters
    L1_4_MISC_PORT_0_GRANT,
    L1_4_MISC_PORT_1_GRANT,
    L1_4_MISC_PORT_2_GRANT,
    L1_4_MISC_PORT_3_GRANT,
    L1_4_MISC_PORT_4_GRANT,
    L1_4_MISC_PORT_5_GRANT,
    L1_4_MISC_PORT_6_GRANT,
    L1_4_MISC_PORT_7_GRANT,
    // L1 Bank 2 (BH only, MUX_CTRL[6:4] = 2, NOC Ring 2 ports 16-23)
    L1_2_NOC_RING2_PORT_0,
    L1_2_NOC_RING2_PORT_1,
    L1_2_NOC_RING2_PORT_2,
    L1_2_NOC_RING2_PORT_3,
    L1_2_NOC_RING2_PORT_4,
    L1_2_NOC_RING2_PORT_5,
    L1_2_NOC_RING2_PORT_6,
    L1_2_NOC_RING2_PORT_7,
    L1_2_NOC_RING2_PORT_0_GRANT,
    L1_2_NOC_RING2_PORT_1_GRANT,
    L1_2_NOC_RING2_PORT_2_GRANT,
    L1_2_NOC_RING2_PORT_3_GRANT,
    L1_2_NOC_RING2_PORT_4_GRANT,
    L1_2_NOC_RING2_PORT_5_GRANT,
    L1_2_NOC_RING2_PORT_6_GRANT,
    L1_2_NOC_RING2_PORT_7_GRANT,
    // L1 Bank 3 (BH only, MUX_CTRL[6:4] = 3, NOC Ring 3 ports 24-31)
    L1_3_NOC_RING3_PORT_0,
    L1_3_NOC_RING3_PORT_1,
    L1_3_NOC_RING3_PORT_2,
    L1_3_NOC_RING3_PORT_3,
    L1_3_NOC_RING3_PORT_4,
    L1_3_NOC_RING3_PORT_5,
    L1_3_NOC_RING3_PORT_6,
    L1_3_NOC_RING3_PORT_7,
    L1_3_NOC_RING3_PORT_0_GRANT,
    L1_3_NOC_RING3_PORT_1_GRANT,
    L1_3_NOC_RING3_PORT_2_GRANT,
    L1_3_NOC_RING3_PORT_3_GRANT,
    L1_3_NOC_RING3_PORT_4_GRANT,
    L1_3_NOC_RING3_PORT_5_GRANT,
    L1_3_NOC_RING3_PORT_6_GRANT,
    L1_3_NOC_RING3_PORT_7_GRANT,
    // Ethernet L1 counters (different port mapping from Tensix)
    // WH Ethernet: no mux, 8 ports. BH Ethernet: 3-bit mux, 4 positions.
    // Mux 0 (WH ports 0-3 / BH ports 0-3)
    ETH_L1_0_NOC_RING1_OUTGOING_0,  // WH port 0 / BH: NOC Ring 0 port 0
    ETH_L1_0_NOC_RING1_OUTGOING_1,  // WH port 1 / BH: NOC Ring 0 port 1
    ETH_L1_0_NOC_RING1_INCOMING_0,  // WH port 2 / BH: NOC Ring 0 port 2
    ETH_L1_0_NOC_RING1_INCOMING_1,  // WH port 3 / BH: NOC Ring 0 port 3
    ETH_L1_0_NOC_RING0_OUTGOING_0,  // WH port 4 / BH: NOC Ring 0 port 4 (not used on BH mux 0)
    ETH_L1_0_NOC_RING0_OUTGOING_1,  // WH port 5
    ETH_L1_0_NOC_RING0_INCOMING_0,  // WH port 6
    ETH_L1_0_NOC_RING0_INCOMING_1,  // WH port 7
    ETH_L1_0_NOC_RING1_OUTGOING_0_GRANT,
    ETH_L1_0_NOC_RING1_OUTGOING_1_GRANT,
    ETH_L1_0_NOC_RING1_INCOMING_0_GRANT,
    ETH_L1_0_NOC_RING1_INCOMING_1_GRANT,
    ETH_L1_0_NOC_RING0_OUTGOING_0_GRANT,
    ETH_L1_0_NOC_RING0_OUTGOING_1_GRANT,
    ETH_L1_0_NOC_RING0_INCOMING_0_GRANT,
    ETH_L1_0_NOC_RING0_INCOMING_1_GRANT,
    // BH Ethernet L1 mux 1 (ports 4-7): NOC Ring 1
    ETH_L1_1_NOC_RING1_OUTGOING_0,
    ETH_L1_1_NOC_RING1_OUTGOING_1,
    ETH_L1_1_NOC_RING1_INCOMING_0,
    ETH_L1_1_NOC_RING1_INCOMING_1,
    ETH_L1_1_PORT_4,
    ETH_L1_1_PORT_5,
    ETH_L1_1_PORT_6,
    ETH_L1_1_PORT_7,
    ETH_L1_1_NOC_RING1_OUTGOING_0_GRANT,
    ETH_L1_1_NOC_RING1_OUTGOING_1_GRANT,
    ETH_L1_1_NOC_RING1_INCOMING_0_GRANT,
    ETH_L1_1_NOC_RING1_INCOMING_1_GRANT,
    ETH_L1_1_PORT_4_GRANT,
    ETH_L1_1_PORT_5_GRANT,
    ETH_L1_1_PORT_6_GRANT,
    ETH_L1_1_PORT_7_GRANT,
    // BH Ethernet L1 mux 2 (ports 8-11): ECC, RISC, ETH0 MAC
    ETH_L1_2_ECC_SCRUBBER,
    ETH_L1_2_RISC,
    ETH_L1_2_ETH0_0,
    ETH_L1_2_ETH0_1,
    ETH_L1_2_PORT_12,
    ETH_L1_2_PORT_13,
    ETH_L1_2_PORT_14,
    ETH_L1_2_PORT_15,
    ETH_L1_2_ECC_SCRUBBER_GRANT,
    ETH_L1_2_RISC_GRANT,
    ETH_L1_2_ETH0_0_GRANT,
    ETH_L1_2_ETH0_1_GRANT,
    ETH_L1_2_PORT_12_GRANT,
    ETH_L1_2_PORT_13_GRANT,
    ETH_L1_2_PORT_14_GRANT,
    ETH_L1_2_PORT_15_GRANT,
    // BH Ethernet L1 mux 3 (ports 12-15): ETH1 MAC, unused
    ETH_L1_3_ETH1_0,
    ETH_L1_3_ETH1_1,
    ETH_L1_3_UNUSED_2,
    ETH_L1_3_UNUSED_3,
    ETH_L1_3_PORT_20,
    ETH_L1_3_PORT_21,
    ETH_L1_3_PORT_22,
    ETH_L1_3_PORT_23,
    ETH_L1_3_ETH1_0_GRANT,
    ETH_L1_3_ETH1_1_GRANT,
    ETH_L1_3_UNUSED_2_GRANT,
    ETH_L1_3_UNUSED_3_GRANT,
    ETH_L1_3_PORT_20_GRANT,
    ETH_L1_3_PORT_21_GRANT,
    ETH_L1_3_PORT_22_GRANT,
    ETH_L1_3_PORT_23_GRANT
};

union PerfCounter {
    struct {
        uint32_t counter_value;
        uint32_t ref_cnt : 23;
        uint32_t counter_type : 9;
    } __attribute__((packed));
    uint64_t raw_data;

    PerfCounter() = delete;
    PerfCounter(uint32_t counter_value, uint32_t ref_cnt, PerfCounterType counter_type) :
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(static_cast<uint32_t>(counter_type)) {}

    PerfCounter(uint64_t raw_data) : raw_data(raw_data) {}
};
static_assert(sizeof(PerfCounter) == sizeof(uint64_t), "PerfCounter must be 64-bit");

// Perf counter start/stop runs on TRISC1 (wraps the compute kernel).
// Counter readout and DRAM push runs on BRISC (has NOC access for DRAM writes).
// ERISC perf counters disabled: MMIO readback hardwired to 0 in RTL.
#if defined(PROFILE_PERF_COUNTERS) && (COMPILE_FOR_TRISC == 1 || defined(COMPILE_FOR_BRISC))

#include "kernel_profiler.hpp"
#include "api/debug/assert.h"

namespace kernel_profiler {

constexpr PerfCounterGroup counter_groups[] = {
    PerfCounterGroup::FPU,
    PerfCounterGroup::PACK,
    PerfCounterGroup::UNPACK,
    PerfCounterGroup::L1_0,
    PerfCounterGroup::L1_1,
    PerfCounterGroup::INSTRN,
    PerfCounterGroup::L1_2,
    PerfCounterGroup::L1_3,
    PerfCounterGroup::L1_4};
constexpr size_t NUM_COUNTER_GROUPS = sizeof(counter_groups) / sizeof(counter_groups[0]);
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 3> fpu_counters = {
    {{PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::SFPU_COUNTER, 1}, {PerfCounterType::MATH_COUNTER, 257}}};
constexpr size_t NUM_FPU_COUNTERS = 3;

// Architecture-specific counter arrays (unpack, pack, l1_0-l1_4, instrn)
#if defined(ARCH_BLACKHOLE)
#include "tt_metal/hw/inc/internal/tt-1xx/blackhole/hw_counters.h"
#else
#include "tt_metal/hw/inc/internal/tt-1xx/wormhole/hw_counters.h"
#endif

// bit masks for the different counter groups
#define PROFILE_PERF_COUNTERS_FPU (1 << 0)
#define PROFILE_PERF_COUNTERS_PACK (1 << 1)
#define PROFILE_PERF_COUNTERS_UNPACK (1 << 2)
#define PROFILE_PERF_COUNTERS_L1_0 (1 << 3)
#define PROFILE_PERF_COUNTERS_L1_1 (1 << 4)
#define PROFILE_PERF_COUNTERS_INSTRN (1 << 5)
#define PROFILE_PERF_COUNTERS_L1_2 (1 << 6)
#define PROFILE_PERF_COUNTERS_L1_3 (1 << 7)
#define PROFILE_PERF_COUNTERS_L1_4 (1 << 8)

/*
Performance Counter registers (to understand programming sequence in start_perf_counter/stop_perf_counter)

Control Registers:

The control registers are RISCV_DEBUG_REG_PERF_CNT_<X>0/1/2 (where X is FPU/TDMA_PACK/TDMA_UNPACK/L1/INSTRN_THREAD)
- RISCV_DEBUG_REG_PERF_CNT_<X>0: Reference period (in cycles)
- RISCV_DEBUG_REG_PERF_CNT_<X>1: Mode register
    - Bits [7:0]: Mode (0=manual, 1=auto-stop, 2=wrap)
        0 = Continuous
        1 = Count until refclk count number of cycles have been hit
        2 = Continuous and don’t maintain refclk count (pretty much same as 0)
    - Bits [12:8]: Bank select (i.e. which counter to read; see PerfCounterType enum for available counters and their
bank select values)
    - Bit [16]: Selects whether to output req or grant count onto RISCV_DEBUG_REG_PERF_CNT_OUT_H_<X>
        Format (0=req_cnt, 1=grant_cnt)
- RISCV_DEBUG_REG_PERF_CNT_<X>2: Control (bit[0]=start, bit[1]=stop)
    - note that these take effect on rising edge only
    - also transitioning the start bit from 0 to 1 clears the counters

Data registers:

- RISCV_DEBUG_REG_PERF_CNT_OUT_L_<X>: contains ref_cnt (i.e. cycles between start and stop)
- RISCV_DEBUG_REG_PERF_CNT_OUT_H_<X>: contains req count if RISCV_DEBUG_REG_PERF_CNT_<X>1[16] is 0
    and grant count if it is 1

Note: Currently only support for obtaining FPU counters (and deriving util metrics from them) is present
but this can be extended easily
*/

#define PERF_CNT_CONTINUOUS_MODE 0
#define PERF_CNT_BANK_SELECT_SHIFT 8
#define PERF_CNT_START_VALUE 1
#define PERF_CNT_STOP_VALUE 2

uint32_t get_cntl_register_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t reg_addr = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: reg_addr = RISCV_DEBUG_REG_PERF_CNT_FPU0; break;
        case PerfCounterGroup::PACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0; break;
        case PerfCounterGroup::UNPACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0; break;
        case PerfCounterGroup::L1_0:
        case PerfCounterGroup::L1_1:
        case PerfCounterGroup::L1_2:
        case PerfCounterGroup::L1_3:
        case PerfCounterGroup::L1_4: reg_addr = RISCV_DEBUG_REG_PERF_CNT_L1_0; break;
        case PerfCounterGroup::INSTRN: reg_addr = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return reg_addr;
}

uint32_t get_read_register_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t reg_addr = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU; break;
        case PerfCounterGroup::PACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK; break;
        case PerfCounterGroup::UNPACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK; break;
        case PerfCounterGroup::L1_0:
        case PerfCounterGroup::L1_1:
        case PerfCounterGroup::L1_2:
        case PerfCounterGroup::L1_3:
        case PerfCounterGroup::L1_4: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1; break;
        case PerfCounterGroup::INSTRN: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return reg_addr;
}

uint32_t get_flag_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t flag = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: flag = PROFILE_PERF_COUNTERS_FPU; break;
        case PerfCounterGroup::PACK: flag = PROFILE_PERF_COUNTERS_PACK; break;
        case PerfCounterGroup::UNPACK: flag = PROFILE_PERF_COUNTERS_UNPACK; break;
        case PerfCounterGroup::L1_0: flag = PROFILE_PERF_COUNTERS_L1_0; break;
        case PerfCounterGroup::L1_1: flag = PROFILE_PERF_COUNTERS_L1_1; break;
        case PerfCounterGroup::INSTRN: flag = PROFILE_PERF_COUNTERS_INSTRN; break;
        case PerfCounterGroup::L1_2: flag = PROFILE_PERF_COUNTERS_L1_2; break;
        case PerfCounterGroup::L1_3: flag = PROFILE_PERF_COUNTERS_L1_3; break;
        case PerfCounterGroup::L1_4: flag = PROFILE_PERF_COUNTERS_L1_4; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return flag;
}

uint32_t get_num_counters_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t num_counters = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: num_counters = NUM_FPU_COUNTERS; break;
        case PerfCounterGroup::UNPACK: num_counters = NUM_UNPACK_COUNTERS; break;
        case PerfCounterGroup::PACK: num_counters = NUM_PACK_COUNTERS; break;
        case PerfCounterGroup::L1_0: num_counters = NUM_L1_0_COUNTERS; break;
        case PerfCounterGroup::L1_1: num_counters = NUM_L1_1_COUNTERS; break;
        case PerfCounterGroup::INSTRN: num_counters = NUM_INSTRN_COUNTERS; break;
#if defined(ARCH_BLACKHOLE) || defined(COMPILE_FOR_ERISC)
        case PerfCounterGroup::L1_2: num_counters = NUM_L1_2_COUNTERS; break;
        case PerfCounterGroup::L1_3: num_counters = NUM_L1_3_COUNTERS; break;
#endif
#if defined(ARCH_BLACKHOLE)
        case PerfCounterGroup::L1_4: num_counters = NUM_L1_4_COUNTERS; break;
#endif
        default: {
            ASSERT(false);
            break;
        }
    }
    return num_counters;
}

FORCE_INLINE const std::pair<PerfCounterType, uint16_t>* get_counters_for_counter_group(
    PerfCounterGroup counter_group) {
    switch (counter_group) {
        case PerfCounterGroup::FPU: return fpu_counters.data();
        case PerfCounterGroup::UNPACK: return unpack_counters.data();
        case PerfCounterGroup::PACK: return pack_counters.data();
        case PerfCounterGroup::L1_0: return l1_0_counters.data();
        case PerfCounterGroup::L1_1: return l1_1_counters.data();
        case PerfCounterGroup::INSTRN: return instrn_counters.data();
#if defined(ARCH_BLACKHOLE) || defined(COMPILE_FOR_ERISC)
        case PerfCounterGroup::L1_2: return l1_2_counters.data();
        case PerfCounterGroup::L1_3: return l1_3_counters.data();
#endif
#if defined(ARCH_BLACKHOLE)
        case PerfCounterGroup::L1_4: return l1_4_counters.data();
#endif
        default: {
            ASSERT(false);
            return fpu_counters.data();
        }
    }
}

void set_l1_mux_ctrl(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr uint32_t* mux_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
    uint32_t mux_val = *mux_reg;
    // Blackhole: 3-bit L1 mux at MUX_CTRL[6:4], values 0-4
    // Wormhole:  1-bit L1 mux at MUX_CTRL[4], values 0-1
#if defined(ARCH_BLACKHOLE)
    constexpr uint32_t L1_MUX_MASK = 0x7 << 4;  // 3 bits [6:4]
#else
    constexpr uint32_t L1_MUX_MASK = 0x1 << 4;  // 1 bit [4]
#endif
    uint32_t mux_sel = 0;
    if (counter_group == PerfCounterGroup::L1_1) {
        mux_sel = 1;
    } else if (counter_group == PerfCounterGroup::L1_2) {
        mux_sel = 2;
    } else if (counter_group == PerfCounterGroup::L1_3) {
        mux_sel = 3;
    } else if (counter_group == PerfCounterGroup::L1_4) {
        mux_sel = 4;
    }
    mux_val = (mux_val & ~L1_MUX_MASK) | (mux_sel << 4);
    *mux_reg = mux_val;
}

__attribute__((noinline)) void start_single_group(PerfCounterGroup counter_group) {
    if (counter_group == PerfCounterGroup::L1_0 || counter_group == PerfCounterGroup::L1_1 ||
        counter_group == PerfCounterGroup::L1_2 || counter_group == PerfCounterGroup::L1_3 ||
        counter_group == PerfCounterGroup::L1_4) {
        set_l1_mux_ctrl(counter_group);
    }
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    cntl_reg[0] = 0xFFFFFFFF;
    cntl_reg[1] = PERF_CNT_CONTINUOUS_MODE;
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_START_VALUE;
}

__attribute__((noinline)) void stop_single_group(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_STOP_VALUE;
}

__attribute__((noinline)) void read_single_group(PerfCounterGroup counter_group) {
    if (counter_group == PerfCounterGroup::L1_0 || counter_group == PerfCounterGroup::L1_1 ||
        counter_group == PerfCounterGroup::L1_2 || counter_group == PerfCounterGroup::L1_3 ||
        counter_group == PerfCounterGroup::L1_4) {
        set_l1_mux_ctrl(counter_group);
    }
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    volatile tt_reg_ptr uint32_t* read_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_read_register_for_counter_group(counter_group));
    const auto* counters = get_counters_for_counter_group(counter_group);
    const uint32_t counters_size = get_num_counters_for_counter_group(counter_group);
    for (unsigned int i = 0; i < counters_size; i++) {
        uint32_t counter_sel = counters[i].second;
        cntl_reg[1] = counter_sel << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
        (void)read_reg[0];
        (void)read_reg[1];
        uint32_t ref_cnt_val = read_reg[0];
        uint32_t counter_val = read_reg[1];
        PerfCounter counter(counter_val, ref_cnt_val, counters[i].first);
        timeStampedData<PERF_COUNTER_PROFILER_ID>(counter.raw_data);
    }
}

// Use preprocessor #if to select groups at compile time, avoiding
// runtime loop optimization issues with the RISC-V compiler.
void start_perf_counter() {
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_FPU)
    start_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_PACK)
    start_single_group(PerfCounterGroup::PACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_UNPACK)
    start_single_group(PerfCounterGroup::UNPACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_0)
    start_single_group(PerfCounterGroup::L1_0);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_1)
    start_single_group(PerfCounterGroup::L1_1);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_INSTRN)
    start_single_group(PerfCounterGroup::INSTRN);
#endif
#if defined(ARCH_BLACKHOLE)
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_2)
    start_single_group(PerfCounterGroup::L1_2);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_3)
    start_single_group(PerfCounterGroup::L1_3);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_4)
    start_single_group(PerfCounterGroup::L1_4);
#endif
#endif
}

// stop_perf_counter: stops all enabled counter groups (freezes hardware counters).
// Called from TRISC1 at the end of the compute kernel scope.
// Does NOT read counter values — that happens on BRISC which has NOC access for DRAM push.
void stop_perf_counter() {
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_FPU)
    stop_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_PACK)
    stop_single_group(PerfCounterGroup::PACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_UNPACK)
    stop_single_group(PerfCounterGroup::UNPACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_0)
    stop_single_group(PerfCounterGroup::L1_0);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_1)
    stop_single_group(PerfCounterGroup::L1_1);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_INSTRN)
    stop_single_group(PerfCounterGroup::INSTRN);
#endif
#if defined(ARCH_BLACKHOLE)
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_2)
    stop_single_group(PerfCounterGroup::L1_2);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_3)
    stop_single_group(PerfCounterGroup::L1_3);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_4)
    stop_single_group(PerfCounterGroup::L1_4);
#endif
#endif
};

// read_perf_counters: reads all enabled counter groups and writes markers to the profiler buffer.
// Called from BRISC after wait_ncrisc_trisc() — BRISC has NOC access so it can push the L1
// profiler buffer to DRAM (via quick_push) between groups when the buffer fills up.
// The perf counter debug registers are shared across all RISCs on the Tensix core,
// so BRISC can read counter values that were started/stopped by TRISC1.
void read_perf_counters() {
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_FPU)
    read_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_PACK)
    read_single_group(PerfCounterGroup::PACK);
    if (!bufferHasRoom(0)) { quick_push(); }
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_UNPACK)
    read_single_group(PerfCounterGroup::UNPACK);
    if (!bufferHasRoom(0)) { quick_push(); }
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_0)
    read_single_group(PerfCounterGroup::L1_0);
    if (!bufferHasRoom(0)) { quick_push(); }
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_1)
    read_single_group(PerfCounterGroup::L1_1);
    if (!bufferHasRoom(0)) { quick_push(); }
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_INSTRN)
    read_single_group(PerfCounterGroup::INSTRN);
    if (!bufferHasRoom(0)) { quick_push(); }
#endif
#if defined(ARCH_BLACKHOLE)
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_2)
    read_single_group(PerfCounterGroup::L1_2);
    if (!bufferHasRoom(0)) { quick_push(); }
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_3)
    read_single_group(PerfCounterGroup::L1_3);
    if (!bufferHasRoom(0)) { quick_push(); }
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_4)
    read_single_group(PerfCounterGroup::L1_4);
#endif
#endif  // ARCH_BLACKHOLE
};

// TRISC1: RAII wrapper that starts counters in constructor and stops in destructor.
// Counter readout happens later on BRISC via read_perf_counters().
struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};

}  // namespace kernel_profiler

#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define ReadPerfCounters() kernel_profiler::read_perf_counters();
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define ReadPerfCounters()
#define RecordPerfCounters()

#endif
