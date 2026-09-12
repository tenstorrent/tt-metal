// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace llk::perf
{

// The ordinal is the wire format: the metal profiler record carries it in an 8-bit field and every host
// decoder resolves names by position. Append new counters at the end, never reorder or remove.
enum PerfCounterType : std::uint16_t
{
    UNDEF = 0,
    // FPU Group
    FPU_COUNTER,
    SFPU_COUNTER,
    MATH_COUNTER,
    // TDMA_UNPACK Group
    MATH_SRC_DATA_READY,
    MATH_NOT_D2S_STALLED,
    MATH_FIDELITY_STALL, // tied off in hardware, kept so the ordinals below do not shift
    MATH_INSTRN_STARTED,
    MATH_INSTRN_AVAILABLE,
    SRCB_WRITE_REQ,
    SRCA_WRITE_REQ,
    UNPACK0_BUSY_THREAD0,
    UNPACK1_BUSY_THREAD0,
    UNPACK0_BUSY_THREAD1,
    UNPACK1_BUSY_THREAD1,
    MATH_INSTRN_HF_1_CYCLE, // fidelity selector is tied off, so this duplicated MATH_INSTRN_STARTED; kept so the
                            // ordinals below do not shift
    MATH_INSTRN_HF_2_CYCLE, // constant zero, fidelity selector is tied off
    MATH_INSTRN_HF_4_CYCLE, // constant zero, fidelity selector is tied off
    // TDMA_PACK Group
    PACKER0_DEST_READ_REQ,
    PACKER_BUSY,
    MATH_NOT_SCOREBOARD_STALLED,
    // INSTRN_THREAD Group
    CFG_INSTRN_AVAILABLE_0,
    CFG_INSTRN_AVAILABLE_1,
    CFG_INSTRN_AVAILABLE_2,
    SYNC_INSTRN_AVAILABLE_0,
    SYNC_INSTRN_AVAILABLE_1,
    SYNC_INSTRN_AVAILABLE_2,
    THCON_INSTRN_AVAILABLE_0,
    THCON_INSTRN_AVAILABLE_1,
    THCON_INSTRN_AVAILABLE_2,
    MOVE_INSTRN_AVAILABLE_0,
    MOVE_INSTRN_AVAILABLE_1,
    MOVE_INSTRN_AVAILABLE_2,
    MATH_INSTRN_AVAILABLE_0,
    MATH_INSTRN_AVAILABLE_1,
    MATH_INSTRN_AVAILABLE_2,
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
    WAITING_FOR_CFG_IDLE_0,
    WAITING_FOR_CFG_IDLE_1,
    WAITING_FOR_CFG_IDLE_2,
    WAITING_FOR_SFPU_IDLE_0,
    WAITING_FOR_SFPU_IDLE_1,
    WAITING_FOR_SFPU_IDLE_2,
    // L1 Bank 0 (mux=0, ports 0-7)
    L1_0_UNPACKER_0,
    L1_0_UNPACKER_1_ECC_PACK1, // Wormhole port 1; Blackhole uses L1_0_UNPACKER_1_ECC
    L1_0_TDMA_BUNDLE_0_RISC,
    L1_0_TDMA_BUNDLE_1_TRISC,
    L1_0_NOC_RING0_OUTGOING_0,
    L1_0_NOC_RING0_OUTGOING_1,
    L1_0_NOC_RING0_INCOMING_0,
    L1_0_NOC_RING0_INCOMING_1,
    // L1 Bank 1 (mux=1, ports 8-15)
    L1_1_TDMA_PACKER_2,  // Wormhole port 8; Blackhole uses L1_1_PACKER_IF_0
    L1_1_EXT_UNPACKER_1, // Wormhole ports 9-11; Blackhole uses L1_1_UNPACKER1_EXT_IF_1-3
    L1_1_EXT_UNPACKER_2,
    L1_1_EXT_UNPACKER_3,
    L1_1_NOC_RING1_OUTGOING_0,
    L1_1_NOC_RING1_OUTGOING_1,
    L1_1_NOC_RING1_INCOMING_0,
    L1_1_NOC_RING1_INCOMING_1,
    L1_0_UNIFIED_PACKER, // retired, kept so the ordinals below do not shift
    L1_1_RISC_CORE,      // retired, kept so the ordinals below do not shift
    // L1 grant counters (reqif_ready)
    L1_0_UNPACKER_0_GRANT,
    L1_0_PORT1_GRANT,
    L1_0_TDMA_BUNDLE_0_GRANT,
    L1_0_TDMA_BUNDLE_1_GRANT,
    L1_0_NOC_RING0_OUTGOING_0_GRANT,
    L1_0_NOC_RING0_OUTGOING_1_GRANT,
    L1_0_NOC_RING0_INCOMING_0_GRANT,
    L1_0_NOC_RING0_INCOMING_1_GRANT,
    L1_1_PORT8_GRANT,
    L1_1_EXT_UNPACKER_1_GRANT,
    L1_1_EXT_UNPACKER_2_GRANT,
    L1_1_EXT_UNPACKER_3_GRANT,
    L1_1_NOC_RING1_OUTGOING_0_GRANT,
    L1_1_NOC_RING1_OUTGOING_1_GRANT,
    L1_1_NOC_RING1_INCOMING_0_GRANT,
    L1_1_NOC_RING1_INCOMING_1_GRANT,
    // === Grant-side counters (accessed via out_fmt bit 16 = 1) ===
    THREAD_INSTRUCTIONS_0,
    THREAD_INSTRUCTIONS_1,
    THREAD_INSTRUCTIONS_2,
    SRCB_WRITE_NOT_BLOCKED_OVR,
    SRCA_WRITE_NOT_BLOCKED_OVR,
    SRCA_WRITE_NOT_BLOCKED_PORT,
    SRCB_WRITE_NOT_BLOCKED_PORT,
    SRCA_WRITE_TID_EVEN,
    SRCB_WRITE_TID_EVEN,
    SRCA_WRITE_TID_ODD,
    SRCB_WRITE_TID_ODD,
    // TDMA_PACK additional req counters (WH only)
    PACKER_DEST_READ_1,
    PACKER_DEST_READ_2,
    PACKER_DEST_READ_3,
    PACKER_BUSY_0,
    PACKER_BUSY_1,
    PACKER_BUSY_2,
    DEST_READ_GRANTED_0,
    DEST_READ_GRANTED_1,
    DEST_READ_GRANTED_2,
    DEST_READ_GRANTED_3,
    MATH_NOT_STALLED_DEST_WR_PORT,
    // L1 Bank 4 (BH only, mux=4, ports 32-39): extended packers 6-7, packer L1 interface 1 (port 34, shared with the
    // tag-search accelerator, debug L1 RAM and timestamp), unpacker 0's extended read interfaces 1-5 (ports 35-39).
    L1_4_EXT_PACKER_6,
    L1_4_EXT_PACKER_7,
    L1_4_PACKER_IF_1_TAG_SEARCH,
    L1_4_UNPACKER0_EXT_IF_1,
    L1_4_UNPACKER0_EXT_IF_2,
    L1_4_UNPACKER0_EXT_IF_3,
    L1_4_UNPACKER0_EXT_IF_4,
    L1_4_UNPACKER0_EXT_IF_5,
    L1_4_EXT_PACKER_6_GRANT,
    L1_4_EXT_PACKER_7_GRANT,
    L1_4_PACKER_IF_1_TAG_SEARCH_GRANT,
    L1_4_UNPACKER0_EXT_IF_1_GRANT,
    L1_4_UNPACKER0_EXT_IF_2_GRANT,
    L1_4_UNPACKER0_EXT_IF_3_GRANT,
    L1_4_UNPACKER0_EXT_IF_4_GRANT,
    L1_4_UNPACKER0_EXT_IF_5_GRANT,
    // L1 Bank 2 (BH only, mux=2, ports 16-23): unpacker 1's extended read interfaces 4-7 (also used by the
    // packer L1-to-L1 read) and NOC ring 0 ports 2-3.
    L1_2_UNPACKER1_EXT_IF_4,
    L1_2_UNPACKER1_EXT_IF_5,
    L1_2_UNPACKER1_EXT_IF_6,
    L1_2_UNPACKER1_EXT_IF_7,
    L1_2_NOC_RING0_OUTGOING_2,
    L1_2_NOC_RING0_OUTGOING_3,
    L1_2_NOC_RING0_INCOMING_2,
    L1_2_NOC_RING0_INCOMING_3,
    L1_2_UNPACKER1_EXT_IF_4_GRANT,
    L1_2_UNPACKER1_EXT_IF_5_GRANT,
    L1_2_UNPACKER1_EXT_IF_6_GRANT,
    L1_2_UNPACKER1_EXT_IF_7_GRANT,
    L1_2_NOC_RING0_OUTGOING_2_GRANT,
    L1_2_NOC_RING0_OUTGOING_3_GRANT,
    L1_2_NOC_RING0_INCOMING_2_GRANT,
    L1_2_NOC_RING0_INCOMING_3_GRANT,
    // L1 Bank 3 (BH only, mux=3, ports 24-31: NOC ring 1 ports 2-3 and extended packers 2-5)
    L1_3_NOC_RING1_OUTGOING_2,
    L1_3_NOC_RING1_OUTGOING_3,
    L1_3_NOC_RING1_INCOMING_2,
    L1_3_NOC_RING1_INCOMING_3,
    L1_3_EXT_PACKER_2,
    L1_3_EXT_PACKER_3,
    L1_3_EXT_PACKER_4,
    L1_3_EXT_PACKER_5,
    L1_3_NOC_RING1_OUTGOING_2_GRANT,
    L1_3_NOC_RING1_OUTGOING_3_GRANT,
    L1_3_NOC_RING1_INCOMING_2_GRANT,
    L1_3_NOC_RING1_INCOMING_3_GRANT,
    L1_3_EXT_PACKER_2_GRANT,
    L1_3_EXT_PACKER_3_GRANT,
    L1_3_EXT_PACKER_4_GRANT,
    L1_3_EXT_PACKER_5_GRANT,
    ANY_THREAD_STALL,
    // L1 Bank 5 (BH only, mux=5, ports 40-41): unpacker 0's extended read interfaces 6-7; slots 2-7 read 0.
    L1_5_UNPACKER0_EXT_IF_6,
    L1_5_UNPACKER0_EXT_IF_7,
    L1_5_UNPACKER0_EXT_IF_6_GRANT,
    L1_5_UNPACKER0_EXT_IF_7_GRANT,
    // Blackhole ports whose client differs from Wormhole (tapeout RTL): port 1 has no packer, port 8 is the
    // packer's L1 interface 0, ports 9-11 are unpacker 1's extended read interfaces (also the packer L1-to-L1 read).
    L1_0_UNPACKER_1_ECC,
    L1_1_PACKER_IF_0,
    L1_1_UNPACKER1_EXT_IF_1,
    L1_1_UNPACKER1_EXT_IF_2,
    L1_1_UNPACKER1_EXT_IF_3,
    L1_0_UNPACKER_1_ECC_GRANT,
    L1_1_PACKER_IF_0_GRANT,
    L1_1_UNPACKER1_EXT_IF_1_GRANT,
    L1_1_UNPACKER1_EXT_IF_2_GRANT,
    L1_1_UNPACKER1_EXT_IF_3_GRANT,
    // === Quasar (A0) ===
    // Thread 3 variants (Quasar NEOs run 4 threads)
    CFG_INSTRN_AVAILABLE_3,
    SYNC_INSTRN_AVAILABLE_3,
    THCON_INSTRN_AVAILABLE_3,
    MATH_INSTRN_AVAILABLE_3,
    UNPACK_INSTRN_AVAILABLE_3,
    PACK_INSTRN_AVAILABLE_3,
    THREAD_STALLS_3,
    THREAD_INSTRUCTIONS_3,
    // Quasar-only instruction classes. XSEARCH requests are tied to 0 in the RTL and its grants alias
    // THREAD_INSTRUCTIONS, so it is not in the table; the enumerators stay so the ordinals below do not shift.
    XSEARCH_INSTRN_AVAILABLE_0,
    XSEARCH_INSTRN_AVAILABLE_1,
    XSEARCH_INSTRN_AVAILABLE_2,
    XSEARCH_INSTRN_AVAILABLE_3,
    INSTISSUE_INSTRN_AVAILABLE_0,
    INSTISSUE_INSTRN_AVAILABLE_1,
    INSTISSUE_INSTRN_AVAILABLE_2,
    INSTISSUE_INSTRN_AVAILABLE_3,
    // Stall reasons, OR-reduced across the 4 threads (Quasar has no per-thread reason counters)
    TILE_COUNTER_STALL_PACK,
    TILE_COUNTER_STALL_UNPACK,
    SRCS_STALL_PACK,
    SRCS_STALL_SFPU,
    SRCS_STALL_UNPACK,
    DEST_STALL_PACK,
    DEST_STALL_SFPU,
    DEST_STALL_MATH,
    DEST_STALL_UNPACK,
    SFPU_DATA_HAZARD_STALL,
    FPU_DATA_HAZARD_STALL,
    SRCB_STALL_UNPACK,
    SRCA_STALL_UNPACK,
    DVALID_STALL_MATH,
    SRCA_STALL_MATH,
    // The l1_client CSR event counter; records carry QUASAR_L1_CLIENT_EVENT_BASE + (subport*8 + event).
    QUASAR_L1_CLIENT_EVENT,
    // Quasar runs 3 unpackers per thread.
    UNPACK2_BUSY_THREAD0,
    // Values stay below 256: the l1_client selections are encoded above them in counter_type.
};

static_assert(UNPACK2_BUSY_THREAD0 <= 255, "PerfCounterType must leave 256 and up for l1_client selections");

// Quasar's l1_client counter has 296 selections (subport*8 + event) behind one enum value. A record
// carries QUASAR_L1_CLIENT_EVENT_BASE + selection in counter_type; the host maps it back.
inline constexpr std::uint32_t QUASAR_L1_CLIENT_EVENT_BASE    = 256;
inline constexpr std::uint32_t QUASAR_L1_CLIENT_NUM_SUBPORTS  = 37;
inline constexpr std::uint32_t QUASAR_L1_CLIENT_NUM_EVENTS    = 8;
inline constexpr std::uint32_t QUASAR_L1_CLIENT_NUM_SELECTIONS = QUASAR_L1_CLIENT_NUM_SUBPORTS * QUASAR_L1_CLIENT_NUM_EVENTS;
static_assert(QUASAR_L1_CLIENT_NUM_SELECTIONS == 296, "the host side hard-codes 296 l1_client selections");

// One physical counter block each. The order is the LLK harness on-wire bank id, so it must not change.
enum class Bank : std::uint8_t
{
    INSTRN_THREAD = 0,
    FPU           = 1,
    TDMA_UNPACK   = 2,
    L1            = 3,
    TDMA_PACK     = 4,
};

inline constexpr std::size_t NUM_BANKS = 5;

// {counter name, hardware select}. Selects 256 and up read the grant side of the slice (mode bit 16).
using Entry = std::pair<PerfCounterType, std::uint16_t>;

} // namespace llk::perf
