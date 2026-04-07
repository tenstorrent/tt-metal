// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Debug checkpoint API for fused kernels.
//
// Provides synchronized barriers where all active RISCs on a core halt together,
// dump circular buffer state via DPRINT, then proceed in unison.
//
// Usage in kernel code:
//   DEBUG_CHECKPOINT(1);                          // barrier + dump CB metadata + barrier
//   DEBUG_CHECKPOINT_EX(2, 4, 8, true);           // dump first 4 CBs, 8 words each, + dest regs
//
// All active RISCs must call the same checkpoint ID. The macro is a no-op when
// DEBUG_CHECKPOINT_ENABLED is not defined.
//
// The checkpoint state (12 bytes) is stored at MEM_LLK_DEBUG_BASE in L1,
// which is a 1024-byte debug region shared by all RISCs on the core.

#pragma once

#include "dev_mem_map.h"
#include "internal/hw_thread.h"
#include "api/debug/waypoint.h"

// Checkpoints work independently of DPRINT/DEVICE_PRINT. When a print backend
// is available, the dump phase prints CB metadata and optionally L1 data / dest
// registers. When no print backend is enabled, the checkpoint still acts as a
// barrier (all RISCs synchronize) but skips the dump.
#if defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
#define CHECKPOINT_PRINT_ENABLED 1
#include "api/debug/dprint.h"
#include "api/debug/device_print.h"
#include "internal/circular_buffer_interface.h"
#if defined(COMPILE_FOR_TRISC) && (COMPILE_FOR_TRISC == 1)
#include "api/debug/dprint_tensix.h"
#endif
#endif

#if defined(DEBUG_CHECKPOINT_ENABLED)

// Checkpoint state lives at a fixed L1 address (start of MEM_LLK_DEBUG region).
// This struct is 12 bytes. All RISCs on the core share this L1 location.
struct debug_checkpoint_state_t {
    volatile uint32_t participant_mask;  // Bitmask of active RISC thread indices
    volatile uint32_t arrived_mask;      // Each RISC sets its bit on arrival
    volatile uint32_t proceed;           // Monotonically increasing release counter
};
static_assert(sizeof(debug_checkpoint_state_t) <= 1024, "Checkpoint state must fit in MEM_LLK_DEBUG region");

inline volatile debug_checkpoint_state_t tt_l1_ptr* get_checkpoint_state() {
    return reinterpret_cast<volatile debug_checkpoint_state_t tt_l1_ptr*>(MEM_LLK_DEBUG_BASE);
}

// ---------------------------------------------------------------------------
// Init: called by BRISC/DM0 before launching subordinate kernels.
// Sets participant_mask and clears barrier state.
// ---------------------------------------------------------------------------
inline void debug_checkpoint_init(uint32_t enables) {
    volatile debug_checkpoint_state_t tt_l1_ptr* ckpt = get_checkpoint_state();
    ckpt->participant_mask = enables;
    ckpt->arrived_mask = 0;
    ckpt->proceed = 0;
}

// ---------------------------------------------------------------------------
// Barrier: all active RISCs synchronize at a checkpoint.
// Uses a monotonically increasing proceed counter to avoid race conditions
// when the barrier is called multiple times (e.g., before and after dump).
// ---------------------------------------------------------------------------
inline void debug_checkpoint_barrier(uint32_t expected_proceed) {
    volatile debug_checkpoint_state_t tt_l1_ptr* ckpt = get_checkpoint_state();
    uint32_t my_idx = internal_::get_hw_thread_idx();
    uint32_t my_bit = 1u << my_idx;
    uint32_t mask = ckpt->participant_mask;

    // Signal arrival by setting our bit
    ckpt->arrived_mask |= my_bit;

    if (my_idx == 0) {
        // Orchestrator (BRISC/DM0): wait for all participants to arrive
        while (ckpt->arrived_mask != mask) {
            invalidate_l1_cache();
        }
        // Clear arrived for next barrier, then release
        ckpt->arrived_mask = 0;
        ckpt->proceed = expected_proceed;
    } else {
        // Subordinate: wait for orchestrator to release
        while (ckpt->proceed != expected_proceed) {
            invalidate_l1_cache();
        }
    }
}

// ---------------------------------------------------------------------------
// CB dump: each RISC prints its view of circular buffer state.
// Supports DPRINT, DEVICE_PRINT, or no-print (barrier-only) mode.
// ---------------------------------------------------------------------------
template <uint8_t num_cbs = 0, uint16_t words_per_cb = 0, bool dump_dest = false>
inline void debug_checkpoint_dump_cbs([[maybe_unused]] uint8_t checkpoint_id) {
#if !defined(CHECKPOINT_PRINT_ENABLED)
    // No print backend available — checkpoint acts as barrier only, skip dump.
    return;
#else
    uint32_t my_idx = internal_::get_hw_thread_idx();

#if defined(USE_DEVICE_PRINT)
    DEVICE_PRINT("=== CKPT {} RISC {} ===", (uint32_t)checkpoint_id, my_idx);
#else
    DPRINT << "=== CKPT " << (uint32_t)checkpoint_id << " RISC " << my_idx << " ===" << ENDL();
#endif

#if defined(COMPILE_FOR_TRISC) && (COMPILE_FOR_TRISC == 1)
    // Math thread cannot access CB interfaces
    if constexpr (dump_dest) {
        // Read and print dest register contents directly (no dbg_halt/dbg_unhalt
        // since the checkpoint barrier already synchronizes all RISCs).
        // Note: dest reg row helpers use DPRINT internally, so dump_dest is only
        // supported with the DPRINT backend.
        uint32_t data_format_reg_field_value = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);
        if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled)) {
            data_format_reg_field_value = (uint32_t)DataFormat::Float32;
        }
        DPRINT << FIXED() << SETW(WIDTH) << SETPRECISION(PRECISION);
        DPRINT << "dest regs tile 0:" << ENDL();
        uint32_t row = 0;
        for (int face_id = 0; face_id < NUM_FACES_PER_TILE; ++face_id) {
            for (int row_id = 0; row_id < NUM_ROWS_PER_FACE; ++row_id) {
                switch (data_format_reg_field_value) {
                    case (uint32_t)DataFormat::Float32: dprint_tensix_dest_reg_row_float32(row); break;
                    case (uint32_t)DataFormat::Int32: dprint_tensix_dest_reg_row_int32(row); break;
                    case (uint32_t)DataFormat::UInt16:
                        dprint_tensix_dest_reg_row_uint16(data_format_reg_field_value, row);
                        break;
                    case (uint32_t)DataFormat::Float16_b:
                        dprint_tensix_dest_reg_row_float16(data_format_reg_field_value, row);
                        break;
                    default: DPRINT << "Unsupported data format: " << data_format_reg_field_value << ENDL(); break;
                }
                row++;
            }
        }
    } else {
#if defined(USE_DEVICE_PRINT)
        DEVICE_PRINT("(math thread, no CB access)");
#else
        DPRINT << "(math thread, no CB access)" << ENDL();
#endif
    }
#else
    // BRISC, NCRISC, TRISC0, TRISC2 can access CB interfaces
    constexpr uint32_t max_cb = (num_cbs == 0) ? NUM_CIRCULAR_BUFFERS : num_cbs;
    for (uint32_t cb = 0; cb < max_cb; cb++) {
        auto& iface = get_local_cb_interface(cb);
        if (iface.fifo_size == 0) {
            continue;  // Skip unconfigured CBs
        }

        // Print CB metadata
#if defined(USE_DEVICE_PRINT)
        DEVICE_PRINT(
            "CB{} sz={} rd={} wr={} ack={} rcv={}",
            cb,
            iface.fifo_size,
            iface.fifo_rd_ptr,
            iface.fifo_wr_ptr,
            (uint32_t)iface.tiles_acked,
            (uint32_t)iface.tiles_received);
#else
        DPRINT << "CB" << cb << " sz=" << iface.fifo_size << " rd=" << iface.fifo_rd_ptr << " wr=" << iface.fifo_wr_ptr
               << " ack=" << iface.tiles_acked << " rcv=" << iface.tiles_received << ENDL();
#endif

        // Optionally dump L1 data starting at read pointer
        if constexpr (words_per_cb > 0) {
            volatile tt_l1_ptr uint32_t* data_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.fifo_rd_ptr << cb_addr_shift);
            for (uint16_t w = 0; w < words_per_cb; w += 4) {
                uint16_t chunk = (words_per_cb - w > 4) ? 4 : (words_per_cb - w);
#if defined(USE_DEVICE_PRINT)
                for (uint16_t j = 0; j < chunk; j++) {
                    DEVICE_PRINT("  [{0}] {1:#010x}", (uint32_t)(w + j), data_ptr[w + j]);
                }
#else
                DPRINT << "  [" << w << "] ";
                for (uint16_t j = 0; j < chunk; j++) {
                    DPRINT << HEX() << data_ptr[w + j] << " ";
                }
                DPRINT << DEC() << ENDL();
#endif
            }
        }
    }
#endif  // COMPILE_FOR_TRISC
#endif  // CHECKPOINT_PRINT_ENABLED
}

// ---------------------------------------------------------------------------
// Combined checkpoint: barrier -> dump -> barrier
// ---------------------------------------------------------------------------
template <uint8_t num_cbs = 0, uint16_t words_per_cb = 0, bool dump_dest = false>
inline void debug_checkpoint(uint8_t checkpoint_id) {
    WAYPOINT("CKW");  // Checkpoint Wait
    // Use checkpoint_id * 2 and checkpoint_id * 2 + 1 as the two barrier phases
    debug_checkpoint_barrier(checkpoint_id * 2);

    debug_checkpoint_dump_cbs<num_cbs, words_per_cb, dump_dest>(checkpoint_id);

    // Second barrier ensures all RISCs finish dumping before any proceeds
    debug_checkpoint_barrier(checkpoint_id * 2 + 1);
    WAYPOINT("CKD");  // Checkpoint Done
}

// ---------------------------------------------------------------------------
// User-facing macros
// ---------------------------------------------------------------------------
#define DEBUG_CHECKPOINT(id) debug_checkpoint<>(id)
#define DEBUG_CHECKPOINT_EX(id, num_cbs, words_per_cb, dump_dest) debug_checkpoint<num_cbs, words_per_cb, dump_dest>(id)

#else  // !DEBUG_CHECKPOINT_ENABLED

#define DEBUG_CHECKPOINT(id)
#define DEBUG_CHECKPOINT_EX(id, num_cbs, words_per_cb, dump_dest)

#endif  // DEBUG_CHECKPOINT_ENABLED
