// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Debug checkpoint API for fused kernels.
//
// Provides synchronized barriers where all active RISCs on a core halt together,
// dump circular buffer state via DPRINT, then proceed in unison.
//
// Usage in kernel code:
//   DEBUG_CHECKPOINT("post_matmul");              // barrier + dump CB metadata + barrier
//   DEBUG_CHECKPOINT_EX("pre_pack", 4, 8, true);  // dump first 4 CBs, 8 words each, + dest regs
//
// All active RISCs must call the same checkpoint. The macro is a no-op when
// DEBUG_CHECKPOINT_ENABLED is not defined.
//
// The checkpoint state (20 bytes) is stored at MEM_LLK_DEBUG_BASE in L1,
// which is a 1024-byte debug region shared by all RISCs on the core.
// No other debug tool should use MEM_LLK_DEBUG_BASE concurrently.

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

// Checkpoints are only supported on Tensix cores (WH/BH) with 5 RISCs.
// On Quasar, get_hw_thread_idx() returns values > 4, which would cause
// out-of-bounds writes to arrived[]. Disable at compile time.
#if defined(ARCH_QUASAR)
static_assert(false, "Debug checkpoints are not supported on Quasar architecture");
#endif

// Checkpoint state lives at a fixed L1 address (start of MEM_LLK_DEBUG region).
// Uses per-RISC byte flags (not a shared bitmask) to avoid read-modify-write
// races — each RISC writes only its own byte.
constexpr uint32_t DEBUG_CHECKPOINT_MAX_RISCS = 5;
struct debug_checkpoint_state_t {
    volatile uint32_t proceed;                             // Monotonically increasing epoch
    volatile uint32_t participant_mask;                    // Bitmask of active RISC thread indices
    volatile uint8_t arrived[DEBUG_CHECKPOINT_MAX_RISCS];  // Per-RISC arrival epoch
    uint8_t orchestrator_idx;                              // Lowest active RISC index
    uint8_t pad[2];
};
static_assert(sizeof(debug_checkpoint_state_t) <= 1024, "Checkpoint state must fit in MEM_LLK_DEBUG region");

inline volatile debug_checkpoint_state_t tt_l1_ptr* get_checkpoint_state() {
    return reinterpret_cast<volatile debug_checkpoint_state_t tt_l1_ptr*>(MEM_LLK_DEBUG_BASE);
}

// ---------------------------------------------------------------------------
// Init: called by BRISC/DM0 before launching subordinate kernels.
// Sets participant_mask, selects orchestrator, and clears barrier state.
// ---------------------------------------------------------------------------
inline void debug_checkpoint_init(uint32_t enables) {
    volatile debug_checkpoint_state_t tt_l1_ptr* ckpt = get_checkpoint_state();
    // Mask to valid RISC indices only
    uint32_t valid_mask = enables & ((1u << DEBUG_CHECKPOINT_MAX_RISCS) - 1u);
    ckpt->participant_mask = valid_mask;
    ckpt->proceed = 0;
    for (uint32_t i = 0; i < DEBUG_CHECKPOINT_MAX_RISCS; i++) {
        ckpt->arrived[i] = 0;
    }
    // Select lowest active RISC as orchestrator
    uint32_t orch = 0;
    while (orch < DEBUG_CHECKPOINT_MAX_RISCS && !(valid_mask & (1u << orch))) {
        orch++;
    }
    ckpt->orchestrator_idx = static_cast<uint8_t>(orch);
}

// ---------------------------------------------------------------------------
// Barrier: all active RISCs synchronize at a checkpoint.
// Each RISC writes its own arrival byte (no shared read-modify-write).
// The orchestrator (lowest active RISC) polls all arrival bytes, then
// increments the proceed epoch. Subordinates spin on the proceed epoch.
// ---------------------------------------------------------------------------
inline void debug_checkpoint_barrier() {
    volatile debug_checkpoint_state_t tt_l1_ptr* ckpt = get_checkpoint_state();
    uint32_t my_idx = internal_::get_hw_thread_idx();
    uint32_t mask = ckpt->participant_mask;
    uint32_t orch = ckpt->orchestrator_idx;
    // Capture current epoch before signaling arrival
    uint8_t current_epoch = static_cast<uint8_t>(ckpt->proceed);
    uint8_t next_epoch = current_epoch + 1;

    // Signal arrival by writing next epoch to our byte
    ckpt->arrived[my_idx] = next_epoch;

    if (my_idx == orch) {
        // Orchestrator: wait for all participants to arrive at next_epoch
        for (uint32_t i = 0; i < DEBUG_CHECKPOINT_MAX_RISCS; i++) {
            if (mask & (1u << i)) {
                while (ckpt->arrived[i] != next_epoch) {
                    invalidate_l1_cache();
                }
            }
        }
        // Release all subordinates
        ckpt->proceed = next_epoch;
    } else {
        // Subordinate: wait for orchestrator to advance epoch.
        // Check for next_epoch OR next_epoch+1 to handle the rare case where
        // the orchestrator has already advanced to the next barrier.
        uint8_t proceed_val;
        do {
            invalidate_l1_cache();
            proceed_val = static_cast<uint8_t>(ckpt->proceed);
        } while (proceed_val != next_epoch && proceed_val != static_cast<uint8_t>(next_epoch + 1));
    }
}

// ---------------------------------------------------------------------------
// CB dump: prints CB state and optionally dest registers.
// CB pointers (rd_ptr, wr_ptr, tiles_acked, tiles_received) are RISC-specific,
// so BRISC, NCRISC, TRISC0, and TRISC2 each print their own view with a
// RISC index prefix. TRISC1 (Math) prints dest registers if dump_dest=true.
// Both DPRINT and DEVICE_PRINT calls are present — the compiler disables
// whichever backend is not active.
// ---------------------------------------------------------------------------
template <uint8_t num_cbs = 0, uint16_t words_per_cb = 0, bool dump_dest = false>
inline void debug_checkpoint_dump_cbs([[maybe_unused]] const char* name) {
#if !defined(CHECKPOINT_PRINT_ENABLED)
    // No print backend available — checkpoint acts as barrier only, skip dump.
    return;
#else

#if defined(COMPILE_FOR_TRISC) && (COMPILE_FOR_TRISC == 1)
    // Math thread: optionally dump dest registers (only Math can access them)
    if constexpr (dump_dest) {
        DPRINT << "=== CKPT " << name << " dest regs ===" << ENDL();
        DEVICE_PRINT("=== CKPT {} dest regs ===\n", name);
        uint32_t data_format_reg_field_value = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);
        if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled)) {
            data_format_reg_field_value = (uint32_t)DataFormat::Float32;
        }
        DPRINT << FIXED() << SETW(WIDTH) << SETPRECISION(PRECISION);
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
                    default:
                        DPRINT << "Unsupported data format: " << data_format_reg_field_value << ENDL();
                        DEVICE_PRINT("Unsupported data format: {}\n", (DataFormat)data_format_reg_field_value);
                        break;
                }
                row++;
            }
        }
    }
    // If dump_dest is false, Math thread prints nothing (no CB access).

#else
    // BRISC, NCRISC, TRISC0 (Unpack), TRISC2 (Pack) print CB metadata.
    // CB pointers (rd_ptr, wr_ptr, tiles_acked, tiles_received) are RISC-specific,
    // so each RISC's view is different. Prefix with RISC index for disambiguation.
    uint32_t risc_idx = internal_::get_hw_thread_idx();
    DPRINT << "=== CKPT " << name << " RISC" << risc_idx << " CBs ===" << ENDL();
    DEVICE_PRINT("=== CKPT {} RISC{} CBs ===\n", name, risc_idx);

    constexpr uint32_t max_cb = (num_cbs == 0) ? NUM_CIRCULAR_BUFFERS : num_cbs;
    for (uint32_t cb = 0; cb < max_cb; cb++) {
        auto& iface = get_local_cb_interface(cb);
        if (iface.fifo_size == 0) {
            continue;
        }

        DPRINT << "CB" << cb << " sz=" << iface.fifo_size << " rd=" << iface.fifo_rd_ptr << " wr=" << iface.fifo_wr_ptr
               << " ack=" << iface.tiles_acked << " rcv=" << iface.tiles_received << ENDL();
        DEVICE_PRINT(
            "CB{} sz={} rd={} wr={} ack={} rcv={}\n",
            cb,
            iface.fifo_size,
            iface.fifo_rd_ptr,
            iface.fifo_wr_ptr,
            (uint32_t)iface.tiles_acked,
            (uint32_t)iface.tiles_received);

        if constexpr (words_per_cb > 0) {
            volatile tt_l1_ptr uint32_t* data_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.fifo_rd_ptr << cb_addr_shift);
            for (uint16_t w = 0; w < words_per_cb; w += 4) {
                uint16_t chunk = (words_per_cb - w > 4) ? 4 : (words_per_cb - w);
                DPRINT << "  [" << w << "] ";
                for (uint16_t j = 0; j < chunk; j++) {
                    DPRINT << HEX() << data_ptr[w + j] << " ";
                    DEVICE_PRINT("  [{}] {:#010x}\n", (uint32_t)(w + j), data_ptr[w + j]);
                }
                DPRINT << DEC() << ENDL();
            }
        }
    }

#endif  // COMPILE_FOR_TRISC == 1 / else
#endif  // CHECKPOINT_PRINT_ENABLED
}

// ---------------------------------------------------------------------------
// Combined checkpoint: barrier -> dump -> barrier
// ---------------------------------------------------------------------------
template <uint8_t num_cbs = 0, uint16_t words_per_cb = 0, bool dump_dest = false>
inline void debug_checkpoint(const char* name) {
    WAYPOINT("CKW");  // Checkpoint Wait
    debug_checkpoint_barrier();

    debug_checkpoint_dump_cbs<num_cbs, words_per_cb, dump_dest>(name);

    // Second barrier ensures all RISCs finish dumping before any proceeds
    debug_checkpoint_barrier();
    WAYPOINT("CKD");  // Checkpoint Done
}

// ---------------------------------------------------------------------------
// Cross-core barrier: synchronize BRISC across all tensix cores via a NOC
// semaphore. Uses a coordinator-based pattern: all cores atomically increment
// the coordinator's semaphore, and each core waits until the monotonically
// increasing expected count is reached.
// Only compiled for BRISC (dataflow RISCs have NOC access).
// ---------------------------------------------------------------------------
#if defined(KERNEL_BUILD) && (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_DM))
#include "api/dataflow/dataflow_api.h"

// Persistent expected count — incremented by num_cores each barrier call.
// Eliminates the need to reset the semaphore between barriers, avoiding the
// race where increments from a new barrier arrive before the reset completes.
static uint32_t cross_core_barrier_expected = 0;

inline void debug_checkpoint_cross_core_barrier(
    uint32_t sem_id, uint32_t barrier_coord_x, uint32_t barrier_coord_y, uint32_t num_cores) {
    uint32_t sem_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint64_t coord_noc_addr = get_noc_addr(barrier_coord_x, barrier_coord_y, sem_addr);
    bool is_coordinator = (my_x[noc_index] == barrier_coord_x && my_y[noc_index] == barrier_coord_y);

    // Advance expected count — no reset needed, semaphore accumulates monotonically
    cross_core_barrier_expected += num_cores;

    // Signal arrival (atomic increment on coordinator)
    noc_semaphore_inc(coord_noc_addr, 1);
    noc_async_atomic_barrier();

    if (is_coordinator) {
        // Coordinator: wait locally for semaphore to reach expected count
        noc_semaphore_wait_min(local_sem, cross_core_barrier_expected);
    } else {
        // Non-coordinator: poll coordinator's semaphore via NOC read
        while (*local_sem < cross_core_barrier_expected) {
            noc_async_read(coord_noc_addr, sem_addr, sizeof(uint32_t));
            noc_async_read_barrier();
            invalidate_l1_cache();
        }
    }
}
#endif  // KERNEL_BUILD && (COMPILE_FOR_BRISC || COMPILE_FOR_NCRISC || COMPILE_FOR_DM)

// ---------------------------------------------------------------------------
// Global checkpoint: barrier across all RISCs on all cores, then dump.
//
// Sequence:
//   1. Intra-core barrier (all RISCs on this core arrive)
//   2. Cross-core barrier (BRISC syncs across all cores via NOC semaphore)
//   3. Intra-core barrier (BRISC releases other RISCs)
//   4. Dump (BRISC prints CB state; Math prints dest regs if dump_dest=true)
//   5. Intra-core barrier (all RISCs wait for dump to finish)
//   6. Cross-core barrier (all cores finish before proceeding)
//   7. Intra-core barrier (final release)
//
// All RISCs on all cores must call this with the same arguments.
// barrier_coord_x/y identify the coordinator core for the NOC semaphore
// barrier — they do NOT affect what gets printed. Each CB-capable RISC
// prints its own view of CB state (pointers are RISC-specific).
// Non-BRISC RISCs ignore the barrier args but participate in intra-core
// barriers. barrier_coord must be the same across all global checkpoints
// in a program (the monotonic semaphore count assumes a fixed coordinator).
// ---------------------------------------------------------------------------
template <uint8_t num_cbs = 0, uint16_t words_per_cb = 0, bool dump_dest = false>
inline void debug_checkpoint_global(
    const char* name,
    [[maybe_unused]] uint32_t sem_id,
    [[maybe_unused]] uint32_t barrier_coord_x,
    [[maybe_unused]] uint32_t barrier_coord_y,
    [[maybe_unused]] uint32_t num_cores) {
    WAYPOINT("GCW");  // Global Checkpoint Wait

    // 1. Intra-core: all RISCs on this core synchronize
    debug_checkpoint_barrier();

    // 2. Cross-core: BRISC on each core synchronizes across all cores
#if defined(KERNEL_BUILD) && defined(COMPILE_FOR_BRISC)
    debug_checkpoint_cross_core_barrier(sem_id, barrier_coord_x, barrier_coord_y, num_cores);
#endif

    // 3. Intra-core: BRISC releases other RISCs after cross-core sync
    debug_checkpoint_barrier();

    // 4. Dump (BRISC/NCRISC/TRISC0/TRISC2 print CB state; Math prints dest regs if enabled)
    debug_checkpoint_dump_cbs<num_cbs, words_per_cb, dump_dest>(name);

    // 5. Intra-core: all RISCs wait for dump to finish
    debug_checkpoint_barrier();

    // 6. Cross-core: all cores finish dumping before any proceeds
    //    (semaphore accumulates monotonically — no reset needed)
#if defined(KERNEL_BUILD) && defined(COMPILE_FOR_BRISC)
    debug_checkpoint_cross_core_barrier(sem_id, barrier_coord_x, barrier_coord_y, num_cores);
#endif

    // 7. Final intra-core release
    debug_checkpoint_barrier();
    WAYPOINT("GCD");  // Global Checkpoint Done
}

// ---------------------------------------------------------------------------
// User-facing macros
// ---------------------------------------------------------------------------
#define DEBUG_CHECKPOINT(name) debug_checkpoint<>(name)
#define DEBUG_CHECKPOINT_EX(name, num_cbs, words_per_cb, dump_dest) \
    debug_checkpoint<num_cbs, words_per_cb, dump_dest>(name)
#define DEBUG_CHECKPOINT_GLOBAL(name, sem_id, barrier_coord_x, barrier_coord_y, num_cores) \
    debug_checkpoint_global<>(name, sem_id, barrier_coord_x, barrier_coord_y, num_cores)

#else  // !DEBUG_CHECKPOINT_ENABLED

#define DEBUG_CHECKPOINT(name)
#define DEBUG_CHECKPOINT_EX(name, num_cbs, words_per_cb, dump_dest)
#define DEBUG_CHECKPOINT_GLOBAL(name, sem_id, barrier_coord_x, barrier_coord_y, num_cores)

#endif  // DEBUG_CHECKPOINT_ENABLED
