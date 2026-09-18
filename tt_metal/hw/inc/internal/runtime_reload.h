// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(BLAZE_RUNTIME_RELOAD)

#include <cstddef>

#include "hostdev/runtime_reload_abi.h"

extern uint8_t noc_mode;

// Runtime binary reload: one launch walks a table of kernel images, re-entering once per stage.
// The table is static; the host guarantees every image fits, so firmware carries no bounds.
enum reload_mode : uint32_t {
    // Walk the table num_rounds times, then stop. blaze's num_iterations.
    RELOAD_MODE_BOUNDED = RELOAD_ABI_MODE_BOUNDED,
    // Walk it until term_sem_addr reads 1. blaze's run_persistently -- and what
    // decode needs, since the token count is not known when the launch goes out.
    RELOAD_MODE_PERSISTENT = RELOAD_ABI_MODE_FIRMWARE_PERSISTENT,
};

struct reload_table_t {
    volatile uint32_t num_stages;
    volatile uint32_t mode;           // reload_mode
    volatile uint32_t num_rounds;     // RELOAD_MODE_BOUNDED
    volatile uint32_t term_sem_addr;  // RELOAD_MODE_PERSISTENT; read once per round
    // RELOAD_ABI_VERSION of the host that built this table. Checked before the first read;
    // a mismatch parks the walk with dbg_phase = RELOAD_PHASE_ABI_MISMATCH rather than
    // indexing a layout this firmware does not have.
    volatile uint32_t abi_version;
    // Cross-core stage barrier, on for every table: a core that runs ahead would overwrite a block
    // a peer is still executing. The addresses are absolute and identical on every core; they name
    // retained semaphores that survive the reload.
    volatile uint32_t sync_arrive_addr;   // counted up on the coordinator
    volatile uint32_t sync_release_addr;  // coordinator sets it, each waiter clears its own
    volatile uint32_t sync_coord_x;       // virtual coords, as src_noc_x/y are
    volatile uint32_t sync_coord_y;
    volatile uint32_t sync_num_cores;  // arrivals to wait for, coordinator included
    // The host says which core coordinates rather than firmware comparing coordinates.
    // A wrong compare gives two coordinators or none, and both hang.
    volatile uint32_t sync_is_coord;
    // The release rectangle, in the same virtual coords. The sender is not its own
    // destination, so sync_mcast_dests is the participant count less the coordinator.
    volatile uint32_t sync_mcast_start_x;
    volatile uint32_t sync_mcast_start_y;
    volatile uint32_t sync_mcast_end_x;
    volatile uint32_t sync_mcast_end_y;
    volatile uint32_t sync_mcast_dests;
    // Reload progress, for the host to read back after the fact: a DPRINT here changes the
    // timing of the very thing being measured, and py-spy cannot see firmware at all. Counted
    // in the table's own L1 shard, so `read_core_l1(core, table_addr, ...)` reports it.
    volatile uint32_t dbg_reload_count;  // installs completed
    volatile uint32_t dbg_last_stage;    // stage index of the last install
    volatile uint32_t dbg_phase;         // where the CURRENT reload is: see RELOAD_PHASE_*
    // Cycles the last reload spent, per core. Written here rather than through the device profiler,
    // whose instrumentation overflows BRISC's text segment.
    volatile uint32_t dbg_weight_cycles;  // the weight fetch, issue through landing
    //! Cycles spent in the stage barrier, per core. Separate from dbg_weight_cycles: the barrier
    //! sits above that zone.
    volatile uint32_t dbg_barrier_cycles;
    // A DRAM-aligned L1 buffer owned by the host table. The launch mailbox itself has weaker
    // alignment, so opaque config reads land here before firmware copies them into the mailbox.
    volatile uint32_t config_scratch_addr;
    // A stage is a whole kernel-config block plus Metal's opaque capture of the launch config.
    // Everything inside the block is addressed as kernel_config_base + offset, so moving the
    // block and restoring its matching launch config brings the program back.
    struct stage_entry_t {
        volatile uint32_t src_noc_x;  // virtual coords; BRISC encodes the NoC address itself
        volatile uint32_t src_noc_y;
        // Which plane fetches the image. The host pairs it with the DRAM port behind src_noc_x/y,
        // planned together with this stage's weight regions: a port is driven from one plane only,
        // and BRISC's own noc_index is whatever the kernel was given, not that plane.
        volatile uint32_t src_noc;
        volatile uint32_t src_addr;  // L1 on some core, or DRAM -- same fetch either way
        volatile uint32_t block_bytes;
        // The opaque kernel_config_msg_t bytes immediately after the block in the image arena.
        volatile uint32_t launch_kernel_config_bytes;
        // RELOAD_ABI_ENABLES_CAPTURED keeps the captured mask; another value overrides it.
        volatile uint32_t enables;
        // The weight regions this stage needs in L1 before it runs, as a range into the
        // weight array that follows stage[]. Zero count is the common case -- a stage whose
        // weights are already resident, and every stage of a binary-only reload.
        volatile uint32_t weight_count;
        volatile uint32_t weight_first;
        // This stage's output tap. Inline, because a stage has at most one output. tap_bytes == 0
        // means no tap. Read off the stage that just ran, at the reload boundary, where the
        // activation is quiescent. No VC (a tap is one small write); tap_noc travels with the DRAM
        // port the host paired it with, because a subchannel endpoint is reachable from one plane.
        volatile uint32_t tap_bytes;
        volatile uint32_t tap_src_l1;     // absolute: the activation lives outside kernel_config
        volatile uint32_t tap_dst_noc_x;  // virtual coords, as the image source's are
        volatile uint32_t tap_dst_noc_y;
        // The append cursor: firmware writes tap_bytes here and advances it, so crossing i lands at
        // base + i*tap_bytes.
        volatile uint32_t tap_dst_addr;
        volatile uint32_t tap_noc;  // which plane issues this write, paired with the port above
        // One past the last writable byte. Appending stops here rather than wrapping: a full
        // log truncates, it does not overwrite what it already recorded, and it never runs
        // into whatever DRAM sits after the buffer.
        volatile uint32_t tap_limit;
    } stage[];
    // Weight regions, indexed by stage[].weight_first .. +weight_count. One entry is one contiguous
    // DRAM->L1 burst; the DRAM side is a byte image of this core's L1 region, so the copy is an
    // identity. ``vc`` and ``noc`` spread the reads over channels and planes.
    struct weight_entry_t {
        volatile uint32_t src_noc_x;  // virtual coords, as the image's are
        volatile uint32_t src_noc_y;
        volatile uint32_t src_addr;
        volatile uint32_t bytes;
        volatile uint32_t dst_l1;  // absolute: weights live outside kernel_config
        volatile uint32_t vc;
        volatile uint32_t noc;  // which plane issues this read
    };
    // The weight array starts right after stage[num_stages] -- the host places it there,
    // the way it places every other part of this table.
    volatile weight_entry_t* weights() const {
        return reinterpret_cast<volatile weight_entry_t*>(
            const_cast<void*>(static_cast<const void*>(&stage[num_stages])));
    }
};

// Check the firmware structs against the wire layout shared with the host.
static_assert(offsetof(reload_table_t, num_stages) == RELOAD_ABI_HDR_NUM_STAGES * 4, "reload_abi: num_stages");
static_assert(offsetof(reload_table_t, mode) == RELOAD_ABI_HDR_MODE * 4, "reload_abi: mode");
static_assert(offsetof(reload_table_t, num_rounds) == RELOAD_ABI_HDR_NUM_ROUNDS * 4, "reload_abi: num_rounds");
static_assert(offsetof(reload_table_t, term_sem_addr) == RELOAD_ABI_HDR_TERM_SEM_ADDR * 4, "reload_abi: term_sem_addr");
static_assert(offsetof(reload_table_t, abi_version) == RELOAD_ABI_HDR_ABI_VERSION * 4, "reload_abi: abi_version");
static_assert(
    offsetof(reload_table_t, sync_arrive_addr) == RELOAD_ABI_HDR_SYNC_ARRIVE_ADDR * 4, "reload_abi: sync_arrive_addr");
static_assert(
    offsetof(reload_table_t, sync_release_addr) == RELOAD_ABI_HDR_SYNC_RELEASE_ADDR * 4,
    "reload_abi: sync_release_addr");
static_assert(offsetof(reload_table_t, sync_coord_x) == RELOAD_ABI_HDR_SYNC_COORD_X * 4, "reload_abi: sync_coord_x");
static_assert(offsetof(reload_table_t, sync_coord_y) == RELOAD_ABI_HDR_SYNC_COORD_Y * 4, "reload_abi: sync_coord_y");
static_assert(
    offsetof(reload_table_t, sync_num_cores) == RELOAD_ABI_HDR_SYNC_NUM_CORES * 4, "reload_abi: sync_num_cores");
static_assert(offsetof(reload_table_t, sync_is_coord) == RELOAD_ABI_HDR_SYNC_IS_COORD * 4, "reload_abi: sync_is_coord");
static_assert(
    offsetof(reload_table_t, sync_mcast_start_x) == RELOAD_ABI_HDR_SYNC_MCAST_START_X * 4,
    "reload_abi: sync_mcast_start_x");
static_assert(
    offsetof(reload_table_t, sync_mcast_start_y) == RELOAD_ABI_HDR_SYNC_MCAST_START_Y * 4,
    "reload_abi: sync_mcast_start_y");
static_assert(
    offsetof(reload_table_t, sync_mcast_end_x) == RELOAD_ABI_HDR_SYNC_MCAST_END_X * 4, "reload_abi: sync_mcast_end_x");
static_assert(
    offsetof(reload_table_t, sync_mcast_end_y) == RELOAD_ABI_HDR_SYNC_MCAST_END_Y * 4, "reload_abi: sync_mcast_end_y");
static_assert(
    offsetof(reload_table_t, sync_mcast_dests) == RELOAD_ABI_HDR_SYNC_MCAST_DESTS * 4, "reload_abi: sync_mcast_dests");
static_assert(
    offsetof(reload_table_t, dbg_reload_count) == RELOAD_ABI_HDR_DBG_RELOAD_COUNT * 4, "reload_abi: dbg_reload_count");
static_assert(
    offsetof(reload_table_t, dbg_last_stage) == RELOAD_ABI_HDR_DBG_LAST_STAGE * 4, "reload_abi: dbg_last_stage");
static_assert(offsetof(reload_table_t, dbg_phase) == RELOAD_ABI_HDR_DBG_PHASE * 4, "reload_abi: dbg_phase");
static_assert(
    offsetof(reload_table_t, dbg_weight_cycles) == RELOAD_ABI_HDR_DBG_WEIGHT_CYCLES * 4,
    "reload_abi: dbg_weight_cycles");
static_assert(
    offsetof(reload_table_t, dbg_barrier_cycles) == RELOAD_ABI_HDR_DBG_BARRIER_CYCLES * 4,
    "reload_abi: dbg_barrier_cycles");
static_assert(
    offsetof(reload_table_t, config_scratch_addr) == RELOAD_ABI_HDR_CONFIG_SCRATCH_ADDR * 4,
    "reload_abi: config_scratch_addr");
static_assert(offsetof(reload_table_t, stage) == RELOAD_ABI_HEADER_WORDS * 4, "reload_abi: header words");
static_assert(sizeof(reload_table_t::stage_entry_t) == RELOAD_ABI_ENTRY_WORDS * 4, "reload_abi: stage entry words");
static_assert(
    offsetof(reload_table_t::stage_entry_t, src_noc_x) == RELOAD_ABI_ENTRY_SRC_NOC_X * 4,
    "reload_abi: stage.src_noc_x");
static_assert(
    offsetof(reload_table_t::stage_entry_t, src_noc_y) == RELOAD_ABI_ENTRY_SRC_NOC_Y * 4,
    "reload_abi: stage.src_noc_y");
static_assert(
    offsetof(reload_table_t::stage_entry_t, src_noc) == RELOAD_ABI_ENTRY_SRC_NOC * 4, "reload_abi: stage.src_noc");
static_assert(
    offsetof(reload_table_t::stage_entry_t, src_addr) == RELOAD_ABI_ENTRY_SRC_ADDR * 4, "reload_abi: stage.src_addr");
static_assert(
    offsetof(reload_table_t::stage_entry_t, block_bytes) == RELOAD_ABI_ENTRY_BLOCK_BYTES * 4,
    "reload_abi: stage.block_bytes");
static_assert(
    offsetof(reload_table_t::stage_entry_t, launch_kernel_config_bytes) ==
        RELOAD_ABI_ENTRY_LAUNCH_KERNEL_CONFIG_BYTES * 4,
    "reload_abi: stage.launch_kernel_config_bytes");
static_assert(
    offsetof(reload_table_t::stage_entry_t, enables) == RELOAD_ABI_ENTRY_ENABLES * 4, "reload_abi: stage.enables");
static_assert(
    offsetof(reload_table_t::stage_entry_t, weight_count) == RELOAD_ABI_ENTRY_WEIGHT_COUNT * 4,
    "reload_abi: stage.weight_count");
static_assert(
    offsetof(reload_table_t::stage_entry_t, weight_first) == RELOAD_ABI_ENTRY_WEIGHT_FIRST * 4,
    "reload_abi: stage.weight_first");
static_assert(
    offsetof(reload_table_t::stage_entry_t, tap_bytes) == RELOAD_ABI_ENTRY_TAP_BYTES * 4,
    "reload_abi: stage.tap_bytes");
static_assert(
    offsetof(reload_table_t::stage_entry_t, tap_src_l1) == RELOAD_ABI_ENTRY_TAP_SRC_L1 * 4,
    "reload_abi: stage.tap_src_l1");
static_assert(
    offsetof(reload_table_t::stage_entry_t, tap_dst_noc_x) == RELOAD_ABI_ENTRY_TAP_DST_NOC_X * 4,
    "reload_abi: stage.tap_dst_noc_x");
static_assert(
    offsetof(reload_table_t::stage_entry_t, tap_dst_noc_y) == RELOAD_ABI_ENTRY_TAP_DST_NOC_Y * 4,
    "reload_abi: stage.tap_dst_noc_y");
static_assert(
    offsetof(reload_table_t::stage_entry_t, tap_dst_addr) == RELOAD_ABI_ENTRY_TAP_DST_ADDR * 4,
    "reload_abi: stage.tap_dst_addr");
static_assert(
    offsetof(reload_table_t::stage_entry_t, tap_noc) == RELOAD_ABI_ENTRY_TAP_NOC * 4, "reload_abi: stage.tap_noc");
static_assert(
    offsetof(reload_table_t::stage_entry_t, tap_limit) == RELOAD_ABI_ENTRY_TAP_LIMIT * 4,
    "reload_abi: stage.tap_limit");
static_assert(sizeof(reload_table_t::weight_entry_t) == RELOAD_ABI_WEIGHT_WORDS * 4, "reload_abi: weight entry words");
static_assert(
    offsetof(reload_table_t::weight_entry_t, src_noc_x) == RELOAD_ABI_WEIGHT_SRC_NOC_X * 4,
    "reload_abi: weight.src_noc_x");
static_assert(
    offsetof(reload_table_t::weight_entry_t, src_noc_y) == RELOAD_ABI_WEIGHT_SRC_NOC_Y * 4,
    "reload_abi: weight.src_noc_y");
static_assert(
    offsetof(reload_table_t::weight_entry_t, src_addr) == RELOAD_ABI_WEIGHT_SRC_ADDR * 4,
    "reload_abi: weight.src_addr");
static_assert(
    offsetof(reload_table_t::weight_entry_t, bytes) == RELOAD_ABI_WEIGHT_BYTES * 4, "reload_abi: weight.bytes");
static_assert(
    offsetof(reload_table_t::weight_entry_t, dst_l1) == RELOAD_ABI_WEIGHT_DST_L1 * 4, "reload_abi: weight.dst_l1");
static_assert(offsetof(reload_table_t::weight_entry_t, vc) == RELOAD_ABI_WEIGHT_VC * 4, "reload_abi: weight.vc");
static_assert(offsetof(reload_table_t::weight_entry_t, noc) == RELOAD_ABI_WEIGHT_NOC * 4, "reload_abi: weight.noc");
static_assert(MaxProcessorsPerCoreType == RELOAD_ABI_NUM_PROCESSORS, "reload_abi: processors per core");
static_assert(
    static_cast<uint32_t>(ProgrammableCoreType::COUNT) == RELOAD_ABI_NUM_CORE_TYPES, "reload_abi: core types");

// Hold every participating core until all of them have finished the stage that just ran, so
// the next stage's fetch never overwrites a block a peer is still executing. Shaped like fast
// dispatch's go-signal multicast: workers count themselves in, the coordinator waits for the
// count and multicasts the release. Per chip; the mesh's rendezvous comes from the socket
// handoff. Both words clear themselves, so the barrier keeps no state across launches.
//! Issue one DRAM->L1 burst for the reload, on a named plane and VC. Image and weight regions
//! take the same path; the caller flushes once, after the last issue.
static void reload_fetch(
    uint32_t noc, uint32_t src_x, uint32_t src_y, uint32_t src_addr, uint32_t dst, uint32_t bytes, uint32_t vc) {
    const uint64_t src = NOC_XY_ADDR(DYNAMIC_NOC_X(noc, src_x), DYNAMIC_NOC_Y(noc, src_y), src_addr);
    if (noc_mode == DM_DYNAMIC_NOC) {
        ncrisc_noc_fast_read_any_len<DM_DYNAMIC_NOC>(noc, DYNAMIC_NOC_BRISC_RD_CMD_BUF, src, dst, bytes, vc);
    } else {
        ncrisc_noc_fast_read_any_len(noc, BRISC_RD_CMD_BUF, src, dst, bytes, vc);
    }
}

// reload_fetch's mirror: L1 -> DRAM, for a stage's output tap. The plane comes from the host,
// paired with the DRAM port: a subchannel endpoint is reachable from one plane only, and a write
// to the wrong one never acks.
static void reload_tap_write(
    uint32_t noc, uint32_t src_l1, uint32_t dst_x, uint32_t dst_y, uint32_t dst_addr, uint32_t bytes) {
    const uint64_t dst = NOC_XY_ADDR(DYNAMIC_NOC_X(noc, dst_x), DYNAMIC_NOC_Y(noc, dst_y), dst_addr);
    if (noc_mode == DM_DYNAMIC_NOC) {
        ncrisc_noc_fast_write_any_len<DM_DYNAMIC_NOC>(
            noc,
            DYNAMIC_NOC_BRISC_WR_CMD_BUF,
            src_l1,
            dst,
            bytes,
            NOC_UNICAST_WRITE_VC,
            false /*mcast*/,
            false /*linked*/,
            1 /*num_dests*/,
            false /*multicast_path_reserve*/,
            false /*posted*/);
    } else {
        ncrisc_noc_fast_write_any_len(
            noc,
            BRISC_WR_CMD_BUF,
            src_l1,
            dst,
            bytes,
            NOC_UNICAST_WRITE_VC,
            false /*mcast*/,
            false /*linked*/,
            1 /*num_dests*/,
            false /*multicast_path_reserve*/,
            false /*posted*/);
    }
}

// The coordinator decides whether the walk continues and distributes the verdict to every core.
static bool reload_stage_barrier(const reload_table_t* reload, bool at_boundary, uint32_t round) {
    const uint64_t arrive_addr = NOC_XY_ADDR(
        DYNAMIC_NOC_X(noc_index, reload->sync_coord_x),
        DYNAMIC_NOC_Y(noc_index, reload->sync_coord_y),
        reload->sync_arrive_addr);

    if (noc_mode == DM_DYNAMIC_NOC) {
        noc_fast_atomic_increment<DM_DYNAMIC_NOC>(
            noc_index, DYNAMIC_NOC_BRISC_AT_CMD_BUF, arrive_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false);
    } else {
        noc_fast_atomic_increment(
            noc_index, BRISC_AT_CMD_BUF, arrive_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false);
    }
    // No wait for our own atomic to be acked: the release cannot come until the coordinator has
    // seen everyone. The counter is drained once, at the end of reload_next_stage.

    volatile tt_l1_ptr uint32_t* const release =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reload->sync_release_addr);

    if (reload->sync_is_coord) {
        volatile tt_l1_ptr uint32_t* const arrivals =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reload->sync_arrive_addr);
        do {
            invalidate_l1_cache();
        } while (*arrivals < reload->sync_num_cores);
        *arrivals = 0;

        // The one stop decision, for both modes: 1 = continue, 2 = stop. The cores are in lockstep
        // here, so the coordinator's round is every core's round.
        bool running = true;
        if (at_boundary) {
            invalidate_l1_cache();
            running = (reload->mode == RELOAD_MODE_PERSISTENT)
                          ? (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reload->term_sem_addr) != 1)
                          : (round < reload->num_rounds);
        }
        const uint32_t verdict = running ? 1u : 2u;

        // The coordinator's own release word is the multicast source; it never reads it, so it need
        // not be cleared.
        *release = verdict;

        // Nobody to release: a single-core grid makes sync_mcast_dests zero, and a multicast with no
        // destinations never completes. The verdict is already in *release.
        if (reload->sync_mcast_dests != 0) {
            // NOC 1 multicasts bottom-left to top-right, so the corners reverse (as
            // Device::get_noc_multicast_encoding does on the host). Get this wrong and the release
            // reaches nobody.
            const uint32_t sx = DYNAMIC_NOC_X(noc_index, reload->sync_mcast_start_x);
            const uint32_t sy = DYNAMIC_NOC_Y(noc_index, reload->sync_mcast_start_y);
            const uint32_t ex = DYNAMIC_NOC_X(noc_index, reload->sync_mcast_end_x);
            const uint32_t ey = DYNAMIC_NOC_Y(noc_index, reload->sync_mcast_end_y);
            const uint64_t release_addr = (noc_index == 0)
                                              ? NOC_MULTICAST_ADDR(sx, sy, ex, ey, reload->sync_release_addr)
                                              : NOC_MULTICAST_ADDR(ex, ey, sx, sy, reload->sync_release_addr);
            // Same arguments noc_semaphore_set_multicast passes: the register command buffer,
            // because this is a 4 B write, and multicast_path_reserve set.
            if (noc_mode == DM_DYNAMIC_NOC) {
                ncrisc_noc_fast_write_any_len<DM_DYNAMIC_NOC>(
                    noc_index,
                    DYNAMIC_NOC_BRISC_WR_REG_CMD_BUF,
                    reload->sync_release_addr,
                    release_addr,
                    sizeof(uint32_t),
                    NOC_MULTICAST_WRITE_VC,
                    true /*mcast*/,
                    false /*linked*/,
                    reload->sync_mcast_dests,
                    true /*multicast_path_reserve*/,
                    false /*posted*/);
            } else {
                ncrisc_noc_fast_write_any_len(
                    noc_index,
                    BRISC_WR_REG_CMD_BUF,
                    reload->sync_release_addr,
                    release_addr,
                    sizeof(uint32_t),
                    NOC_MULTICAST_WRITE_VC,
                    true /*mcast*/,
                    false /*linked*/,
                    reload->sync_mcast_dests,
                    true /*multicast_path_reserve*/,
                    false /*posted*/);
            }
        }
        // Not drained here: this core gates every other, and the waiters read their own release
        // word. The counter is drained at the end of reload_next_stage.
        return verdict != 2u;
    } else {
        // One local word, no NoC traffic: the coordinator sets it when everyone is in, and it
        // carries the verdict as well as the release.
        do {
            invalidate_l1_cache();
        } while (*release == 0);
        const uint32_t verdict = *release;
        *release = 0;
        return verdict != 2u;
    }
}

// Where a reload is, for the host to read out of dbg_phase when nothing is moving. Firmware
// spins in several places here (the cross-core barrier, the block fetch), and none of them are
// visible to a host debugger or survivable by DPRINT, which perturbs the timing.
enum : uint32_t {
    RELOAD_PHASE_IDLE = RELOAD_ABI_PHASE_IDLE,
    RELOAD_PHASE_ENTER = RELOAD_ABI_PHASE_ENTER,
    RELOAD_PHASE_BARRIER = RELOAD_ABI_PHASE_BARRIER,
    RELOAD_PHASE_FETCH = RELOAD_ABI_PHASE_FETCH,
    RELOAD_PHASE_INSTALLED = RELOAD_ABI_PHASE_INSTALLED,
    RELOAD_PHASE_DONE = RELOAD_ABI_PHASE_DONE,
    // Inside the output tap's flush. A marker, not a failure state: an unacked NoC write never
    // errors and never flushes, so this names the drain on a wedged rank.
    RELOAD_PHASE_TAP_ENTER = RELOAD_ABI_PHASE_TAP_ENTER,
    RELOAD_PHASE_TAP_DONE = RELOAD_ABI_PHASE_TAP_DONE,
    // The table version or captured-config size does not match this firmware; it did not walk it.
    RELOAD_PHASE_ABI_MISMATCH = RELOAD_ABI_PHASE_ABI_MISMATCH,
};

// Advance the walk: decide whether another stage runs and, if so, install its text. Runs once
// per launch for every program; one without a table runs its launch once, as it always has.
// Safe only past wait_ncrisc_trisc(), when no RISC executes kernel text.
static bool reload_next_stage(launch_msg_t* launch_msg_address, uint32_t& stage, uint32_t& round) {
    reload_table_t* const reload =
        reinterpret_cast<reload_table_t*>(launch_msg_address->kernel_config.reload_table_addr);
    if (reload == nullptr) {
        return false;
    }
    // Refuse a table whose host layout differs from this firmware.
    if (reload->abi_version != RELOAD_ABI_VERSION) {
        reload->dbg_phase = RELOAD_PHASE_ABI_MISMATCH;
        return false;
    }

    uint32_t next = stage + 1;
    bool at_boundary = false;
    if (next == reload->num_stages) {
        // Round boundary: one walk of the table is one microbatch through this core's
        // layers. This is where blaze's generated kernel checks its own loop, so it is
        // where the same decision belongs.
        next = 0;
        round++;
        at_boundary = true;
        // No exit here, in EITHER mode. The stop decision belongs to the coordinator, past the
        // barrier below, so that every core on this chip retires on the same walk.
    }

    // The output tap, before the `!keep_going` return so the last crossing is captured too.
    // Indexed by `stage`, the outgoing one.
    {
        // NON-const, because the cursor below is written back through it.
        auto& out = reload->stage[stage];
        const uint32_t tap_bytes = out.tap_bytes;
        // Append: write at the cursor, then advance it. Bounded by tap_limit rather than wrapped, so
        // the early history survives; unwritten entries stay zero, which is the reader's end marker.
        const uint32_t tap_dst = out.tap_dst_addr;
        if (tap_bytes != 0) {
            if (tap_dst + tap_bytes <= out.tap_limit) {
                reload->dbg_phase = RELOAD_PHASE_TAP_ENTER;
                const uint32_t tnoc = out.tap_noc;
                reload_tap_write(tnoc, out.tap_src_l1, out.tap_dst_noc_x, out.tap_dst_noc_y, tap_dst, tap_bytes);
                // Advanced BEFORE the drain below, not after: the drain can only be reached by
                // finishing this write, and leaving the bump past a spin would repeat the entry
                // if that spin were ever cut short.
                out.tap_dst_addr = tap_dst + tap_bytes;
                // Drained here: the fetch path's flush is past the return below, and the end-of-kernel
                // checks assert no outstanding writes. Phase either side of the drain, so a wedged rank names
                // this spin. tnoc, not noc_index: the plane that issued the write acks it.
                if (noc_mode == DM_DYNAMIC_NOC) {
                    do {
                        invalidate_l1_cache();
                    } while (!ncrisc_dynamic_noc_nonposted_writes_flushed(tnoc));
                } else {
                    while (!ncrisc_noc_nonposted_writes_flushed(tnoc)) {
                    }
                }
                // Past the drain: the tap landed. Back to BARRIER, which is where the walk
                // actually is -- the stop check has not run yet and FETCH is set below it.
                reload->dbg_phase = RELOAD_PHASE_TAP_DONE;
            }
        }
    }

    // Wait for every core to finish this stage before the fetch below overwrites the block they may
    // still be running from. Above the timing zone. Every core arrives, in every mode, including one
    // about to stop: that is what makes the exit collective.
    //
    reload->dbg_phase = RELOAD_PHASE_BARRIER;
    const uint32_t t_barrier_start = get_timestamp_32b();
    // Both modes stop on the coordinator's verdict carried out of the barrier, so every core
    // retires on the same walk. Deciding per core was a race against the host's asynchronous flag
    // write and stranded cores in the barrier.
    const bool keep_going = reload_stage_barrier(reload, at_boundary, round);
    reload->dbg_barrier_cycles = get_timestamp_32b() - t_barrier_start;
    if (!keep_going) {
        reload->dbg_phase = RELOAD_PHASE_DONE;
        return false;
    }

    // What a reload costs: fetch plus install, per stage, per core, measured with the wall clock
    // (the profiler's instrumentation does not fit BRISC's text). Scoped below the early returns so
    // it records reloads only.

    reload->dbg_phase = RELOAD_PHASE_FETCH;

    const auto& entry = reload->stage[next];
    auto& kernel_config = launch_msg_address->kernel_config;
    // Land the image on the base the first image runs from: metal's kernel config region, which the
    // blocks were compiled for. A program cannot run from an arbitrary L1 address.
    const uint32_t config_base = kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];
    const uint32_t reload_table_addr = kernel_config.reload_table_addr;

    // Metal owns the captured config's layout. Firmware knows that layout because it compiles
    // against the same kernel_config_msg_t; the Blaze host only transports the opaque bytes.
    if (entry.block_bytes != 0 && entry.launch_kernel_config_bytes != sizeof(kernel_config_msg_t)) {
        reload->dbg_phase = RELOAD_PHASE_ABI_MISMATCH;
        return false;
    }

    // Start the clock before the image is issued: the image is the other half of a reload, over the
    // same links as the weights.
    const uint32_t t_reload_start = get_timestamp_32b();

    // Fetch the block and its adjacent opaque launch config from wherever the host put the image
    // (L1 or DRAM), on the plane the host paired with that endpoint -- not noc_index, which is the
    // kernel's plane and shares no endpoint with the plan. Both reads are issued here and drained
    // with the weights below.
    const uint32_t other_noc = 1 - noc_index;
    bool reads_on_other_noc = false;
    if (entry.block_bytes != 0) {
        const uint32_t inoc = entry.src_noc;
        reads_on_other_noc = (inoc != noc_index);
        reload_fetch(inoc, entry.src_noc_x, entry.src_noc_y, entry.src_addr, config_base, entry.block_bytes, 1);
        reload_fetch(
            inoc,
            entry.src_noc_x,
            entry.src_noc_y,
            entry.src_addr + entry.block_bytes,
            reload->config_scratch_addr,
            entry.launch_kernel_config_bytes,
            1);
    }

    // The next stage's weights, issued into the same quiesced window as the image. Disjoint
    // destinations, so the two overlap and one flush covers both. Each region names its own plane
    // and VC; reads on the other plane are waited for separately.
    {
        for (uint32_t w = 0; w < entry.weight_count; w++) {
            volatile reload_table_t::weight_entry_t& wt = reload->weights()[entry.weight_first + w];
            const uint32_t wnoc = wt.noc;
            if (wnoc != noc_index) {
                reads_on_other_noc = true;
            }
            reload_fetch(wnoc, wt.src_noc_x, wt.src_noc_y, wt.src_addr, wt.dst_l1, wt.bytes, wt.vc);
        }
    }

    (void)reload_stage_barrier(reload, /*at_boundary=*/false, round);

    // The block must be whole before anything runs from it. Each read bumps BRISC's own issued
    // counter, so the standard barrier is exact.
    if (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_reads_flushed(noc_index));
        if (reads_on_other_noc) {
            do {
                invalidate_l1_cache();
            } while (!ncrisc_dynamic_noc_reads_flushed(other_noc));
        }
        // The barrier's atomic and release multicast are drained here, folded into a barrier that
        // happens anyway, instead of on the critical path where they were issued.
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_index));
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_nonposted_writes_flushed(noc_index));
    } else {
        while (!ncrisc_noc_reads_flushed(noc_index)) {
        }
        if (reads_on_other_noc) {
            while (!ncrisc_noc_reads_flushed(other_noc)) {
            }
        }
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index)) {
        }
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
        }
    }

    invalidate_l1_cache();

    if (entry.block_bytes == 0) {
        kernel_config.enables = 0;
    } else {
        auto* kernel_config_words = reinterpret_cast<volatile uint32_t*>(&kernel_config);
        auto* captured_kernel_config_words = reinterpret_cast<volatile uint32_t*>(reload->config_scratch_addr);
        for (uint32_t i = 0; i < sizeof(kernel_config_msg_t) / sizeof(uint32_t); ++i) {
            kernel_config_words[i] = captured_kernel_config_words[i];
        }
        // The capture contains its old source base and may contain the value present in the launch
        // message at capture time. Keep this run's relocated base and live table address.
        kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX] = config_base;
        kernel_config.reload_table_addr = reload_table_addr;
        if (entry.enables != RELOAD_ABI_ENABLES_CAPTURED) {
            kernel_config.enables = entry.enables;
        }
    }
    reload->dbg_reload_count = reload->dbg_reload_count + 1;
    reload->dbg_last_stage = next;
    reload->dbg_phase = RELOAD_PHASE_INSTALLED;

    // Issue through landing, measured off the barrier above rather than a second flush that would
    // serialise the weight fetch against the image fetch.
    reload->dbg_weight_cycles = get_timestamp_32b() - t_reload_start;

    stage = next;
    return true;
}

#else

static inline bool reload_next_stage(launch_msg_t*, uint32_t&, uint32_t&) { return false; }

#endif
