// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch Kernel (subordinate)
// Required to asynchronously send go signals to workers, upon recieving a program
// completion signal. This allows program dispatch (for subsequent programs) to overlap
// with worker execution (for current program), leading to a lower dispatch latency.
// - Handles the following commands:
//  - CQ_DISPATCH_CMD_SEND_GO_SIGNAL: "multicast" go signal to all workers
//  - CQ_DISPATCH_CMD_WAIT: Wait for workers to complete and reset wait count
//    and instead need a unicast for the go signal

#include "debug/assert.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"

// dispatch_s has a customized command buffer allocation for NOC 1.
// Cmd Buf 0 is used for regular writes.
// Cmd Buf 1 is used for small (inline) writes.
// Cmd Buf 2 is used for atomics.
// Cmd Buf 3 is unavailable (used by dispatch_d).
// Reads cannot be issued by dispatch_s.
constexpr uint32_t DISPATCH_S_WR_REG_CMD_BUF = 1;
constexpr uint32_t DISPATCH_S_ATOMIC_CMD_BUF = 2;
constexpr uint32_t cb_base = CB_BASE;
constexpr uint32_t cb_log_page_size = CB_LOG_PAGE_SIZE;
constexpr uint32_t cb_size = CB_SIZE;
constexpr uint32_t my_dispatch_cb_sem_id = MY_DISPATCH_CB_SEM_ID;
constexpr uint32_t upstream_dispatch_cb_sem_id = UPSTREAM_DISPATCH_CB_SEM_ID;
constexpr uint32_t dispatch_s_sync_sem_base_addr = DISPATCH_S_SYNC_SEM_BASE_ADDR;
constexpr uint32_t mcast_go_signal_addr = MCAST_GO_SIGNAL_ADDR;
constexpr uint32_t unicast_go_signal_addr = UNICAST_GO_SIGNAL_ADDR;
constexpr uint32_t distributed_dispatcher =
    DISTRIBUTED_DISPATCHER;  // dispatch_s and dispatch_d running on different cores
constexpr uint32_t first_stream_used = FIRST_STREAM_USED;
constexpr uint32_t max_num_worker_sems = MAX_NUM_WORKER_SEMS;
constexpr uint32_t max_num_go_signal_noc_data_entries = MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES;
constexpr uint32_t virtualize_unicast_cores = VIRTUALIZE_UNICAST_CORES;
constexpr uint32_t num_virtual_unicast_cores = NUM_VIRTUAL_UNICAST_CORES;
constexpr uint32_t num_physical_unicast_cores = NUM_PHYSICAL_UNICAST_CORES;

constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t dispatch_d_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint8_t my_noc_index = NOC_INDEX;

constexpr uint32_t cb_page_size = 1 << cb_log_page_size;
constexpr uint32_t cb_end = cb_base + cb_size;
static uint32_t num_pages_acquired = 0;
static uint32_t num_mcasts_sent[max_num_worker_sems] = {0};
static uint32_t cmd_ptr;

// When dispatch_d and dispatch_s run on separate cores, dispatch_s gets the go signal update from workers.
// dispatch_s is responsible for sending the latest worker completion count to dispatch_d.
// To minimize the number of writes from dispatch_s to dispatch_d, locally track dispatch_d's copy.
static uint32_t worker_count_update_for_dispatch_d[max_num_worker_sems] = {0};

static uint32_t go_signal_noc_data[max_num_go_signal_noc_data_entries];

static uint32_t num_worker_sems = 1;

FORCE_INLINE
void dispatch_s_wr_reg_cmd_buf_init() {
    uint64_t xy_local_addr = get_noc_addr_helper(my_noc_xy, 0);
    NOC_CMD_BUF_WRITE_REG(
        my_noc_index,
        DISPATCH_S_WR_REG_CMD_BUF,
        NOC_TARG_ADDR_COORDINATE,
        (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));
}

FORCE_INLINE
void dispatch_s_atomic_cmd_buf_init() {
    uint64_t atomic_ret_addr = get_noc_addr_helper(my_noc_xy, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    NOC_CMD_BUF_WRITE_REG(
        my_noc_index, DISPATCH_S_ATOMIC_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
    NOC_CMD_BUF_WRITE_REG(
        my_noc_index,
        DISPATCH_S_ATOMIC_CMD_BUF,
        NOC_RET_ADDR_COORDINATE,
        (uint32_t)(atomic_ret_addr >> NOC_ADDR_COORD_SHIFT));
}

FORCE_INLINE
void dispatch_s_noc_semaphore_inc(uint64_t addr, uint32_t incr, uint8_t noc_id) {
    // dispatch_s specific atomic inc API, which will use DISPATCH_S_ATOMIC_CMD_BUF to ensure that
    // ncrisc and brisc don't clobber each other's resources when dispatch_s and dispatch_d are on
    // the same tensix core
    WAYPOINT("NSIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    DEBUG_INSERT_DELAY(TransactionAtomic);
    noc_fast_atomic_increment(
        noc_id,
        DISPATCH_S_ATOMIC_CMD_BUF,
        addr,
        NOC_UNICAST_WRITE_VC,
        incr,
        31 /*wrap*/,
        false /*linked*/,
        false /*posted*/);
    WAYPOINT("NSID");
}

FORCE_INLINE
void dispatch_s_noc_inline_dw_write(uint64_t addr, uint32_t val, uint8_t noc_id, uint8_t be = 0xF) {
    WAYPOINT("NWIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    // Workaround for BH inline writes does not apply here because this writes to a stream register.
    // See comment in `noc_get_interim_inline_value_addr` for more details.
    noc_fast_write_dw_inline<noc_mode>(
        noc_id,
        DISPATCH_S_WR_REG_CMD_BUF,
        val,
        addr,
        be,  // byte-enable
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        false   // posted
    );
    WAYPOINT("NWID");
}

FORCE_INLINE
uint32_t stream_wrap_gt(uint32_t a, uint32_t b) {
    constexpr uint32_t shift = 32 - MEM_WORD_ADDR_WIDTH;
    // Careful below: have to take the signed diff for 2s complement to handle the wrap
    // Below relies on taking the diff first then the compare to move the wrap
    // to 2^31 away
    int32_t diff = a - b;
    return (diff << shift) > 0;
}

FORCE_INLINE
void wait_for_workers(volatile CQDispatchCmd tt_l1_ptr* cmd) {
    volatile uint32_t* worker_sem =
        (volatile uint32_t*)STREAM_REG_ADDR(cmd->mcast.wait_stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    while (stream_wrap_gt(cmd->mcast.wait_count, *worker_sem)) {
    }
}

template <bool flush_write = false>
FORCE_INLINE void update_worker_completion_count_on_dispatch_d() {
    if constexpr (distributed_dispatcher) {
        bool write = false;
        for (uint32_t i = 0; i < num_worker_sems; i++) {
            uint32_t num_workers_signalling_completion =
                NOC_STREAM_READ_REG(i + first_stream_used, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
            if (num_workers_signalling_completion != worker_count_update_for_dispatch_d[i]) {
                worker_count_update_for_dispatch_d[i] = num_workers_signalling_completion;
                // Writing to STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX sets
                // STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX (rather than incrementing it).
                uint64_t dispatch_d_dst = get_noc_addr_helper(
                    dispatch_d_noc_xy, STREAM_REG_ADDR(i + first_stream_used, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
                dispatch_s_noc_inline_dw_write(dispatch_d_dst, num_workers_signalling_completion, my_noc_index);
                write = true;
            }
        }
        if constexpr (flush_write) {
            if (write) {
                noc_async_writes_flushed();
            }
        }
    }
}

template <uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE void cb_acquire_pages_dispatch_s(uint32_t n) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    WAYPOINT("DAPW");
    uint32_t heartbeat = 0;
    // Stall until the number of pages already acquired + the number that need to be acquired is greater
    // than the number available
    while (wrap_gt(num_pages_acquired + n, *sem_addr)) {
        invalidate_l1_cache();
        update_worker_completion_count_on_dispatch_d();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("DAPD");
    num_pages_acquired += n;
}

template <uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE void cb_release_pages_dispatch_s(uint32_t n) {
    dispatch_s_noc_semaphore_inc(get_noc_addr_helper(noc_xy, get_semaphore<fd_core_type>(sem_id)), n, my_noc_index);
}

FORCE_INLINE
void process_go_signal_mcast_cmd() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t sync_index = cmd->mcast.wait_stream - first_stream_used;
    // Get semaphore that will be update by dispatch_d, signalling that it's safe to send a go signal

    volatile tt_l1_ptr uint32_t* sync_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dispatch_s_sync_sem_base_addr + sync_index * L1_ALIGNMENT);

    // Wait for notification from dispatch_d, signalling that it's safe to send the go signal
    uint32_t& mcasts_sent = num_mcasts_sent[sync_index];
    while (wrap_ge(mcasts_sent, *sync_sem_addr)) {
        invalidate_l1_cache();
        // Update dispatch_d with the latest num_workers
        update_worker_completion_count_on_dispatch_d();
    }
    mcasts_sent++;  // Go signal sent -> update counter

    // The location of the go signal embedded in the command does not meet NOC alignment requirements.
    // cmd_ptr is guaranteed to meet the alignment requirements, since it is written to by prefetcher over NOC.
    // Copy the go signal from an unaligned location to an aligned (cmd_ptr) location. This is safe as long as we
    // can guarantee that copying the go signal does not corrupt any other command fields, which is true (see
    // CQDispatchGoSignalMcastCmd).
    volatile uint32_t tt_l1_ptr* aligned_go_signal_storage = (volatile uint32_t tt_l1_ptr*)cmd_ptr;
    *aligned_go_signal_storage = cmd->mcast.go_signal;
    uint8_t go_signal_noc_data_idx = cmd->mcast.noc_data_start_index;

    if (cmd->mcast.num_mcast_txns > 0) {
        // Setup registers before waiting for workers so only the NOC_CMD_CTRL register needs to be touched after.
        uint64_t dst_noc_addr_multicast =
            get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], mcast_go_signal_addr);
        uint32_t num_dests = go_signal_noc_data[go_signal_noc_data_idx++];
        cq_noc_async_write_init_state<CQ_NOC_SNDL, true>(
            (uint32_t)aligned_go_signal_storage, dst_noc_addr_multicast, sizeof(uint32_t));
        noc_nonposted_writes_acked[noc_index] += num_dests;

        wait_for_workers(cmd);
        cq_noc_async_write_with_state<CQ_NOC_sndl, CQ_NOC_wait>(0, 0, 0);
        // Send GO signal to remaining destinations. Only the destination NOC needs to be modified.
        for (uint32_t i = 1, num_mcasts = cmd->mcast.num_mcast_txns; i < num_mcasts; ++i) {
            uint64_t dst = get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], mcast_go_signal_addr);
            uint32_t num_dests = go_signal_noc_data[go_signal_noc_data_idx++];
            cq_noc_async_write_with_state<CQ_NOC_sNdl>(0, dst, 0);
            noc_nonposted_writes_acked[noc_index] += num_dests;
        }
        noc_nonposted_writes_num_issued[noc_index] += cmd->mcast.num_mcast_txns;
    } else {
        wait_for_workers(cmd);
    }

    uint32_t num_unicasts = cmd->mcast.num_unicast_txns;
    if constexpr (virtualize_unicast_cores) {
        // Issue #19729: Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
        // This chip is virtualizing cores the go signal is unicasted to
        // In this case, the number of unicasts specified in the command can exceed
        // the number of actual cores on this chip.
        if (num_unicasts > num_physical_unicast_cores) {
            // If this is the case, cap the number of unicasts to avoid invalid NOC txns
            num_unicasts = num_physical_unicast_cores;
            // Fake updates from non-existent workers here. The dispatcher expects an ack from
            // the number of cores specified inside cmd->mcast.num_unicast_txns. If this is
            // greater than the number of cores actually on the chip, we must account for acks
            // from non-existent cores here.
            NOC_STREAM_WRITE_REG(
                first_stream_used,
                STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
                (num_virtual_unicast_cores - num_physical_unicast_cores) << REMOTE_DEST_BUF_WORDS_FREE_INC);
        }
    }

    for (uint32_t i = 0; i < num_unicasts; ++i) {
        uint64_t dst = get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], unicast_go_signal_addr);
        noc_async_write_one_packet((uint32_t)(aligned_go_signal_storage), dst, sizeof(uint32_t));
    }

    update_worker_completion_count_on_dispatch_d();
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void process_dispatch_s_wait_cmd() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    // Limited Usage of Wait CMD: dispatch_s should get a wait command only if it's not on the
    // same core as dispatch_d and is used to clear the worker count
    ASSERT(
        (cmd->wait.flags == (CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM)) &&
        distributed_dispatcher);
    uint32_t stream = cmd->wait.stream;
    uint32_t index = stream - first_stream_used;
    volatile uint32_t* worker_sem =
        (volatile uint32_t*)STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

    // Wait for workers to complete
    while (stream_wrap_gt(cmd->wait.count, *worker_sem)) {
    }
    // Send updated worker count to dispatch_d and wait for updated count to get picked up by NOC before clearing the
    // counter. dispatch_d will clear it's own counter
    update_worker_completion_count_on_dispatch_d<true>();
    // Reset SPACE_AVAILABLE to 0.
    NOC_STREAM_WRITE_REG(stream, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
    worker_count_update_for_dispatch_d[index] =
        0;  // Local worker count update for dispatch_d should reflect state of worker semaphore on dispatch_s
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void set_num_worker_sems() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    num_worker_sems = cmd->set_num_worker_sems.num_worker_sems;
    ASSERT(num_worker_sems <= max_num_worker_sems);
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void set_go_signal_noc_data() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t num_words = cmd->set_go_signal_noc_data.num_words;
    ASSERT(num_words <= max_num_go_signal_noc_data_entries);
    volatile tt_l1_ptr uint32_t* data_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cmd_ptr + sizeof(CQDispatchCmd));
    for (uint32_t i = 0; i < num_words; ++i) {
        go_signal_noc_data[i] = *(data_ptr++);
    }
    cmd_ptr = round_up_pow2((uint32_t)data_ptr, L1_ALIGNMENT);
}

void kernel_main() {
    DPRINT << "dispatch_s : start" << ENDL();
    // Initialize customized command buffers.
    dispatch_s_wr_reg_cmd_buf_init();
    dispatch_s_atomic_cmd_buf_init();
    if constexpr (distributed_dispatcher) {
        for (size_t i = 0; i < max_num_worker_sems; i++) {
            uint32_t index = i + first_stream_used;

            NOC_STREAM_WRITE_REG(
                index,
                STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
                -NOC_STREAM_READ_REG(index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX)
                    << REMOTE_DEST_BUF_WORDS_FREE_INC);
        }
    }

    cmd_ptr = cb_base;
    bool done = false;
    uint32_t total_pages_acquired = 0;
    while (!done) {
        DeviceZoneScopedN("CQ-DISPATCH-SUBORDINATE");
        cb_acquire_pages_dispatch_s<my_noc_xy, my_dispatch_cb_sem_id>(1);

        volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
        DeviceTimestampedData("process_cmd_d_dispatch_subordinate", (uint32_t)cmd->base.cmd_id);
        switch (cmd->base.cmd_id) {
            case CQ_DISPATCH_CMD_SEND_GO_SIGNAL: process_go_signal_mcast_cmd(); break;
            case CQ_DISPATCH_SET_NUM_WORKER_SEMS: set_num_worker_sems(); break;
            case CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA: set_go_signal_noc_data(); break;
            case CQ_DISPATCH_CMD_WAIT: process_dispatch_s_wait_cmd(); break;
            case CQ_DISPATCH_CMD_TERMINATE: done = true; break;
            default: DPRINT << "dispatcher_s invalid command" << ENDL(); ASSERT(0);
        }
        // Dispatch s only supports single page commands for now
        ASSERT(cmd_ptr <= ((uint32_t)cmd + cb_page_size));
        cmd_ptr = round_up_pow2(cmd_ptr, cb_page_size);
        // Release a single page to prefetcher. Assumption is that all dispatch_s commands fit inside a single page for
        // now.
        cb_release_pages_dispatch_s<upstream_noc_xy, upstream_dispatch_cb_sem_id>(1);
        if (cmd_ptr == cb_end) {
            cmd_ptr = cb_base;
        }
        total_pages_acquired++;
    }
    // Confirm expected number of pages, spinning here is a leak
    cb_wait_all_pages<my_dispatch_cb_sem_id>(total_pages_acquired);
#ifdef COMPILE_FOR_IDLE_ERISC
    // Wait for all transactions to complete, to avoid hitting the asserts in
    // idle_erisck.cc if there are outstanding transactions. These barriers
    // don't work on worker cores, because there cq_dispatch is on the same core
    // and shares use of this noc, but doesn't update this risc's transaction
    // counts. However, we don't have the barrier checks in brisck.cc, so we can
    // skip this for now.
    noc_async_full_barrier();
#endif
    DPRINT << "dispatch_s : done" << ENDL();
}
