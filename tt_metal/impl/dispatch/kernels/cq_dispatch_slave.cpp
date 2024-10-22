// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch Kernel (slave)
// Required to asynchronously send go signals to workers, upon recieving a program
// completion signal. This allows program dispatch (for subsequent programs) to overlap
// with worker execution (for current program), leading to a lower dispatch latency.
// - Handles the following commands:
//  - CQ_DISPATCH_CMD_SEND_GO_SIGNAL: "multicast" go signal to all workers
//  - CQ_DISPATCH_CMD_WAIT: Wait for workers to complete and reset wait count
//    and instead need a unicast for the go signal

#include "debug/assert.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"

// dispatch_s has a customized command buffer allocation for NOC 1.
// Cmd Buf 0 is used for regular writes.
// Cmd Buf 1 is used for small (inline) writes.
// Cmd Buf 2 is used for atomics.
// Cmd Buf 3 is unavailable (used by dispatch_d).
// Reads cannot be issued by dispatch_s.
constexpr uint32_t DISPATCH_S_WR_REG_CMD_BUF = 1;
constexpr uint32_t DISPATCH_S_ATOMIC_CMD_BUF = 2;
constexpr uint32_t cb_base = get_compile_time_arg_val(0);
constexpr uint32_t cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_size = get_compile_time_arg_val(2);
constexpr uint32_t my_dispatch_cb_sem_id = get_compile_time_arg_val(3);
constexpr uint32_t upstream_dispatch_cb_sem_id = get_compile_time_arg_val(4);
constexpr uint32_t dispatch_s_sync_sem_base_addr = get_compile_time_arg_val(5);
constexpr uint32_t mcast_go_signal_addr = get_compile_time_arg_val(6);
constexpr uint32_t unicast_go_signal_addr = get_compile_time_arg_val(7);
constexpr uint32_t distributed_dispatcher = get_compile_time_arg_val(8); // dispatch_s and dispatch_d running on different cores
constexpr uint32_t worker_sem_base_addr = get_compile_time_arg_val(9); // workers update the semaphore at this location to signal completion
constexpr uint32_t max_num_worker_sems = get_compile_time_arg_val(10); // maximum number of worker semaphores

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

static uint32_t num_worker_sems = 1;

FORCE_INLINE
void dispatch_s_wr_reg_cmd_buf_init() {
    uint64_t xy_local_addr = get_noc_addr_helper(my_noc_xy, 0);
    NOC_CMD_BUF_WRITE_REG(my_noc_index, DISPATCH_S_WR_REG_CMD_BUF, NOC_TARG_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));
}

FORCE_INLINE
void dispatch_s_atomic_cmd_buf_init() {
    uint64_t atomic_ret_addr = get_noc_addr_helper(my_noc_xy, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    NOC_CMD_BUF_WRITE_REG(my_noc_index, DISPATCH_S_ATOMIC_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
    NOC_CMD_BUF_WRITE_REG(my_noc_index, DISPATCH_S_ATOMIC_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(atomic_ret_addr >> NOC_ADDR_COORD_SHIFT));
}

FORCE_INLINE
void dispatch_s_noc_semaphore_inc(uint64_t addr, uint32_t incr, uint8_t noc_id) {
    // dispatch_s specific atomic inc API, which will use DISPATCH_S_ATOMIC_CMD_BUF to ensure that
    // ncrisc and brisc don't clobber each other's resources when dispatch_s and dispatch_d are on
    // the same tensix core
    WAYPOINT("NSIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    DEBUG_INSERT_DELAY(TransactionAtomic);
    noc_fast_atomic_increment(noc_id, DISPATCH_S_ATOMIC_CMD_BUF, addr, NOC_UNICAST_WRITE_VC, incr, 31 /*wrap*/, false /*linked*/, false /*posted*/);
    WAYPOINT("NSID");
}

FORCE_INLINE
void dispatch_s_noc_inline_dw_write(uint64_t addr, uint32_t val, uint8_t noc_id, uint8_t be = 0xF) {
    WAYPOINT("NWIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    noc_fast_write_dw_inline(
                noc_id,
                DISPATCH_S_WR_REG_CMD_BUF,
                val,
                addr,
                be, // byte-enable
                NOC_UNICAST_WRITE_VC,
                false, // mcast
                false // posted
            );
    WAYPOINT("NWID");
}

FORCE_INLINE
void wait_for_workers(volatile CQDispatchCmd tt_l1_ptr *cmd) {
    uint8_t dispatch_message_offset = *((uint8_t *)&cmd->mcast.go_signal + offsetof(go_msg_t, dispatch_message_offset));
    volatile tt_l1_ptr uint32_t* worker_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_base_addr + dispatch_message_offset);
    while (wrap_gt(cmd->mcast.wait_count, *worker_sem));
}

template<bool flush_write = false>
FORCE_INLINE
void update_worker_completion_count_on_dispatch_d() {
    if constexpr(distributed_dispatcher) {
        bool write = false;
        for (uint32_t i = 0, worker_sem_addr = worker_sem_base_addr; i < num_worker_sems; ++i, worker_sem_addr += L1_ALIGNMENT) {
            uint32_t num_workers_signalling_completion = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr);
            if (num_workers_signalling_completion != worker_count_update_for_dispatch_d[i]) {
                worker_count_update_for_dispatch_d[i] = num_workers_signalling_completion;
                uint64_t dispatch_d_dst = get_noc_addr_helper(dispatch_d_noc_xy, worker_sem_addr);
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

template<uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void cb_acquire_pages_dispatch_s(uint32_t n) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    WAYPOINT("DAPW");
    uint32_t heartbeat = 0;
    // Stall until the number of pages already acquired + the number that need to be acquired is greater
    // than the number available
    while (wrap_gt(num_pages_acquired + n, *sem_addr)) {
        update_worker_completion_count_on_dispatch_d();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("DAPD");
    num_pages_acquired += n;
}

template<uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void cb_release_pages_dispatch_s(uint32_t n) {
    dispatch_s_noc_semaphore_inc(get_noc_addr_helper(noc_xy, get_semaphore<fd_core_type>(sem_id)), n, my_noc_index);
}

FORCE_INLINE
void process_go_signal_mcast_cmd() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    // Get semaphore that will be update by dispatch_d, signalling that it's safe to send a go signal
    volatile tt_l1_ptr uint32_t* sync_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dispatch_s_sync_sem_base_addr + (cmd->mcast.wait_addr - worker_sem_base_addr));

    // Wait for notification from dispatch_d, signalling that it's safe to send the go signal
    uint32_t& mcasts_sent = num_mcasts_sent[(cmd->mcast.wait_addr - worker_sem_base_addr) / L1_ALIGNMENT];
    while (wrap_ge(mcasts_sent, *sync_sem_addr)) {
        // Update dispatch_d with the latest num_workers
        update_worker_completion_count_on_dispatch_d();
    }
    mcasts_sent++; // Go signal sent -> update counter
    // Wait until workers have completed before sending go signal
    wait_for_workers(cmd);

    // The location of the go signal embedded in the command does not meet NOC alignment requirements.
    // cmd_ptr is guaranteed to meet the alignment requirements, since it is written to by prefetcher over NOC.
    // Copy the go signal from an unaligned location to an aligned (cmd_ptr) location. This is safe as long as we
    // can guarantee that copying the go signal does not corrupt any other command fields, which is true (see CQDispatchGoSignalMcastCmd).
    volatile uint32_t tt_l1_ptr* aligned_go_signal_storage = (volatile uint32_t tt_l1_ptr*)cmd_ptr;
    *aligned_go_signal_storage = cmd->mcast.go_signal;

    // send go signal update here
    volatile uint32_t tt_l1_ptr *data_ptr = reinterpret_cast<volatile uint32_t tt_l1_ptr *>(cmd_ptr + sizeof(CQDispatchCmd));
    for (uint32_t i = 0, num_mcasts = cmd->mcast.num_mcast_txns; i < num_mcasts; ++i) {
        uint64_t dst = get_noc_addr_helper(*(data_ptr++), mcast_go_signal_addr);
        // packed_write_max_unicast_sub_cmds is the total number of compute cores (num_mcast_dests for this txn)
        noc_async_write_multicast_one_packet((uint32_t)(aligned_go_signal_storage), dst, sizeof(uint32_t), *(data_ptr++));
    }
    for (uint32_t i = 0, num_unicasts = cmd->mcast.num_unicast_txns; i < num_unicasts; ++i) {
        uint64_t dst = get_noc_addr_helper(*(data_ptr++), unicast_go_signal_addr);
        noc_async_write_one_packet((uint32_t)(aligned_go_signal_storage), dst, sizeof(uint32_t));
    }
    update_worker_completion_count_on_dispatch_d();
    cmd_ptr = round_up_pow2((uint32_t)data_ptr, L1_ALIGNMENT);
}

FORCE_INLINE
void process_dispatch_s_wait_cmd() {
    static constexpr uint32_t worker_sem_max_addr = worker_sem_base_addr + (max_num_worker_sems - 1) * L1_ALIGNMENT;

    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    // Limited Usage of Wait CMD: dispatch_s should get a wait command only if it's not on the
    // same core as dispatch_d and is used to clear the worker count
    ASSERT(cmd->wait.clear_count && distributed_dispatcher);
    uint32_t worker_sem_addr = cmd->wait.addr;
    ASSERT(worker_sem_addr >= worker_sem_base_addr && worker_sem_addr <= worker_sem_max_addr);
    uint32_t index = (worker_sem_addr - worker_sem_base_addr) / L1_ALIGNMENT;
    volatile tt_l1_ptr uint32_t* worker_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr);
    // Wait for workers to complete
    while (wrap_gt(cmd->wait.count, *worker_sem));
    // Send updated worker count to dispatch_d and wait for updated count to get picked up by NOC before clearing the counter.
    // dispatch_d will clear it's own counter
    update_worker_completion_count_on_dispatch_d<true>();
    *worker_sem = 0;
    worker_count_update_for_dispatch_d[index] = 0; // Local worker count update for dispatch_d should reflect state of worker semaphore on dispatch_s
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void set_num_worker_sems() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    num_worker_sems = cmd->set_num_worker_sems.num_worker_sems;
    ASSERT(num_worker_sems <= max_num_worker_sems);
    cmd_ptr += sizeof(CQDispatchCmd);
}

void kernel_main() {
    DPRINT << "dispatch_s : start" << ENDL();
    // Initialize customized command buffers.
    dispatch_s_wr_reg_cmd_buf_init();
    dispatch_s_atomic_cmd_buf_init();
    cmd_ptr = cb_base;
    bool done = false;
    uint32_t total_pages_acquired = 0;
    while(!done) {
        cb_acquire_pages_dispatch_s<my_noc_xy, my_dispatch_cb_sem_id>(1);

        volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
        switch (cmd->base.cmd_id) {
            case CQ_DISPATCH_CMD_SEND_GO_SIGNAL:
                process_go_signal_mcast_cmd();
                break;
            case CQ_DISPATCH_SET_NUM_WORKER_SEMS:
                set_num_worker_sems();
                break;
            case CQ_DISPATCH_CMD_WAIT:
                process_dispatch_s_wait_cmd();
                break;
            case CQ_DISPATCH_CMD_TERMINATE:
                done = true;
                break;
            default:
                DPRINT << "dispatcher_s invalid command" << ENDL();
                ASSERT(0);
        }
        cmd_ptr = round_up_pow2(cmd_ptr, cb_page_size);
        // Release a single page to prefetcher. Assumption is that all dispatch_s commands fit inside a single page for now.
        cb_release_pages_dispatch_s<upstream_noc_xy, upstream_dispatch_cb_sem_id>(1);
        if (cmd_ptr == cb_end) {
            cmd_ptr = cb_base;
        }
        total_pages_acquired++;
    }
    // Confirm expected number of pages, spinning here is a leak
    cb_wait_all_pages<my_dispatch_cb_sem_id>(total_pages_acquired);
    DPRINT << "dispatch_s : done" << ENDL();
}
