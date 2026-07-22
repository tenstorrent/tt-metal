// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch Kernel (subordinate)
// Required to asynchronously send go signals to workers, upon receiving a program
// completion signal. This allows program dispatch (for subsequent programs) to overlap
// with worker execution (for current program), leading to a lower dispatch latency.
// - Handles the following commands:
//  - CQ_DISPATCH_CMD_SEND_GO_SIGNAL: "multicast" go signal to all workers
//  - CQ_DISPATCH_CMD_WAIT: Wait for workers to complete and reset wait count
//    and instead need a unicast for the go signal

#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#if DEVICE_PRINT_DISPATCH_ENABLED
#include "tt_metal/impl/dispatch/kernels/device_print_dispatch.h"
#endif
#include "tt_metal/impl/dispatch/kernels/realtime_profiler.hpp"
#include "hostdevcommon/profiler_common.h"
#include "hostdevcommon/dispatch_telemetry_types.hpp"
#include "hostdev/dev_msgs.h"
#include "risc_common.h"

#include <array>

// dispatch_s has a customized command buffer allocation for NOC 1.
// Cmd Buf 0 is used for regular writes.
// Cmd Buf 1 is used for small (inline) writes.
// Cmd Buf 2 is used for atomics.
// Cmd Buf 3 is unavailable (used by dispatch_d).
// Reads cannot be issued by dispatch_s.
constexpr uint32_t DISPATCH_S_WR_REG_CMD_BUF = 1;
constexpr uint32_t DISPATCH_S_ATOMIC_CMD_BUF = 2;
constexpr uintptr_t cb_base = CB_BASE;
constexpr uint32_t cb_log_page_size = CB_LOG_PAGE_SIZE;
constexpr uint32_t cb_size = CB_SIZE;
constexpr uint32_t my_dispatch_cb_sem_id = MY_DISPATCH_CB_SEM_ID;
constexpr uint32_t upstream_dispatch_cb_sem_id = UPSTREAM_DISPATCH_CB_SEM_ID;
constexpr uint32_t dispatch_d_shutdown_sem_id = DISPATCH_D_SHUTDOWN_SEM_ID;
constexpr uintptr_t dispatch_s_sync_sem_base_addr = DISPATCH_S_SYNC_SEM_BASE_ADDR;
constexpr uint32_t mcast_go_signal_addr = MCAST_GO_SIGNAL_ADDR;
constexpr uint32_t unicast_go_signal_addr = UNICAST_GO_SIGNAL_ADDR;
constexpr uint32_t distributed_dispatcher =
    DISTRIBUTED_DISPATCHER;  // dispatch_s and dispatch_d running on different cores
constexpr uint32_t first_stream_used = FIRST_STREAM_USED;
constexpr uint32_t completion_counter_offset = COMPLETION_COUNTER_OFFSET;
constexpr uint32_t max_num_worker_sems = MAX_NUM_WORKER_SEMS;
constexpr uint32_t max_num_go_signal_noc_data_entries = MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES;
constexpr uintptr_t dispatch_telemetry_control_addr = DISPATCH_TELEMETRY_CONTROL_ADDR;
constexpr bool telemetry_enabled = !DISPATCH_TELEMETRY_DISABLED;
constexpr uintptr_t dispatch_telemetry_base = DISPATCH_TELEMETRY_ADDR;
constexpr uint32_t virtualize_unicast_cores = VIRTUALIZE_UNICAST_CORES;
constexpr uint32_t num_virtual_unicast_cores = NUM_VIRTUAL_UNICAST_CORES;
constexpr uint32_t num_physical_unicast_cores = NUM_PHYSICAL_UNICAST_CORES;
volatile tt_l1_ptr tt::tt_metal::dispatch_telemetry_types::DispatchTelemetryControl* dispatch_telemetry_control =
    reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::dispatch_telemetry_types::DispatchTelemetryControl*>(
        dispatch_telemetry_control_addr);

constexpr uint32_t worker_mcast_grid = WORKER_MCAST_GRID;
constexpr uint32_t num_worker_cores_to_mcast = NUM_WORKER_CORES_TO_MCAST;

#if DEVICE_PRINT_DISPATCH_ENABLED
constexpr uint32_t device_print_noc_locations_addr = DEVICE_PRINT_NOC_LOCATIONS_ADDR;
constexpr uint32_t device_print_noc_locations_count = DEVICE_PRINT_NOC_LOCATIONS_COUNT;
constexpr uint32_t device_print_l1_cache_addr = DEVICE_PRINT_L1_CACHE_ADDR;
constexpr uint32_t device_print_l1_cache_size = DEVICE_PRINT_L1_CACHE_SIZE;
constexpr uint16_t device_print_dram_x = DEVICE_PRINT_DRAM_X;
constexpr uint16_t device_print_dram_y = DEVICE_PRINT_DRAM_Y;
constexpr uint64_t device_print_dram_rw_ptrs = DEVICE_PRINT_DRAM_RW_PTRS;
constexpr uint64_t device_print_dram_buf_addr = DEVICE_PRINT_DRAM_BUF_ADDR;
constexpr uint32_t device_print_dram_buf_size = DEVICE_PRINT_DRAM_BUF_SIZE;
constexpr uint64_t device_print_cycles_for_stall = DEVICE_PRINT_CYCLES_FOR_STALL;
constexpr uint64_t device_print_cycles_for_full = DEVICE_PRINT_CYCLES_FOR_FULL;

// RAII guard for dispatch_s's NOC cmd_buf state on cmd_buf 0 (NCRISC_WR_CMD_BUF, used by
// dispatch_s for regular writes) and cmd_buf 1 (NCRISC_RD_CMD_BUF, used by dispatch_s for
// inline writes via dispatch_s_noc_inline_dw_write).
//
// Why we save state: dispatch_s issues `cq_noc_async_write_with_state` calls that rely on
// cmd_buf state programmed by a preceding `_init_state`. If our execute() runs between an
// init_state and with_state pair, our init_state + with_state operations would clobber
// dispatch_s's NOC_CTRL / NOC_*_ADDR_COORDINATE. The wait_for_workers reorder in
// process_go_signal_mcast_cmd protects today's known site, but we save/restore for
// defense-in-depth so future dispatch_s code can rely on cmd_buf state being preserved
// across an execute() / shutdown() call.
//
// Constructor: snapshot the regs, then call the new-API _init_state functions to program
// NOC_CTRL for our reads (cmd_buf 1) and writes (cmd_buf 0).
// Destructor: restore the saved regs.
struct DispatchSNocCmdBufGuard {
    uint32_t saved_rd_ctrl;
    uint32_t saved_rd_targ_coord;
    uint32_t saved_wr_ctrl;
    uint32_t saved_wr_ret_coord;

    DispatchSNocCmdBufGuard() {
        saved_rd_ctrl = NOC_CMD_BUF_READ_REG(NOC_INDEX, NCRISC_RD_CMD_BUF, NOC_CTRL);
        saved_rd_targ_coord = NOC_CMD_BUF_READ_REG(NOC_INDEX, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE);
        saved_wr_ctrl = NOC_CMD_BUF_READ_REG(NOC_INDEX, NCRISC_WR_CMD_BUF, NOC_CTRL);
        saved_wr_ret_coord = NOC_CMD_BUF_READ_REG(NOC_INDEX, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE);
    }

    ~DispatchSNocCmdBufGuard() {
        NOC_CMD_BUF_WRITE_REG(NOC_INDEX, NCRISC_RD_CMD_BUF, NOC_CTRL, saved_rd_ctrl);
        NOC_CMD_BUF_WRITE_REG(NOC_INDEX, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, saved_rd_targ_coord);
        NOC_CMD_BUF_WRITE_REG(NOC_INDEX, NCRISC_WR_CMD_BUF, NOC_CTRL, saved_wr_ctrl);
        NOC_CMD_BUF_WRITE_REG(NOC_INDEX, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, saved_wr_ret_coord);
    }

    DispatchSNocCmdBufGuard(const DispatchSNocCmdBufGuard&) = delete;
    DispatchSNocCmdBufGuard& operator=(const DispatchSNocCmdBufGuard&) = delete;
};

static DevicePrintDispatch<
    true,
    DEVICE_PRINT_MAX_NOC_LOCATIONS,
    device_print_dispatch::NOC_L1_TO_L1_ALIGNMENT,
    device_print_dispatch::NOC_L1_TO_DRAM_ALIGNMENT,
    DispatchSNocCmdBufGuard>
    device_print_dispatcher;

void device_print_dispatcher_execute_hook() {
    // This function shouldn't be called unless there are lots of DEVICE_PRINT
    // calls inside dispatch_s kernel. We are not optimizing this path as it is
    // fairly unlikely to be hit. When DEVICE_PRINT buffer is full, dispatcher
    // will be executed to drain this buffer and check for all others as well.
    // Here we force stall detection execution to avoid waiting for timer.
    ::device_print_dispatcher.execute(true);
}
#endif

constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t dispatch_d_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint8_t my_noc_index = NOC_INDEX;

constexpr uint32_t cb_page_size = 1 << cb_log_page_size;
constexpr uintptr_t cb_end = cb_base + cb_size;

// Dispatch-core-local L1 region assigned by DispatchMemMap via
// CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG. Address comes from host through the
// REALTIME_PROFILER_MSG_ADDR compile-time define. See cq_dispatch.cpp for the full mailbox
// description; this kernel is the consumer side of the embedded program_id_fifo.
volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

static bool rt_profiler_enabled = false;

static uint32_t num_pages_acquired = 0;
static uint32_t num_mcasts_sent[max_num_worker_sems] = {0};
static uintptr_t cmd_ptr;

extern "C" {
// These variables are used by triage to help report dispatcher state.
volatile uint32_t last_wait_count = 0;
volatile uint32_t last_wait_stream = 0;
constexpr uint32_t stream_addr0 = STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
constexpr uint32_t stream_addr1 = STREAM_REG_ADDR(1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
constexpr uint32_t stream_width = MEM_WORD_ADDR_WIDTH;
}

// When dispatch_d and dispatch_s run on separate cores, dispatch_s gets the go signal update from workers.
// dispatch_s is responsible for sending the latest worker completion count to dispatch_d.
// To minimize the number of writes from dispatch_s to dispatch_d, locally track dispatch_d's copy.
static uint32_t worker_count_update_for_dispatch_d[max_num_worker_sems] = {0};

static uint32_t go_signal_noc_data[max_num_go_signal_noc_data_entries];

static uint32_t num_worker_sems = 1;

// The dispatch message entry limit also bounds the number of sub-devices.
static std::array<uint32_t, max_num_worker_sems> workers_per_sub_device = {0};

FORCE_INLINE
void dispatch_s_wr_reg_cmd_buf_init() {
    uint64_t xy_local_addr = get_noc_addr_helper(my_noc_xy, 0);
    noc_cmd_buf_set_targ_addr_coordinate(
        my_noc_index, DISPATCH_S_WR_REG_CMD_BUF, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));
}

FORCE_INLINE
void dispatch_s_atomic_cmd_buf_init() {
    uint64_t atomic_ret_addr = get_noc_addr_helper(my_noc_xy, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    noc_cmd_buf_set_ret_addr(my_noc_index, DISPATCH_S_ATOMIC_CMD_BUF, atomic_ret_addr);
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
void signal_realtime_profiler_and_switch(volatile tt_l1_ptr realtime_profiler_msg_t* msg) {
    RealtimeProfilerState current_state = static_cast<RealtimeProfilerState>(msg->realtime_profiler_state);
    bool used_buffer_a = (current_state == REALTIME_PROFILER_STATE_PUSH_B);

    RealtimeProfilerState new_state = used_buffer_a ? REALTIME_PROFILER_STATE_PUSH_A : REALTIME_PROFILER_STATE_PUSH_B;
    msg->realtime_profiler_state = new_state;

    if (msg->realtime_profiler_core_noc_xy != 0) {
        uint64_t realtime_profiler_addr =
            get_noc_addr_helper(msg->realtime_profiler_core_noc_xy, msg->realtime_profiler_remote_state_addr);
        dispatch_s_noc_inline_dw_write(realtime_profiler_addr, static_cast<uint32_t>(new_state), my_noc_index);
    }
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
void wait_for_workers(uint32_t wait_count, uint32_t wait_stream) {
    WAYPOINT("WCW");
    last_wait_count = wait_count;
    last_wait_stream = wait_stream;
#ifdef ARCH_QUASAR
    volatile uint32_t* worker_sem =
        worker_completion_sem_addr(wait_stream, first_stream_used, completion_counter_offset);
#else
    volatile uint32_t* worker_sem = reinterpret_cast<volatile uint32_t*>(
        static_cast<uintptr_t>(STREAM_REG_ADDR(wait_stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX)));
#endif
    DPRINT("DISPATCH_S: wait_for_workers: wait_count: {}, worker_sem: {}\n", wait_count, *worker_sem);
#ifdef ARCH_QUASAR
    while (wrap_gt(wait_count, *worker_sem)) {
#else
    while (stream_wrap_gt(wait_count, *worker_sem)) {
#endif
        if (rt_profiler_enabled) {
            record_realtime_timestamp(rt_profiler_msg, false);
        }
#if DEVICE_PRINT_DISPATCH_ENABLED
        device_print_dispatcher.execute();
#endif
    }

    WAYPOINT("WCD");
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
    volatile tt_l1_ptr uint32_t* sem_addr = uncached_l1_ptr<uint32_t>(get_semaphore<fd_core_type>(sem_id));

    WAYPOINT("DAPW");
    uint32_t heartbeat = 0;
    // Stall until the number of pages already acquired + the number that need to be acquired is greater
    // than the number available
    while (wrap_gt(num_pages_acquired + n, *sem_addr)) {
        invalidate_l1_cache();
        update_worker_completion_count_on_dispatch_d();
#if DEVICE_PRINT_DISPATCH_ENABLED
        device_print_dispatcher.execute();
#endif
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("DAPD");
    num_pages_acquired += n;
}

template <uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE void cb_release_pages_dispatch_s(uint32_t n) {
#ifdef ARCH_QUASAR
    Semaphore<fd_core_type>(sem_id).up(n);
#else
    dispatch_s_noc_semaphore_inc(get_noc_addr_helper(noc_xy, get_semaphore<fd_core_type>(sem_id)), n, my_noc_index);
#endif
}

FORCE_INLINE
void process_go_signal_mcast_cmd() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = uncached_l1_ptr<CQDispatchCmd>(cmd_ptr);
    uint32_t sync_index = cmd->mcast.wait_stream - first_stream_used;
    // Get semaphore that will be update by dispatch_d, signalling that it's safe to send a go signal

    volatile tt_l1_ptr uint32_t* sync_sem_addr =
        uncached_l1_ptr<uint32_t>(dispatch_s_sync_sem_base_addr + sync_index * L1_ALIGNMENT);

    WAYPOINT("DCW");
    // Wait for notification from dispatch_d, signalling that it's safe to send the go signal
    uint32_t& mcasts_sent = num_mcasts_sent[sync_index];
    while (wrap_ge(mcasts_sent, *sync_sem_addr)) {
        invalidate_l1_cache();
        // Update dispatch_d with the latest num_workers
        update_worker_completion_count_on_dispatch_d();
#if DEVICE_PRINT_DISPATCH_ENABLED
        device_print_dispatcher.execute();
#endif
    }
    mcasts_sent++;  // Go signal sent -> update counter

    // The location of the go signal embedded in the command does not meet NOC alignment requirements.
    // cmd_ptr is guaranteed to meet the alignment requirements, since it is written to by prefetcher over NOC.
    // Copy the go signal from an unaligned location to an aligned (cmd_ptr) location. This is safe as long as we
    // can guarantee that copying the go signal does not corrupt any other command fields, which is true (see
    // CQDispatchGoSignalMcastCmd).
    // NOC source addresses must be raw L1 byte offsets (cached-alias form), so keep
    // aligned_go_signal_storage at the cached alias for the NOC sources below.
    // CPU writes go through a separate uncached pointer so the value lands in L1 SRAM directly;
    // the NOC then reads the same physical location via the cached-form source address.
    volatile uint32_t tt_l1_ptr* aligned_go_signal_storage = (volatile uint32_t tt_l1_ptr*)cmd_ptr;
    volatile uint32_t tt_l1_ptr* aligned_go_signal_storage_uncached = uncached_l1_ptr<uint32_t>(cmd_ptr);
    uint32_t go_signal_value = cmd->mcast.go_signal;
    uint8_t go_signal_noc_data_idx = cmd->mcast.noc_data_start_index;
    uint32_t multicast_go_offset = cmd->mcast.multicast_go_offset;
    uint32_t num_unicasts = cmd->mcast.num_unicast_txns;
    uint32_t wait_count = cmd->mcast.wait_count;
    uint32_t wait_stream = cmd->mcast.wait_stream;

    if (multicast_go_offset != CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET) {
        // Setup registers before waiting for workers so only the NOC_CMD_CTRL register needs to be touched after.
        uint64_t dst_noc_addr_multicast =
            get_noc_addr_helper(worker_mcast_grid, mcast_go_signal_addr + sizeof(uint32_t) * multicast_go_offset);
        uint32_t num_dests = num_worker_cores_to_mcast;
        // Ensure the offset with respect to L1_ALIGNMENT is the same for the source and destination.
        uint32_t storage_offset = multicast_go_offset % (L1_ALIGNMENT / sizeof(uint32_t));
        aligned_go_signal_storage_uncached[storage_offset] = go_signal_value;

#if DEVICE_PRINT_DISPATCH_ENABLED
        // wait_for_workers polls device_print_dispatcher.execute() inside its busy loop when
        // DEVICE_PRINT dispatch is enabled. That dispatcher may issue writes using
        // NCRISC_WR_REG_CMD_BUF, which would clobber the state programmed by
        // cq_noc_async_write_init_state below. Wait first in that build so init_state's state
        // is the last thing touching this command buffer before the cq_noc_async_write_with_state
        // call. In the non-DEVICE_PRINT build keep the original ordering so init_state + write
        // accounting overlap with the worker wait.
        wait_for_workers(wait_count, wait_stream);
#endif

        cq_noc_async_write_init_state<CQ_NOC_SNDL, true>(
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&aligned_go_signal_storage[storage_offset])),
            dst_noc_addr_multicast,
            sizeof(uint32_t));

        // Multicast write accounting: increment counters for num_dests acks and one issued transaction.
        noc_increment_nonposted_writes_acked(noc_index, num_dests);

#if !DEVICE_PRINT_DISPATCH_ENABLED
        wait_for_workers(wait_count, wait_stream);
#endif
        cq_noc_async_write_with_state<CQ_NOC_sndl, CQ_NOC_wait>(0, 0, 0);
        noc_increment_nonposted_writes_issued(noc_index, 1);
    } else {
        wait_for_workers(wait_count, wait_stream);
    }

    *aligned_go_signal_storage_uncached = go_signal_value;
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
#ifdef ARCH_QUASAR
            *worker_completion_sem_addr(first_stream_used, first_stream_used, completion_counter_offset) +=
                (num_virtual_unicast_cores - num_physical_unicast_cores);
#else
            NOC_STREAM_WRITE_REG(
                first_stream_used,
                STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
                (num_virtual_unicast_cores - num_physical_unicast_cores) << REMOTE_DEST_BUF_WORDS_FREE_INC);
#endif
        }
    }

    for (uint32_t i = 0; i < num_unicasts; ++i) {
        uint64_t dst = get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], unicast_go_signal_addr);
        noc_async_write_one_packet(
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(aligned_go_signal_storage)), dst, sizeof(uint32_t));
    }

    if (telemetry_enabled) {
        static uint32_t local_launch_seq_counter = 0;
        const uint32_t stream_index = wait_stream - first_stream_used;
        ASSERT(stream_index < max_num_worker_sems);
        auto dispatch_telemetry =
            reinterpret_cast<volatile tt_l1_ptr tt::tt_metal::dispatch_telemetry_types::DispatchCoreTelemetry*>(
                dispatch_telemetry_base);

        dispatch_telemetry_control->launched_work_sequence_counter[stream_index] = ++local_launch_seq_counter;
        dispatch_telemetry->last_work_launch_timestamp[stream_index] = get_timestamp();
        dispatch_telemetry_control->launched_work_start_stream_sem[stream_index] = wait_count;
        dispatch_telemetry_control->launched_work_sequence_counter[stream_index] = ++local_launch_seq_counter;
    }

#if DEVICE_PRINT_DISPATCH_ENABLED
    // Workers have just been notified to start a new program; reset the stall-detection
    // window so it measures THIS program's print activity rather than dispatch_s's
    // overall lifetime.
    device_print_dispatcher.notify_kernel_start();
#endif

    update_worker_completion_count_on_dispatch_d();
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void process_dispatch_s_wait_cmd() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = uncached_l1_ptr<CQDispatchCmd>(cmd_ptr);
    // Limited Usage of Wait CMD: dispatch_s should get a wait command only if it's not on the
    // same core as dispatch_d and is used to clear the worker count
    ASSERT(
        (cmd->wait.flags == (CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM)) &&
        distributed_dispatcher);
    uint32_t stream = cmd->wait.stream;
    uint32_t index = stream - first_stream_used;
    volatile uint32_t* worker_sem = reinterpret_cast<volatile uint32_t*>(
        static_cast<uintptr_t>(STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX)));

    // Wait for workers to complete
    while (stream_wrap_gt(cmd->wait.count, *worker_sem)) {
#if DEVICE_PRINT_DISPATCH_ENABLED
        device_print_dispatcher.execute();
#endif
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
    volatile CQDispatchCmd tt_l1_ptr* cmd = uncached_l1_ptr<CQDispatchCmd>(cmd_ptr);
    num_worker_sems = cmd->set_num_worker_sems.num_worker_sems;
    ASSERT(num_worker_sems <= max_num_worker_sems);
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void set_go_signal_noc_data() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = uncached_l1_ptr<CQDispatchCmd>(cmd_ptr);
    uint32_t num_words = cmd->set_go_signal_noc_data.num_words;
    ASSERT(num_words <= max_num_go_signal_noc_data_entries);
    volatile tt_l1_ptr uint32_t* data_ptr = uncached_l1_ptr<uint32_t>(cmd_ptr + sizeof(CQDispatchCmd));
    for (uint32_t i = 0; i < num_words; ++i) {
        go_signal_noc_data[i] = *(data_ptr++);
    }
    cmd_ptr = round_up_pow2(l1_cached_addr(reinterpret_cast<uintptr_t>(data_ptr)), L1_ALIGNMENT);
}

// When dispatch_d runs on the same core, it issues transactions on dispatch_s's dedicated NOC
// that we never count locally. This function will wait for dispatch_d to publish its NOC 1 deltas into dispatch_d's
// normal counter slots, then merge any non-zero deltas into our local counters before the barrier.
FORCE_INLINE
void merge_dispatch_d_noc_counter_deltas() {
    if constexpr (distributed_dispatcher) {
        DPRINT("merge_dispatch_d_noc_counter_deltas is only supported when dispatch_d runs on the same core\n");
        ASSERT(0);
        return;
    }

    constexpr auto dispatch_d_proc_type = static_cast<decltype(proc_type)>(TensixProcessorTypes::DM0);

    volatile tt_l1_ptr uint32_t* shutdown_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(dispatch_d_shutdown_sem_id));
    noc_semaphore_wait(shutdown_sem_addr, 1);

    invalidate_l1_cache();
    const uint32_t reads_delta =
        get_noc_counter_val<dispatch_d_proc_type, NocBarrierType::READS_NUM_ISSUED>(my_noc_index);
    const uint32_t nonposted_writes_delta =
        get_noc_counter_val<dispatch_d_proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(my_noc_index);
    const uint32_t nonposted_writes_acked_delta =
        get_noc_counter_val<dispatch_d_proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(my_noc_index);
    const uint32_t nonposted_atomics_acked_delta =
        get_noc_counter_val<dispatch_d_proc_type, NocBarrierType::NONPOSTED_ATOMICS_ACKED>(my_noc_index);
    const uint32_t posted_writes_delta =
        get_noc_counter_val<dispatch_d_proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(my_noc_index);

    if (reads_delta != 0) {
        noc_reads_num_issued[my_noc_index] += reads_delta;
    }
    if (nonposted_writes_delta != 0) {
        noc_nonposted_writes_num_issued[my_noc_index] += nonposted_writes_delta;
    }
    if (nonposted_writes_acked_delta != 0) {
        noc_nonposted_writes_acked[my_noc_index] += nonposted_writes_acked_delta;
    }
    if (nonposted_atomics_acked_delta != 0) {
        noc_nonposted_atomics_acked[my_noc_index] += nonposted_atomics_acked_delta;
    }
    if (posted_writes_delta != 0) {
        noc_posted_writes_num_issued[my_noc_index] += posted_writes_delta;
    }
}

void kernel_main() {
    set_l1_data_cache<true>();
    DPRINT("dispatch_s : start\n");
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

    // realtime_profiler_msg_t signalling + FIFO + kernel_* .id fields are zeroed on the host in
    // DispatchSKernel::ConfigureCore() before CQ kernels launch.

    cmd_ptr = cb_base;
    bool done = false;
    uint32_t total_pages_acquired = 0;
#if DEVICE_PRINT_DISPATCH_ENABLED
    device_print_dispatcher.init(
        device_print_noc_locations_addr,
        device_print_noc_locations_count,
        device_print_l1_cache_addr,
        device_print_l1_cache_size,
        device_print_dram_x,
        device_print_dram_y,
        device_print_dram_rw_ptrs,
        device_print_dram_buf_addr,
        device_print_dram_buf_size,
        device_print_cycles_for_stall,
        device_print_cycles_for_full);
    // notify_kernel_start() is invoked from process_go_signal_mcast_cmd, after the
    // go signal is sent — the stall-detection window is per-program, not per-dispatch_s.
#endif
    while (!done) {
        DeviceZoneScopedN("CQ-DISPATCH-SUBORDINATE");
        rt_profiler_enabled = (rt_profiler_msg->realtime_profiler_core_noc_xy != 0);
        uint32_t popped_pid = 0;
        if (rt_profiler_enabled) {
            record_realtime_timestamp(rt_profiler_msg, true);
            popped_pid = pop_program_id(rt_profiler_msg);
        }
#if DEVICE_PRINT_DISPATCH_ENABLED
        device_print_dispatcher.execute();
#endif
        cb_acquire_pages_dispatch_s<my_noc_xy, my_dispatch_cb_sem_id>(1);

        volatile CQDispatchCmd tt_l1_ptr* cmd = uncached_l1_ptr<CQDispatchCmd>(cmd_ptr);
        DeviceTimestampedData("process_cmd_d_dispatch_subordinate", (uint32_t)cmd->base.cmd_id);
        if (rt_profiler_enabled) {
            const bool is_profiled_cmd = cmd->base.cmd_id == CQ_DISPATCH_CMD_SEND_GO_SIGNAL ||
                                         cmd->base.cmd_id == CQ_DISPATCH_CMD_RT_PROFILER_FLUSH;
            write_buffer_id(
                rt_profiler_msg,
                is_profiled_cmd ? popped_pid : static_cast<uint32_t>(REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID));
        }
        switch (cmd->base.cmd_id) {
            case CQ_DISPATCH_CMD_SEND_GO_SIGNAL:
                DPRINT("CQ_DISPATCH_CMD_SEND_GO_SIGNAL\n");
                process_go_signal_mcast_cmd();
                break;
            case CQ_DISPATCH_SET_NUM_WORKER_SEMS:
                DPRINT("CQ_DISPATCH_SET_NUM_WORKER_SEMS\n");
                set_num_worker_sems();
                break;
            case CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA:
                DPRINT("CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA\n");
                set_go_signal_noc_data();
                break;
            case CQ_DISPATCH_SET_SUB_DEVICE_WORKER_COUNTS:
                DPRINT("CQ_DISPATCH_SET_SUB_DEVICE_WORKER_COUNTS\n");
                cmd_ptr += set_sub_device_worker_counts<telemetry_enabled>(
                    cmd_ptr,
                    workers_per_sub_device,
                    &dispatch_telemetry_control->sub_device_worker_counts_update,
                    dispatch_telemetry_base);
                break;
            case CQ_DISPATCH_CMD_WAIT:
                DPRINT("CQ_DISPATCH_CMD_WAIT\n");
                process_dispatch_s_wait_cmd();
                break;
            case CQ_DISPATCH_CMD_RT_PROFILER_FLUSH:
                DPRINT("CQ_DISPATCH_CMD_RT_PROFILER_FLUSH\n");
                wait_for_workers(cmd->rt_profiler_flush.wait_count, cmd->rt_profiler_flush.wait_stream);
                cmd_ptr += sizeof(CQDispatchCmd);
                break;
            case CQ_DISPATCH_CMD_TERMINATE:
                DPRINT("CQ_DISPATCH_CMD_TERMINATE\n");
                if (rt_profiler_enabled) {
                    signal_realtime_profiler_and_switch(rt_profiler_msg);
                    noc_async_writes_flushed();
                    for (volatile uint32_t delay = 0; delay < 5000; delay++) {
                    }
                }

                rt_profiler_msg->realtime_profiler_state = REALTIME_PROFILER_STATE_TERMINATE;
                if (rt_profiler_enabled) {
                    uint64_t realtime_profiler_terminate_addr = get_noc_addr_helper(
                        rt_profiler_msg->realtime_profiler_core_noc_xy,
                        rt_profiler_msg->realtime_profiler_remote_state_addr);
                    dispatch_s_noc_inline_dw_write(
                        realtime_profiler_terminate_addr, REALTIME_PROFILER_STATE_TERMINATE, my_noc_index);
                }
                if constexpr (telemetry_enabled) {
                    dispatch_telemetry_control->compute_terminate = 1;
                }
                done = true;
                break;
            default: DPRINT("dispatcher_s invalid command\n"); ASSERT(0);
        }
        // Dispatch s only supports single page commands for now
        ASSERT(cmd_ptr <= (l1_cached_addr(reinterpret_cast<uintptr_t>(cmd)) + cb_page_size));
        cmd_ptr = round_up_pow2(cmd_ptr, cb_page_size);
        // Release a single page to prefetcher. Assumption is that all dispatch_s commands fit inside a single page for
        // now.
        cb_release_pages_dispatch_s<upstream_noc_xy, upstream_dispatch_cb_sem_id>(1);
        if (cmd_ptr == cb_end) {
            cmd_ptr = cb_base;
        }
        total_pages_acquired++;

        if (!done && rt_profiler_enabled) {
            signal_realtime_profiler_and_switch(rt_profiler_msg);
        }
    }
    // Confirm expected number of pages, spinning here is a leak
    cb_wait_all_pages<my_dispatch_cb_sem_id>(total_pages_acquired);

#ifndef ARCH_QUASAR
    if constexpr (!distributed_dispatcher) {
        merge_dispatch_d_noc_counter_deltas();
    }
#endif

    noc_async_full_barrier();

    DPRINT("dispatch_s : done\n");
#if DEVICE_PRINT_DISPATCH_ENABLED
    device_print_dispatcher.shutdown();
#endif
    set_l1_data_cache<false>();
}
