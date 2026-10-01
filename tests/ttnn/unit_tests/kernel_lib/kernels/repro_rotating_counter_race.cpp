// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Repro: rotating sender + DataReadySignal::Counter + handshake=false loses a counter increment.
//
// Every core on a line is both a sender (on its own phase) and a receiver (on all other phases) of
// one rotating channel and only exchanges control signals (send_signal / receive_signal), so the next
// round's sender turns around as fast as possible after its receive.
//
// After each send, SenderPipe bumps its own data_ready counter with a NON-atomic L1 load/add/store
// (Semaphore::up under LOCAL_NONATOMIC). If the next sender's inc_multicast lands between that load
// and store, one increment is lost and this core's counter stays one short forever.
//
// Instead of hanging in wait_min(), every receive is preceded by a bounded poll of the same counter.
// A timeout is recorded (round, observed vs expected) and the core leaves the loop, so the program
// always completes and the host can report which core lost the increment.
#include <stdint.h>
#include <optional>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

// Status record layout (16 words = 64 B, one DRAM page per core).
enum : uint32_t {
    ST_MAGIC = 0,
    ST_CORE,
    ST_FAILED,
    ST_FAIL_ROUND,
    ST_OBSERVED,
    ST_EXPECTED,
    ST_FINAL_COUNTER,
    ST_ROUNDS_SENT,
    ST_ROUNDS_DONE,
    ST_WORDS = 16
};

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(1);
    constexpr uint32_t max_delay = get_compile_time_arg_val(2);  // receive->next-op jitter, in nops
    constexpr uint32_t timeout_polls = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    constexpr uint32_t mcast_ct = get_named_compile_time_arg_val("mcast_ct_offset");
    constexpr auto mc = McastArgs<mcast_ct, get_named_compile_time_arg_val("mcast_rt_offset")>();
    static_assert(mc.active && mc.rotating, "repro needs a rotating channel");
    static_assert(mc.signal == DataReadySignal::Counter, "repro needs the Counter data-ready signal");

    // The exact word SenderPipe::up(1) and the remote inc_multicast both modify.
    constexpr uint32_t data_ready_id = dataflow_kernel_lib::detail::
        positional_mcast_semaphore<mcast_ct, dataflow_kernel_lib::mcast_wire::DATA_READY>();
    volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_id));

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_index = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_obj(cb);
    cb_obj.reserve_back(1);
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_obj.get_write_ptr());

    auto send_pipe = mc.optional_sender(noc);
    auto recv_pipe = mc.optional_receiver(noc);

    uint32_t failed = 0, fail_round = 0, observed = 0, expected = 0, sent = 0, r = 0;
    for (; r < num_rounds; ++r) {
        if (mc.should_send(r)) {
            send_pipe->send_signal();  // inc_multicast, atomic barrier, then the local non-atomic up(1)
            ++sent;
        } else {
            if constexpr (!mc.pre_handshake) {
                // Counter receive waits for value >= r + 1 (ReceiverPipe::receive_signal -> wait_min(r + 1)).
                // Bounded here so a lost increment is reported instead of hanging. Only valid without a
                // handshake: with one, the sender waits for this core's ack inside receive_signal().
                const uint32_t want = r + 1;
                uint32_t v = *counter;
                for (uint32_t polls = 0; v < want && polls < timeout_polls; ++polls) {
                    v = *counter;
                }
                if (v < want) {
                    failed = 1;
                    fail_round = r;
                    observed = v;
                    expected = want;
                    break;
                }
            }
            // No handshake: returns immediately. Handshake control: acks, then waits (unbounded).
            recv_pipe->receive_signal(r);
            // Sweep the receive -> send turnaround so the next atomic lands at varying offsets
            // relative to the previous sender's local load/add/store.
            if constexpr (max_delay > 0) {
                const uint32_t d = ((r * 2654435761u) >> 16) % (max_delay + 1);
                for (uint32_t i = 0; i < d; ++i) {
                    asm volatile("nop");
                }
            }
        }
    }

    noc.async_atomic_barrier();
    for (uint32_t i = 0; i < ST_WORDS; ++i) {
        status[i] = 0;
    }
    status[ST_MAGIC] = 0xC0FFEE00u;
    status[ST_CORE] = core_index;
    status[ST_FAILED] = failed;
    status[ST_FAIL_ROUND] = fail_round;
    status[ST_OBSERVED] = observed;
    status[ST_EXPECTED] = expected;
    status[ST_FINAL_COUNTER] = *counter;
    status[ST_ROUNDS_SENT] = sent;
    status[ST_ROUNDS_DONE] = r;
    noc.async_write(cb_obj, out, ST_WORDS * sizeof(uint32_t), {.offset_bytes = 0}, {.page_id = core_index});
    noc.async_write_barrier();
}
