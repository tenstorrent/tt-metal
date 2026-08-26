// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe + mcast_host END-TO-END rotating-LINE test kernel.
//
// Every core on a 1D line runs this ONE kernel and plays BOTH faces of the channel over
// `num_rounds`, decoding the host::Mcast1D(rotating) wire with McastArgs<CT=1, RT=4>.
//
// Sender selection cycles every mc.num_senders rounds. Receiver-capable cores receive every other
// round; an independent sender outside that rectangle stays sender-only. This is the 1D mirror of
// the block-sharded matmul in0 reader.
//
// The dest rect is the FULL line (it includes the sender), so the sender does an IN-PLACE mcast: it
// stages its own shard into cb (the landing region the receivers also use) and calls send(dst, dst)
// with src == dst, which the pipe resolves to a plain EXCLUDE_SRC broadcast (no loopback) to the
// other span-1 cores.
#include <stdint.h>
#include <optional>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);  // mcast + landing region (one per core)
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/4>();
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();  // = 8
    constexpr uint32_t num_rounds = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t payload_pages = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto in_args = TensorAccessorArgs<SCALARS + 3>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_start_id = get_arg_val<uint32_t>(3);  // first DRAM slot this core writes

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    CircularBuffer cb_obj(cb);
    const auto in = TensorAccessor(in_args, input_addr);
    const auto out = TensorAccessor(out_args, output_addr);

    cb_obj.reserve_back(payload_pages);  // fixed scratch region; write_ptr == base == the mcast address
    const uint32_t cb_addr = cb_obj.get_write_ptr();

    const bool can_send = mc.can_send();
    const bool can_receive = mc.can_receive();

    // Both faces built ONCE and reused every round: on its own round the core sends, on the others it
    // receives — over the SAME data_ready cell. (Rotating, so the pipe resets that cell to INVALID after
    // each broadcast; without that the next receive would return on this core's own stale VALID.) The
    // faces are optional because a core may hold only one role, and the pipes are not default-constructible.
    using SendPipe = decltype(mc.sender(noc));
    using ReceivePipe = decltype(mc.receiver(noc));
    std::optional<SendPipe> send_pipe;
    std::optional<ReceivePipe> recv_pipe;
    if (can_send) {
        send_pipe.emplace(mc.sender(noc));
    }
    if (can_receive) {
        recv_pipe.emplace(mc.receiver(noc));
    }

    for (uint32_t r = 0; r < num_rounds; ++r) {
        if (mc.should_send(r)) {
            // SENDER: stage my shard into cb, then broadcast it IN PLACE (src == dst => EXCLUDE_SRC) to
            // the other cores on the line.
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_read(
                    in, cb_obj, page_bytes, {.page_id = input_start_id + i}, {.offset_bytes = i * page_bytes});
            }
            noc.async_read_barrier();
            send_pipe->send(cb_addr, cb_addr, payload_bytes);
        } else if (can_receive) {
            // RECEIVER: the shard the round-r sender broadcasts lands in cb.
            recv_pipe->receive(r);
        } else {
            continue;  // sender-only core outside its sender phase
        }
        for (uint32_t i = 0; i < payload_pages; ++i) {
            noc.async_write(
                cb_obj,
                out,
                page_bytes,
                {.offset_bytes = i * page_bytes},
                {.page_id = output_start_id + r * payload_pages + i});
        }
        noc.async_write_barrier();
    }
}
