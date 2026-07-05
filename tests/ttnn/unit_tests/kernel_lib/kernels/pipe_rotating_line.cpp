// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// mcast_pipe + mcast_host END-TO-END rotating-LINE test kernel.
//
// Every core on a 1D line runs this ONE kernel and plays BOTH faces of the channel over `span`
// rounds, decoding the host::Mcast1D(rotating) wire with McastArgs<CT=1, RT=5, SPAN=span>.
//
// On round r == my_index this core is the SENDER; on every other round it RECEIVES the shard the
// round-r sender broadcasts and records it to DRAM. This is the 1D mirror of the block-sharded
// matmul in0 reader.
//
// The dest rect is the FULL line (it includes the sender), so the sender does an IN-PLACE mcast: it
// stages its own shard into cb (the landing region the receivers also use) and calls send(dst, dst)
// with src == dst, which the pipe resolves to a plain EXCLUDE_SRC broadcast (no loopback) to the
// other span-1 cores.
#include <stdint.h>
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
    // `span` is the first scalar after the mcast CT block; read it via a SPAN-less view of the args,
    // then form the SPAN-typed McastArgs (rotating).
    constexpr uint32_t SCALARS = McastArgs</*CT=*/1, /*RT=*/5>::next_compile_time_args_offset();  // = 6
    constexpr uint32_t span = get_compile_time_arg_val(SCALARS + 0);
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/5, span>();
    constexpr uint32_t payload_pages = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto in_args = TensorAccessorArgs<SCALARS + 3>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_start_id = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_start_id = get_arg_val<uint32_t>(3);  // first DRAM slot this core writes
    const uint32_t my_index = get_arg_val<uint32_t>(4);         // this core's axis position == its sender round

    constexpr uint32_t payload_bytes = payload_pages * page_bytes;

    Noc noc;
    CircularBuffer cb_obj(cb);
    const auto in = TensorAccessor(in_args, input_addr);
    const auto out = TensorAccessor(out_args, output_addr);

    cb_obj.reserve_back(payload_pages);  // fixed scratch region; write_ptr == base == the mcast address
    const uint32_t cb_addr = cb_obj.get_write_ptr();

    // Both faces built ONCE and reused every round: on its own round the core sends, on the others it
    // receives — over the SAME data_ready cell. (Rotating, so the pipe resets that cell to INVALID after
    // each broadcast; without that the next receive would return on this core's own stale VALID.)
    auto send_pipe = mc.sender(noc);
    auto recv_pipe = mc.receiver(noc);

    for (uint32_t r = 0; r < span; ++r) {
        if (r == my_index) {
            // SENDER: stage my shard into cb, then broadcast it IN PLACE (src == dst => EXCLUDE_SRC) to
            // the other span-1 cores on the line.
            for (uint32_t i = 0; i < payload_pages; ++i) {
                noc.async_read(
                    in, cb_obj, page_bytes, {.page_id = input_start_id + i}, {.offset_bytes = i * page_bytes});
            }
            noc.async_read_barrier();
            send_pipe.send(cb_addr, cb_addr, payload_bytes);
        } else {
            // RECEIVER: the shard the round-r sender broadcasts lands in cb.
            recv_pipe.receive(r);
        }
        // record this round's shard (mine when sending, the sender's when receiving) to DRAM, before
        // the next round overwrites the landing buffer.
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
