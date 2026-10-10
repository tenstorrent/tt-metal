// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t payload_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = 2048;
    constexpr uint32_t payload_bytes = payload_pages * page_bytes;
    constexpr auto input_args = TensorAccessorArgs<1>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto mcast = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1));
    const uint32_t receiver_rank = get_arg_val<uint32_t>(2);

    Noc noc;
    CircularBuffer source(0);
    CircularBuffer destination(1);
    auto sender = mcast.optional_sender(noc);
    auto receiver = mcast.optional_receiver(noc);

    if (sender) {
        source.reserve_back(payload_pages);
        const uint32_t source_address = source.get_write_ptr();
        for (uint32_t page = 0; page < payload_pages; ++page) {
            noc.async_read(
                input, CoreLocalMem<uint32_t>(source_address + page * page_bytes), page_bytes, {.page_id = page}, {});
        }
        noc.async_read_barrier();

        const uint32_t destination_address = destination.get_write_ptr();
        sender->send<SourceL1Guard::CallerManaged>(source_address, destination_address, payload_bytes);
        noc.async_write_barrier();
    } else if (receiver) {
        destination.reserve_back(payload_pages);
        const uint32_t destination_address = destination.get_write_ptr();
        receiver->receive_and_forward<SourceL1Guard::CallerManaged>(destination_address, payload_bytes, 0);

        for (uint32_t page = 0; page < payload_pages; ++page) {
            noc.async_write(
                CoreLocalMem<uint32_t>(destination_address + page * page_bytes),
                output,
                page_bytes,
                {},
                {.page_id = receiver_rank * payload_pages + page});
        }
        noc.async_write_barrier();
    }
}
