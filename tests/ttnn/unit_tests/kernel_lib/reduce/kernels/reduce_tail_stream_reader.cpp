// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

// The test stores padded stream packets in resident CB 3. Copy them through the
// planner-sized input FIFO, exercising real producer/consumer synchronization
// and repeated wraparound. Runtime arguments select full or tail packet counts;
// both auxiliary recipes are prepared independently of that choice.
void kernel_main() {
    using Auxiliary = ttnn::kernel_lib::ReduceAuxiliaryArgs<1>;
    using Call = ttnn::kernel_lib::ReduceCallArgs<Auxiliary::next_compile_time_args_offset()>;
    static_assert(Call::has_tail_variant);
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();
    const auto runtime = Call::runtime_shape();
    const uint32_t rows = runtime.has_override() ? (runtime.height + 31) / 32 : Call::rows;
    const uint32_t columns = runtime.has_override() ? (runtime.width + 31) / 32 : Call::columns;
    const uint32_t batches = runtime.has_override() ? runtime.batches : Call::batches;
    constexpr uint32_t axis_chunk = Call::reduce_axis_chunk_tiles;
    constexpr uint32_t output_chunk = Call::output_chunk_tiles;
    constexpr uint32_t packet_tiles = axis_chunk * output_chunk;
    const uint32_t packets =
        Call::reduce_dim == ckernel::ReduceDim::REDUCE_ROW
            ? batches * rows * ((columns + axis_chunk - 1) / axis_chunk)
            : batches * ((columns + output_chunk - 1) / output_chunk) * ((rows + axis_chunk - 1) / axis_chunk);
    DataflowBuffer source(3);
    DataflowBuffer input(Call::input_cb_id);
    Noc noc;
    UnicastEndpoint self;
    const uint32_t packet_bytes = packet_tiles * get_tile_size(Call::input_cb_id);
    for (uint32_t packet = 0; packet < packets; ++packet) {
        input.reserve_back(packet_tiles);
        noc.async_read(
            self,
            input,
            packet_bytes,
            dataflow_kernel_lib::local_addr(source.get_read_ptr() + packet * packet_bytes, noc.get_noc_id()),
            {});
        noc.async_read_barrier();
        input.push_back(packet_tiles);
    }
}
