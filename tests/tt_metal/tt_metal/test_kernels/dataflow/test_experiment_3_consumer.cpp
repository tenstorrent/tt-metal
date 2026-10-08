// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const std::uint32_t tiles_per_neo = get_arg(args::tiles_per_neo);  // rt arg per neo cluster
    const std::uint32_t start_page = get_arg(args::start_page);

    const std::uint32_t num_neo_dm = get_num_threads();
    const std::uint32_t my_neo_id = get_my_thread_id();
    const std::uint32_t tile_count = tiles_per_neo / num_neo_dm + (my_neo_id < (tiles_per_neo % num_neo_dm) ? 1 : 0);

    const auto tensor_accessor = TensorAccessor(tensor::dst);

    DataflowBuffer dfb_out(dfb::out);
    Noc noc;

    std::uint32_t page_id = start_page + my_neo_id;
    for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
        noc.async_write<NocOptions::TXN_ID>(dfb_out, tensor_accessor, {}, {.page_id = page_id});

        page_id += num_neo_dm;
    }

    dfb_out.finish();
    dfb_out.write_barrier(noc);
}
