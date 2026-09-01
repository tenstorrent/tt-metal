// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) BLOCKED DFB producer.
// Parallel to dfb_producer_2_0.cpp, but moves block_size contiguous entries per NoC
// transaction and posts their credits together, then strides by block_size * num_producers.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);

    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    const uint32_t num_blocks = num_entries_per_producer / block_size;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        // This thread's b-th block: block_size contiguous pages, blocks interleaved across producers.
        const uint32_t block_base_page = chunk_offset + (b * num_producers + producer_idx) * block_size;
        if (block_base_page >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_read<NocOptions::TXN_ID>(
                tensor_accessor, dfb, {.page_id = block_base_page}, {.num_tiles = block_size});
#endif
        } else {
            dfb.reserve_back(block_size);
            noc.async_read(tensor_accessor, dfb, block_size * entry_size, {.page_id = block_base_page}, {});
            noc.async_read_barrier();
            dfb.push_back(block_size);
        }
    }
    dfb.finish();
}
