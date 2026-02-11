// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// ScatterHeads micro-op
//
// Scatters data from few input cores to many output cores.
// Each output core reads its portion of data from one input core.
//
// Input: 8 cores, each with shard shape (8, 512) = 8 rows × 512 elements
// Output: 64 cores, each with shard shape (1, 512) = 1 row × 512 elements
// Each input core's 8 rows are scattered to 8 different output cores.
//
// - NCRISC: Performs the noc_async_read to fetch data from source core
// - BRISC: No-op
// - TRISC: No-op (dataflow-only operation)
//
// CB States:
//   NCRISC (Output cores):
//     - Reserves: dst_cb (dst_num_pages)
//     - Performs: noc_async_read from source core
//     - Pushes: dst_cb (dst_num_pages)
//   BRISC: No-op
//   TRISC: No-op
//
// ============================================================================
struct ScatterHeads {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC)
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC)
    struct ComputeCTArgs {};

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC - output cores): source info for reading
    struct ReaderArgs {
        uint32_t src_noc_x;        // NOC X coordinate of source core
        uint32_t src_noc_y;        // NOC Y coordinate of source core
        uint32_t src_addr;         // L1 address on source core to read from
        uint32_t src_row_offset;   // Which row (0-7) of the source shard to read
        uint32_t data_size_bytes;  // Size of data to read (1 row × 512 elements)
        uint32_t dst_cb;           // Destination CB index
        uint32_t dst_num_pages;    // Number of pages to push to CB
    };

    // Writer args (BRISC): used for input core CB setup
    struct WriterArgs {};

    // Compute args (TRISC) - not used for scatter (dataflow only)
    struct ComputeArgs {};

    // Note: For scatter, NCRISC=Reader (on output cores), BRISC=Writer (setup on input cores)
    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // IsOutputCore: compile-time flag for output cores that read data
    // ========================================================================
    template <bool IsOutputCore>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - Output cores read from input cores
            // ================================================================
            if constexpr (IsOutputCore) {
                // Reserve space in destination CB
                cb_reserve_back(args.dst_cb, args.dst_num_pages);

                uint32_t dst_addr = get_write_ptr(args.dst_cb);

                // Calculate source address with row offset
                uint32_t src_addr_with_offset = args.src_addr + (args.src_row_offset * args.data_size_bytes);
                uint64_t src_noc_addr = get_noc_addr(args.src_noc_x, args.src_noc_y, src_addr_with_offset);

                // Read data from source core
                noc_async_read(src_noc_addr, dst_addr, args.data_size_bytes);
                noc_async_read_barrier();

                // Push to destination CB
                cb_push_back(args.dst_cb, args.dst_num_pages);
            }
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer) - No-op
            // ================================================================
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - No-op (scatter is dataflow only)
            // ================================================================
#endif
        }
    };  // class Op

};  // struct ScatterHeads

}  // namespace deepseek_b1_ops
