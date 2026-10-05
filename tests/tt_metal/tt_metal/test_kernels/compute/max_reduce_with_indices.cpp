// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Drives max_reduce_with_indices the way the pool compute_mpwi kernels do: a values tile and an
// indices tile of a different format are copied into Dest, reduced in place into their row 0, and
// packed back out. With MPWI_ACCUMULATE, num_chunks values/indices tile pairs are folded under a
// single Dest acquire, so the running max carried in Dest tiles 1 and 3 is exercised as well.
//
// Dest layout follows compute_mpwi: values = 0, running values = 1, indices = 2, running indices = 3.
void kernel_main() {
    constexpr std::uint32_t num_chunks = get_arg(args::num_chunks);

    constexpr std::uint32_t values_dst = 0;
    constexpr std::uint32_t indices_dst = 2;
    // Each output carries its operand tile and the running tile above it: the running pair is part
    // of the result when accumulating, and the host ignores it otherwise.
    constexpr std::uint32_t out_tiles = 2;

    DataflowBuffer values_in(dfb::values_in);
    DataflowBuffer indices_in(dfb::indices_in);
    DataflowBuffer values_out(dfb::values_out);
    DataflowBuffer indices_out(dfb::indices_out);

    compute_kernel_hw_startup(values_in.get_id(), values_out.get_id());
    copy_init(values_in.get_id());
    max_reduce_with_indices_init<MPWI_LAYOUT>();

    values_in.wait_front(num_chunks);
    indices_in.wait_front(num_chunks);
    values_out.reserve_back(out_tiles);
    indices_out.reserve_back(out_tiles);

    tile_regs_acquire();
    for (std::uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // A copy_init per operand: on Quasar it also reprograms the unpack buffer descriptor, which
        // reconfig_data_format_srca leaves alone.
        reconfig_data_format_srca(indices_in.get_id());
        copy_init(indices_in.get_id());
        copy_tile(indices_in.get_id(), chunk, indices_dst);

        reconfig_data_format_srca(values_in.get_id());
        copy_init(values_in.get_id());
        copy_tile(values_in.get_id(), chunk, values_dst);

        max_reduce_with_indices<MPWI_NUM_ROWS, MPWI_LAYOUT, MPWI_ACCUMULATE>(values_dst, indices_dst, chunk);
    }
    tile_regs_commit();
    tile_regs_wait();

    // Likewise a pack_init per output: on Quasar, pack_reconfig_data_format alone leaves the packer
    // writing to the previous output's buffer.
    pack_reconfig_data_format(values_out.get_id());
    pack_init(values_out.get_id());
    pack_tile(values_dst, values_out.get_id());
    pack_tile(values_dst + 1, values_out.get_id());

    pack_reconfig_data_format(indices_out.get_id());
    pack_init(indices_out.get_id());
    pack_tile(indices_dst, indices_out.get_id());
    pack_tile(indices_dst + 1, indices_out.get_id());

    tile_regs_release();

    values_out.push_back(out_tiles);
    indices_out.push_back(out_tiles);
    values_in.pop_front(num_chunks);
    indices_in.pop_front(num_chunks);
}
