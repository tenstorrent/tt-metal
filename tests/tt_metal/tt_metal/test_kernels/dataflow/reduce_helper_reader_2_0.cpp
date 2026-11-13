// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

constexpr uint32_t auxiliary_tiles = get_arg(args::auxiliary_tiles);
constexpr auto aux0_type = static_cast<ttnn::kernel_lib::ReduceAuxiliaryTileType>(get_arg(args::aux0_type));
constexpr uint32_t aux0_valid = get_arg(args::aux0_valid);
constexpr uint32_t aux0_value = get_arg(args::aux0_value);
constexpr auto aux1_type = static_cast<ttnn::kernel_lib::ReduceAuxiliaryTileType>(get_arg(args::aux1_type));
constexpr uint32_t aux1_valid = get_arg(args::aux1_valid);
constexpr uint32_t aux1_value = get_arg(args::aux1_value);

static_assert(auxiliary_tiles == 1 || auxiliary_tiles == 2);

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg(args::src_addr);
    const uint32_t src_bank_id = get_arg(args::src_bank_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t dram_page_stride = get_arg(args::dram_page_stride);

    // reduce() waits on the auxiliary tiles before its first input tile.
    DataflowBuffer aux(dfb::aux);
    dataflow_kernel_lib::prepare_reduce_auxiliary_tile<dfb::aux, aux0_type, aux0_valid, aux0_value>(aux);
    if constexpr (auxiliary_tiles == 2) {
        dataflow_kernel_lib::prepare_reduce_auxiliary_tile<dfb::aux, aux1_type, aux1_valid, aux1_value>(aux);
    }

    AllocatorBank<AllocatorBankType::DRAM> src_dram;
    Noc noc;
    DataflowBuffer in(dfb::in);
    const uint32_t tile_bytes = in.get_entry_size();
    uint32_t addr = src_addr;
    for (uint32_t i = 0; i < num_tiles; ++i) {
#ifdef ARCH_QUASAR
        noc.async_read<NocOptions::TXN_ID>(src_dram, in, {.bank_id = src_bank_id, .addr = addr}, {});
#else
        in.reserve_back(1);
        noc.async_read(src_dram, in, tile_bytes, {.bank_id = src_bank_id, .addr = addr}, {});
        noc.async_read_barrier();
        in.push_back(1);
#endif
        addr += dram_page_stride;
    }
}
