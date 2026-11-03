// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

// Reads one data tile (src_tensor -> out_data) and one scaler tile (scaler_tensor -> out_scaler) from DRAM.
// Unlike the readers that build a bf16 scaler tile in L1, this lets the host supply the scaler in any
// format (e.g. MxFp4), since the tile is copied as-is.
void kernel_main() {
    constexpr std::uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_data(dfb::out_data);
    DataflowBuffer dfb_scaler(dfb::out_scaler);
    const auto src = TensorAccessor(tensor::src_tensor);
    const auto scaler = TensorAccessor(tensor::scaler_tensor);

    dfb_scaler.reserve_back(onetile);
    noc.async_read(scaler, dfb_scaler, dfb_scaler.get_entry_size(), {.page_id = 0}, {});
    dfb_data.reserve_back(onetile);
    noc.async_read(src, dfb_data, dfb_data.get_entry_size(), {.page_id = 0}, {});
    noc.async_read_barrier();
    dfb_scaler.push_back(onetile);
    dfb_data.push_back(onetile);
}
