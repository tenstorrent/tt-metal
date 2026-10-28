// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"

void kernel_main() {
    LocalTensorAccessor<uint32_t> report(tensor::report);

    Scratchpad<uint32_t> pad(scratch::pad);
    {
        auto lock = report.scoped_lock(0, 1);
        report[0] = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(pad.local_mem()));
    }
    {
        auto lock = pad.scoped_lock(0, pad.size());
        for (uint32_t i = 0; i < pad.size(); ++i) {
            pad[i] = get_arg(args::scratch_seed) + i;
        }
    }
}
