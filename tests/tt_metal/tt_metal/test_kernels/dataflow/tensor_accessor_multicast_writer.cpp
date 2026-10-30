// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multicasts this node's shard of the bound `source` tensor into the bound `target` tensor, so Watcher checks the
// rectangle against `target`'s nodes.

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    LocalTensorAccessor<uint32_t> source(tensor::source);
    const auto target = TensorAccessor(tensor::target);

    Noc noc;
    noc.async_write_multicast(
        source,
        target,
        get_arg(args::size_bytes),
        get_arg(args::num_targets),
        {},
        {
            .noc_x_start = get_arg(args::noc_x_start),
            .noc_y_start = get_arg(args::noc_y_start),
            .noc_x_end = get_arg(args::noc_x_end),
            .noc_y_end = get_arg(args::noc_y_end),
        });
    noc.async_write_barrier();
}
