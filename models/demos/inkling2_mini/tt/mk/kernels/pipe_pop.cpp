// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Drops one slab of a tensor prefetcher pipe (tt/mk/pipe_stage.py) on a ring core that a linear does not use, so
// the slabs behind it can stream. Runtime args: pipe config page address.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/prefetcher_pipe.h"

constexpr uint32_t ENTRY = get_named_compile_time_arg_val("entry");
constexpr uint32_t N_ENTRIES = get_named_compile_time_arg_val("n_entries");

void kernel_main() {
    experimental::PrefetcherPipe pipe(experimental::ReceiverConfigPage{get_arg_val<uint32_t>(0)}, ENTRY);
    pipe.wait_front(N_ENTRIES);
    pipe.pop_front(N_ENTRIES);
}
