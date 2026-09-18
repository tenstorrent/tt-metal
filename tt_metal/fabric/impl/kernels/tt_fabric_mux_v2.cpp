// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_forwarder.hpp"
#include "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_manager.hpp"

void kernel_main() {
    set_l1_data_cache<false>();

#ifdef COMPILE_FOR_NCRISC
    tt::tt_fabric::mux_v2::kernel::run_manager();
#else
    auto context = tt::tt_fabric::mux_v2::kernel::make_forwarder_context();
    tt::tt_fabric::mux_v2::kernel::run_forwarder(context);
#endif
}
