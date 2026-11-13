// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_forwarder.hpp"
#include "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_manager.hpp"

void kernel_main() {
    set_l1_data_cache<false>();
    if constexpr (tt::tt_fabric::mux_v2::ct_args::in_program_local_l1) {
        tt::tt_fabric::mux_v2::kernel::l1_base_address =
            get_write_ptr(tt::tt_fabric::mux_v2::ct_args::program_local_l1_cb_index);
    }

#ifdef COMPILE_FOR_NCRISC
    if constexpr (tt::tt_fabric::mux_v2::ct_args::in_program_local_l1) {
        // Publish the map's address before anything else: an eager client stages into its channel before READY, and
        // waits for this. The status word sits in a fresh circular buffer, so clear it first so that a stale value
        // cannot read as READY. Dispatch zeroes the semaphore at every launch.
        tt::tt_fabric::mux_v2::kernel::get_status_ptr()[0] = tt::tt_fabric::FabricMuxStatus::STARTED;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<ProgrammableCoreType::TENSIX>(
            tt::tt_fabric::mux_v2::ct_args::l1_base_semaphore_id)) = tt::tt_fabric::mux_v2::kernel::l1_base_address;
    }
    tt::tt_fabric::mux_v2::kernel::run_manager();
#else
    auto context = tt::tt_fabric::mux_v2::kernel::make_forwarder_context();
    tt::tt_fabric::mux_v2::kernel::add_l1_base_to_forwarder_context(context);
    tt::tt_fabric::mux_v2::kernel::run_forwarder(context);
#endif
}
