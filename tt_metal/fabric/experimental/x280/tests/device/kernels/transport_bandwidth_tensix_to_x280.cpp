// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// transport_bandwidth_tensix_to_x280.cpp -- Tensix -> X280 addressable
// bandwidth kernel, and with TRANSPORT_BW_SENDER_IDLE_ERISC its idle-ERISC
// variant.
//
// The Metal core is the initiator; X280 is a passive addressable target -- no
// X280 image is submitted for this suite. The kernel is a thin entry point: it
// reads its runtime args via the Metal backend under
// kernels/transport_bandwidth/backends/tensix_to_x280.hpp, then hands the
// state to the shared frontend at kernels/shared/. The state machine,
// warm-up discipline, timed loop, boundary specialisation, and result
// publication all live in the shared frontend, so this member shares one
// implementation with the X280 -> Tensix half of the suite.
//
// See docs/benchmarks/design.md § Transport bandwidth.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Relative includes: an EthernetConfig kernel gets no include paths.
#include "../../../kernels/transport_bandwidth/backends/tensix_to_x280.hpp"
#include "../../../kernels/transport_bandwidth/tensix_to_x280.h"
#include "../../../kernels/transport_bandwidth/transport_bandwidth_common.h"
#include "../../../kernels/transport_bandwidth/transport_bandwidth_frontend_cxx.hpp"

void kernel_main() {
    transport_bandwidth::backend_tensix_to_x280::state_t state{};
    if (!transport_bandwidth::backend_tensix_to_x280::initialize(state)) {
        // initialize() writes nothing on refusal, so the host observes a
        // zeroed result region and decodes DONE-not-set as
        // ARG_LAYOUT_MISMATCH.
        return;
    }

    transport_bandwidth::run_member<transport_bandwidth::backend_tensix_to_x280::Backend>(state);
}
