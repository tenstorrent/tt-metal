// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK stub phase body (data-movement RISCs): the placeholder every phase of the family skeleton runs until its piece
// registers the real body (tt/pmk_layer.py register_phase). Included by the generated composite inside the phase's
// namespace with `kernel_main` renamed to `phase_main` and the phase's argument bases PH_CT0 / ph_rt0 in scope — the
// same convention as the real bodies (kernels/mk/phase_args.hpp). It touches no CB and no tensor, so the stub
// program is launch-safe on a mock and on a chip (every hand-off wait in the main loop is on a counter nobody raises:
// the composite only emits waits for phases whose body is registered -- see composite_source).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // nothing: the real body reads its args at PH_CT0 + i / ph_rt0 + i
    (void)PH_CT0;
    (void)ph_rt0;
}
