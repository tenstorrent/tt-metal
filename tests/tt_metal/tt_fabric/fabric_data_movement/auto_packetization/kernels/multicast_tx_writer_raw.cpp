// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile-probe kernel for auto-packetization validation.
// Including these headers exercises the mesh and linear raw-size API declarations
// through the device toolchain. The kernel_main body is minimal — the purpose is
// to verify that API renames and wrapper additions in plans 02-06 compile correctly.

#include <cstdint>
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

void kernel_main() {
    // TODO: Will call the renamed _single_packet or auto-packetizing wrapper
    // once Plans 02-05 are done. For now this is a compile-only stub that
    // validates the API headers are syntactically correct with the device toolchain.
    //
    // Placeholder references to ensure the mesh namespace symbols are visible:
    // tt::tt_fabric::mesh::experimental::fabric_multicast_noc_unicast_write(...)
    (void)0;
}
