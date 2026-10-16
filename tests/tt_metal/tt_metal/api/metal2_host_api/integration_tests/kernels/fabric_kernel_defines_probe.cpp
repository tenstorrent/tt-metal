// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: reports what the fabric kernel defines expand to in this kernel's compilation.
//
// The host passes FABRIC_DEFINE_PROBES = PROBE(<name>)PROBE(<name>)... with one PROBE per define in
// ControlPlane::get_fabric_kernel_defines(), so this kernel names no fabric define itself. Each PROBE
// writes the stringified expansion of <name> into its own fixed-size slot at a host-known L1 address
// (a named RTA): the define's value when the build forwarded it, or the bare name when it did not.
// The host compares every slot against the control plane's value.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#ifndef FABRIC_DEFINE_PROBES
#error "The host must define FABRIC_DEFINE_PROBES as PROBE(<name>) for each fabric kernel define"
#endif

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

constexpr uint32_t kSlotBytes = 64;  // must match the host

void write_slot(volatile tt_l1_ptr char* slot, const char* text) {
    uint32_t i = 0;
    for (; i + 1 < kSlotBytes && text[i] != '\0'; i++) {
        slot[i] = text[i];
    }
    slot[i] = '\0';
}

void kernel_main() {
    auto* report = reinterpret_cast<volatile tt_l1_ptr char*>(get_arg(args::report_addr));
    uint32_t slot = 0;
    // `name` is macro-expanded before STRINGIFY sees it, so a forwarded define reads back as its value.
#define PROBE(name) write_slot(report + (slot++) * kSlotBytes, STRINGIFY(name));
    FABRIC_DEFINE_PROBES
#undef PROBE
}
