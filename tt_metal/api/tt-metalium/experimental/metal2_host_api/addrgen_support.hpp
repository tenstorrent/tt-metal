// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared support / capability predicates for TensorAccessor hardware addrgen.
// Used by both program_spec.cpp (host validation) and gtest fixtures so the
// "what's supported" decision lives in exactly one place.
//
// See tt_metal/hw/inc/api/tensor/tensor_accessor_addrgen.h for the device-side
// configuration each support category maps to.

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// Why the HW addrgen path may not be usable for a given tensor layout.
// kSupported means addrgen_mode != NONE is fine for this tensor.
enum class AddrgenSupport {
    kSupported,
    kSkipNonQuasar,       // Hardware addrgen is Quasar-only.
    kSkipSharded,         // Sharded tensors deferred until a later addrgen phase.
};

// Returns kSupported iff the tensor's memory layout is compatible with HW addrgen on the
// current build. Includes the architecture check so callers don't need a separate guard.
AddrgenSupport addrgen_support_for(const tt::tt_metal::TensorSpec& tensor_spec);

// Human-readable explanation for a non-supported AddrgenSupport value.
// Returns nullptr for kSupported.
const char* describe_skip_reason(AddrgenSupport support);

}  // namespace tt::tt_metal::experimental::metal2_host_api
