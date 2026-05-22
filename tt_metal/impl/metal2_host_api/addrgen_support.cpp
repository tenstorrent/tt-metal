// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/addrgen_support.hpp>

#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

namespace {

bool current_arch_is_quasar() { return tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR; }

}  // namespace

AddrgenSupport addrgen_support_for(const tt::tt_metal::TensorSpec& tensor_spec) {
    if (!current_arch_is_quasar()) {
        return AddrgenSupport::kSkipNonQuasar;
    }

    const auto& memory_config = tensor_spec.memory_config();
    if (!memory_config.is_sharded()) {
        return AddrgenSupport::kSupported;
    }

    return AddrgenSupport::kSkipSharded;
}

const char* describe_skip_reason(AddrgenSupport support) {
    switch (support) {
        case AddrgenSupport::kSupported: return nullptr;
        case AddrgenSupport::kSkipNonQuasar:
            return "HW addrgen is Quasar-only (current architecture is not Gen2).";
        case AddrgenSupport::kSkipSharded:
            return "Sharded tensors with HW addrgen are deferred (Phase 1 supports interleaved only).";
    }
    return "Unknown AddrgenSupport value.";
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
