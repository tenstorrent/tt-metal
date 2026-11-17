// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Compile-time handle for a fabric connection accessor (KernelAdvancedOptions::FabricConnectionBinding),
// emitted into kernel_bindings_generated.h in the `fabric::` namespace. Carries the index of the first word
// of the binding's section in the kernel's runtime args. On each node, the section holds:
//   [connected (0 or 1), router channel, teardown semaphore id, buffer index semaphore id]
//
// Usage example:
//   // (Host code declares "fwd" as the fabric connection accessor name for this kernel.)
//   if (tt::tt_fabric::has_fabric_connection(fabric::fwd)) {
//       auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::fwd);
//   }
//
// This header holds only the token, so the generated bindings header does not have to pull in the fabric
// headers; the kernel includes tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp itself.
struct FabricConnectionBindingToken {
    explicit constexpr FabricConnectionBindingToken(uint32_t rta_word) noexcept : rta_word_(rta_word) {}

    constexpr uint32_t rta_word() const noexcept { return rta_word_; }

private:
    uint32_t rta_word_;
};
