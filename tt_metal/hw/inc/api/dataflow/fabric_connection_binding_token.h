// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// A codegen-emitted handle for a Metal 2.0 kernel's fabric connection binding (KernelSpec::FabricConnectionBinding).
// The user never constructs this type; they build a fabric sender or connection manager from the generated token:
//   auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::to_receiver);
// The token has one connection slot for each target of the bound FabricConnectionSpec, in target order.
class FabricConnectionBindingToken {
public:
    explicit constexpr FabricConnectionBindingToken(uint32_t rta_word, uint32_t num_targets) noexcept :
        rta_word_(rta_word), num_targets_(num_targets) {}

    // The index of the binding's first word in the kernel's runtime args.
    constexpr uint32_t rta_word() const noexcept { return rta_word_; }

    // The number of connection slots: one for each target of the bound FabricConnectionSpec.
    constexpr uint32_t num_targets() const noexcept { return num_targets_; }

private:
    uint32_t rta_word_;
    uint32_t num_targets_;
};
