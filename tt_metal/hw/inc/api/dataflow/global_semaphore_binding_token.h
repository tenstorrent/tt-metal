// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Compile-time handle for a GlobalSemaphore accessor (KernelAdvancedOptions::GlobalSemaphoreBinding),
// emitted into kernel_bindings_generated.h in the `global_sem::` namespace. Carries the word index,
// within the kernel's common runtime args, of the bound GlobalSemaphore's address.
//
// Usage example:
//   // (Host code declares "signal" as the GlobalSemaphore accessor name for this kernel.)
//   experimental::GlobalSemaphore signal(global_sem::signal);
//
// This header holds only the token, so the generated bindings header does not have to pull in
// api/dataflow/global_semaphore.h; the kernel includes that itself to construct the GlobalSemaphore.
struct GlobalSemaphoreBindingToken {
    explicit constexpr GlobalSemaphoreBindingToken(uint32_t crta_word) noexcept : crta_word_(crta_word) {}

    constexpr uint32_t crta_word() const noexcept { return crta_word_; }

private:
    uint32_t crta_word_;
};
