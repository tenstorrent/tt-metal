// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Compile-time handle for a PrefetcherPipe accessor (KernelAdvancedOptions::PrefetcherPipeBinding),
// emitted into kernel_bindings_generated.h in the `pipe::` namespace. Carries the program's
// PrefetcherPipe slot id; the slot resolves, on the executing node, to whichever pipe of the
// accessor's group is present there (the host fills every node's slot record from that pipe).
//
// Usage example:
//   // (Host code declares "weights" as the PrefetcherPipe accessor name for this kernel.)
//   experimental::PrefetcherPipe weights(pipe::weights);
//
// This header holds only the token, with no dependency beyond <cstdint>, so the generated
// bindings header does not have to pull in api/dataflow/prefetcher_pipe.h; the kernel includes
// that itself to construct the PrefetcherPipe.
struct PrefetcherPipeBindingToken {
    explicit constexpr PrefetcherPipeBindingToken(uint8_t prefetcher_pipe_id) noexcept :
        prefetcher_pipe_id_(prefetcher_pipe_id) {}

    constexpr uint8_t prefetcher_pipe_id() const noexcept { return prefetcher_pipe_id_; }

private:
    uint8_t prefetcher_pipe_id_;
};
