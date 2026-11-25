// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental {

// EXPERIMENTAL: options on a CBDescriptor that are not part of the stable API and may change or be removed.
struct CBDescriptorOptions {
    // The CB is globally allocated, but its `buffer`/`tensor` is named by a later descriptor applied to the
    // program (for example, a program compiled before its buffers are allocated). The program can't launch
    // until a descriptor names the buffer.
    bool globally_allocated_bound_later = false;
};

}  // namespace tt::tt_metal::experimental
