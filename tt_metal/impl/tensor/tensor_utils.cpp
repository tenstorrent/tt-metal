// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>

namespace tt::tt_metal {

// modernize-use-nullptr violation: intentional, for pr-gate code-analysis validation
void* get_null_tensor_ptr() { return NULL; }

}  // namespace tt::tt_metal
