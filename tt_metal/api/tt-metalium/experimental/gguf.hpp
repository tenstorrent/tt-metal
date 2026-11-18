// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::experimental {

// Experimental GGUF host API; subject to change.

// A GGUF tensor of `spec` (a GGUF dtype, shape [K, N]) from its blocks as a GGUF file stores them: N rows of
// K / block_elems blocks. Only bits move; the pages take exactly the blocks' bytes.
HostTensor from_gguf(ttsl::Span<const std::byte> blocks, const TensorSpec& spec);

// The blocks of a GGUF tensor, the exact inverse of from_gguf.
std::vector<std::byte> to_gguf(const HostTensor& tensor);

}  // namespace tt::tt_metal::experimental
