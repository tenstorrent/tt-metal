// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn::generic_op (ttnn/cpp/ttnn/operations/generic/generic_op.hpp): exported by TT-NN's library, but its header is
// not among TT-NN's installed headers, so its declaration is repeated here.
#pragma once
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn {
Tensor generic_op(const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor);
}  // namespace ttnn
