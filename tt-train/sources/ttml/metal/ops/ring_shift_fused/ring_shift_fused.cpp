// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_shift_fused.hpp"

#include "device/ring_shift_fused_device_operation.hpp"

namespace ttml::metal {

std::vector<ttnn::Tensor> ring_shift_fused(
    const std::vector<ttnn::Tensor>& inputs,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& send_sockets,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& recv_sockets,
    const std::vector<ttnn::Tensor>& preallocated_outputs) {
    return ttnn::prim::ttml_ring_shift_fused(inputs, send_sockets, recv_sockets, preallocated_outputs);
}

}  // namespace ttml::metal
