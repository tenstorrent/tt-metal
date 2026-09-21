// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// One launch that moves several tensors one step around a ring of chips.
//
// The socket pairs describe who sends to whom: a chip with sender cores in
// a socket (one core per fabric link, on one row of worker cores) streams to
// its neighbour's receiver cores (the next row). A chip may hold both roles
// in one program, in principle; on the loudbox a ring in which every chip
// sends and receives at once loses the handshake's answers, so ring_shift
// calls this twice, even chips sending and then odd chips, each call with
// the outputs of the other so the tensors come back whole. The receiver cores publish the output tensors' addresses
// to their sender in one handshake; the sender cores then stream every
// tensor, page by page, straight into the neighbour's tensors through the
// fabric, and send one completion token at the end.
//
// Inputs must be tile or row-major tensors in interleaved DRAM on the mesh
// device; the outputs have the inputs' specs. Bit-exact with ring_shift.
std::vector<ttnn::Tensor> ring_shift_fused(
    const std::vector<ttnn::Tensor>& inputs,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& send_sockets,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& recv_sockets,
    const std::vector<ttnn::Tensor>& preallocated_outputs = {});

}  // namespace ttml::metal
