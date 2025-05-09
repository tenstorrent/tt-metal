// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed/mpi_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"
namespace ttml::core::distributed {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor) {
    auto* device = &autograd::ctx().get_device();
    auto devices_count = device->get_devices().size();
    assert(devices_count >= 1U);
    // no need to synchronize if there is only one device
    if (devices_count == 1U) {
        return tensor;
    }

    // all_reduce Mean is not supported, use sum and divide by #devices
    auto result = ttnn_fixed::distributed::all_reduce(tensor);
    result = ttnn::multiply(result, 1.0F / static_cast<float>(devices_count));
    return result;
}

void synchronize_parameters(const serialization::NamedParameters& parameters) {
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(synchronize_tensor(tensor->get_grad()));
        }
    }
}

void send_tensor(const ttnn::Tensor& tensor, int dest, int tag) {
    auto* device = &autograd::ctx().get_device();
    auto& mpi_context = autograd::ctx().get_mpi_context();
    auto devices_count = device->get_devices().size();

    auto cpu_tensor = tensor.cpu();
    auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);
    for (auto buffer : buffers) {
        mpi_context.send(buffer, dest, tag);
    }
}

void recv_tensor(ttnn::Tensor& tensor, int source, int tag) {
    auto* device = &autograd::ctx().get_device();
    auto& mpi_context = autograd::ctx().get_mpi_context();

    auto cpu_tensor = tensor.cpu();

    auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);
    for (auto buffer : buffers) {
        mpi_context.recv(buffer, source, tag);
    }

    ttnn::assign(cpu_tensor.to_device(tensor.device()), tensor);
}
void broadcast_tensor(ttnn::Tensor& tensor, int root) {
    auto* device = &autograd::ctx().get_device();
    auto& mpi_context = autograd::ctx().get_mpi_context();

    auto cpu_tensor = tensor.cpu();

    auto buffers = ttml::core::get_bytes_from_cpu_tensor(cpu_tensor);

    for (auto buffer : buffers) {
        mpi_context.broadcast(buffer, root);
    }
    if (mpi_context.get_rank() != root) {
        ttnn::assign(cpu_tensor.to_device(tensor.device()), tensor);
    }
}

// @dmakoviichuk TODO: optimize it using split and broadcast
void broadcast_tensor_to_group(ttnn::Tensor& tensor, int root, std::span<int> client_ranks) {
    auto* device = &autograd::ctx().get_device();
    auto& mpi_context = autograd::ctx().get_mpi_context();
    int rank = mpi_context.get_rank();

    if (rank == root) {
        for (int client : client_ranks) {
            send_tensor(tensor, client);
        }
    } else if (std::find(client_ranks.begin(), client_ranks.end(), rank) != client_ranks.end()) {
        recv_tensor(tensor, root);
    }
}

// @dmakoviichuk TODO:
// optimize this code
void reduce_tensor(ttnn::Tensor& tensor, std::span<int> client_ranks) {
    bool is_first = true;
    ttnn::Tensor temp = ttnn::empty_like(tensor);
    for (int rank : client_ranks) {
        if (is_first) {
            // First client: receive directly into `tensor`
            recv_tensor(tensor, rank);
            is_first = false;
        } else {
            recv_tensor(temp, rank);

            // Accumulate into the output tensor
            tensor = ttnn::add(tensor, temp);
        }
    }
}

}  // namespace ttml::core::distributed
