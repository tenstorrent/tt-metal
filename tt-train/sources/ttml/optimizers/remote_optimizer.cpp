// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "remote_optimizer.hpp"

#include <numeric>
#include <vector>

namespace ttml::optimizers {

namespace {
// Helper function to create a vector of ranks from 0 to num_workers (inclusive)
std::vector<int> get_workers_and_aggregator_ranks(uint32_t num_workers) {
    std::vector<int> ranks(num_workers + 1U);
    std::iota(ranks.begin(), ranks.end(), 0);
    return ranks;
}
}  // namespace

RemoteOptimizer::RemoteOptimizer(serialization::NamedParameters parameters, int aggregator_rank) :
    OptimizerBase(std::move(parameters)) {
    require_bf16_parameters(m_parameters, "RemoteOptimizer");
    m_aggregator_rank = core::distributed::Rank{aggregator_rank};
    m_sorted_parameters = SortedParameters(m_parameters.begin(), m_parameters.end());
    init_param_groups({{"lr", 0.0F}}, {});

    auto workers_and_aggregator_ranks = get_workers_and_aggregator_ranks(static_cast<uint32_t>(*m_aggregator_rank));
    m_distributed_ctx = autograd::ctx().get_distributed_context()->create_sub_context(workers_and_aggregator_ranks);
}

void RemoteOptimizer::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            // Set gradient to empty tensor
            tensor_ptr->set_grad(ttnn::Tensor());
        }
    }
}

void RemoteOptimizer::step() {
    m_steps++;
    send_gradients();
    receive_weights();
}

serialization::StateDict RemoteOptimizer::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    save_param_groups(dict);
    return dict;
}

void RemoteOptimizer::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
}

size_t RemoteOptimizer::get_steps() const {
    return m_steps;
}

void RemoteOptimizer::set_steps(size_t steps) {
    m_steps = steps;
}

SortedParameters RemoteOptimizer::get_sorted_parameters() const {
    return m_sorted_parameters;
}

void RemoteOptimizer::send_gradients() {
    auto& socket_manager = autograd::ctx().get_socket_manager();
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            auto grad = tensor_ptr->get_grad();
            socket_manager.send(grad, m_distributed_ctx, m_aggregator_rank);
        }
    }
}

void RemoteOptimizer::receive_weights() {
    auto& socket_manager = autograd::ctx().get_socket_manager();
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        // Frozen fp32 tensors, which the bf16 check skips, keep their dtype too.
        socket_manager.recv_value(*tensor_ptr, m_distributed_ctx, m_aggregator_rank);
    }
}

}  // namespace ttml::optimizers
