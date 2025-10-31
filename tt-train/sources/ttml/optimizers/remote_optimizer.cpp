// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    m_aggregator_rank = core::distributed::Rank{aggregator_rank};
    m_sorted_parameters = SortedParameters(m_parameters.begin(), m_parameters.end());

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
    return dict;
}

void RemoteOptimizer::set_state_dict(const serialization::StateDict& dict) {
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
        auto tensor = tensor_ptr->get_value();
        tensor = socket_manager.recv(tensor, m_distributed_ctx, m_aggregator_rank);
        tensor_ptr->set_value(tensor);
    }
}

void RemoteOptimizer::set_lr(float lr) {
    // No-op: learning rate is managed by the remote optimizer
}

float RemoteOptimizer::get_lr() const {
    // Remote optimizer doesn't store learning rate locally
    return 0.F;
}

}  // namespace ttml::optimizers
