// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "remote_optimizer.hpp"

#include "common.hpp"

RemoteOptimizer::RemoteOptimizer(ttml::serialization::NamedParameters parameters, int aggregator_rank) :
    ttml::optimizers::OptimizerBase(std::move(parameters)) {
    m_aggregator_rank = ttml::core::distributed::Rank{aggregator_rank};
    m_sorted_parameters = SortedParameters(m_parameters.begin(), m_parameters.end());

    auto workers_and_aggregator_ranks =
        three_tier_arch::get_workers_and_aggregator_ranks(static_cast<uint32_t>(*m_aggregator_rank));
    m_distributed_ctx =
        ttml::autograd::ctx().get_distributed_context().create_sub_context(workers_and_aggregator_ranks);
}
void RemoteOptimizer::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            // i don't see a reason why not to set it to empty
            tensor_ptr->set_grad(ttnn::Tensor());
        }
    }
}
void RemoteOptimizer::step() {
    m_steps++;
    send_gradients();
    receive_weights();
}
ttml::serialization::StateDict RemoteOptimizer::get_state_dict() const {
    ttml::serialization::StateDict dict;
    dict["steps"] = m_steps;
    return dict;
}
void RemoteOptimizer::set_state_dict(const ttml::serialization::StateDict& dict) {
    m_steps = ttml::serialization::get_value_type<size_t>(dict, "steps");
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
    auto& ctx = ttml::autograd::ctx();
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            auto grad = tensor_ptr->get_grad();
            ttml::core::distributed::send_tensor(*m_distributed_ctx, grad, m_aggregator_rank);
        }
    }
}
void RemoteOptimizer::receive_weights() {
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        auto tensor = tensor_ptr->get_value();
        ttml::core::distributed::broadcast_tensor(*m_distributed_ctx, tensor, m_aggregator_rank);
        tensor_ptr->set_value(tensor);
    }
}
void RemoteOptimizer::set_lr(float lr) {
}
float RemoteOptimizer::get_lr() const {
    return 0.F;
}
