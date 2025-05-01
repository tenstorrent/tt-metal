// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "worker.hpp"

#include <string>

#include "core/distributed/distributed.hpp"
#include "core/distributed/mpi_context.hpp"
#include "ops/losses.hpp"
namespace roles {

RemoteOptimizer::RemoteOptimizer(ttml::serialization::NamedParameters parameters, int aggregator_rank) :
    OptimizerBase(std::move(parameters)), m_aggregator_rank(aggregator_rank) {
    m_sorted_parameters = get_sorted_parameters();
}

void RemoteOptimizer::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(ttml::core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void RemoteOptimizer::step() {
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
    this->m_steps = steps;
}

SortedParameters RemoteOptimizer::get_sorted_parameters() const {
    return {m_parameters.begin(), m_parameters.end()};
}

void RemoteOptimizer::send_gradients() {
    auto& ctx = ttml::autograd::ctx();
    auto& mpi_ctx = ctx.get_mpi_context();
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            auto grad = tensor_ptr->get_grad();
            fmt::print("Rank {}: sending gradient for parameter {}\n", mpi_ctx.get_rank(), name);
            ttml::core::distributed::send_tensor(grad, m_aggregator_rank);
        }
    }
}

void RemoteOptimizer::receive_weights() {
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        auto tensor = tensor_ptr->get_value();
        ttml::core::distributed::recv_tensor(tensor, m_aggregator_rank);
    }
}

Worker::Worker(DataLoader train_dataloader, std::shared_ptr<ttml::modules::LinearLayer> model, int aggregator_rank) :
    m_train_dataloader(std::move(train_dataloader)), m_model(std::move(model)), m_aggregator_rank(aggregator_rank) {
    m_optimizer = std::make_shared<RemoteOptimizer>(m_model->parameters(), m_aggregator_rank);
}
void Worker::training_step() {
    auto& ctx = ttml::autograd::ctx();
    auto& mpi_ctx = ctx.get_mpi_context();
    auto& device = ctx.get_device();

    m_model->train();

    for (auto [features, target] : m_train_dataloader) {
        m_optimizer->zero_grad();
        auto output = (*m_model)(features);
        auto loss = ttml::ops::mse_loss(output, target);
        fmt::print("Rank {}: Loss: {}\n", mpi_ctx.get_rank(), ttml::core::to_vector(loss->get_value())[0]);
        loss->backward();
        m_optimizer->step();
        ttml::autograd::ctx().reset_graph();
        m_training_step++;
    }
}

}  // namespace roles
