// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "worker.hpp"

#include "ops/losses.hpp"
namespace roles {

RemoteOptimizer::RemoteOptimizer(ttml::serialization::NamedParameters parameters) :
    OptimizerBase(std::move(parameters)) {
}

void RemoteOptimizer::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(ttml::core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void RemoteOptimizer::step() {
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

Worker::Worker(DataLoader train_dataloader, std::shared_ptr<ttml::modules::LinearLayer> model) :
    m_train_dataloader(std::move(train_dataloader)), m_model(std::move(model)) {
}
void Worker::training_step() {
    auto& ctx = ttml::autograd::ctx();
    auto& device = ctx.get_device();

    m_model->train();

    for (auto [features, target] : m_train_dataloader) {
        auto output = (*m_model)(features);
        auto loss = ttml::ops::mse_loss(output, target);
        loss->backward();
    }
}

void Worker::send_gradients(int aggregator_rank) {
}

void receive_weights(int aggregator_rank) {
}

}  // namespace roles
