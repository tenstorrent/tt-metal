// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizer.hpp"

namespace roles {
Optimizer::Optimizer(std::shared_ptr<ttml::modules::LinearLayer> model, int rank, int worker_rank) :
    m_model(model), m_rank(rank), m_worker_rank(worker_rank) {
    auto sgd_config = ttml::optimizers::SGDConfig{.lr = 0.1F, .momentum = 0.0F};
    m_optimizer = std::make_shared<ttml::optimizers::SGD>(model->parameters(), sgd_config);
    auto params = model->parameters();
    m_sorted_parameters = SortedParameters{params.begin(), params.end()};
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        if (tensor_ptr->get_requires_grad() && !tensor_ptr->is_grad_initialized()) {
            auto tensor = tensor_ptr->get_value();
            tensor_ptr->set_grad(ttml::core::zeros_like(tensor));
        }
    }
}
void Optimizer::optimization_step() {
    for (auto& [name, tensor_ptr] : m_sorted_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            auto grad = tensor_ptr->get_grad();
            fmt::print("Rank {}, Worker Rank {}: recv gradient for parameter {}\n", m_rank, m_worker_rank, name);
            ttml::core::distributed::recv_tensor(grad, m_worker_rank);
            tensor_ptr->set_grad(grad);  // probably not needed
        }
    }
    m_optimizer->step();
}
void Optimizer::send_weights() {
    for (auto& it : m_sorted_parameters) {
        auto& [name, tensor_ptr] = it;
        {
            auto tensor = tensor_ptr->get_value();
            ttml::core::distributed::send_tensor(tensor, m_worker_rank);
        }
    }
}

}  // namespace roles
