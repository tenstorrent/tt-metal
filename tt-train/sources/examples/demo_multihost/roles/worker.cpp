// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "worker.hpp"

#include "ops/losses.hpp"

namespace roles {
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
