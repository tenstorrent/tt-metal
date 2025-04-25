// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <span>

#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/generators.hpp"
#include "models/linear_regression.hpp"
#include "modules/linear_module.hpp"

namespace roles {
using DatasetSample = std::pair<std::vector<float>, std::vector<float>>;
using BatchType = std::pair<ttml::autograd::TensorPtr, ttml::autograd::TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryFloatVecDataset,
    std::function<BatchType(std::vector<DatasetSample>&& samples)>,
    BatchType>;
class Worker {
public:
    Worker(DataLoader train_dataloader, std::shared_ptr<ttml::modules::LinearLayer> model);
    void training_step();

    void send_gradients(int aggregator_rank);
    void receive_weights(int aggregator_rank);

private:
    DataLoader m_train_dataloader;

    std::shared_ptr<ttml::modules::LinearLayer> m_model;

    int m_training_step = 0;
};
}  // namespace roles
