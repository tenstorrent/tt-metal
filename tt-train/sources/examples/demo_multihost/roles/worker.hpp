// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <span>

#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/generators.hpp"
#include "models/linear_regression.hpp"
#include "modules/linear_module.hpp"
#include "optimizers/optimizer_base.hpp"

namespace roles {
using DatasetSample = std::pair<std::vector<float>, std::vector<float>>;
using BatchType = std::pair<ttml::autograd::TensorPtr, ttml::autograd::TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryFloatVecDataset,
    std::function<BatchType(std::vector<DatasetSample>&& samples)>,
    BatchType>;

using SortedParameters = std::map<std::string, ttml::autograd::TensorPtr>;
class RemoteOptimizer : public ttml::optimizers::OptimizerBase {
public:
    explicit RemoteOptimizer(ttml::serialization::NamedParameters parameters, int aggregator_rank = 0);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] ttml::serialization::StateDict get_state_dict() const override;
    void set_state_dict(const ttml::serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    SortedParameters get_sorted_parameters() const;
    void send_gradients();
    void receive_weights();

    void set_lr(float lr) override {
    }
    [[nodiscard]] float get_lr() const override {
        return 0.F;
    }

private:
    size_t m_steps{0};
    int m_aggregator_rank{0};
    SortedParameters m_sorted_parameters;
};

class Worker {
public:
    Worker(DataLoader train_dataloader, std::shared_ptr<ttml::modules::LinearLayer> model, int aggregator_rank);
    void training_step();

private:
    DataLoader m_train_dataloader;

    std::shared_ptr<ttml::modules::LinearLayer> m_model;
    std::shared_ptr<RemoteOptimizer> m_optimizer;

    int m_aggregator_rank = 0;
    int m_training_step = 0;
};

}  // namespace roles
