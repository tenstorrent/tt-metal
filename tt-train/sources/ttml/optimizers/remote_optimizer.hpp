// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/distributed/distributed.hpp"
#include "core/distributed/socket_manager.hpp"
#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

using SortedParameters = std::map<std::string, autograd::TensorPtr>;

class RemoteOptimizer : public OptimizerBase {
public:
    RemoteOptimizer(serialization::NamedParameters parameters, int aggregator_rank);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;

    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;

    void set_steps(size_t steps) override;

    [[nodiscard]] SortedParameters get_sorted_parameters() const;

    void send_gradients();

    void receive_weights();

    void set_lr(float lr) override;

    [[nodiscard]] float get_lr() const override;

private:
    size_t m_steps{0};
    SortedParameters m_sorted_parameters;
    core::distributed::Rank m_aggregator_rank{0};
    std::shared_ptr<core::distributed::DistributedContext> m_distributed_ctx;
};

}  // namespace ttml::optimizers
