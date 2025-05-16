// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "optimizers/optimizer_base.hpp"

using SortedParameters = std::map<std::string, ttml::autograd::TensorPtr>;

class RemoteOptimizer : public ttml::optimizers::OptimizerBase {
public:
    RemoteOptimizer(ttml::serialization::NamedParameters parameters, int aggregator_rank);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] ttml::serialization::StateDict get_state_dict() const override;

    void set_state_dict(const ttml::serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;

    void set_steps(size_t steps) override;

    SortedParameters get_sorted_parameters() const;

    void send_gradients();

    void receive_weights();

    void set_lr(float lr) override;

    [[nodiscard]] float get_lr() const override;

private:
    size_t m_steps{0};
    SortedParameters m_sorted_parameters;
    ttml::core::distributed::Rank m_aggregator_rank{0};
    std::shared_ptr<ttml::autograd::DistributedContext> m_distributed_ctx;
};
