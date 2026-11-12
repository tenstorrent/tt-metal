// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

// Defaults for every param group. Groups can override lr and momentum; ns_steps applies to the whole
// optimizer.
struct MuonConfig {
    float lr{1e-3F};
    float momentum{0.95F};
    int ns_steps{5};
};

class MuonComposite : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override {
        return "MuonComposite";
    }

    explicit MuonComposite(
        ttml::serialization::NamedParameters parameters,
        const MuonConfig& config,
        std::vector<ParamGroup> param_groups = {});

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

private:
    size_t m_steps{0};
    int m_ns_steps{5};
    ttml::serialization::NamedParameters m_momentum_buffer;
};

}  // namespace ttml::optimizers
