// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

// Defaults for every param group. Groups can override lr, momentum, dampening and weight_decay;
// nesterov applies to the whole optimizer.
struct SGDConfig {
    float lr{1e-3F};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};
};

class SGD : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override {
        return "SGD";
    }

    explicit SGD(
        ttml::serialization::NamedParameters parameters,
        const SGDConfig& config,
        std::vector<ParamGroup> param_groups = {});

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    [[nodiscard]] bool get_nesterov() const;
    void set_nesterov(bool nesterov);

private:
    // Checks every group's momentum / dampening against nesterov
    void validate_config() const;

    size_t m_steps{0};
    bool m_nesterov{false};
    ttml::serialization::NamedParameters m_momentum;
    // Buffers that have received at least one update. PyTorch seeds a fresh momentum
    // buffer with the raw gradient (buf = g), so each buffer's first update must skip
    // dampening — tracked per buffer because a lazily-unfrozen parameter can take its
    // first step at any global step count.
    std::unordered_set<std::string> m_momentum_initialized;
};

}  // namespace ttml::optimizers
