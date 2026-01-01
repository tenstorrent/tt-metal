// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "optimizers/optimizer_base.hpp"

namespace ttml::optimizers {

// beta1^t and beta2^t for each param group (AdamW's bias correction).
//
// - advance(): once per group in every step(), after m_steps++.
// - reset(): whenever the step count, a group's beta1/beta2, or the param groups change.
class BetaPowers {
public:
    void reset(const std::vector<ParamGroup>& groups, size_t steps) {
        m_beta1_pows.clear();
        m_beta2_pows.clear();
        m_beta1_pows.reserve(groups.size());
        m_beta2_pows.reserve(groups.size());
        for (const auto& group : groups) {
            m_beta1_pows.push_back(static_cast<float>(std::pow(group.options.at("beta1"), steps)));
            m_beta2_pows.push_back(static_cast<float>(std::pow(group.options.at("beta2"), steps)));
        }
    }

    std::pair<float, float> advance(size_t group_idx, float beta1, float beta2) {
        return {m_beta1_pows[group_idx] *= beta1, m_beta2_pows[group_idx] *= beta2};
    }

    [[nodiscard]] static bool depends_on(const std::string& key) {
        return key == "beta1" || key == "beta2";
    }

private:
    std::vector<float> m_beta1_pows;
    std::vector<float> m_beta2_pows;
};

}  // namespace ttml::optimizers
