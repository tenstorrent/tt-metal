// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
            // The float product advance() builds, so a restored step count continues bit for bit. std::pow rounds
            // differently: 0.9^3 is 0.7289999 as a float product and 0.729 from pow.
            const float beta1 = group.options.at("beta1");
            const float beta2 = group.options.at("beta2");
            float beta1_pow = 1.0F;
            float beta2_pow = 1.0F;
            for (size_t step = 0; step < steps; ++step) {
                beta1_pow *= beta1;
                beta2_pow *= beta2;
            }
            m_beta1_pows.push_back(beta1_pow);
            m_beta2_pows.push_back(beta2_pow);
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
