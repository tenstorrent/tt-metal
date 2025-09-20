// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hal.hpp"

namespace tt::tt_metal::hal_2xx {

class HalJitBuildQueryBase : public HalJitBuildQueryInterface {
public:
    std::vector<std::string> defines(const Params& params) const override;
    std::vector<std::string> srcs(const Params& params) const override;
    std::string target_name(const Params& params) const override;
};

}  // namespace tt::tt_metal::hal_2xx
