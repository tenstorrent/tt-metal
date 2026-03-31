// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::moe_dispatch {

struct MoeDispatchParams {
    uint32_t cluster_axis;  // which mesh axis to dispatch along (0 or 1)
    uint32_t E_local;       // experts per device

    // Per dispatch-device counts and offsets (local to each device's sorted_hidden).
    // Indexed by dispatch_axis_index (0 .. num_EP_devices-1).
    // expert_counts_per_device[d][e] = tile-rows of device d's tokens going to expert e.
    // expert_offsets_per_device[d][e] = start tile-row in device d's sorted_hidden.
    std::vector<std::vector<uint32_t>> expert_counts_per_device;
    std::vector<std::vector<uint32_t>> expert_offsets_per_device;

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("E_local", E_local);
        attrs.emplace_back(
            "num_experts",
            static_cast<uint32_t>(expert_counts_per_device.empty() ? 0 : expert_counts_per_device[0].size()));
        return attrs;
    }
};

struct MoeDispatchTensorArgs {
    const ttnn::Tensor& sorted_hidden;  // [1, 1, N_padded, D] sorted tokens
    const ttnn::Tensor& w_up;           // [E_local, 1, D, ffn_dim] local expert weights
};

}  // namespace ttml::metal::ops::moe_dispatch
