// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data_processor.hpp"

namespace tt::noc_estimator::offline {
std::map<common::GroupKey, std::vector<DataPoint>> group_by_parameters(const std::vector<DataPoint>& data_points) {
    std::map<common::GroupKey, std::vector<DataPoint>> groups;

    for (const auto& point : data_points) {
        common::GroupKey key{
            .mechanism = point.mechanism,
            .pattern = point.pattern,
            .memory = point.memory,
            .arch = point.arch,
            .num_transactions = point.num_transactions,
            .num_subordinates = point.num_subordinates,
            .same_axis = point.same_axis,
            .linked = point.linked,
        };

        groups[key].push_back(point);
    }

    return groups;
}

}  // namespace tt::noc_estimator::offline
