// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data_processor.hpp"

namespace tt::tt_metal::experimental::noc_estimator::offline {

std::map<GroupKey, std::vector<DataPoint>> group_by_parameters(const std::vector<DataPoint>& data_points) {
    std::map<GroupKey, std::vector<DataPoint>> groups;

    for (const auto& point : data_points) {
        GroupKey key{
            .mechanism = point.mechanism,
            .pattern = point.pattern,
            .memory = point.memory,
            .arch = point.arch,
            .num_transactions = point.num_transactions,
            .num_subordinates = point.num_subordinates,
            .same_axis = point.same_axis,
            .stateful = point.stateful,
            .loopback = point.loopback,
            .noc_index = point.noc_index,
        };

        groups[key].push_back(point);
    }

    return groups;
}

}  // namespace tt::tt_metal::experimental::noc_estimator::offline
