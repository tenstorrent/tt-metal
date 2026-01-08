#pragma once

#include <map>
#include <vector>
#include "csv_reader.hpp"
#include "../common/types.hpp"

namespace tt::noc_estimator::offline {

// Group data points by all parameters except transaction_size
std::map<common::GroupKey, std::vector<DataPoint>> group_by_parameters(const std::vector<DataPoint>& data_points);

}  // namespace tt::noc_estimator::offline
