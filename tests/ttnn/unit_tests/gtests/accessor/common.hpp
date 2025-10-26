// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "hostdevcommon/tensor_accessor/arg_config.hpp"

std::vector<tensor_accessor::ArgsConfig> get_all_sharded_args_configs();
