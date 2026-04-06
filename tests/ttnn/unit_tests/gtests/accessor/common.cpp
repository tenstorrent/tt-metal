// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/ttnn/unit_tests/gtests/accessor/common.hpp"

std::vector<tensor_accessor::ArgsConfig> get_all_sharded_args_configs() {
    using tensor_accessor::ArgConfig;
    using tensor_accessor::ArgsConfig;
    std::vector<tensor_accessor::ArgsConfig> args_combinations;
    for (int runtime_rank = 0; runtime_rank < 2; ++runtime_rank) {
        for (int runtime_num_bank = 0; runtime_num_bank < 2; ++runtime_num_bank) {
            for (int runtime_tensor_shape = 0; runtime_tensor_shape < 2; ++runtime_tensor_shape) {
                for (int runtime_shard_shape = 0; runtime_shard_shape < 2; ++runtime_shard_shape) {
                    for (int runtime_bank_coords = 0; runtime_bank_coords < 2; ++runtime_bank_coords) {
                        if (runtime_rank and (!runtime_tensor_shape or !runtime_shard_shape)) {
                            // If rank is runtime, tensor and shard shapes must also be runtime
                            continue;
                        }
                        if (runtime_num_bank and !runtime_bank_coords) {
                            // If number of banks is runtime, bank coordinates must also be runtime
                            continue;
                        }
                        ArgsConfig args_config{ArgConfig::Sharded};
                        args_config.set(ArgConfig::RuntimeRank, runtime_rank);
                        args_config.set(ArgConfig::RuntimeNumBanks, runtime_num_bank);
                        args_config.set(ArgConfig::RuntimeTensorShape, runtime_tensor_shape);
                        args_config.set(ArgConfig::RuntimeShardShape, runtime_shard_shape);
                        args_config.set(ArgConfig::RuntimeBankCoords, runtime_bank_coords);
                        args_combinations.push_back(args_config);
                    }
                }
            }
        }
    }
    return args_combinations;
}
