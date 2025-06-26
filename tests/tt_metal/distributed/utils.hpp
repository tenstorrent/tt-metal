// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <memory>
#include <unordered_set>
#include <vector>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>

namespace tt {
namespace tt_metal {
namespace distributed {
class MeshBuffer;
class MeshDevice;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed::test::utils {

std::vector<std::shared_ptr<Program>> create_eltwise_bin_programs(
    std::shared_ptr<MeshDevice>& mesh_device,
    std::vector<std::shared_ptr<MeshBuffer>>& src0_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& src1_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& output_bufs);

std::vector<std::shared_ptr<Program>> create_random_programs(
    uint32_t num_programs,
    CoreCoord worker_grid_size,
    uint32_t seed,
    const std::unordered_set<CoreCoord>& active_eth_cores = {});

// RAII guard for managing a single environment variable
class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value);
    ~ScopedEnvVar();

    // Delete copy/move to ensure RAII semantics
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;
    ScopedEnvVar(ScopedEnvVar&&) = delete;
    ScopedEnvVar& operator=(ScopedEnvVar&&) = delete;

private:
    const char* name_;
    std::string original_value_;
    bool had_original_ = false;
};

}  // namespace tt::tt_metal::distributed::test::utils
