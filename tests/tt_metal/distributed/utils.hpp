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

// RAII class to create and delete a temporary file.
class TemporaryFile {
public:
    explicit TemporaryFile(const std::string& filename) :
        path_(std::filesystem::temp_directory_path() / filename) {}

    ~TemporaryFile() {
        if (std::filesystem::exists(path_)) {
            std::filesystem::remove(path_);
        }
    }

    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    TemporaryFile(TemporaryFile&&) = delete;
    TemporaryFile& operator=(TemporaryFile&&) = delete;

    std::string string() const { return path_.string(); }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace tt::tt_metal::distributed::test::utils
