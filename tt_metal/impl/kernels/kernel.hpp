/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>
#include <map>
#include <optional>
#include <variant>
#include <type_traits>

#include "build_kernels_for_riscv/build_kernel_options.hpp"
#include "common/base_types.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/llrt/tt_memory.h"

namespace tt {

namespace tt_metal {

using Config = std::variant<DataMovementConfig, ComputeConfig>;

class Kernel {
   public:
    Kernel(const std::string &kernel_path_file_name, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &compile_args, const std::map<std::string, std::string> &defines);

    virtual ~Kernel() {}

    const uintptr_t id() const { return id_; }

    std::string kernel_path_file_name() const { return kernel_path_file_name_; }

    std::string name() const;

    CoreRangeSet core_range_set() const { return core_range_set_; }

    std::set<CoreCoord> logical_cores() const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    std::vector<ll_api::memory> const &binaries() const;

    std::vector<u32> compile_time_args() const { return compile_time_args_; }

    std::map<CoreCoord, std::vector<u32>> const &runtime_args() const { return core_to_runtime_args_; }

    std::vector<u32> const &runtime_args(const CoreCoord &logical_core);

    std::map<std::string, std::string> defines() const { return defines_; }

    virtual RISCV processor() const = 0;

    virtual bool configure(Device *device, const CoreCoord &logical_core) const = 0;

    virtual Config config() const = 0;

    std::string compute_hash() const;
    virtual void set_build_options(build_kernel_for_riscv_options_t &build_options) const = 0;
    virtual void generate_binaries(Device *device, build_kernel_for_riscv_options_t *build_options, const std::string &op_path_suffix) const = 0;
    void set_binary_path ( const std::string & binary_path) { binary_path_ = binary_path; }
    void set_binaries(std::vector<ll_api::memory> &&binaries);
    virtual void read_binaries(int pcie_slot) = 0;

    void set_runtime_args(const CoreCoord &logical_core, const std::vector<u32> &runtime_args);

   protected:
    const uintptr_t id_;
    std::string kernel_path_file_name_;                 // Full kernel path and file name
    CoreRangeSet core_range_set_;
    std::string binary_path_;
    std::vector<ll_api::memory> binaries_;              // DataMovement kernels have one binary each and Compute kernels have three binaries
    std::vector<u32> compile_time_args_;
    std::map<CoreCoord, std::vector<u32>> core_to_runtime_args_;
    std::map<std::string, std::string> defines_;        // preprocessor defines. this is to be able to generate generic instances.

    virtual uint8_t expected_num_binaries() const = 0;

    virtual std::string config_hash() const = 0;

    virtual std::pair<u64, u64> get_runtime_args_range() const = 0;
};

class DataMovementKernel : public Kernel {
   public:
    DataMovementKernel(const std::string &kernel_path, const CoreRangeSet &cr_set, const DataMovementConfig &config) : Kernel(kernel_path, cr_set, config.compile_args, config.defines), config_(config) {}

    ~DataMovementKernel() {}

    RISCV processor() const;

    void set_build_options(build_kernel_for_riscv_options_t &build_options) const;
    void generate_binaries(Device *device, build_kernel_for_riscv_options_t *build_options, const std::string &op_path_suffix) const;
    void read_binaries(int pcie_slot);

    bool configure(Device *device, const CoreCoord &logical_core) const;

    Config config() const { return this->config_; }

   private:
    const DataMovementConfig config_;

    uint8_t expected_num_binaries() const;

    std::string config_hash() const;

    std::pair<u64, u64> get_runtime_args_range() const;
};

class ComputeKernel : public Kernel {
   public:
    ComputeKernel(const std::string &kernel_path, const CoreRangeSet &cr_set, const ComputeConfig &config) : Kernel(kernel_path, cr_set, config.compile_args, config.defines), config_(config) {}

    ~ComputeKernel() {}

    RISCV processor() const;

    void set_build_options(build_kernel_for_riscv_options_t &build_options) const;
    void generate_binaries(Device *device, build_kernel_for_riscv_options_t *build_options, const std::string &op_path_suffix) const;
    void read_binaries(int pcie_slot);

    bool configure(Device *device, const CoreCoord &logical_core) const;

    Config config() const { return this->config_; }

   private:
    const ComputeConfig config_;

    uint8_t expected_num_binaries() const;

    std::string config_hash() const;

    std::pair<u64, u64> get_runtime_args_range() const;
};

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor);

struct KernelDefinesHash {
    KernelDefinesHash() {}

    size_t operator()(const std::map<std::string, std::string> &c_defines) const;
};

}  // namespace tt_metal

}  // namespace tt
