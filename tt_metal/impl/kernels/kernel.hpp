// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>
#include <variant>
#include <type_traits>
#include <memory>

#include "jit_build/build.hpp"
#include "common/base_types.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/llrt/tt_memory.h"
#include "tt_metal/tt_stl/span.hpp"
#include "runtime_args_data.hpp"

namespace tt {

namespace tt_metal {
inline namespace v0 {

class Device;

}  // namespace v0

constexpr uint32_t max_runtime_args = 256;
constexpr uint32_t idle_eth_max_runtime_args = eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_SIZE / sizeof(uint32_t);

using Config = std::variant<DataMovementConfig, EthernetConfig, ComputeConfig>;
struct KernelSource {
    enum SourceType { FILE_PATH, SOURCE_CODE };

    std::string source_;
    SourceType source_type_;

    KernelSource(const std::string &source, const SourceType &source_type) :
        source_(source), source_type_(source_type) {}

    std::string name() const {
        std::string name;
        if (this->source_type_ == SourceType::FILE_PATH) {
            const std::size_t start_pos_of_name = this->source_.rfind("/") + 1;
            const std::size_t pos_of_dot = this->source_.rfind(".");
            name = this->source_.substr(start_pos_of_name, (pos_of_dot - start_pos_of_name));
        } else {
            name = "Kernel_Source_Code";
        }
        return name;
    }
};

inline namespace v0 {

class Kernel : public JitBuildSettings {
   public:
    Kernel(
        const KernelSource &kernel_src,
        const CoreRangeSet &core_range_set,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines);

    virtual ~Kernel() {}

    std::string name() const;

    const KernelSource &kernel_source() const { return kernel_src_; }

    const CoreRangeSet &core_range_set() const { return core_range_set_; }

    const std::set<CoreCoord> &logical_cores() const;

    std::vector<CoreRange> logical_coreranges() const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    std::vector<ll_api::memory> const &binaries(uint32_t build_key) const;

    std::vector<uint32_t> compile_time_args() const { return compile_time_args_; }

    const std::set<CoreCoord> &cores_with_runtime_args() const { return core_with_runtime_args_; }

    std::vector<uint32_t> & runtime_args(const CoreCoord &logical_core);
    RuntimeArgsData & runtime_args_data(const CoreCoord &logical_core);
    std::vector< std::vector< std::vector<uint32_t>> > & runtime_args();
    std::vector< std::vector< RuntimeArgsData > > & runtime_args_data();
    void set_runtime_args_count(CoreRangeSet& core_ranges, uint32_t count);
    std::vector<uint32_t> & common_runtime_args();
    RuntimeArgsData & common_runtime_args_data();
    void set_common_runtime_args_count(uint32_t count);
    uint32_t get_common_runtime_args_count() const { return this->common_runtime_args_count_; }

    std::map<std::string, std::string> defines() const { return defines_; }

    virtual RISCV processor() const = 0;
    dispatch_core_processor_classes dispatch_class() { return this->dispatch_class_; }

    virtual bool configure(Device *device, const CoreCoord &logical_core) const = 0;

    virtual Config config() const = 0;

    std::string compute_hash() const;
    virtual void set_build_options(JitBuildOptions &build_options) const = 0;
    virtual void generate_binaries(Device *device, JitBuildOptions &build_options) const = 0;
    uint32_t get_binary_packed_size(Device *device, int index) const;
    uint32_t get_binary_text_size(Device *device, int index) const;
    void set_binary_path(const std::string &binary_path) { binary_path_ = binary_path; }
    void set_binaries(uint32_t build_key, std::vector<ll_api::memory> &&binaries);
    virtual void read_binaries(Device *device) = 0;

    void validate_runtime_args_size(size_t num_unique_rt_args, size_t num_common_rt_args, const CoreCoord& logical_core);
    void set_runtime_args(const CoreCoord &logical_core, stl::Span<const uint32_t> runtime_args);
    void set_common_runtime_args(stl::Span<const uint32_t> runtime_args);

    int get_watcher_kernel_id() { return watcher_kernel_id_; }

    CoreType get_kernel_core_type() const;
    void set_full_name(const string& s) { kernel_full_name_ = s; }
    const string& get_full_kernel_name() const override;
    void process_defines(const std::function<void (const string& define, const string &value)>) const override;
    void process_compile_time_args(const std::function<void (int i, uint32_t value)>) const override;

    bool is_idle_eth();

   protected:
    int watcher_kernel_id_;
    KernelSource kernel_src_;
    std::string kernel_full_name_;  // Name + hash
    CoreRangeSet core_range_set_;
    std::string binary_path_;
    // DataMovement kernels have one binary each and Compute kernels have three binaries
    // Different set of binaries per device because kernel compilation is device dependent
    // TODO: break this dependency by https://github.com/tenstorrent/tt-metal/issues/3381
    std::unordered_map<chip_id_t, std::vector<ll_api::memory>> binaries_;
    dispatch_core_processor_classes dispatch_class_;
    std::vector<uint32_t> compile_time_args_;
    std::vector< std::vector< std::vector<uint32_t>> > core_to_runtime_args_;
    std::vector< std::vector< RuntimeArgsData> > core_to_runtime_args_data_;
    uint32_t common_runtime_args_count_;
    std::vector<uint32_t> common_runtime_args_;
    RuntimeArgsData common_runtime_args_data_;
    std::set<CoreCoord> core_with_runtime_args_;
    std::size_t max_runtime_args_per_core_;             // For validation
    CoreCoord core_with_max_runtime_args_;              // For validation
    std::map<std::string, std::string> defines_;        // preprocessor defines. this is to be able to generate generic instances.
    std::set<CoreCoord> logical_cores_;

    virtual uint8_t expected_num_binaries() const = 0;

    virtual std::string config_hash() const = 0;

   private:
    void register_kernel_with_watcher();
};

class DataMovementKernel : public Kernel {
   public:
    DataMovementKernel(const KernelSource &kernel_src, const CoreRangeSet &cr_set, const DataMovementConfig &config) :
        Kernel(kernel_src, cr_set, config.compile_args, config.defines), config_(config) {
        this->dispatch_class_ = (config.processor == DataMovementProcessor::RISCV_0) ? DISPATCH_CLASS_TENSIX_DM0
                                                                                     : DISPATCH_CLASS_TENSIX_DM1;
    }

    ~DataMovementKernel() {}

    RISCV processor() const override;

    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(Device *device, JitBuildOptions& build_options) const override;
    void read_binaries(Device *device) override;

    bool configure(Device *device, const CoreCoord &logical_core) const override;

    Config config() const override { return this->config_; }

    void process_defines(const std::function<void (const string& define, const string &value)>) const override;

   private:
    const DataMovementConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;
};

class EthernetKernel : public Kernel {
   public:
    EthernetKernel(const KernelSource &kernel_src, const CoreRangeSet &cr_set, const EthernetConfig &config) :
        Kernel(kernel_src, cr_set, config.compile_args, config.defines), config_(config) {
        this->dispatch_class_ = DISPATCH_CLASS_ETH_DM0;
    }

    ~EthernetKernel() {}

    RISCV processor() const override;

    void set_build_options(JitBuildOptions &build_options) const override;
    void generate_binaries(Device *device, JitBuildOptions &build_options) const override;
    void read_binaries(Device *device) override;

    bool configure(Device *device, const CoreCoord &logical_core) const override;

    Config config() const override { return this->config_; }

    void process_defines(const std::function<void(const string &define, const string &value)>) const override;

   private:
    const EthernetConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;
};

class ComputeKernel : public Kernel {
   public:
    ComputeKernel(const KernelSource &kernel_src, const CoreRangeSet &cr_set, const ComputeConfig &config) :
        Kernel(kernel_src, cr_set, config.compile_args, config.defines), config_(config) {
        this->dispatch_class_ = DISPATCH_CLASS_TENSIX_COMPUTE;
    }

    ~ComputeKernel() {}

    RISCV processor() const override;

    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(Device *device, JitBuildOptions& build_options) const override;
    void read_binaries(Device *device) override;

    bool configure(Device *device, const CoreCoord &logical_core) const override;

    Config config() const override { return this->config_; }

    void process_defines(const std::function<void (const string& define, const string &value)>) const override;

   private:
    const ComputeConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;
};

}  // namespace v0

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor);

struct KernelDefinesHash {
    KernelDefinesHash() {}

    size_t operator()(const std::map<std::string, std::string> &c_defines) const;
};

}  // namespace tt_metal

}  // namespace tt
