// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <enchantum/enchantum.hpp>
#include <stdint.h>
#include <tt_stl/span.hpp>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>
#include <tt-metalium/utils.hpp>

namespace ll_api {
class memory;
}  // namespace ll_api

namespace tt {

namespace tt_metal {

class IDevice;
enum class DataMovementProcessor;
class KernelImpl;

constexpr uint32_t max_runtime_args = 256;

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

class Kernel {
public:
    virtual ~Kernel() {}

    std::string name() const;

    const KernelSource &kernel_source() const { return kernel_src_; }

    const CoreRangeSet &core_range_set() const { return core_range_set_; }

    const std::set<CoreCoord> &logical_cores() const;

    std::vector<CoreRange> logical_coreranges() const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

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

    const std::map<std::string, std::string>& defines() const { return defines_; }

    virtual RISCV processor() const = 0;
    uint32_t dispatch_class() { return this->dispatch_class_; }

    virtual bool configure(IDevice* device, const CoreCoord &logical_core, uint32_t base_address, const uint32_t offsets[]) const = 0;

    virtual Config config() const = 0;

    std::string compute_hash() const;

    virtual const std::string& get_full_kernel_name() const = 0;

    void validate_runtime_args_size(size_t num_unique_rt_args, size_t num_common_rt_args, const CoreCoord& logical_core);
    void set_runtime_args(const CoreCoord &logical_core, stl::Span<const uint32_t> runtime_args);
    void set_common_runtime_args(stl::Span<const uint32_t> runtime_args);

    int get_watcher_kernel_id() const { return watcher_kernel_id_; }

    HalProgrammableCoreType get_kernel_programmable_core_type() const;
    CoreType get_kernel_core_type() const;
    void set_full_name(const std::string& s) { kernel_full_name_ = s; }
    void add_defines(const std::map<std::string, std::string>& defines);
    virtual uint32_t get_binary_packed_size(IDevice* device, int index) const = 0;

    bool is_idle_eth() const;

protected:
    int watcher_kernel_id_;
    KernelSource kernel_src_;
    std::string kernel_full_name_;  // Name + hash
    CoreRangeSet core_range_set_;
    uint8_t dispatch_class_;
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

    virtual std::string config_hash() const = 0;

private:
    void register_kernel_with_watcher();

    Kernel(
        const KernelSource& kernel_src,
        const CoreRangeSet& core_range_set,
        const std::vector<uint32_t>& compile_args,
        const std::map<std::string, std::string>& defines);

    // Only allow KernelImpl to inherit from Kernel.
    friend class KernelImpl;
};

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor);

}  // namespace tt_metal

}  // namespace tt
