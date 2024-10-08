// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/kernels/kernel.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <set>

#include "jit_build/build.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/common/utils.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/jit_build/genfiles.hpp"
namespace tt {

namespace tt_metal {

Kernel::Kernel(
    const KernelSource &kernel_src,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &compile_args,
    const std::map<std::string, std::string> &defines) :
    kernel_src_(kernel_src),
    core_range_set_(core_range_set),
    max_runtime_args_per_core_(0),
    core_with_max_runtime_args_({0, 0}),
    compile_time_args_(compile_args),
    defines_(defines) {
    this->register_kernel_with_watcher();

    size_t max_x = 0, max_y = 0;
    for (auto core_range : this->core_range_set_.ranges()) {
        auto start = core_range.start_coord;
        auto end = core_range.end_coord;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core({x, y});
                this->logical_cores_.insert(logical_core);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
    }
    this->core_to_runtime_args_ = {max_x + 1, std::vector<std::vector<uint32_t>>(max_y + 1, std::vector<uint32_t>())};
    this->core_to_runtime_args_data_ = {max_x + 1, std::vector<RuntimeArgsData>(max_y + 1, RuntimeArgsData{})};
    for (auto &runtime_args_data_x : this->core_to_runtime_args_data_) {
        for (auto &runtime_args_data : runtime_args_data_x) {
            runtime_args_data.rt_args_data = nullptr;
            runtime_args_data.rt_args_count = 0;
        }
    }
    this->common_runtime_args_count_ = 0;
}

void Kernel::register_kernel_with_watcher() {
    if (this->kernel_src_.source_type_ == KernelSource::FILE_PATH) {
        this->watcher_kernel_id_ = watcher_register_kernel(this->kernel_src_.source_);
    } else {
        TT_FATAL(this->kernel_src_.source_type_ == KernelSource::SOURCE_CODE, "Unsupported kernel source type!");
        this->watcher_kernel_id_ = watcher_register_kernel(this->name());
    }
}

std::string Kernel::name() const { return this->kernel_src_.name(); }

const std::set<CoreCoord> &Kernel::logical_cores() const { return this->logical_cores_; }

std::vector<CoreRange> Kernel::logical_coreranges() const {
    auto crs = this->core_range_set_.ranges();
    return {crs.begin(), crs.end()};
}

bool Kernel::is_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}

CoreType Kernel::get_kernel_core_type() const {
    RISCV riscv_processor = this->processor();
    switch (riscv_processor) {
        case RISCV::BRISC:
        case RISCV::NCRISC:
        case RISCV::COMPUTE: return CoreType::WORKER;
        case RISCV::ERISC: return CoreType::ETH;
        default: TT_ASSERT(false, "Unsupported kernel processor!");
    }
    return CoreType::WORKER;
}

const string &Kernel::get_full_kernel_name() const { return this->kernel_full_name_; }

void Kernel::process_defines(const std::function<void(const string &define, const string &value)> callback) const {
    for (const auto &[define, value] : this->defines_) {
        callback(define, value);
    }
}

void DataMovementKernel::process_defines(
    const std::function<void(const string &define, const string &value)> callback) const {
    Kernel::process_defines(callback);
    callback("NOC_INDEX", std::to_string(this->config_.noc));
    callback("NOC_MODE", std::to_string(this->config_.noc_mode));
}

void ComputeKernel::process_defines(
    const std::function<void(const string &define, const string &value)> callback) const {
    for (const auto &[define, value] : this->defines_) {
        callback(define, value);
    }
    // pass default noc mode as compute does not need it, just for compile to pass
    callback("NOC_MODE", std::to_string(NOC_MODE::DM_DEDICATED_NOC));
}

void EthernetKernel::process_defines(
    const std::function<void(const string &define, const string &value)> callback) const {
    Kernel::process_defines(callback);
    callback("NOC_INDEX", std::to_string(this->config_.noc));
    // pass default noc mode as eth does not need it, just for compile to pass
    callback("NOC_MODE", std::to_string(NOC_MODE::DM_DEDICATED_NOC));
}

void Kernel::process_compile_time_args(const std::function<void(int i, uint32_t value)> callback) const {
    for (int i = 0; i < this->compile_time_args_.size(); i++) {
        callback(i, this->compile_time_args_[i]);
    }
}

uint8_t DataMovementKernel::expected_num_binaries() const { return 1; }

uint8_t EthernetKernel::expected_num_binaries() const { return 1; }

uint8_t ComputeKernel::expected_num_binaries() const {
    // Compute kernels generate binaries for all three TRISC processors
    return 3;
}

std::vector<ll_api::memory> const &Kernel::binaries(uint32_t build_key) const {
    int expected_num_binaries = this->expected_num_binaries();
    if (this->binaries_.find(build_key) != this->binaries_.end() and
        this->binaries_.at(build_key).size() != expected_num_binaries) {
        TT_THROW(
            "Expected {} binaries but have {} for kernel {}",
            expected_num_binaries,
            this->binaries_.at(build_key).size(),
            this->name());
    }
    return this->binaries_.at(build_key);
}

std::string DataMovementKernel::config_hash() const {
    return fmt::format("{}", magic_enum::enum_name(this->config_.noc));
}

// Add "eth_" to the hash to differentiate between erisc and brisc.
std::string EthernetKernel::config_hash() const {
    return fmt::format("eth_{}_{}", magic_enum::enum_name(this->config_.noc), this->config_.eth_mode);
}

std::string ComputeKernel::config_hash() const {
    return fmt::format(
        "{}_{}_{}_{}",
        magic_enum::enum_name(this->config_.math_fidelity),
        this->config_.fp32_dest_acc_en,
        this->config_.math_approx_mode,
        this->config_.dst_full_sync_en);
}

std::string Kernel::compute_hash() const {
    return fmt::format(
        "{}_{}_{}_{}",
        std::hash<std::string>{}(this->kernel_src_.source_),
        fmt::join(this->compile_time_args_, "_"),
        KernelDefinesHash{}(this->defines_),
        this->config_hash());
}

std::vector<uint32_t> &Kernel::runtime_args(const CoreCoord &logical_core) {
    // TODO (abhullar): Should this check only be enabled in debug mode?
    TT_FATAL(
        logical_core.x < this->core_to_runtime_args_.size() &&
            logical_core.y < this->core_to_runtime_args_[logical_core.x].size(),
        "Cannot get runtime args for kernel {} that is not placed on core {}",
        this->name(),
        logical_core.str());
    return this->core_to_runtime_args_[logical_core.x][logical_core.y];
}

RuntimeArgsData &Kernel::runtime_args_data(const CoreCoord &logical_core) {
    // TODO (abhullar): Should this check only be enabled in debug mode?
    TT_FATAL(
        logical_core.x < this->core_to_runtime_args_.size() &&
            logical_core.y < this->core_to_runtime_args_[logical_core.x].size(),
        "Cannot get runtime args for kernel {} that is not placed on core {}",
        this->name(),
        logical_core.str());
    return this->core_to_runtime_args_data_[logical_core.x][logical_core.y];
}

std::vector<std::vector<std::vector<uint32_t>>> &Kernel::runtime_args() { return this->core_to_runtime_args_; }

std::vector<std::vector<RuntimeArgsData>> &Kernel::runtime_args_data() { return this->core_to_runtime_args_data_; }

std::vector<uint32_t> &Kernel::common_runtime_args() { return this->common_runtime_args_; }

RuntimeArgsData &Kernel::common_runtime_args_data() { return this->common_runtime_args_data_; }

// Ensure that unique and common runtime args do not overflow reserved region in L1.
void Kernel::validate_runtime_args_size(
    size_t num_unique_rt_args, size_t num_common_rt_args, const CoreCoord &logical_core) {
    uint32_t total_rt_args = (num_unique_rt_args + num_common_rt_args);
    uint32_t max_rt_args = is_idle_eth() ? idle_eth_max_runtime_args : max_runtime_args;

    if (total_rt_args > max_rt_args) {
        log_warning(tt::LogMetal, "Too many runtime args, unique: {} common: {} on {}", num_unique_rt_args, num_common_rt_args, this->processor());
        TT_THROW("{} unique+common runtime args targeting kernel {} on {} are too large. Max allowable is {}", total_rt_args, this->name(), logical_core.str(), max_runtime_args);
    }
}

void Kernel::set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    // TODO (abhullar): If we don't include this check then user can write runtime args to a core that the kernel is not
    // placed on.
    //                  Should this check only be enabled in debug mode?
    TT_ASSERT(
        this->is_on_logical_core(logical_core),
        "Cannot set runtime args for core {} since kernel {} is not placed on it!",
        logical_core.str(),
        this->name());

    // Keep state for validation, to be able to check from both set_runtime_args() and set_common_runtime_args() APIs.

    auto &set_rt_args = this->core_to_runtime_args_[logical_core.x][logical_core.y];
    // TODO: Only allow setting once
    if (set_rt_args.empty()) {
        if (runtime_args.size() > max_runtime_args_per_core_) {
            max_runtime_args_per_core_ = runtime_args.size();
            core_with_max_runtime_args_ = logical_core;
        }
        this->validate_runtime_args_size(runtime_args.size(), this->common_runtime_args_.size(), logical_core);
        set_rt_args = runtime_args;
        this->core_to_runtime_args_data_[logical_core.x][logical_core.y] =
            RuntimeArgsData{set_rt_args.data(), set_rt_args.size()};
        this->core_with_runtime_args_.insert(logical_core);
    } else {
        TT_FATAL(
            set_rt_args.size() == runtime_args.size(),
            "Illegal Runtime Args on {}: Number of runtime args cannot be modified from {} to {}!", logical_core.str(), set_rt_args.size(), runtime_args.size());
        std::memcpy(
            this->core_to_runtime_args_data_[logical_core.x][logical_core.y].rt_args_data,
            runtime_args.data(),
            runtime_args.size() * sizeof(uint32_t));
    }
}

void Kernel::set_common_runtime_args(const std::vector<uint32_t> &common_runtime_args) {
    auto &set_rt_args = this->common_runtime_args_;
    TT_FATAL(
        set_rt_args.empty(),
        "Illegal Common Runtime Args: Can only set common runtime args once. Get and modify args in place instead.");
    this->validate_runtime_args_size(
        max_runtime_args_per_core_, common_runtime_args.size(), core_with_max_runtime_args_);
    set_rt_args = common_runtime_args;
    this->common_runtime_args_data_ = RuntimeArgsData{set_rt_args.data(), set_rt_args.size()};
}

// Pads runtime args to count
void Kernel::set_runtime_args_count(CoreRangeSet& core_ranges, uint32_t count) {

    for (const CoreRange &core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                auto &set_rt_args = this->core_to_runtime_args_[x][y];
                if (set_rt_args.empty()) continue;

                TT_ASSERT(count >= core_to_runtime_args_data_[x][y].size());
                core_to_runtime_args_data_[x][y].rt_args_count = count;
            }
        }
    }
}

void Kernel::set_common_runtime_args_count(uint32_t count) {
    TT_ASSERT(count >= this->common_runtime_args_.size());

    this->common_runtime_args_count_ = count;
    this->common_runtime_args_data_.rt_args_count = count;
}

bool Kernel::is_idle_eth() {
    return std::holds_alternative<EthernetConfig>(this->config()) && std::get<EthernetConfig>(this->config()).eth_mode == Eth::IDLE;
}

uint32_t Kernel::get_binary_packed_size(Device *device, int index) const {
    // In testing situations we can query the size w/o a binary
    return (this->binaries_.find(device->build_key()) != this->binaries_.end()) ?
        this->binaries_.at(device->build_key())[index].get_packed_size() :
        0;
}

uint32_t Kernel::get_binary_text_size(Device *device, int index) const {
    // In testing situations we can query the size w/o a binary
    return (this->binaries_.find(device->build_key()) != this->binaries_.end()) ?
        this->binaries_.at(device->build_key())[index].get_text_size() :
        0;
}

void DataMovementKernel::set_build_options(JitBuildOptions &build_options) const {
    ZoneScoped;
    switch (this->config_.processor) {
        case DataMovementProcessor::RISCV_0: {
            build_options.brisc_defines = this->defines_;
        } break;
        case DataMovementProcessor::RISCV_1: {
            build_options.ncrisc_defines = this->defines_;
        } break;
        default: TT_THROW("Unsupported data movement processor!"); break;
    }
}

void EthernetKernel::set_build_options(JitBuildOptions &build_options) const {
    build_options.erisc_defines = this->defines_;
}

void ComputeKernel::set_build_options(JitBuildOptions &build_options) const {
    build_options.set_hlk_math_fidelity_all_cores(this->config_.math_fidelity);
    build_options.set_hlk_math_approx_mode_all_cores(this->config_.math_approx_mode);
    build_options.fp32_dest_acc_en = this->config_.fp32_dest_acc_en;
    build_options.dst_full_sync_en = this->config_.dst_full_sync_en;
    build_options.unpack_to_dest_mode = this->config_.unpack_to_dest_mode;
    build_options.hlk_defines = this->defines_;
}

void DataMovementKernel::generate_binaries(Device *device, JitBuildOptions &build_options) const {
    jit_build_genfiles_kernel_include(device->build_env(), *this, this->kernel_src_);
    device->generate_device_headers(build_options.path);
    int riscv_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(this->config_.processor);
    jit_build(device->build_kernel_state(JitBuildProcessorType::DATA_MOVEMENT, riscv_id), this);
}

void EthernetKernel::generate_binaries(Device *device, JitBuildOptions &build_options) const {
    jit_build_genfiles_kernel_include(device->build_env(), *this, this->kernel_src_);
    device->generate_device_headers(build_options.path);
    int erisc_id = this->config_.eth_mode == Eth::IDLE ? 1 : 0;
    jit_build(device->build_kernel_state(JitBuildProcessorType::ETHERNET, erisc_id), this);
}

void ComputeKernel::generate_binaries(Device *device, JitBuildOptions &build_options) const {
    jit_build_genfiles_triscs_src(device->build_env(), *this, this->kernel_src_);
    JitBuildStateSubset build_states = device->build_kernel_states(JitBuildProcessorType::COMPUTE);
    jit_build_subset(build_states, this);
}

void Kernel::set_binaries(uint32_t build_key, std::vector<ll_api::memory> &&binaries) {
    if (this->binaries_.find(build_key) != this->binaries_.end()) {
        TT_ASSERT(this->binaries_.at(build_key) == binaries);
    } else {
        this->binaries_[build_key] = std::move(binaries);
    }
}

void DataMovementKernel::read_binaries(Device *device) {
    TT_ASSERT(!binary_path_.empty(), "Path to Kernel binaries not set!");
    std::vector<ll_api::memory> binaries;

    // TODO(pgk): move the procssor types into the build system.  or just use integer indicies
    // TODO(pgk): consolidate read_binaries where possible
    int riscv_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(this->config_.processor);
    const JitBuildState &build_state = device->build_kernel_state(JitBuildProcessorType::DATA_MOVEMENT, riscv_id);
    ll_api::memory binary_mem = llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_), riscv_id, llrt::PackSpans::PACK);
    binaries.push_back(binary_mem);
    uint32_t binary_size = binary_mem.get_packed_size();
    log_debug(LogLoader, "RISC {} kernel binary size: {} in bytes", riscv_id, binary_size);
    this->set_binaries(device->build_key(), std::move(binaries));
}

void EthernetKernel::read_binaries(Device *device) {
    // untested
    TT_ASSERT(!binary_path_.empty(), "Path to Kernel binaries not set!");
    std::vector<ll_api::memory> binaries;
    int erisc_id = this->config_.eth_mode == Eth::IDLE ? 1 : 0;
    const JitBuildState &build_state = device->build_kernel_state(JitBuildProcessorType::ETHERNET, erisc_id);
    ll_api::memory binary_mem = llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_), erisc_id + 5, llrt::PackSpans::PACK);
    binaries.push_back(binary_mem);
    uint32_t binary_size = binary_mem.get_packed_size();
    log_debug(LogLoader, "ERISC {} kernel binary size: {} in bytes", erisc_id, binary_size);
    this->set_binaries(device->build_key(), std::move(binaries));
}

void ComputeKernel::read_binaries(Device *device) {
    TT_ASSERT(!binary_path_.empty(), "Path to Kernel binaries not set!");
    std::vector<ll_api::memory> binaries;
    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        const JitBuildState &build_state = device->build_kernel_state(JitBuildProcessorType::COMPUTE, trisc_id);
        ll_api::memory binary_mem = llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_), trisc_id + 2, llrt::PackSpans::PACK);
        binaries.push_back(binary_mem);
        uint32_t binary_size = binary_mem.get_packed_size();
        log_debug(LogLoader, "RISC {} kernel binary size: {} in bytes", trisc_id + 2, binary_size);
    }
    this->set_binaries(device->build_key(), std::move(binaries));
}

RISCV DataMovementKernel::processor() const {
    switch (this->config_.processor) {
        case DataMovementProcessor::RISCV_0: return RISCV::BRISC;
        case DataMovementProcessor::RISCV_1: return RISCV::NCRISC;
        default: TT_THROW("Unsupported data movement processor");
    }
    return RISCV::BRISC;
}

RISCV EthernetKernel::processor() const { return RISCV::ERISC; }

RISCV ComputeKernel::processor() const { return RISCV::COMPUTE; }

bool DataMovementKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core {}", logical_core.str());
    }
    auto device_id = device->id();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    ll_api::memory binary_mem = this->binaries(device->build_key()).at(0);

    int riscv_id;
    switch (this->config_.processor) {
        case (DataMovementProcessor::RISCV_0): {
            riscv_id = 0;
        } break;
        case (DataMovementProcessor::RISCV_1): {
            riscv_id = 1;
        } break;
        default: TT_THROW("Unsupported data movement processor!");
    }

    pass &= tt::llrt::test_load_write_read_risc_binary(binary_mem, device_id, worker_core, riscv_id);
    return pass;
}

bool EthernetKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    auto device_id = device->id();
    auto ethernet_core = device->ethernet_core_from_logical_core(logical_core);
    ll_api::memory binary_mem = this->binaries(device->build_key()).at(0);
    int riscv_id = this->config_.eth_mode == Eth::IDLE ? 6 : 5;
    pass &= tt::llrt::test_load_write_read_risc_binary(binary_mem, device_id, ethernet_core, riscv_id);
    return pass;
}

bool ComputeKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core {}", logical_core.str());
    }
    auto device_id = device->id();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    std::vector<ll_api::memory> binaries = this->binaries(device->build_key());

    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        pass &= tt::llrt::test_load_write_read_trisc_binary(binaries.at(trisc_id), device_id, worker_core, trisc_id);
    }

    return pass;
}

std::ostream &operator<<(std::ostream &os, const DataMovementProcessor &processor) {
    switch (processor) {
        case DataMovementProcessor::RISCV_0: os << "RISCV_0"; break;
        case DataMovementProcessor::RISCV_1: os << "RISCV_1"; break;
        default: TT_THROW("Unknown data movement processor");
    }
    return os;
}

size_t KernelDefinesHash::operator()(const std::map<std::string, std::string> &c_defines) const {
    size_t hash_value = 0;
    for (auto it = c_defines.begin(); it != c_defines.end(); ++it)
        tt::utils::hash_combine(hash_value, std::hash<std::string>{}(it->first + it->second));
    return hash_value;
}

}  // namespace tt_metal

}  // namespace tt
