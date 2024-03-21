// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "jit_build/build.hpp"
#include "llrt/llrt.hpp"

#include <fmt/ranges.h>
#include <set>
#include <unordered_set>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"

namespace tt {

namespace tt_metal {

Kernel::Kernel(const std::string &kernel_path_file_name, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &compile_args, const std::map<std::string, std::string> &defines) :
    watcher_kernel_id_(watcher_register_kernel(kernel_path_file_name)),
    kernel_path_file_name_(kernel_path_file_name),
    core_range_set_(core_range_set),
    binary_size16_(0),
    compile_time_args_(compile_args), defines_(defines) {
    size_t max_x = 0, max_y = 0;
    for (auto core_range : this->core_range_set_.ranges()) {
        auto start = core_range.start;
        auto end = core_range.end;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core({x, y});
                this->logical_cores_.insert(logical_core);
                max_x = std::max( max_x, x );
                max_y = std::max( max_y, y);
            }
        }
    }
    this->core_to_runtime_args_ = { max_x+1, std::vector< std::vector<uint32_t > > (max_y+1, std::vector<uint32_t>() ) };
}

std::string Kernel::name() const {
    auto pos_of_name = kernel_path_file_name_.rfind("/") + 1;
    auto pos_of_dot = kernel_path_file_name_.rfind(".");
    std::string kernel_name = kernel_path_file_name_.substr(pos_of_name, (pos_of_dot - pos_of_name));
    return kernel_name;
}

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

const string& Kernel::get_full_kernel_name() const {
    return this->kernel_full_name_;
}

void Kernel::process_defines(const std::function<void (const string& define, const string &value)> callback) const {
    for (const auto &[define, value]: this->defines_) {
        callback(define, value);
    }
}

void DataMovementKernel::process_defines(const std::function<void (const string& define, const string &value)>callback) const {
    Kernel::process_defines(callback);
    callback("NOC_INDEX", std::to_string(this->config_.noc));
}

void ComputeKernel::process_defines(const std::function<void (const string& define, const string &value)>callback) const {
    for (const auto &[define, value]: this->defines_) {
        callback(define, value);
    }
}

void EthernetKernel::process_defines(const std::function<void (const string& define, const string &value)>callback) const {
    Kernel::process_defines(callback);
    callback("NOC_INDEX", std::to_string(this->config_.noc));
}

void Kernel::process_compile_time_args(const std::function<void (int i, uint32_t value)>callback) const {
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

std::vector<ll_api::memory> const &Kernel::binaries(chip_id_t device_id) const {
    int expected_num_binaries = this->expected_num_binaries();
    if (this->binaries_.find(device_id) != this->binaries_.end() and this->binaries_.at(device_id).size() != expected_num_binaries) {
        TT_THROW("Expected " + std::to_string(expected_num_binaries) + " binaries but have "
                    + std::to_string(this->binaries_.at(device_id).size()) + " for kernel " + this->name());
    }
    return this->binaries_.at(device_id);
}

std::string DataMovementKernel::config_hash() const {
    return fmt::format("{}", magic_enum::enum_name(this->config_.noc));
}

// Add "eth_" to the hash to differentiate between erisc and brisc.
std::string EthernetKernel::config_hash() const { return fmt::format("eth_{}", magic_enum::enum_name(this->config_.noc)); }

std::string ComputeKernel::config_hash() const {
    return fmt::format("{}_{}_{}", magic_enum::enum_name(this->config_.math_fidelity), this->config_.fp32_dest_acc_en, this->config_.math_approx_mode);
}

std::string Kernel::compute_hash() const {
    return fmt::format(
        "{}_{}_{}_{}",
        std::hash<std::string>{}(this->name()),
        fmt::join(this->compile_time_args_, "_"),
        KernelDefinesHash{}(this->defines_),
        this->config_hash()
    );
}

void Kernel::update_runtime_arg( const CoreCoord &logical_core, size_t idx, uint32_t value){
    ZoneScoped;
    auto & v = this->core_to_runtime_args_[logical_core.x][logical_core.y];
    TT_ASSERT( idx < v.size(), "Runtime arg offset {} for Core {} out of bounds", idx, logical_core.str());
    v[idx] = value;
}

std::vector<uint32_t>& Kernel::runtime_args(const CoreCoord &logical_core) {
    // TODO (abhullar): Should this check only be enabled in debug mode?
    TT_FATAL( logical_core.x < this->core_to_runtime_args_.size() && logical_core.y < this->core_to_runtime_args_[logical_core.x].size(), "Cannot get runtime args for kernel {} that is not placed on core {}", this->name(), logical_core.str());
    return this->core_to_runtime_args_[logical_core.x][logical_core.y];
}

std::pair<uint64_t, uint64_t> DataMovementKernel::get_runtime_args_range() const {
    std::pair<uint64_t, uint64_t> arg_base_to_result_base;
    switch (this->config_.processor) {
        case DataMovementProcessor::RISCV_0: {
            arg_base_to_result_base = {BRISC_L1_ARG_BASE, BRISC_L1_RESULT_BASE};
        }
        break;
        case DataMovementProcessor::RISCV_1: {
            arg_base_to_result_base = {NCRISC_L1_ARG_BASE, NCRISC_L1_RESULT_BASE};
        }
        break;
        default:
            arg_base_to_result_base = {BRISC_L1_ARG_BASE, BRISC_L1_RESULT_BASE};
        break;
    }
    return arg_base_to_result_base;
}

std::pair<uint64_t, uint64_t> EthernetKernel::get_runtime_args_range() const {
    std::pair<uint64_t, uint64_t> arg_base_to_result_base;
    if (this->config_.eth_mode == Eth::IDLE) {
        arg_base_to_result_base = {IDLE_ERISC_L1_ARG_BASE, IDLE_ERISC_L1_RESULT_BASE};
    } else {
        arg_base_to_result_base = {eth_l1_mem::address_map::ERISC_L1_ARG_BASE, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE};
    }
    return arg_base_to_result_base;
}

std::pair<uint64_t, uint64_t> ComputeKernel::get_runtime_args_range() const {
    std::pair<uint64_t, uint64_t> arg_base_to_result_base = {TRISC_L1_ARG_BASE, TRISC_L1_ARG_BASE + 1024};
    return arg_base_to_result_base;
}

void Kernel::set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    auto validate_runtime_args_size = [&]() {
        uint32_t runtime_args_size = runtime_args.size() * sizeof(uint32_t);
        auto[l1_arg_base, result_base] = this->get_runtime_args_range();
        if (l1_arg_base + runtime_args_size > result_base) {
            TT_THROW(std::to_string(runtime_args_size / 1024) + "KB runtime args targeting kernel " + this->name() + " on " + logical_core.str() + " are too large.\
                Cannot be written as they will run into memory region reserved for result. Max allowable size is " + std::to_string((result_base - l1_arg_base)/1024) + " KB.");
        }
    };

    // TODO (abhullar): If we don't include this check then user can write runtime args to a core that the kernel is not placed on.
    //                  Should this check only be enabled in debug mode?
    // TT_FATAL(this->is_on_logical_core(logical_core), "Cannot set runtime args for core {} since kernel {} is not placed on it!", logical_core.str(), this->name());
    validate_runtime_args_size();
    auto &set_rt_args = this->core_to_runtime_args_[logical_core.x][logical_core.y];
    TT_ASSERT(set_rt_args.empty() or set_rt_args.size() == runtime_args.size(), "Illegal Runtime Args: Number of runtime args cannot be modified!");
    set_rt_args = runtime_args;
    this->core_with_runtime_args_.insert( logical_core );
}

void DataMovementKernel::set_build_options(JitBuildOptions& build_options) const {
    ZoneScoped;
    switch (this->config_.processor) {
        case DataMovementProcessor::RISCV_0: {
            build_options.brisc_kernel_file_name = this->kernel_path_file_name_;
            build_options.brisc_defines = this->defines_;
        }
        break;
        case DataMovementProcessor::RISCV_1: {
            build_options.ncrisc_kernel_file_name = this->kernel_path_file_name_;
            build_options.ncrisc_defines = this->defines_;
        }
        break;
        default:
            TT_THROW("Unsupported data movement processor!");
        break;
    }
}

void EthernetKernel::set_build_options(JitBuildOptions& build_options) const {
    build_options.erisc_kernel_file_name = this->kernel_path_file_name_;
    build_options.erisc_defines = this->defines_;
}

void ComputeKernel::set_build_options(JitBuildOptions& build_options) const {
    build_options.set_hlk_file_name_all_cores(this->kernel_path_file_name_);
    build_options.set_hlk_math_fidelity_all_cores(this->config_.math_fidelity);
    build_options.set_hlk_math_approx_mode_all_cores(this->config_.math_approx_mode);
    build_options.fp32_dest_acc_en = this->config_.fp32_dest_acc_en;
    build_options.hlk_defines = this->defines_;
}

void DataMovementKernel::generate_binaries(Device *device, JitBuildOptions& build_options) const {
    detail::GenerateDeviceHeaders(device, build_options.path);
    int riscv_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(this->config_.processor);
    jit_build(device->build_kernel_state(JitBuildProcessorType::DATA_MOVEMENT, riscv_id), this, this->kernel_path_file_name_);
}

void EthernetKernel::generate_binaries(Device *device, JitBuildOptions& build_options) const {
    detail::GenerateDeviceHeaders(device, build_options.path);
    int erisc_id = this->config_.eth_mode == Eth::IDLE ? 1 : 0;
    jit_build(device->build_kernel_state(JitBuildProcessorType::ETHERNET, erisc_id), this, this->kernel_path_file_name_);
}

void ComputeKernel::generate_binaries(Device *device, JitBuildOptions& build_options) const {
    jit_build_genfiles_triscs_src(device->build_env(), *this, this->kernel_path_file_name_);
    JitBuildStateSubset build_states = device->build_kernel_states(JitBuildProcessorType::COMPUTE);
    jit_build_subset(build_states, this, this->kernel_path_file_name_);
}

void Kernel::set_binaries(chip_id_t device_id, std::vector<ll_api::memory> &&binaries) {
    if (this->binaries_.find(device_id) != this->binaries_.end()) {
        TT_ASSERT(this->binaries_.at(device_id) == binaries);
    } else {
        this->binaries_[device_id] = std::move(binaries);
    }
}

void DataMovementKernel::read_binaries(Device *device) {
    TT_ASSERT ( !binary_path_.empty(), "Path to Kernel binaries not set!" );
    std::vector<ll_api::memory> binaries;

    // TODO(pgk): move the procssor types into the build system.  or just use integer indicies
    // TODO(pgk): consolidate read_binaries where possible
    int riscv_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(this->config_.processor);
    const JitBuildState& build_state = device->build_kernel_state(JitBuildProcessorType::DATA_MOVEMENT, riscv_id);
    ll_api::memory binary_mem = llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_));
    this->binary_size16_ = llrt::get_binary_code_size16(binary_mem, riscv_id);
    log_debug(LogLoader, "RISC {} kernel binary size: {} in bytes", riscv_id, this->binary_size16_ * 16);

    binaries.push_back(binary_mem);
    this->set_binaries(device->id(), std::move(binaries));
}

void EthernetKernel::read_binaries(Device *device) {
   // untested
    TT_ASSERT ( !binary_path_.empty(), "Path to Kernel binaries not set!" );
    std::vector<ll_api::memory> binaries;
    int erisc_id = this->config_.eth_mode == Eth::IDLE ? 1 : 0;
    const JitBuildState& build_state = device->build_kernel_state(JitBuildProcessorType::ETHERNET, erisc_id);
    ll_api::memory binary_mem = llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_));
    binaries.push_back(binary_mem);
    this->set_binaries(device->id(), std::move(binaries));
}

void ComputeKernel::read_binaries(Device *device) {
    TT_ASSERT ( !binary_path_.empty(), "Path to Kernel binaries not set!" );
    std::vector<ll_api::memory> binaries;
    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        const JitBuildState& build_state = device->build_kernel_state(JitBuildProcessorType::COMPUTE, trisc_id);
        ll_api::memory binary_mem = llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_));
        this->binary_size16_ = llrt::get_binary_code_size16(binary_mem, trisc_id + 2);
        log_debug(LogLoader, "RISC {} kernel binary size: {} in bytes", trisc_id + 2, this->binary_size16_ * 16);
        binaries.push_back(binary_mem);
    }
    this->set_binaries(device->id(), std::move(binaries));
}

RISCV DataMovementKernel::processor() const {
    switch (this->config_.processor) {
        case DataMovementProcessor::RISCV_0: return RISCV::BRISC;
        case DataMovementProcessor::RISCV_1: return RISCV::NCRISC;
        default:
            TT_THROW("Unsupported data movement processor");
    }
    return RISCV::BRISC;
}

RISCV EthernetKernel::processor() const { return RISCV::ERISC; }

RISCV ComputeKernel::processor() const { return RISCV::COMPUTE; }

bool DataMovementKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core " + logical_core.str());
    }
    auto device_id = device->id();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    ll_api::memory binary_mem = this->binaries(device_id).at(0);

    int riscv_id;
    switch (this->config_.processor) {
        case (DataMovementProcessor::RISCV_0): {
            riscv_id = 0;
        }
        break;
        case (DataMovementProcessor::RISCV_1): {
            riscv_id = 1;
        }
        break;
        default:
            TT_THROW("Unsupported data movement processor!");
    }

    pass &= tt::llrt::test_load_write_read_risc_binary(binary_mem, device_id, worker_core, riscv_id);
    return pass;
}

bool EthernetKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    auto device_id = device->id();
    auto ethernet_core = device->ethernet_core_from_logical_core(logical_core);
    ll_api::memory binary_mem = this->binaries(device_id).at(0);
    int riscv_id = this->config_.eth_mode == Eth::IDLE ? 6 : 5;
    pass &= tt::llrt::test_load_write_read_risc_binary(binary_mem, device_id, ethernet_core, riscv_id);
    return pass;
}

bool ComputeKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core " + logical_core.str());
    }
    auto device_id = device->id();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    std::vector<ll_api::memory> binaries = this->binaries(device_id);

    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        pass &= tt::llrt::test_load_write_read_trisc_binary(
            binaries.at(trisc_id),
            device_id,
            worker_core,
            trisc_id);
    }

    return pass;
}

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor) {
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
        boost::hash_combine(hash_value, std::hash<std::string>{}(it->first + it->second));
    return hash_value;
}


}  // namespace tt_metal

}  // namespace tt
