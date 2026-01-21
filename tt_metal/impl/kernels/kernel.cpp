// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core_coord.hpp>
#include <device.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <kernel_types.hpp>
#include <enchantum/enchantum.hpp>
#include <tt_stl/tt_stl/reflection.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
#include <string_view>
#include <type_traits>
#include <utility>

#include <tt_stl/assert.hpp>
#include "base_types.hpp"
#include "data_types.hpp"
#include "hal_types.hpp"
#include "jit_build/build.hpp"
#include "jit_build/jit_build_options.hpp"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_memory.h"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/arch.hpp>
#include "kernel.hpp"
#include <impl/debug/watcher_server.hpp>

namespace tt::tt_metal {

namespace fs = std::filesystem;

namespace {
// Kernel path resolve:
//
// If the path is not an absolute path, then it must be resolved relative to:
// 1. CWD
// 2. TT_METAL_KERNEL_PATH
// 3. System Kernel Directory
// 4. TT_METAL_HOME / SetRootDir (API)
fs::path resolve_path(const fs::path& given_file_name) {
    // Priority 0: Absolute path
    if (given_file_name.is_absolute()) {
        return given_file_name;
    }

    // Priority 1: Current working directory
    {
        auto current_working_dir_search_path = fs::current_path() / given_file_name;
        if (fs::exists(current_working_dir_search_path)) {
            return current_working_dir_search_path;
        }
    }

    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();

    // Priority 2: Kernel directory
    if (rtoptions.is_kernel_dir_specified()) {
        auto kernel_dir_search_path = fs::absolute(fs::path(rtoptions.get_kernel_dir()) / given_file_name);
        if (fs::exists(kernel_dir_search_path)) {
            return kernel_dir_search_path;
        }
    }

    // Priority 3: System kernel directory
    auto system_kernel_dir_search_path = fs::absolute(fs::path(rtoptions.get_system_kernel_dir()) / given_file_name);
    if (fs::exists(system_kernel_dir_search_path)) {
        return system_kernel_dir_search_path;
    }

    // Priority 4: Root directory
    {
        auto root_dir_search_path = fs::absolute(fs::path(rtoptions.get_root_dir()) / given_file_name);
        if (fs::exists(root_dir_search_path)) {
            return root_dir_search_path;
        }
    }

    // Not found
    TT_THROW("Kernel file {} doesn't exist in any of the searched paths!", given_file_name);
}
}  // namespace

KernelSource::KernelSource(const std::string& source, const SourceType& source_type) :
    source_(source), source_type_(source_type) {
    if (source_type == FILE_PATH) {
        path_ = resolve_path(source);
    }
};

Kernel::Kernel(
    HalProgrammableCoreType programmable_core_type,
    HalProcessorClassType processor_class,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const std::vector<uint32_t>& compile_args,
    const std::map<std::string, std::string>& defines,
    const std::unordered_map<std::string, uint32_t>& named_compile_args) :
    programmable_core_type_(programmable_core_type),
    processor_class_(processor_class),
    kernel_src_(kernel_src),
    core_range_set_(core_range_set),
    compile_time_args_(compile_args),
    named_compile_time_args_(named_compile_args),

    core_with_max_runtime_args_({0, 0}),
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
    for (auto& runtime_args_data_x : this->core_to_runtime_args_data_) {
        for (auto& runtime_args_data : runtime_args_data_x) {
            runtime_args_data.rt_args_data = nullptr;
            runtime_args_data.rt_args_count = 0;
        }
    }
}

void Kernel::register_kernel_with_watcher() {
    if (this->kernel_src_.source_type_ == KernelSource::FILE_PATH) {
        this->watcher_kernel_id_ =
            MetalContext::instance().watcher_server()->register_kernel(this->kernel_src_.source_);
    } else {
        TT_FATAL(this->kernel_src_.source_type_ == KernelSource::SOURCE_CODE, "Unsupported kernel source type!");
        this->watcher_kernel_id_ = MetalContext::instance().watcher_server()->register_kernel(this->name());
    }
}

void Kernel::register_kernel_elf_paths_with_watcher(IDevice& device) const {
    TT_ASSERT(!this->kernel_full_name_.empty(), "Kernel full name not set!");
    auto paths = this->file_paths(device);
    MetalContext::instance().watcher_server()->register_kernel_elf_paths(this->watcher_kernel_id_, paths);
}

std::string Kernel::name() const { return this->kernel_src_.name(); }

const std::set<CoreCoord>& Kernel::logical_cores() const { return this->logical_cores_; }

std::vector<CoreRange> Kernel::logical_coreranges() const {
    auto crs = this->core_range_set_.ranges();
    return {crs.begin(), crs.end()};
}

bool Kernel::is_on_logical_core(const CoreCoord& logical_core) const {
    return this->core_range_set_.contains(logical_core);
}

CoreType Kernel::get_kernel_core_type() const {
    switch (programmable_core_type_) {
        case HalProgrammableCoreType::TENSIX: return CoreType::WORKER;
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: return CoreType::ETH;
        case HalProgrammableCoreType::COUNT: TT_THROW("Bad programmable core type!");
    }
    TT_THROW("Unreachable");
}

const std::string& Kernel::get_full_kernel_name() const { return this->kernel_full_name_; }

void Kernel::add_defines(const std::map<std::string, std::string>& defines) {
    this->defines_.insert(defines.begin(), defines.end());
}

void Kernel::process_defines(
    const std::function<void(const std::string& define, const std::string& value)> callback) const {
    for (const auto& [define, value] : this->defines_) {
        callback(define, value);
    }
}

void DataMovementKernel::process_defines(
    const std::function<void(const std::string& define, const std::string& value)> callback) const {
    Kernel::process_defines(callback);
    callback("NOC_INDEX", std::to_string(this->config_.noc));
    callback("NOC_MODE", std::to_string(this->config_.noc_mode));
}

void ComputeKernel::process_defines(
    const std::function<void(const std::string& define, const std::string& value)> callback) const {
    for (const auto& [define, value] : this->defines_) {
        callback(define, value);
    }
    // pass default noc mode as compute does not need it, just for compile to pass
    callback("NOC_MODE", std::to_string(NOC_MODE::DM_DEDICATED_NOC));
}

void EthernetKernel::process_defines(
    const std::function<void(const std::string& define, const std::string& value)> callback) const {
    Kernel::process_defines(callback);
    callback("NOC_INDEX", std::to_string(this->config_.noc));
    callback("NOC_MODE", std::to_string(this->config_.noc_mode));
}

std::string_view DataMovementKernel::get_compiler_opt_level() const {
    return enchantum::to_string(this->config_.opt_level);
}

std::string_view DataMovementKernel::get_linker_opt_level() const { return this->get_compiler_opt_level(); }

std::string_view ComputeKernel::get_compiler_opt_level() const {
    return enchantum::to_string(this->config_.opt_level);
}

std::string_view ComputeKernel::get_linker_opt_level() const { return this->get_compiler_opt_level(); }

std::string_view EthernetKernel::get_compiler_opt_level() const {
    return enchantum::to_string(this->config_.opt_level);
}

std::string_view EthernetKernel::get_linker_opt_level() const { return this->get_compiler_opt_level(); }

void Kernel::process_compile_time_args(const std::function<void(const std::vector<uint32_t>& values)> callback) const {
    callback(this->compile_time_args());
}

void Kernel::process_named_compile_time_args(
    const std::function<void(const std::unordered_map<std::string, uint32_t>& named_args)> callback) const {
    callback(this->named_compile_time_args());
}

void Kernel::process_include_paths(const std::function<void(const std::string& path)>& callback) const {
    // For FILE_PATH kernels, add the kernel source directory to the include path.
    // This enables relative includes (e.g., #include "foo.inc") to work when the kernel
    // source is transformed and inlined (as with simplified compute kernel syntax).
    if (kernel_src_.source_type_ == KernelSource::FILE_PATH) {
        callback(kernel_src_.path_.parent_path().string());
    }
}

bool Kernel::binaries_exist_on_disk(const IDevice* device) const {
    const uint32_t core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    const uint32_t processor_class = enchantum::to_underlying(this->get_kernel_processor_class());
    std::optional<std::string> output_path = std::nullopt;
    for (int i = 0; i < this->expected_num_binaries(); ++i) {
        const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), core_type, processor_class, this->get_kernel_processor_type(i));
        if (output_path.has_value()) {
            TT_ASSERT(build_state.get_out_path() == output_path.value());
        } else {
            output_path = build_state.get_out_path();
        }
    }
    // Note: this->get_full_kernel_name() already has a '/' at the end.
    const std::string build_success_marker_path =
        fmt::format("{}{}{}", output_path.value(), this->get_full_kernel_name(), SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME);
    return std::filesystem::exists(build_success_marker_path);
}

std::vector<std::string> Kernel::file_paths(IDevice& device) const {
    std::vector<std::string> file_paths;
    const auto& hal = MetalContext::instance().hal();
    uint32_t core_type = hal.get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    uint32_t processor_class = enchantum::to_underlying(this->get_kernel_processor_class());
    for (int i = 0; i < this->expected_num_binaries(); i++) {
        const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
            device.build_id(), core_type, processor_class, this->get_kernel_processor_type(i));
        file_paths.push_back(build_state.get_target_out_path(this->kernel_full_name_));
    }
    return file_paths;
}

uint8_t DataMovementKernel::expected_num_binaries() const { return 1; }

uint8_t EthernetKernel::expected_num_binaries() const { return 1; }

uint8_t ComputeKernel::expected_num_binaries() const {
    // Compute kernels generate binaries for all three TRISC processors
    return 3;
}

const std::vector<const ll_api::memory*>& Kernel::binaries(uint64_t build_key) const {
    auto iter = binaries_.find(build_key);
    TT_FATAL(iter != binaries_.end(), "binary not found");
    if (iter->second.size() != expected_num_binaries()) {
        TT_THROW(
            "Expected {} binaries but have {} for kernel {}",
            expected_num_binaries(),
            iter->second.size(),
            this->name());
    }
    return iter->second;
}

uint32_t DataMovementKernel::get_kernel_processor_type(int index) const {
    TT_ASSERT(0 <= index && index < expected_num_binaries(), "index out of bounds");
    return enchantum::to_underlying(this->config_.processor);
}

uint32_t EthernetKernel::get_kernel_processor_type(int index) const {
    TT_ASSERT(0 <= index && index < expected_num_binaries(), "index out of bounds");
    return enchantum::to_underlying(this->config_.processor);
}

uint32_t ComputeKernel::get_kernel_processor_type(int index) const {
    TT_ASSERT(0 <= index && index < expected_num_binaries(), "index out of bounds");
    return index;
}

std::string DataMovementKernel::config_hash() const {
    return fmt::format(
        "{}_{}_{}",
        enchantum::to_string(this->config_.processor),
        enchantum::to_string(this->config_.noc),
        enchantum::to_string(this->config_.noc_mode));
}

// Add "eth_" to the hash to differentiate between erisc and brisc.
std::string EthernetKernel::config_hash() const {
    return fmt::format(
        "eth_{}_{}_{}_{}",
        enchantum::to_string(this->config_.processor),
        enchantum::to_string(this->config_.noc),
        enchantum::to_string(this->config_.noc_mode),
        enchantum::to_string(this->config_.eth_mode));
}

std::string ComputeKernel::config_hash() const {
    // This handles config hashing for unpack_to_dest_mode which can be:
    // 1. Having specific configuration (e.g. some CBs are set to UnpackToDestFp32)
    // 2. Having explicit default configuration (vector full of Default)
    // 3. Having implicit default configuration (vector empty)
    std::string unpack_mode_descriptor = "default";
    const auto& unpack_modes = this->config_.unpack_to_dest_mode;
    if (std::ranges::any_of(this->config_.unpack_to_dest_mode, [](auto v) { return v != UnpackToDestMode::Default; })) {
        unpack_mode_descriptor = fmt::format("{}", fmt::join(unpack_modes, "."));
    }

    return fmt::format(
        "{}_{}_{}_{}_{}",
        enchantum::to_string(this->config_.math_fidelity),
        this->config_.fp32_dest_acc_en,
        this->config_.math_approx_mode,
        this->config_.dst_full_sync_en,
        unpack_mode_descriptor);
}

std::string Kernel::compute_hash() const {
    size_t define_hash_value = 0;
    for (const auto& [define, value] : this->defines_) {
        ttsl::hash::hash_combine(define_hash_value, std::hash<std::string>{}(define + value));
    }

    size_t named_args_hash_value = ttsl::hash::hash_objects_with_default_seed(this->named_compile_time_args_);

    return fmt::format(
        "{}_{}_{}_{}_{}",
        std::hash<std::string>{}(this->kernel_src_.source_),
        fmt::join(this->compile_time_args_, "_"),
        define_hash_value,
        named_args_hash_value,
        this->config_hash());
}

std::vector<uint32_t>& Kernel::runtime_args(const CoreCoord& logical_core) {
    // TODO (abhullar): Should this check only be enabled in debug mode?
    TT_FATAL(
        logical_core.x < this->core_to_runtime_args_.size() &&
            logical_core.y < this->core_to_runtime_args_[logical_core.x].size(),
        "Cannot get runtime args for kernel {} that is not placed on core {}",
        this->name(),
        logical_core.str());
    return this->core_to_runtime_args_[logical_core.x][logical_core.y];
}

RuntimeArgsData& Kernel::runtime_args_data(const CoreCoord& logical_core) {
    // TODO (abhullar): Should this check only be enabled in debug mode?
    TT_FATAL(
        logical_core.x < this->core_to_runtime_args_.size() &&
            logical_core.y < this->core_to_runtime_args_[logical_core.x].size(),
        "Cannot get runtime args for kernel {} that is not placed on core {}",
        this->name(),
        logical_core.str());
    return this->core_to_runtime_args_data_[logical_core.x][logical_core.y];
}

std::vector<std::vector<std::vector<uint32_t>>>& Kernel::runtime_args() { return this->core_to_runtime_args_; }

std::vector<std::vector<RuntimeArgsData>>& Kernel::runtime_args_data() { return this->core_to_runtime_args_data_; }

std::vector<uint32_t>& Kernel::common_runtime_args() { return this->common_runtime_args_; }

RuntimeArgsData& Kernel::common_runtime_args_data() { return this->common_runtime_args_data_; }

// Ensure that unique and common runtime args do not overflow reserved region in L1.
void Kernel::validate_runtime_args_size(
    size_t num_unique_rt_args, size_t num_common_rt_args, const CoreCoord& logical_core) const {
    uint32_t total_rt_args = (num_unique_rt_args + num_common_rt_args);
    uint32_t expected_max_rt_args = 0;

    switch (this->get_kernel_programmable_core_type()) {
        case HalProgrammableCoreType::TENSIX: expected_max_rt_args = max_runtime_args; break;
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH:
            expected_max_rt_args = MetalContext::instance().hal().get_dev_size(
                                       this->get_kernel_programmable_core_type(), HalL1MemAddrType::KERNEL_CONFIG) /
                                   sizeof(uint32_t);
            break;
        default: TT_THROW("Invalid programmable core type: {}", this->get_kernel_programmable_core_type());
    }

    if (total_rt_args > expected_max_rt_args) {
        TT_THROW(
            "{} unique+common runtime args targeting kernel {} on {} are too large. Max allowable is {}",
            total_rt_args,
            this->name(),
            logical_core.str(),
            expected_max_rt_args);
    }
}

void Kernel::set_runtime_args(const CoreCoord& logical_core, stl::Span<const uint32_t> runtime_args) {
    // TODO (abhullar): If we don't include this check then user can write runtime args to a core that the kernel is not
    // placed on.
    //                  Should this check only be enabled in debug mode?
    TT_ASSERT(
        this->is_on_logical_core(logical_core),
        "Cannot set runtime args for core {} since kernel {} is not placed on it. core range set: {}!",
        logical_core.str(),
        this->name(),
        core_range_set_.str());

    // Keep state for validation, to be able to check from both set_runtime_args() and set_common_runtime_args() APIs.

    auto& set_rt_args = this->core_to_runtime_args_[logical_core.x][logical_core.y];
    // TODO: Only allow setting once
    if (set_rt_args.empty()) {
        if (runtime_args.size() > max_runtime_args_per_core_) {
            max_runtime_args_per_core_ = runtime_args.size();
            core_with_max_runtime_args_ = logical_core;
        }
        this->validate_runtime_args_size(runtime_args.size(), this->common_runtime_args_.size(), logical_core);
        set_rt_args.assign(runtime_args.begin(), runtime_args.end());
        this->core_to_runtime_args_data_[logical_core.x][logical_core.y] =
            RuntimeArgsData{set_rt_args.data(), set_rt_args.size()};
        this->core_with_runtime_args_.insert(logical_core);
    } else {
        TT_FATAL(
            set_rt_args.size() == runtime_args.size(),
            "Illegal Runtime Args on {}: Number of runtime args cannot be modified from {} to {}!",
            logical_core.str(),
            set_rt_args.size(),
            runtime_args.size());
        std::memcpy(
            this->core_to_runtime_args_data_[logical_core.x][logical_core.y].rt_args_data,
            runtime_args.data(),
            runtime_args.size() * sizeof(uint32_t));
    }
}

void Kernel::set_common_runtime_args(stl::Span<const uint32_t> common_runtime_args) {
    auto& set_rt_args = this->common_runtime_args_;
    TT_FATAL(
        set_rt_args.empty(),
        "Illegal Common Runtime Args: Can only set common runtime args once. Get and modify args in place instead.");
    this->validate_runtime_args_size(
        max_runtime_args_per_core_, common_runtime_args.size(), core_with_max_runtime_args_);
    set_rt_args.assign(common_runtime_args.begin(), common_runtime_args.end());
    this->common_runtime_args_data_ = RuntimeArgsData{set_rt_args.data(), set_rt_args.size()};
}

// Pads runtime args to count
void Kernel::set_runtime_args_count(CoreRangeSet& core_ranges, uint32_t count) {
    for (const CoreRange& core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                auto& set_rt_args = this->core_to_runtime_args_[x][y];
                if (set_rt_args.empty()) {
                    continue;
                }

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

bool Kernel::is_idle_eth() const { return this->programmable_core_type_ == HalProgrammableCoreType::IDLE_ETH; }

detail::KernelMeta Kernel::meta(IDevice* device) const {
    detail::KernelMeta result {
        .name = this->kernel_full_name_,
        .source = this->kernel_src_.source_,
        .processor_class = get_kernel_processor_class(),
        .programmable_core_type = get_kernel_programmable_core_type(),
    };

    if (get_kernel_processor_class() == HalProcessorClassType::COMPUTE) {
        result.math_fidelity = std::get<ComputeConfig>(config()).math_fidelity;
    }

    if (device != nullptr) {
        result.binary_meta.reserve(this->expected_num_binaries());
        for (int i = 0; i < this->expected_num_binaries(); i++) {
            result.binary_meta.push_back({
                .processor_type = this->get_kernel_processor_type(i),
                .packed_size = this->get_binary_packed_size(device, i),
            });
        }
    }

    return result;
}

uint32_t Kernel::get_binary_packed_size(IDevice* device, int index) const {
    // In testing situations we can query the size w/o a binary
    auto iter = binaries_.find(BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key());
    return iter != this->binaries_.end() ? iter->second[index]->get_packed_size() : 0;
}

uint32_t Kernel::get_binary_text_size(IDevice* device, int index) const {
    // In testing situations we can query the size w/o a binary
    auto iter = binaries_.find(BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key());
    return iter != this->binaries_.end() ? iter->second[index]->get_text_size() : 0;
}

void ComputeKernel::set_build_options(JitBuildOptions& build_options) const {
    build_options.set_hlk_math_fidelity_all_cores(this->config_.math_fidelity);
    build_options.set_hlk_math_approx_mode_all_cores(this->config_.math_approx_mode);
    build_options.fp32_dest_acc_en = this->config_.fp32_dest_acc_en;
    build_options.dst_full_sync_en = this->config_.dst_full_sync_en;
    build_options.unpack_to_dest_mode = this->config_.unpack_to_dest_mode;
    build_options.bfp8_pack_precise = this->config_.bfp8_pack_precise;
}

void DataMovementKernel::generate_binaries(IDevice* device, JitBuildOptions& /*build_options*/) const {
    jit_build_genfiles_kernel_include(
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env, *this, this->kernel_src_);
    uint32_t tensix_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
    int riscv_id = static_cast<std::underlying_type_t<DataMovementProcessor>>(this->config_.processor);
    jit_build(
        BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, dm_class_idx, riscv_id),
        this);
}

void EthernetKernel::generate_binaries(IDevice* device, JitBuildOptions& /*build_options*/) const {
    jit_build_genfiles_kernel_include(
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env, *this, this->kernel_src_);
    uint32_t erisc_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
    int erisc_id = enchantum::to_underlying(this->config_.processor);
    jit_build(
        BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), erisc_core_type, dm_class_idx, erisc_id),
        this);
}

void ComputeKernel::generate_binaries(IDevice* device, JitBuildOptions& /*build_options*/) const {
    jit_build_genfiles_triscs_src(
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env, *this, this->kernel_src_);
    uint32_t tensix_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    uint32_t compute_class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);
    auto build_states = BuildEnvManager::get_instance().get_kernel_build_states(
        device->build_id(), tensix_core_type, compute_class_idx);
    jit_build_subset(build_states, this);
}

void Kernel::set_binaries(uint64_t build_key, std::vector<const ll_api::memory*>&& binaries) {
    // Try inserting an empty vector, as that is cheap to construct
    // and avoids an additional move.
    auto pair = binaries_.insert({build_key, {}});
    if (pair.second) {
        pair.first->second = std::move(binaries);
    } else {
        TT_ASSERT(pair.first->second == binaries);
    }
}

void DataMovementKernel::read_binaries(IDevice* device) {
    TT_ASSERT(this->binaries_exist_on_disk(device));
    std::vector<const ll_api::memory*> binaries;

    // TODO(pgk): move the procssor types into the build system.  or just use integer indicies
    // TODO(pgk): consolidate read_binaries where possible
    uint32_t tensix_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
    int riscv_id = static_cast<std::underlying_type_t<DataMovementProcessor>>(this->config_.processor);
    const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
        device->build_id(), tensix_core_type, dm_class_idx, riscv_id);
    auto load_type =
        MetalContext::instance().hal().get_jit_build_config(tensix_core_type, dm_class_idx, riscv_id).memory_load;
    const ll_api::memory& binary_mem =
        llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_), load_type);
    binaries.push_back(&binary_mem);
    [[maybe_unused]] uint32_t binary_size = binary_mem.get_packed_size();
    log_debug(LogLoader, "RISC={}, name={}, size={} (bytes)", riscv_id, this->name(), binary_size);
    this->set_binaries(
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key(), std::move(binaries));
}

void EthernetKernel::read_binaries(IDevice* device) {
    // untested
    TT_ASSERT(this->binaries_exist_on_disk(device));
    std::vector<const ll_api::memory*> binaries;
    uint32_t erisc_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    constexpr auto k_EthDmClassIndex = enchantum::to_underlying(HalProcessorClassType::DM);
    int erisc_id = enchantum::to_underlying(this->config_.processor);
    const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
        device->build_id(), erisc_core_type, k_EthDmClassIndex, erisc_id);
    // TODO: fix when active eth supports relo
    auto load_type =
        MetalContext::instance().hal().get_jit_build_config(erisc_core_type, k_EthDmClassIndex, erisc_id).memory_load;
    const ll_api::memory& binary_mem = llrt::get_risc_binary(
        build_state.get_target_out_path(this->kernel_full_name_), load_type, [this](ll_api::memory& binary_mem) {
            if (tt::tt_metal::MetalContext::instance().rtoptions().get_erisc_iram_enabled() &&
                this->config_.eth_mode != Eth::IDLE) {
                // text_addr and some of span's addr point to IRAM base address.
                // However it need to be placed L1 kernel base address for FW to copy it to IRAM then kick off
                // The kernel can run with IRAM base address once it started.
                binary_mem.set_text_addr(
                    tt::tt_metal::MetalContext::instance().hal().erisc_iram_relocate_dev_addr(
                        (uint64_t)binary_mem.get_text_addr()));
                std::function<void(uint64_t& addr)> update_callback = [](uint64_t& addr) {
                    addr = tt::tt_metal::MetalContext::instance().hal().erisc_iram_relocate_dev_addr(addr);
                };
                binary_mem.update_spans(update_callback);
            }
        });
    binaries.push_back(&binary_mem);
    [[maybe_unused]] uint32_t binary_size = binary_mem.get_packed_size();
    log_debug(LogLoader, "ERISC={}, name={}, size={} (bytes)", erisc_id, this->name(), binary_size);
    this->set_binaries(
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key(), std::move(binaries));
}

void ComputeKernel::read_binaries(IDevice* device) {
    TT_ASSERT(this->binaries_exist_on_disk(device));
    std::vector<const ll_api::memory*> binaries;
    uint32_t tensix_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(this->get_kernel_programmable_core_type());
    uint32_t compute_class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);
    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, compute_class_idx, trisc_id);
        auto load_type = MetalContext::instance()
                             .hal()
                             .get_jit_build_config(tensix_core_type, compute_class_idx, trisc_id)
                             .memory_load;
        const ll_api::memory& binary_mem =
            llrt::get_risc_binary(build_state.get_target_out_path(this->kernel_full_name_), load_type);
        binaries.push_back(&binary_mem);
    }
    this->set_binaries(
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key(), std::move(binaries));
}

bool DataMovementKernel::configure(
    IDevice* device, const CoreCoord& logical_core, uint32_t base_address, const uint32_t offsets[]) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core {}", logical_core.str());
    }
    auto device_id = device->id();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    const ll_api::memory& binary_mem =
        *this->binaries(BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key())[0];
    int riscv_id = static_cast<std::underlying_type_t<DataMovementProcessor>>(this->config_.processor);
    llrt::write_binary_to_address(binary_mem, device_id, worker_core, base_address + offsets[riscv_id]);

    return true;
}

bool EthernetKernel::configure(
    IDevice* device, const CoreCoord& logical_core, uint32_t base_address, const uint32_t offsets[]) const {
    const auto& hal = MetalContext::instance().hal();
    auto device_id = device->id();
    auto ethernet_core = device->ethernet_core_from_logical_core(logical_core);
    const ll_api::memory& binary_mem =
        *this->binaries(BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key())[0];

    if (hal.get_core_kernel_stored_in_config_buffer(this->get_kernel_programmable_core_type())) {
        uint32_t offset_idx = hal.get_processor_index(
            this->get_kernel_programmable_core_type(),
            HalProcessorClassType::DM,
            enchantum::to_underlying(this->config_.processor));
        llrt::write_binary_to_address(binary_mem, device_id, ethernet_core, base_address + offsets[offset_idx]);
    } else {
        const auto erisc_core_index = hal.get_programmable_core_type_index(this->get_kernel_programmable_core_type());
        uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
        int erisc_id = enchantum::to_underlying(this->config_.processor);
        tt::llrt::test_load_write_read_risc_binary(
            binary_mem, device_id, ethernet_core, erisc_core_index, dm_class_idx, erisc_id);
    }

    return true;
}

bool ComputeKernel::configure(
    IDevice* device, const CoreCoord& logical_core, uint32_t base_address, const uint32_t offsets[]) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core {}", logical_core.str());
    }
    auto device_id = device->id();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    const std::vector<const ll_api::memory*>& binaries =
        this->binaries(BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key());
    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        llrt::write_binary_to_address(
            *binaries[trisc_id], device_id, worker_core, base_address + offsets[2 + trisc_id]);
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

}  // namespace tt::tt_metal
