// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include <circular_buffer.hpp>
#include <circular_buffer_constants.h>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>
#include <cstdint>
#include "context/context_types.hpp"
#include "context/metal_env_accessor.hpp"
#include "device/device_manager.hpp"
#include "impl/dispatch/host_device_transfer.hpp"
#include "host_api/helpers.hpp"
#include <global_circular_buffer.hpp>
#include <global_semaphore.hpp>
#include "impl/buffers/global_semaphore_impl.hpp"
#include <host_api.hpp>
#include <experimental/dispatch_context.hpp>
#include <enchantum/enchantum.hpp>
#include <memory>
#include <sub_device_types.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include <variant>

#include "buffer_types.hpp"
#include "circular_buffer_config.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-umd-workload/cluster.hpp>
#include <tt-umd/cluster_descriptor.hpp>
#include <filesystem>
#include "device.hpp"
#include "context/metal_context.hpp"
#include "kernels/kernel.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "device/device_impl.hpp"
#include "hal_types.hpp"
#include "kernel_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include <tt-metalium/experimental/lightmetal/lightmetal_binary.hpp>
#include <tt-metalium/experimental/lightmetal/lightmetal_api.hpp>
#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <fmt/format.h>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <program.hpp>
#include "program/dispatch.hpp"
#include "program/program_impl.hpp"
#include "program/slow_dispatch.hpp"
#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include "impl/buffers/semaphore.hpp"
#include "tracy/Tracy.hpp"
#include <tt-umd/types/xy_pair.hpp>
#include <tt_stl/enum.hpp>
#include <graph_tracking.hpp>
#include <tt_stl/overloaded.hpp>
#include "get_platform_architecture.hpp"
#include "common/tt_backend_api_types.hpp"
#include <experimental/fabric/control_plane.hpp>
#include "impl/buffers/circular_buffer.hpp"
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <internal/service/service_core_manager.hpp>

#ifdef TT_METAL_USE_EMULE
#include "emulated_program_runner.hpp"
#endif
#include "impl/emulation/host_sanitizers.hpp"
#include "impl/emulation/emule_live_ranges.hpp"

namespace tt::tt_metal {
struct RuntimeArgsData;
struct TraceDescriptor;

namespace {

void ValidateLegacyRuntimeArgsAPI(const Program& program, std::string_view api_name) {
    TT_FATAL(
        !program.impl().has_metal2_registry(),
        "{} cannot be used with a Program created from a Metal 2.0 ProgramSpec. "
        "Use experimental::SetProgramRunArgs or experimental::UpdateProgramRunArgs instead.",
        api_name);
}

inline void SetRuntimeArgsImpl(
    const Program& program, KernelHandle kernel_id, const CoreCoord& c, ttsl::Span<const uint32_t> runtime_args) {
    if (!runtime_args.empty()) {
        program.impl().get_kernel(kernel_id)->set_runtime_args(c, runtime_args);
    }
}

inline void SetRuntimeArgsImpl(
    const Program& program,
    KernelHandle kernel_id,
    const CoreRange& core_range,
    ttsl::Span<const uint32_t> runtime_args) {
    if (!runtime_args.empty()) {
        auto kernel = program.impl().get_kernel(kernel_id);
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
            }
        }
    }
}

inline void SetRuntimeArgsImpl(
    const Program& program,
    KernelHandle kernel_id,
    const CoreRangeSet& core_range_set,
    ttsl::Span<const uint32_t> runtime_args) {
    if (!runtime_args.empty()) {
        auto kernel = program.impl().get_kernel(kernel_id);
        for (const auto& core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                    kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
                }
            }
        }
    }
}

}  // namespace

namespace detail {

bool DispatchStateCheck(bool isFastDispatch) {
    MetalContext::instance().device_manager()->check_dispatch_mode(isFastDispatch);
    return isFastDispatch;
}

bool WriteToDeviceDRAMChannel(
    IDevice* device, int dram_channel, uint32_t address, std::span<const std::uint8_t> host_buffer) {
    return slow_dispatch::WriteToDeviceDRAMChannel(*device, dram_channel, address, host_buffer);
}

bool WriteToDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer) {
    return slow_dispatch::WriteToDeviceDRAMChannel(*device, dram_channel, address, host_buffer);
}

bool ReadFromDeviceDRAMChannel(IDevice* device, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer) {
    return slow_dispatch::ReadFromDeviceDRAMChannel(*device, dram_channel, address, host_buffer);
}

bool ReadFromDeviceDRAMChannel(
    IDevice* device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer) {
    return slow_dispatch::ReadFromDeviceDRAMChannel(*device, dram_channel, address, size, host_buffer);
}

bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<const std::uint8_t> host_buffer,
    CoreType core_type) {
    ZoneScoped;
    if constexpr (emule::kEmuleAsanBuild) {
        if (emule::emule_asan_enabled()) {
            emule::LiveL1HostPokeRanges::add(
                device->id(), address, address + static_cast<uint32_t>(host_buffer.size()));
        }
    }
    auto worker_core = device->virtual_core_from_logical_core(logical_core, core_type);
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    metal_ctx.get_cluster().write_core(device->id(), worker_core, host_buffer, address);
    return true;
}

bool WriteToDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type) {
    return WriteToDeviceL1(
        device,
        logical_core,
        address,
        std::span(reinterpret_cast<const std::uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(uint32_t)),
        core_type);
}

bool WriteRegToDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, const uint32_t& regval) {
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    metal_ctx.get_cluster().write_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    std::span<uint8_t> host_buffer,
    CoreType core_type) {
    if constexpr (emule::kEmuleAsanBuild) {
        if (emule::emule_asan_enabled()) {
            emule::LiveL1HostPokeRanges::add(
                device->id(), address, address + static_cast<uint32_t>(host_buffer.size()));
        }
    }
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    metal_ctx.get_cluster().l1_barrier(device->id());
    auto virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    metal_ctx.get_cluster().read_core(
        host_buffer.data(), host_buffer.size(), tt_cxy_pair(device->id(), virtual_core), address);
    return true;
}

bool ReadFromDeviceL1(
    IDevice* device,
    const CoreCoord& logical_core,
    uint32_t address,
    uint32_t size,
    std::vector<uint32_t>& host_buffer,
    CoreType core_type) {
    if constexpr (emule::kEmuleAsanBuild) {
        if (emule::emule_asan_enabled()) {
            emule::LiveL1HostPokeRanges::add(device->id(), address, address + size);
        }
    }
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    metal_ctx.get_cluster().l1_barrier(device->id());
    auto virtual_core = device->virtual_core_from_logical_core(logical_core, core_type);
    host_buffer = metal_ctx.get_cluster().read_core(device->id(), virtual_core, address, size);
    return true;
}

bool ReadRegFromDevice(IDevice* device, const CoreCoord& logical_core, uint32_t address, uint32_t& regval) {
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    metal_ctx.get_cluster().l1_barrier(device->id());
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    metal_ctx.get_cluster().read_reg(&regval, tt_cxy_pair(device->id(), worker_core), address);
    return true;
}

std::string get_platform_architecture_name() { return tt::get_string_lowercase(get_platform_architecture({})); }

IDevice* GetActiveDevice(ChipId device_id) {
    IDevice* device = nullptr;
    if (MetalContext::instance().device_manager()->is_device_active(device_id)) {
        device = MetalContext::instance().device_manager()->get_active_device(device_id);
    }
    return device;
}

}  // namespace detail

namespace experimental {

void ConfigureProgramWithoutLaunch(IDevice* device, Program& program) {
    ZoneScoped;
    // Debug breadcrumbs, one per step: a hang in this path has no Python frame below it, and the
    // step name is what says where it stopped.
    log_debug(tt::LogMetal, "ConfigureProgramWithoutLaunch[{}] enter", device->id());

    // Same prologue as LaunchProgram: compile, finalize offsets, write configs and binaries, then
    // runtime args (configure first: it allocates the scratchpads whose addresses become CRTAs).
    program.impl().compile(device);
    log_debug(tt::LogMetal, "ConfigureProgramWithoutLaunch[{}] compiled", device->id());
    program.impl().finalize_dataflow_buffer_configs();
    if (!program.impl().is_finalized()) {
        program.impl().finalize_offsets(device);
    }
    log_debug(tt::LogMetal, "ConfigureProgramWithoutLaunch[{}] finalized", device->id());
    slow_dispatch::ConfigureDeviceWithProgram(*device, program, /*force_slow_dispatch=*/false);
    log_debug(tt::LogMetal, "ConfigureProgramWithoutLaunch[{}] configured", device->id());
    slow_dispatch::WriteRuntimeArgsToDevice(*device, program, /*force_slow_dispatch=*/false);
    log_debug(tt::LogMetal, "ConfigureProgramWithoutLaunch[{}] rtargs written", device->id());

    auto device_id = device->id();
    MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    metal_ctx.get_cluster().dram_barrier(device_id);
    metal_ctx.get_cluster().l1_barrier(device_id);
    log_debug(tt::LogMetal, "ConfigureProgramWithoutLaunch[{}] barriers done", device->id());

    // Only the launch message: it names kernel_config_base and the text offsets. send_go=false
    // leaves firmware parked.
    const auto& hal = metal_ctx.hal();
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.impl().logical_cores();
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            auto* kg = program.impl().kernels_on_core(logical_core, programmable_core_type_index);
            dev_msgs::launch_msg_t local_launch_msg = kg->launch_msg;
            local_launch_msg.view().kernel_config().host_assigned_id() = program.get_runtime_id();
            auto physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
            tt::llrt::write_launch_msg_to_core(
                MetalEnvAccessor(metal_ctx.get_env()).impl(),
                device_id,
                physical_core,
                local_launch_msg.view(),
                kg->go_msg.view(),
                /*send_go=*/false);
        }
    }
    log_debug(tt::LogMetal, "ConfigureProgramWithoutLaunch[{}] launch msgs written, done", device_id);
}

void DispatchCompiledProgramToDevice(IDevice* device, Program& program) {
    ZoneScoped;

    auto device_id = device->id();
    MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));

#ifdef TT_METAL_USE_EMULE
    if (metal_ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Emule) {
        // Emule lazily JIT-compiles inside execute_program_emulated, so is_finalized()/is_compiled()
        // are never set — skip both asserts (HW-only) and run synchronously, mirroring LaunchProgram.
        // Configure before writing runtime args: ConfigureDeviceWithProgram allocates the ephemeral
        // scratchpad buffers whose addresses are passed as common runtime args, so the write must see
        // them (matches LaunchProgram and the non-emule path below).
        slow_dispatch::ConfigureDeviceWithProgram(*device, program, /*force_slow_dispatch=*/false);
        slow_dispatch::WriteRuntimeArgsToDevice(*device, program, /*force_slow_dispatch=*/false);
        emule::execute_program_emulated(device, program);
        return;
    }
#endif

    // Verify program was prepared by prior LaunchProgram call
    TT_FATAL(
        program.impl().is_finalized(),
        "Program must be finalized before calling DispatchCompiledProgramToDevice (target device {}). "
        "Call LaunchProgram on another device first.",
        device_id);

    TT_FATAL(
        program.impl().is_compiled(),
        "Program must be compiled on at least one device before calling DispatchCompiledProgramToDevice (target device "
        "{}). "
        "Call LaunchProgram on another device first.",
        device_id);

    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.impl().logical_cores();
    TT_FATAL(
        !logical_cores_used_in_program.empty(),
        "Program has no logical cores to dispatch to device {}. Ensure the program has kernels.",
        device_id);

    // First configure (allocate buffers + write configs/binaries), then write runtime args.
    // This allows us to allocate ephemeral scratchpad buffers, and pass their locations as implicit CRTAs.
    slow_dispatch::ConfigureDeviceWithProgram(*device, program, /*force_slow_dispatch=*/false);
    slow_dispatch::WriteRuntimeArgsToDevice(*device, program, /*force_slow_dispatch=*/false);

    metal_ctx.get_cluster().dram_barrier(device_id);
    metal_ctx.get_cluster().l1_barrier(device_id);
    const auto& hal = metal_ctx.hal();
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            auto* kg = program.impl().kernels_on_core(logical_core, programmable_core_type_index);

            // Use a thread-local copy of launch_msg to avoid racing on the shared KernelGroup state
            dev_msgs::launch_msg_t local_launch_msg = kg->launch_msg;
            local_launch_msg.view().kernel_config().host_assigned_id() = program.get_runtime_id();

            auto physical_core = device->virtual_core_from_logical_core(logical_core, core_type);
            tt::llrt::write_launch_msg_to_core(
                MetalEnvAccessor(metal_ctx.get_env()).impl(),
                device_id,
                physical_core,
                local_launch_msg.view(),
                kg->go_msg.view(),
                hal.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
        }
    }
}

struct CapturedKernelConfig::Impl {
    uint32_t kernel_config_base;
    uint32_t kernel_config_size;
    std::vector<uint8_t> launch_kernel_config;
};

CapturedKernelConfig::CapturedKernelConfig(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

uint32_t CapturedKernelConfig::kernel_config_base() const { return impl_->kernel_config_base; }

uint32_t CapturedKernelConfig::kernel_config_size() const { return impl_->kernel_config_size; }

const std::vector<uint8_t>& CapturedKernelConfig::launch_kernel_config() const { return impl_->launch_kernel_config; }

CapturedKernelConfig CaptureKernelConfig(IDevice* device, const CoreCoord& logical_core) {
    // Decode through the generated view firmware compiles against, so the layout is stated once.
    const auto& hal = MetalContext::instance(extract_context_id(device)).hal();
    const auto core_type = HalProgrammableCoreType::TENSIX;
    auto factory = hal.get_dev_msgs_factory(core_type);
    auto launch = factory.create<dev_msgs::launch_msg_t>();

    std::vector<uint32_t> raw;
    detail::ReadFromDeviceL1(
        device,
        logical_core,
        hal.get_dev_addr(core_type, HalL1MemAddrType::MAILBOX) +
            factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::launch),
        launch.size(),
        raw);

    auto view = factory.create_view<dev_msgs::launch_msg_t>(reinterpret_cast<const std::byte*>(raw.data()));
    auto kc = view.kernel_config();

    uint32_t kernel_config_size = 0;
    for (uint32_t i = 0; i < kc.kernel_text_offset().size(); i++) {
        kernel_config_size = std::max(kernel_config_size, kc.kernel_text_offset()[i] + kc.kernel_text_size()[i]);
    }

    std::vector<uint8_t> launch_kernel_config(kc.size());
    std::memcpy(launch_kernel_config.data(), kc.data(), kc.size());
    return CapturedKernelConfig(std::make_shared<CapturedKernelConfig::Impl>(
        kc.kernel_config_base()[0], kernel_config_size, std::move(launch_kernel_config)));
}

}  // namespace experimental

namespace detail {

void ReleaseOwnership() {
    experimental::DispatchContext::get().reset();
    MetalContext::destroy_all_instances();
}

void print_page(
    uint32_t dev_page_id,
    CoreCoord core,
    uint32_t host_page_id,
    CoreCoord noc_coordinates,
    uint32_t l1_address,
    uint32_t bank_id,
    const std::vector<uint32_t>& page) {
    std::cout << "dev_page_index " << dev_page_id << " on core " << core.str() << std::endl;
    std::cout << "host_page_index " << host_page_id << std::endl;
    std::cout << "noc coordinates " << noc_coordinates.str() << std::endl;
    std::cout << "l1_address " << l1_address << std::endl;
    std::cout << "bank id " << bank_id << std::endl;

    std::cout << "0x";
    for (auto entry : page) {
        std::cout << std::hex << entry << std::dec;
    }
    std::cout << std::dec << std::endl;
}

void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer) {
    slow_dispatch::WriteToBuffer(buffer, host_buffer);
}

void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer) { slow_dispatch::ReadFromBuffer(buffer, host_buffer); }

void ReadShard(Buffer& buffer, uint8_t* host_buffer, const uint32_t& core_id) {
    slow_dispatch::ReadShard(buffer, host_buffer, core_id);
}

void WaitProgramDone(IDevice* device, Program& program, bool read_device_profiler_results) {
    slow_dispatch::WaitProgramDone(*device, program);
    if (read_device_profiler_results) {
        detail::ReadDeviceProfilerResults(device);
    }
}

bool ConfigureDeviceWithProgram(IDevice* device, Program& program, bool force_slow_dispatch) {
    slow_dispatch::ConfigureDeviceWithProgram(*device, program, force_slow_dispatch);
    return true;
}

void WriteRuntimeArgsToDevice(IDevice* device, Program& program, bool force_slow_dispatch) {
    slow_dispatch::WriteRuntimeArgsToDevice(*device, program, force_slow_dispatch);
}

}  // namespace detail

namespace experimental::core_subset_write {

void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer, const CoreRangeSet& logical_core_filter) {
    slow_dispatch::WriteToBuffer(buffer, host_buffer, logical_core_filter);
}

}  // namespace experimental::core_subset_write

size_t GetNumAvailableDevices() { return MetalContext::instance().get_cluster().number_of_user_devices(); }

bool IsGalaxyCluster() { return MetalContext::instance().get_cluster().is_galaxy_cluster(); }

size_t GetNumPCIeDevices() { return MetalContext::instance().get_cluster().number_of_pci_devices(); }

ChipId GetPCIeDeviceID(ChipId device_id) {
    return MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
}

ClusterType GetClusterType() { return MetalContext::instance().get_cluster().get_cluster_type(); }

std::string SerializeClusterDescriptor() {
    // Serialize the descriptor the cluster already holds. The cluster makes the
    // mock-vs-real (emule) decision once when it builds the descriptor at init
    // (see tt_cluster.cpp, gated on rtoptions.get_mock_enabled()), so this layer
    // must not re-create it: a fresh real-UMD create here would open the physical
    // card and bypass the emulator. Assumes the cluster is initialized (this API's
    // only caller, ttnn serialize_cluster_descriptor, runs in an active session).
    return MetalContext::instance().get_cluster().get_cluster_desc()->serialize_to_file().string();
}

// This function is used to set a default root directory for the tt_metal library.
void SetRootDir(const std::string& root_dir) { tt::llrt::RunTimeOptions::set_root_dir(root_dir); }

Program CreateProgram() { return Program(); }

Program CreateProgram(distributed::MeshDevice& mesh_device) {
    return Program(std::make_shared<detail::ProgramImpl>(extract_context_id(&mesh_device)));
}

void ValidateKernelConfigDefines(const std::map<std::string, std::string>& defines) {
    for (const auto& [key, value] : defines) {
        if (value.find('\0') != std::string::npos) {
            throw std::invalid_argument("Define value for key '" + key + "' contains null character");
        }
    }
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config) {
    std::visit([](const auto& cfg) { ValidateKernelConfigDefines(cfg.defines); }, config);

    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_path(program.impl().get_context_id(), file_name);
    KernelHandle kernel = std::visit(
        [&](const auto& cfg) -> KernelHandle {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                return program.impl().builder().create_data_movement_kernel(kernel_src, core_ranges, cfg);
            } else {
                return program.impl().builder().create_compute_kernel(kernel_src, core_ranges, cfg);
            }
        },
        config);

    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> cfg_variant = std::visit(
        [&](const auto& cfg) -> std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> { return cfg; },
        config);
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureCreateKernel, kernel, program, file_name, core_spec, cfg_variant);

    return kernel;
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const EthernetConfig& config) {
    ValidateKernelConfigDefines(config.defines);
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_path(program.impl().get_context_id(), file_name);
    return program.impl().builder().create_ethernet_kernel(kernel_src, core_ranges, config);
}

KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config) {
    std::visit([](const auto& cfg) { ValidateKernelConfigDefines(cfg.defines); }, config);
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_source(kernel_src_code);
    return std::visit(
        [&](const auto& cfg) -> KernelHandle {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                return program.impl().builder().create_data_movement_kernel(kernel_src, core_ranges, cfg);
            } else {
                return program.impl().builder().create_compute_kernel(kernel_src, core_ranges, cfg);
            }
        },
        config);
}

KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const EthernetConfig& config) {
    ValidateKernelConfigDefines(config.defines);
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_source(kernel_src_code);
    return program.impl().builder().create_ethernet_kernel(kernel_src, core_ranges, config);
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DramConfig& config) {
    ValidateKernelConfigDefines(config.defines);
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_path(program.impl().get_context_id(), file_name);
    return program.impl().builder().create_dram_kernel(kernel_src, core_ranges, config);
}

KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DramConfig& config) {
    ValidateKernelConfigDefines(config.defines);
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_source(kernel_src_code);
    return program.impl().builder().create_dram_kernel(kernel_src, core_ranges, config);
}

CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    auto cb_handle = program.impl().builder().add_circular_buffer(core_ranges, config);
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureCreateCircularBuffer, cb_handle, program, core_spec, config);
    return cb_handle;
}

const CircularBufferConfig& GetCircularBufferConfig(Program& program, CBHandle cb_handle) {
    return program.impl().builder().get_circular_buffer(cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program& program, CBHandle cb_handle, uint32_t total_size) {
    program.impl().builder().update_circular_buffer_total_size(cb_handle, total_size);
}

void UpdateCircularBufferPageSize(Program& program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    program.impl().builder().update_circular_buffer_page_size(cb_handle, buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const Buffer& buffer) {
    program.impl().builder().update_dynamic_circular_buffer_address(cb_handle, buffer);
}

void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t address_offset) {
    program.impl().builder().update_dynamic_circular_buffer_address(cb_handle, buffer, address_offset);
}

void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const MeshTensor& tensor) {
    program.impl().builder().update_dynamic_circular_buffer_address(cb_handle, tensor);
}

void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t total_size) {
    program.impl().builder().update_dynamic_circular_buffer_address_and_total_size(cb_handle, buffer, total_size);
}

void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const MeshTensor& tensor, uint32_t total_size) {
    program.impl().builder().update_dynamic_circular_buffer_address_and_total_size(cb_handle, tensor, total_size);
}

uint32_t CreateSemaphore(
    Program& program, const std::variant<CoreRange, CoreRangeSet>& core_spec, uint32_t initial_value) {
    return CreateSemaphore(program, core_spec, initial_value, CoreType::WORKER);
}

uint32_t CreateSemaphore(
    Program& program,
    const std::variant<CoreRange, CoreRangeSet>& core_spec,
    uint32_t initial_value,
    CoreType core_type) {
    return program.impl().builder().create_semaphore(core_spec, initial_value, core_type);
}

GlobalSemaphore CreateGlobalSemaphore(
    distributed::MeshDevice& device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type) {
    return detail::Metal1ProgramBuilder::create_global_semaphore(device, std::move(cores), initial_value, buffer_type);
}

void AssignGlobalBufferToProgram(const std::shared_ptr<Buffer>& buffer, Program& program) {
    program.impl().builder().assign_global_buffer(buffer);
}

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    ttsl::Span<const uint32_t> runtime_args) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureSetRuntimeArgsUint32, program, kernel_id, core_spec, runtime_args);
    ValidateLegacyRuntimeArgsAPI(program, "SetRuntimeArgs");
    std::visit([&](auto&& core_spec) { SetRuntimeArgsImpl(program, kernel_id, core_spec, runtime_args); }, core_spec);
}

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    std::initializer_list<uint32_t> runtime_args) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureSetRuntimeArgsUint32, program, kernel_id, core_spec, runtime_args);
    ZoneScoped;
    ValidateLegacyRuntimeArgsAPI(program, "SetRuntimeArgs");
    std::visit([&](auto&& core_spec) { SetRuntimeArgsImpl(program, kernel_id, core_spec, runtime_args); }, core_spec);
}

void SetRuntimeArgs(
    const Program& program,
    KernelHandle kernel,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::vector<uint32_t>>& runtime_args) {
    ZoneScoped;
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureSetRuntimeArgsUint32VecPerCore, program, kernel, core_spec, runtime_args);
    ValidateLegacyRuntimeArgsAPI(program, "SetRuntimeArgs");
    TT_FATAL(
        core_spec.size() == runtime_args.size(),
        "Mismatch between number of cores {} and number of runtime args {} getting updated",
        core_spec.size(),
        runtime_args.size());
    auto k = program.impl().get_kernel(kernel);
    for (size_t i = 0; i < core_spec.size(); i++) {
        k->set_runtime_args(core_spec[i], runtime_args[i]);
    }
}

void SetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id, ttsl::Span<const uint32_t> runtime_args) {
    ZoneScoped;
    ValidateLegacyRuntimeArgsAPI(program, "SetCommonRuntimeArgs");
    if (!runtime_args.empty()) {
        program.impl().get_kernel(kernel_id)->set_common_runtime_args(runtime_args);
    }
}

void SetCommonRuntimeArgs(
    const Program& program, KernelHandle kernel_id, std::initializer_list<uint32_t> runtime_args) {
    ZoneScoped;
    ValidateLegacyRuntimeArgsAPI(program, "SetCommonRuntimeArgs");
    if (runtime_args.size() != 0) {
        program.impl().get_kernel(kernel_id)->set_common_runtime_args(runtime_args);
    }
}

RuntimeArgsData& GetRuntimeArgs(const Program& program, KernelHandle kernel_id, const CoreCoord& logical_core) {
    ValidateLegacyRuntimeArgsAPI(program, "GetRuntimeArgs");
    return program.impl().get_kernel(kernel_id)->runtime_args_data(logical_core);
}

std::vector<std::vector<RuntimeArgsData>>& GetRuntimeArgs(const Program& program, KernelHandle kernel_id) {
    ValidateLegacyRuntimeArgsAPI(program, "GetRuntimeArgs");
    return program.impl().get_kernel(kernel_id)->runtime_args_data();
}

RuntimeArgsData& GetCommonRuntimeArgs(const Program& program, KernelHandle kernel_id) {
    ValidateLegacyRuntimeArgsAPI(program, "GetCommonRuntimeArgs");
    return program.impl().get_kernel(kernel_id)->common_runtime_args_data();
}

namespace experimental::lightmetal {

// This is nop if compile time define not set.
void LightMetalBeginCapture() {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    log_debug(tt::LogMetalTrace, "Begin LightMetalBinary Capture");
    auto& lm_capture_ctx = LightMetalCaptureContext::get();
    lm_capture_ctx.reset();            // Clear previous traces if any, ensure tracing disabled
    lm_capture_ctx.set_tracing(true);  // Enable tracing
#else
    log_warning(tt::LogMetalTrace, "TT_ENABLE_LIGHT_METAL_TRACE!=1, ignoring LightMetalBeginCapture()");
#endif
}

// This is nop if compile time define not set, return empty vector.
LightMetalBinary LightMetalEndCapture() {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    log_debug(tt::LogMetalTrace, "End LightMetalBinary Capture");
    auto& lm_capture_ctx = LightMetalCaptureContext::get();
    TT_ASSERT(lm_capture_ctx.is_tracing(), "Light Metal Capture was not enabled.");
    lm_capture_ctx.set_tracing(false);  // Disable tracing
    return lm_capture_ctx.create_light_metal_binary();
#else
    log_warning(tt::LogMetalTrace, "TT_ENABLE_LIGHT_METAL_TRACE!=1, ignoring LightMetalEndCapture()");
    return {};
#endif
}

}  // namespace experimental::lightmetal

void PushCurrentCommandQueueIdForThread(uint8_t cq_id) {
    auto& cq_stack = MetalContext::instance().get_command_queue_id_stack_for_thread();
    cq_stack.push_back(cq_id);
}

uint8_t PopCurrentCommandQueueIdForThread() {
    auto& cq_stack = MetalContext::instance().get_command_queue_id_stack_for_thread();
    TT_FATAL(!cq_stack.empty(), "Current command queue id stack is empty!");
    uint8_t cq_id = cq_stack.back();
    cq_stack.pop_back();
    return cq_id;
}

uint8_t GetCurrentCommandQueueIdForThread() {
    // TODO: Make GetCurrentCommandQueueIdForThread work for non-default contexts
    // https://github.com/tenstorrent/tt-metal/issues/39819
    if (!MetalContext::instance_exists(DEFAULT_CONTEXT_ID)) {
        return 0;
    }
    const auto& cq_stack = MetalContext::instance().get_command_queue_id_stack_for_thread();
    if (cq_stack.empty()) {
        return 0;
    }
    return cq_stack.back();
}

namespace experimental {

CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    return program.impl().builder().add_circular_buffer(core_ranges, config, global_circular_buffer);
}

void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer) {
    program.impl().builder().update_dynamic_circular_buffer_address(cb_handle, global_circular_buffer);
}

PrecompiledKernelNotFoundError::PrecompiledKernelNotFoundError(
    std::string kernel_name,
    size_t compile_hash,
    std::string precompiled_dir,
    PrecompiledKernelConfig::FallbackPolicy fallback_policy) :
    std::runtime_error(fmt::format(
        "Precompiled kernel binary not found. Kernel: \"{}\", compile_hash: {:#x}, searched in: \"{}\". "
        "Either build/install the offline binaries there, set PrecompiledKernelConfig::fallback_policy to "
        "JitCompile, or catch PrecompiledKernelNotFoundError for details.",
        kernel_name,
        compile_hash,
        precompiled_dir)),
    kernel_name_(std::move(kernel_name)),
    compile_hash_(compile_hash),
    precompiled_dir_(std::move(precompiled_dir)),
    fallback_policy_(fallback_policy) {}

KernelHandle CreateKernelFromPrecompiled(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const PrecompiledKernelConfig& precompiled_config) {
    std::visit([](const auto& cfg) { ValidateKernelConfigDefines(cfg.defines); }, config);

    KernelHandle kernel_handle = CreateKernel(program, file_name, core_spec, config);
    program.impl().get_kernel(kernel_handle)->set_precompiled_config(precompiled_config);
    return kernel_handle;
}

}  // namespace experimental

}  // namespace tt::tt_metal
