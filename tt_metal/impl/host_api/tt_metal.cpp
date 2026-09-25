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
#include <umd/device/cluster.hpp>
#include <umd/device/cluster_descriptor.hpp>
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
#include <umd/device/types/xy_pair.hpp>
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

struct DataMovementConfigStatus {
    bool riscv0_in_use;
    bool riscv1_in_use;
    bool noc0_in_use;
    bool noc1_in_use;
};

DataMovementConfigStatus CheckDataMovementConfig(
    const HalProgrammableCoreType& programmable_core, Program& program, const CoreRangeSet& core_ranges) {
    DataMovementConfigStatus data_movement_config_status{
        .riscv0_in_use = false, .riscv1_in_use = false, .noc0_in_use = false, .noc1_in_use = false};

    auto set_global_and_local_noc_usage =
        [&](const std::shared_ptr<Kernel>& kernel, bool& local_noc0_usage, bool& local_noc1_usage) {
            int noc_value;
            switch (programmable_core) {
                case HalProgrammableCoreType::TENSIX:
                    noc_value = enchantum::to_underlying(std::get<DataMovementConfig>(kernel->config()).noc);
                    break;
                case HalProgrammableCoreType::ACTIVE_ETH:
                case HalProgrammableCoreType::IDLE_ETH:
                    noc_value = enchantum::to_underlying(std::get<EthernetConfig>(kernel->config()).noc);
                    break;
                default:
                    TT_THROW(
                        "Checking NoC and DataMovementProcessor is unsupported for programmable core {}",
                        enchantum::to_string(programmable_core));
            }
            local_noc0_usage = noc_value == 0;
            local_noc1_usage = noc_value == 1;
            data_movement_config_status.noc0_in_use = local_noc0_usage;
            data_movement_config_status.noc1_in_use = local_noc1_usage;
        };

    const auto& hal = MetalContext::instance(program.impl().get_context_id()).hal();
    for (const auto& core_range : core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                const KernelGroup* kernel_group = program.impl().kernels_on_core(
                    CoreCoord(x, y), hal.get_programmable_core_type_index(programmable_core));
                if (kernel_group != nullptr) {
                    bool local_noc0_in_use = false;
                    bool local_noc1_in_use = false;
                    bool has_dm0 = false;
                    bool has_dm1 = false;
                    for (auto kernel_id : kernel_group->kernel_ids) {
                        const auto kernel = program.impl().get_kernel(kernel_id);
                        if (kernel->get_kernel_processor_class() == HalProcessorClassType::DM) {
                            switch (kernel->get_kernel_processor_type(0)) {
                                case 0:
                                    has_dm0 = true;
                                    data_movement_config_status.riscv0_in_use = true;
                                    set_global_and_local_noc_usage(kernel, local_noc0_in_use, local_noc1_in_use);
                                    break;
                                case 1:
                                    has_dm1 = true;
                                    data_movement_config_status.riscv1_in_use = true;
                                    set_global_and_local_noc_usage(kernel, local_noc0_in_use, local_noc1_in_use);
                                    break;
                                default: TT_THROW("Unknown DataMovementProcessor type"); break;
                            }
                        }
                    }
                    if (has_dm0 and has_dm1) {
                        TT_FATAL(
                            local_noc0_in_use and local_noc1_in_use,
                            "Illegal NOC usage: data movement kernels on logical core {} cannot use the same NOC, "
                            "doing so results in hangs!",
                            CoreCoord(x, y).str());
                    }
                }
            }
        }
    }

    return data_movement_config_status;
}

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
        emule::check_host_l1_alignment(device, address, static_cast<uint32_t>(host_buffer.size()), "WriteToDeviceL1");
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
        emule::check_host_l1_alignment(device, address, static_cast<uint32_t>(host_buffer.size()), "ReadFromDeviceL1");
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
        emule::check_host_l1_alignment(device, address, size, "ReadFromDeviceL1");
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

std::map<ChipId, IDevice*> CreateDevices(
    const std::vector<ChipId>& device_ids,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& /*l1_bank_remap*/,
    const size_t worker_l1_size,
    bool init_profiler,
    [[maybe_unused]] bool ignored,
    bool initialize_fabric_and_dispatch_fw) {
    ZoneScoped;
    bool is_galaxy = MetalContext::instance().get_cluster().is_galaxy_cluster();
    MetalContext::instance().initialize_device_manager(
        device_ids,
        num_hw_cqs,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        {},
        worker_l1_size,
        init_profiler,
        initialize_fabric_and_dispatch_fw);

    const auto devices = MetalContext::instance().device_manager()->get_all_active_devices();
    std::map<ChipId, IDevice*> ret_devices;
    // Only include the mmio device in the active devices set returned to the caller if we are not running
    // on a Galaxy cluster.
    // On Galaxy, gateway (mmio devices) cannot run compute workloads.

    for (IDevice* dev : devices) {
        if (is_galaxy and dev->is_mmio_capable()) {
            continue;
        }
        ret_devices.insert({dev->id(), dev});
    }

    return ret_devices;
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

void CloseDevices(const std::map<ChipId, IDevice*>& devices) {
    std::vector<IDevice*> devices_to_close;
    devices_to_close.reserve(devices.size());
    for (const auto& [id, device] : devices) {
        devices_to_close.push_back(device);
    }
    if (devices.empty()) {
        MetalContext::instance().device_manager()->close_devices(devices_to_close);
    } else {
        MetalContext::instance(extract_context_id(devices.begin()->second))
            .device_manager()
            ->close_devices(devices_to_close);
    }
}

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

void LaunchProgram(
    IDevice* device, const std::shared_ptr<Program>& program, bool wait_until_cores_done, bool force_slow_dispatch) {
    LaunchProgram(device, *program, wait_until_cores_done, force_slow_dispatch);
}

void LaunchProgram(IDevice* device, Program& program, bool wait_until_cores_done, bool force_slow_dispatch) {
    slow_dispatch::LaunchProgramAsync(*device, program, force_slow_dispatch);
#ifdef TT_METAL_USE_EMULE
    // Emulated mode executes synchronously inside slow_dispatch::LaunchProgramAsync.
    if (MetalContext::instance(extract_context_id(device)).get_cluster().get_target_device_type() ==
        tt::TargetDevice::Emule) {
        return;
    }
#endif
    if (wait_until_cores_done) {
        slow_dispatch::WaitProgramDone(*device, program);
        detail::ReadDeviceProfilerResults(device);
    }
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

void CompileProgram(IDevice* device, Program& program, bool force_slow_dispatch) {
    program.impl().compile(device, force_slow_dispatch);
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

IDevice* CreateDevice(
    ChipId device_id,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& l1_bank_remap,
    const size_t worker_l1_size) {
    ZoneScoped;

    // MMIO devices do not support dispatch on galaxy clusters.
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        TT_FATAL(
            !(MetalContext::instance().get_cluster().is_galaxy_cluster() &&
              MetalContext::instance().get_cluster().get_cluster_desc()->is_chip_mmio_capable(device_id)),
            "Galaxy clusters do not support dispatch on MMIO devices. Use "
            "distributed::MeshDevice::create_unit_meshes to open all devices for dispatch.");
    }

    // This API may not be used to create single remote device or multi chip clusters
    // MeshDevice should be used instead to ensure proper initialization and teardown.
    TT_FATAL(
        MetalContext::instance().get_cluster().get_associated_mmio_device(device_id) == device_id,
        "CreateDevice(device_id={}) may only be used for opening single MMIO capable devices. For multi chip clusters, "
        "use distributed::MeshDevice::create_unit_meshes().",
        device_id);

    MetalContext::instance().initialize_device_manager(
        {device_id}, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_config, l1_bank_remap, worker_l1_size);
    auto* dev = MetalContext::instance().device_manager()->get_active_device(device_id);
    return dev;
}

IDevice* CreateDeviceMinimal(
    ChipId device_id, const uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config) {
    ZoneScoped;
    auto& ctx = MetalContext::instance();  // runtime state
    auto& env = ctx.get_env();             // default low level state
    ctx.initialize(dispatch_core_config, num_hw_cqs, {}, DEFAULT_L1_SMALL_SIZE, true);
    auto* dev =
        new Device(&env, &ctx, device_id, num_hw_cqs, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, {}, true);
    auto& control_plane = MetalEnvAccessor(env).impl().get_control_plane();
    MetalEnvAccessor(env).impl().get_cluster().set_internal_routing_info_for_ethernet_cores(control_plane, true);
    return dev;
}

bool CloseDevice(IDevice* device) {
    ZoneScoped;
    auto device_id = device->id();
    MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));

    // This API may not be used to close a single remote device or multi-chip cluster.
    // MeshDevice RAII should be used instead to ensure proper teardown.
    TT_FATAL(
        metal_ctx.get_cluster().get_associated_mmio_device(device_id) == device_id,
        "CloseDevice(device_id={}) may only be used for closing single MMIO capable devices. For multi chip clusters, "
        "use MeshDevice RAII or MeshDevice::close().",
        device_id);

    return metal_ctx.device_manager()->close_device(device_id);
}

Program CreateProgram() { return Program(); }

KernelHandle CreateDataMovementKernel(
    Program& program,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const DataMovementConfig& config) {
    const DataMovementConfigStatus& data_movement_config_status =
        CheckDataMovementConfig(HalProgrammableCoreType::TENSIX, program, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    std::string kernel_name;
    if (kernel_src.source_type_ == KernelSource::FILE_PATH) {
        kernel_name = kernel_src.source_;
    } else {
        TT_FATAL(kernel_src.source_type_ == KernelSource::SOURCE_CODE, "Unsupported kernel source type!");
        kernel_name = "kernel";
    }

    TT_FATAL(
        !(are_both_riscv_in_use),
        "DataMovementKernel creation failure: Cannot create data movement kernel for {} across specified "
        "cores because both data movement processors are in use!",
        kernel_name);
    TT_FATAL(
        !(are_both_noc_in_use),
        "DataMovementKernel creation failure: Cannot create data movement kernels for {} across specified "
        "cores because both NOCs are in use!",
        kernel_name);

    TT_FATAL(
        config.processor == DataMovementProcessor::RISCV_0 || config.processor == DataMovementProcessor::RISCV_1,
        "DataMovementKernel creation failure: Data movement kernels can only be created on DM0 or DM1 processors.");

    const ContextId context_id = program.impl().get_context_id();
    std::shared_ptr<Kernel> kernel =
        std::make_shared<DataMovementKernel>(context_id, kernel_src, core_range_set, config);

    // Inject all fabric-related defines (routing mode, UDM mode, dynamic header sizes, etc.)
    const auto fabric_defines = MetalContext::instance(context_id).get_control_plane().get_fabric_kernel_defines();
    if (!fabric_defines.empty()) {
        kernel->add_defines(fabric_defines);
    }

    return program.impl().add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateComputeKernel(
    Program& program, const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const ComputeConfig& config) {
    std::shared_ptr<Kernel> kernel =
        std::make_shared<ComputeKernel>(program.impl().get_context_id(), kernel_src, core_range_set, config);
    return program.impl().add_kernel(kernel, HalProgrammableCoreType::TENSIX);
}

KernelHandle CreateEthernetKernel(
    Program& program,
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const EthernetConfig& config) {
    HalProgrammableCoreType eth_core_type =
        config.eth_mode == Eth::IDLE ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    const DataMovementConfigStatus& data_movement_config_status =
        CheckDataMovementConfig(eth_core_type, program, core_range_set);
    const bool are_both_riscv_in_use =
        data_movement_config_status.riscv0_in_use && data_movement_config_status.riscv1_in_use;
    const bool are_both_noc_in_use = data_movement_config_status.noc0_in_use && data_movement_config_status.noc1_in_use;

    TT_FATAL(
        config.processor == DataMovementProcessor::RISCV_0 || config.processor == DataMovementProcessor::RISCV_1,
        "EthernetKernel creation failure: Ethernet kernels can only be created on DM0 or DM1 processors.");

    const ContextId context_id = program.impl().get_context_id();
    auto& metal_context = MetalContext::instance(context_id);
    std::shared_ptr<Kernel> kernel = std::make_shared<EthernetKernel>(context_id, kernel_src, core_range_set, config);

    // Inject all fabric-related defines (routing mode, UDM mode, dynamic header sizes, etc.)
    const auto fabric_defines = metal_context.get_control_plane().get_fabric_kernel_defines();
    if (!fabric_defines.empty()) {
        kernel->add_defines(fabric_defines);
    }

    // Disable watcher on ethernet cores if requested
    const auto& rt_options = metal_context.rtoptions();
    if (rt_options.watcher_eth_disabled()) {
        kernel->add_defines({{"FORCE_WATCHER_OFF", "1"}});
    }

    TT_FATAL(
        ttsl::as_underlying_type<DataMovementProcessor>(config.processor) <
            metal_context.hal().get_num_risc_processors(eth_core_type),
        "EthernetKernel creation failure: {} kernel cannot target processor {} because Ethernet core only has {} "
        "processors. "
        "Update DataMovementProcessor in the config.",
        kernel->name(),
        enchantum::to_string(config.processor),
        metal_context.hal().get_num_risc_processors(eth_core_type));
    TT_FATAL(
        !(are_both_riscv_in_use),
        "EthernetKernel creation failure: Cannot create data movement kernel for {} across specified "
        "cores because both data movement processors are in use!",
        kernel->name());
    TT_FATAL(
        !(are_both_noc_in_use),
        "EthernetKernel creation failure: Cannot create data movement kernels for {} across specified "
        "cores because both NOCs are in use!",
        kernel->name());

    //
    // Valid configurations for Blackhole ERISC
    //
    // |                | Valid NOC Configuration     |                             |
    // |----------------|-----------------------------|-----------------------------|
    // | **ERISC Mode** | **Physical ERISC0**         | **Physical ERISC1**         |
    // | Single         | Not enabled for dispatch    | Dedicated NOC1              |
    // | Dual           | Dedicated NOC0, Dynamic NOC | Dedicated NOC1, Dynamic NOC |
    //
    if (!metal_context.hal().get_eth_fw_is_cooperative() && config.eth_mode != Eth::IDLE &&
        config.noc_mode != NOC_MODE::DM_DYNAMIC_NOC) {
        bool is_dual_erisc_mode = metal_context.rtoptions().get_enable_2_erisc_mode();
        bool is_erisc0 = (config.processor == DataMovementProcessor::RISCV_0);
        bool is_erisc1 = (config.processor == DataMovementProcessor::RISCV_1);

        if (is_dual_erisc_mode) {
            // Dual ERISC mode: ERISC0 uses NOC0, ERISC1 uses NOC1 (when in dedicated mode)
            if (is_erisc0) {
                TT_FATAL(
                    config.noc == NOC::NOC_0,
                    "EthernetKernel creation failure: In dual ERISC mode, ERISC0 in dedicated mode must use NOC0. "
                    "Kernel: {}, Current NOC: {}, Required NOC: NOC_0. Use Dynamic NOC mode for flexible routing.",
                    kernel->name(),
                    config.noc);
            } else if (is_erisc1) {
                TT_FATAL(
                    config.noc == NOC::NOC_1,
                    "EthernetKernel creation failure: In dual ERISC mode, ERISC1 in dedicated mode must use NOC1. "
                    "Kernel: {}, Current NOC: {}, Required NOC: NOC_1. Use Dynamic NOC mode for flexible routing.",
                    kernel->name(),
                    config.noc);
            }
        } else {
            // ERISC1 must use NOC1 in dedicated mode
            TT_FATAL(
                config.noc == NOC::NOC_1,
                "EthernetKernel creation failure: In single ERISC mode, ERISC0 must use NOC1. "
                "Kernel: {}, Current NOC: {}, Required NOC: NOC_1.",
                kernel->name(),
                config.noc);
        }
    }

    // Dynamic noc is not supported on single erisc mode
    if (!metal_context.hal().get_eth_fw_is_cooperative() && !metal_context.rtoptions().get_enable_2_erisc_mode()) {
        TT_FATAL(
            config.noc_mode == NOC_MODE::DM_DEDICATED_NOC,
            "EthernetKernel creation failure: Dynamic NOC is not supported on single ERISC mode. "
            "Kernel: {}, Current NOC Mode: {}, Required NOC Mode: DM_DEDICATED_NOC.",
            kernel->name(),
            config.noc_mode);
    }

    if (metal_context.hal().get_eth_fw_is_cooperative()) {
        // Dynamic NOC is not supported with this configuration
        TT_FATAL(
            config.noc_mode != NOC_MODE::DM_DYNAMIC_NOC,
            "EthernetKernel creation failure: Cannot create data movement kernels for {} across specified "
            "cores because NOC Mode {} is not supported on this platform",
            kernel->name(),
            config.noc_mode);
    }
    return program.impl().add_kernel(kernel, eth_core_type);
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
                return CreateDataMovementKernel(program, kernel_src, core_ranges, cfg);
            } else {
                return CreateComputeKernel(program, kernel_src, core_ranges, cfg);
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
    return CreateEthernetKernel(program, kernel_src, core_ranges, config);
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
                return CreateDataMovementKernel(program, kernel_src, core_ranges, cfg);
            } else {
                return CreateComputeKernel(program, kernel_src, core_ranges, cfg);
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
    return CreateEthernetKernel(program, kernel_src, core_ranges, config);
}

static KernelHandle CreateDramKernel(
    Program& program, const KernelSource& kernel_src, const CoreRangeSet& core_range_set, const DramConfig& config) {
    const ContextId context_id = program.impl().get_context_id();
    auto& metal_context = MetalContext::instance(context_id);
    TT_FATAL(metal_context.get_cluster().arch() == ARCH::BLACKHOLE, "DramKernel is only supported on Blackhole.");
    TT_FATAL(
        metal_context.hal().has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM programmable cores are not enabled; they auto-enable on Blackhole with firmware >= 19.12.0.0.");
    std::shared_ptr<Kernel> kernel = std::make_shared<DramKernel>(context_id, kernel_src, core_range_set, config);
    return program.impl().add_kernel(kernel, HalProgrammableCoreType::DRAM);
}

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DramConfig& config) {
    ValidateKernelConfigDefines(config.defines);
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_path(program.impl().get_context_id(), file_name);
    return CreateDramKernel(program, kernel_src, core_ranges, config);
}

KernelHandle CreateKernelFromString(
    Program& program,
    const std::string& kernel_src_code,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DramConfig& config) {
    ValidateKernelConfigDefines(config.defines);
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    KernelSource kernel_src = KernelSource::from_source(kernel_src_code);
    return CreateDramKernel(program, kernel_src, core_ranges, config);
}

CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    auto cb_handle = program.impl().add_circular_buffer(core_ranges, config);
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureCreateCircularBuffer, cb_handle, program, core_spec, config);
    return cb_handle;
}

const CircularBufferConfig& GetCircularBufferConfig(Program& program, CBHandle cb_handle) {
    return program.impl().get_circular_buffer(cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program& program, CBHandle cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBufferImpl> circular_buffer = program.impl().get_circular_buffer(cb_handle);
    if (not circular_buffer->globally_allocated()) {
        program.impl().invalidate_circular_buffer_allocation();
    }
    circular_buffer->set_total_size(total_size);
}

void UpdateCircularBufferPageSize(Program& program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    circular_buffer->set_page_size(buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const Buffer& buffer) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->set_global_buffer(buffer, circular_buffer->size(), circular_buffer->config().address_offset());
}

void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t address_offset) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->set_global_buffer(buffer, circular_buffer->size(), address_offset);
}

void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const MeshTensor& tensor) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->set_global_buffer(
        *tensor.mesh_buffer().get_reference_buffer(),
        circular_buffer->size(),
        circular_buffer->config().address_offset());
}

void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const Buffer& buffer, uint32_t total_size) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    circular_buffer->set_global_buffer(buffer, total_size, circular_buffer->config().address_offset());
}

void UpdateDynamicCircularBufferAddressAndTotalSize(
    Program& program, CBHandle cb_handle, const MeshTensor& tensor, uint32_t total_size) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    circular_buffer->set_global_buffer(
        *tensor.mesh_buffer().get_reference_buffer(), total_size, circular_buffer->config().address_offset());
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
    CoreRangeSet crs = std::visit(
        ttsl::overloaded{
            [](const CoreRange& c) { return CoreRangeSet(c); },
            [](const CoreRangeSet& c) {
                // Merge ranges to reduce the number of multicasts needed to initialize semaphores.
                return c.merge_ranges();
            },
        },
        core_spec);
    return program.impl().create_semaphore(crs, initial_value, core_type);
}

GlobalSemaphore CreateGlobalSemaphore(
    distributed::MeshDevice& device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type) {
    return GlobalSemaphore(GlobalSemaphoreImpl(device, std::move(cores), initial_value, buffer_type));
}

std::shared_ptr<Buffer> CreateBuffer(const BufferConfig& config) {
    return BufferImpl::create(config.device, config.size, config.page_size, config.buffer_type);
}
std::shared_ptr<Buffer> CreateBuffer(const BufferConfig& config, DeviceAddr address) {
    return BufferImpl::create(config.device, address, config.size, config.page_size, config.buffer_type);
}
std::shared_ptr<Buffer> CreateBuffer(const BufferConfig& config, SubDeviceId sub_device_id) {
    return BufferImpl::create(
        config.device, config.size, config.page_size, config.buffer_type, std::nullopt, std::nullopt, sub_device_id);
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config) {
    return BufferImpl::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout));
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address) {
    return BufferImpl::create(
        config.device,
        address,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout));
}
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id) {
    return BufferImpl::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout),
        std::nullopt,
        sub_device_id);
}

void DeallocateBuffer(Buffer& buffer) { buffer.impl().deallocate(buffer); }

void AssignGlobalBufferToProgram(const std::shared_ptr<Buffer>& buffer, Program& program) {
    MetalContext& metal_ctx = MetalContext::instance(program.impl().get_context_id());
    metal_ctx.device_manager()->check_dispatch_mode(metal_ctx.rtoptions().get_fast_dispatch());
    program.impl().add_buffer(buffer);
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
    return program.impl().add_circular_buffer(core_ranges, config, global_circular_buffer);
}

void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    TT_FATAL(circular_buffer->is_global_circular_buffer(), "CircularBuffer must be linked to a GlobalCircularBuffer!");
    circular_buffer->set_global_circular_buffer(global_circular_buffer);
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
