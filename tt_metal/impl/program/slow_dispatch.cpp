// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slow_dispatch.hpp"

#include <cstdint>
#include <exception>
#include <span>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "circular_buffer_constants.h"
#include "core_coord.hpp"
#include "device.hpp"
#include "hal_types.hpp"
#include "hostdev/remote_dfb_constants.h"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/emulation/host_sanitizers.hpp"
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "kernels/kernel.hpp"
#include "llrt.hpp"
#include "llrt/tt_cluster.hpp"
#include "program_impl.hpp"
#include "tt_metal.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include "tracy/Tracy.hpp"
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/program.hpp>

#ifdef TT_METAL_USE_EMULE
#include "emulated_program_runner.hpp"
#endif

namespace tt::tt_metal::slow_dispatch {

namespace {

void ConfigureKernelGroup(
    Program& program,
    uint32_t programmable_core_type_index,
    const KernelGroup* kernel_group,
    IDevice& device,
    const CoreCoord& logical_core,
    const Hal& hal) {
    uint32_t kernel_config_base =
        hal.get_dev_addr(hal.get_programmable_core_type(programmable_core_type_index), HalL1MemAddrType::KERNEL_CONFIG);
    for (auto kernel_id : kernel_group->kernel_ids) {
        // Need the individual offsets of each bin
        // TODO: make configure take a std::span
        program.impl().get_kernel(kernel_id)->configure(
            &device, logical_core, kernel_config_base, kernel_group->kernel_text_offsets.data());
    }
}

// Returns true iff the program has kernels and every core it targets is a DRAM programmable core.
// Such programs (e.g. the persistent tensor-prefetcher DRISC senders) are disjoint from the FD
// worker grid and dispatch column, so launching them via slow dispatch does not perturb an active
// FD session. Used to scope the force-slow-dispatch guard in LaunchProgram.
bool program_targets_only_dram_cores(const Program& program, const Hal& hal) {
    const auto& logical_cores_used_in_program = program.impl().logical_cores();
    bool has_any_core = false;
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        if (logical_cores_used_in_program[programmable_core_type_index].empty()) {
            continue;
        }
        has_any_core = true;
        if (hal.get_programmable_core_type(programmable_core_type_index) != HalProgrammableCoreType::DRAM) {
            return false;
        }
    }
    return has_any_core;
}

}  // namespace

void ConfigureDeviceWithProgram(IDevice& device, Program& program, bool force_slow_dispatch) {
    ZoneScoped;
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(&device));
    // This function is shared between FD and SD.
    // We call this function when initializing HW Command Queues or when reading Profiler Device to Device
    // sync information from the accelerators.
    // Must be set by the user only when its safe to mix slow dispatch with fast dispatch (advanced feature).
    if (!force_slow_dispatch) {
        detail::DispatchStateCheck(false);
    }

    auto device_id = device.id();

    // Individual device allocators don't track mesh buffer allocations, so use the
    // MeshDevice for validation when available to correctly detect CB/L1 buffer overlaps.
    auto mesh_device = device.get_mesh_device();
    const IDevice* validation_device = mesh_device ? mesh_device.get() : &device;

    bool is_emulated = false;
#ifdef TT_METAL_USE_EMULE
    is_emulated = metal_ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Emule;
#endif

    try {
        program.impl().allocate_circular_buffers(validation_device);
        program.impl().validate_circular_buffer_core_ranges(validation_device);
        program.impl().validate_circular_buffer_region(validation_device);
        program.impl().allocate_dataflow_buffers(validation_device);
        // Pre-size Metal 2.0 RTA/CRTA buffers from schema when not already reserved
        // (e.g. MakeProgramFromSpec). Idempotent if SetProgramRunArgs already sized them.
        program.impl().reserve_runtime_arg_buffers();
        // Metal 2.0 scratchpads stack on the DFB allocations, so allocate them AFTER the DFBs are placed.
        // Scratchpads are passed as implicit CRTAs, so they must be allocated before the CRTAs are committed.
        program.impl().allocate_scratchpads(validation_device);
        program.impl().validate_dataflow_buffer_region(validation_device);

        // Emule-only static KERNEL_CONFIG-window overflow sanitizer (no-op on
        // hardware); a throw here is surfaced as an ASAN panic by the catch below.
        if constexpr (emule::kEmuleAsanBuild) {
            emule::check_program_metadata_size(program);
        }
    } catch (const std::exception& e) {
        // Surface the overflow as an ASAN panic when emulating; no-op otherwise.
        // Routed through the facade so this TU carries no __emule_asan_panic
        // reference in a non-emule build. Always rethrows.
        if constexpr (emule::kEmuleAsanBuild) {
            emule::report_metadata_overflow(is_emulated, e.what());
        }
        throw;
    }

    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.impl().logical_cores();
    const auto& hal = metal_ctx.hal();
    uint32_t max_dfbs = hal.get_num_dataflow_buffers();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto& logical_cores = logical_cores_used_in_program[index];
        CoreType core_type = hal.get_core_type(index);
        for (const auto& logical_core : logical_cores) {
            KernelGroup* kernel_group = program.impl().kernels_on_core(logical_core, index);
            CoreCoord physical_core = device.virtual_core_from_logical_core(logical_core, core_type);
            // Skip binary writing for emulated mode (JIT compilation happens in execute_program_emulated)
            if (!is_emulated) {
                ConfigureKernelGroup(program, index, kernel_group, device, logical_core, hal);
            }
            // TODO: add support for CB for ethernet cores
            if (core_type == CoreType::WORKER) {
                uint64_t kernel_config_base =
                    hal.get_dev_addr(hal.get_programmable_core_type(index), HalL1MemAddrType::KERNEL_CONFIG);
                const auto& cbs_on_core = program.impl().circular_buffers_on_core(logical_core);
                const auto& dfbs_on_core = program.impl().dataflow_buffers_on_core(logical_core);
                const bool scans_remote_cb_configs =
                    kernel_group->launch_msg.view().kernel_config().min_remote_cb_start_index() < max_dfbs;
                if (!cbs_on_core.empty() || scans_remote_cb_configs) {
                    // CircularBufferConfigVec -- common across all kernels, so written once to the core
                    std::vector<uint32_t> circular_buffer_config_vec(
                        program.impl().get_program_config(index).cb_size / sizeof(uint32_t));

                    uint32_t remote_offset_index =
                        program.impl().get_program_config(index).local_cb_size / sizeof(uint32_t);
                    for (const auto& circular_buffer : cbs_on_core) {
                        for (uint32_t buffer_index : circular_buffer->local_buffer_indices()) {
                            uint32_t base_index = buffer_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;
                            uint32_t addr_in_bytes = circular_buffer->address();
                            uint32_t size_in_bytes = circular_buffer->size();
                            uint32_t num_pages = circular_buffer->num_pages(buffer_index);
                            uint32_t page_size = size_in_bytes / num_pages;
                            circular_buffer_config_vec[base_index] = addr_in_bytes;      // convert to addr in 16B words
                            circular_buffer_config_vec[base_index + 1] = size_in_bytes;  // convert to addr in 16B words
                            circular_buffer_config_vec[base_index + 2] = num_pages;
                            circular_buffer_config_vec[base_index + 3] = page_size;
                        }
                        for (uint32_t buffer_index : circular_buffer->remote_buffer_indices()) {
                            uint32_t base_index =
                                remote_offset_index +
                                ((max_dfbs - 1 - buffer_index) * UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG);
                            uint32_t config_address = circular_buffer->config_address();
                            circular_buffer_config_vec[base_index] = config_address;
                            circular_buffer_config_vec[base_index + 1] = circular_buffer->page_size(buffer_index);
                        }
                    }  // PROF_END("CBS")
                    uint64_t addr = kernel_config_base + program.impl().get_program_config(index).cb_offset;
                    metal_ctx.get_cluster().write_core(device_id, physical_core, circular_buffer_config_vec, addr);
                }

                if (!dfbs_on_core.empty()) {
                    log_info(tt::LogMetal, "DFB size: {}", program.impl().get_program_config(index).dfb_size);
                    std::vector<uint8_t> dfb_config_vec(
                        program.impl().get_program_config(index).dfb_size / sizeof(uint8_t));

                    const size_t bytes_written = tt::tt_metal::experimental::dfb::detail::serialize_dfb_config_for_core(
                        logical_core, dfbs_on_core, dfb_config_vec);

                    uint64_t addr = kernel_config_base + program.impl().get_program_config(index).dfb_offset;
                    log_info(
                        tt::LogMetal,
                        "Writing DFB config to core {} at addr 0x{:x} (kernel_config_base=0x{:x}, dfb_offset=0x{:x}) "
                        "size: {}",
                        physical_core.str(),
                        addr,
                        kernel_config_base,
                        program.impl().get_program_config(index).dfb_offset,
                        bytes_written);
                    metal_ctx.get_cluster().write_core(
                        device_id, physical_core, std::span<const uint8_t>(dfb_config_vec.data(), bytes_written), addr);
                }

                // CrossNodeDFB dense index in the worker kernel-config window. Full host
                // pages (including zeroed credits) are materialized into the dedicated
                // program-owned config Buffers at launch.
                const auto& program_config = program.impl().get_program_config(index);
                const uint32_t cross_node_dfb_offset = program_config.cross_node_dfb_offset;
                const auto& per_core_cross_node_dfbs = program.impl().get_per_core_cross_node_dfbs();
                auto it = per_core_cross_node_dfbs.find(logical_core);
                if (it != per_core_cross_node_dfbs.end() && !it->second.empty()) {
                    TT_FATAL(
                        cross_node_dfb_offset != REMOTE_DFB_OFFSET_NONE,
                        "CrossNodeDFB participants present but cross_node_dfb_offset is NONE");
                    const uint8_t num_program_slots = program.impl().num_cross_node_dfb_slots();
                    const uint32_t payload_words = remote_dfb_config_region_words(num_program_slots);
                    std::vector<uint32_t> cross_node_dfb_vec(payload_words, 0u);
                    cross_node_dfb_vec[0] = num_program_slots;
                    for (const auto& participant : it->second) {
                        TT_FATAL(
                            participant.remote_dfb_id < num_program_slots,
                            "CrossNodeDFB sparse participant remote_dfb_id {} exceeds program slot count {}",
                            participant.remote_dfb_id,
                            num_program_slots);
                        const uint32_t base = REMOTE_DFB_REGION_HEADER_WORDS +
                                              participant.remote_dfb_id * UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
                        cross_node_dfb_vec[base + 0] = participant.config_page_addr;
                        cross_node_dfb_vec[base + 1] = participant.entry_size;
                        cross_node_dfb_vec[base + 2] = participant.relay_dfb_id;

                        // Write the full host config page (credits already zero) to this
                        // core's shard of the dedicated config Buffer.
                        const auto& page =
                            program.impl().get_cross_node_dfb(participant.remote_dfb_id).config_page(logical_core);
                        metal_ctx.get_cluster().write_core(
                            device_id, physical_core, page, participant.config_page_addr);
                    }
                    uint64_t addr = kernel_config_base + cross_node_dfb_offset;
                    metal_ctx.get_cluster().write_core(device_id, physical_core, cross_node_dfb_vec, addr);
                }

                // PrefetcherPipe dense index only config pages written to device when the PrefetcherPipe is created.
                const uint32_t prefetcher_pipe_offset = program_config.prefetcher_pipe_offset;
                const auto& per_core_prefetcher_pipes = program.impl().get_per_core_prefetcher_pipes();
                auto persistent_it = per_core_prefetcher_pipes.find(logical_core);
                if (persistent_it != per_core_prefetcher_pipes.end() && !persistent_it->second.empty()) {
                    TT_FATAL(
                        prefetcher_pipe_offset != REMOTE_DFB_OFFSET_NONE,
                        "PrefetcherPipe participants present but prefetcher_pipe_offset is NONE");
                    // Same encoding as fast dispatch (relay word carries the active lane count).
                    std::vector<uint32_t> prefetcher_pipe_vec =
                        program_dispatch::build_prefetcher_pipe_config_payload(program.impl(), persistent_it->second);
                    uint64_t addr = kernel_config_base + prefetcher_pipe_offset;
                    metal_ctx.get_cluster().write_core(device_id, physical_core, prefetcher_pipe_vec, addr);
                }
            }
            program.impl().init_semaphores(device, logical_core, index);
        }
    }
}

void WriteRuntimeArgsToDevice(IDevice& device, Program& program, bool force_slow_dispatch) {
    ZoneScoped;
    auto device_id = device.id();
    // This function is shared between FD and SD.
    // We call this function when initializing HW Command Queues or when reading Profiler Device to Device
    // sync information from the accelerators.
    // Must be set by the user only when its safe to mix slow dispatch with fast dispatch (advanced feature).
    if (!force_slow_dispatch) {
        detail::DispatchStateCheck(false);
    }

    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(&device));
    const auto& hal = metal_ctx.hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        CoreType core_type = hal.get_core_type(index);
        HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(index);
        uint64_t l1_noc_offset = hal.get_l1_noc_offset(programmable_core_type);
        for (const auto& kg : program.impl().get_kernel_groups(index)) {
            auto kernel_config = kg->launch_msg.view().kernel_config();
            uint64_t kernel_config_base =
                static_cast<uint64_t>(kernel_config.kernel_config_base()[index]) + l1_noc_offset;
            for (const CoreRange& core_range : kg->core_ranges.ranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                        CoreCoord logical_core(x, y);
                        auto physical_core = device.virtual_core_from_logical_core(logical_core, core_type);
                        for (auto kernel_id : kg->kernel_ids) {
                            const auto& kernel = program.impl().get_kernel(kernel_id);
                            const auto& rt_args = kernel->runtime_args(logical_core);

                            // RTA/CRTA offsets are the same for all binaries of the kernel, pick any binary.
                            uint32_t processor_index = hal.get_processor_index(
                                kernel->get_kernel_programmable_core_type(),
                                kernel->get_kernel_processor_class(),
                                kernel->get_kernel_processor_type(0));
                            auto rta_offset = kernel_config.rta_offset()[processor_index];
                            if (!rt_args.empty()) {
                                auto rt_args_addr = kernel_config_base + rta_offset.rta_offset();
                                log_trace(
                                    tt::LogMetal,
                                    "{} - Writing {} unique rtargs to core {} (physical: {}) addr 0x{:x} => args: "
                                    "{}",
                                    __FUNCTION__,
                                    rt_args.size(),
                                    logical_core.str(),
                                    physical_core.str(),
                                    rt_args_addr,
                                    rt_args);
                                metal_ctx.get_cluster().write_core(device_id, physical_core, rt_args, rt_args_addr);
                            }

                            const auto& common_rt_args = kernel->common_runtime_args();
                            if (!common_rt_args.empty()) {
                                auto common_rt_args_addr = kernel_config_base + rta_offset.crta_offset();
                                log_trace(
                                    tt::LogMetal,
                                    "{} - Writing {} common rtargs to core {} (physical: {}) addr 0x{:x} => args: "
                                    "{}",
                                    __FUNCTION__,
                                    common_rt_args.size(),
                                    logical_core.str(),
                                    physical_core.str(),
                                    common_rt_args_addr,
                                    common_rt_args);
                                metal_ctx.get_cluster().write_core(
                                    device_id, physical_core, common_rt_args, common_rt_args_addr);
                            }
                        }
                    }
                }
            }
        }
    }
}

void WaitProgramDone(IDevice& device, const Program& program) {
    auto& metal_ctx = MetalContext::instance(extract_context_id(&device));
    llrt::internal_::wait_for_idle(metal_ctx, device.id(), program.impl().logical_cores());
}

void LaunchProgram(IDevice& device, Program& program, bool force_slow_dispatch) {
    ZoneScoped;
    MetalContext& metal_ctx = MetalContext::instance(extract_context_id(&device));
    // Must be set by the user only when its safe to mix slow dispatch with fast dispatch (advanced feature).
    if (!force_slow_dispatch) {
        detail::DispatchStateCheck(false);
    } else {
        auto& dm = metal_ctx.device_manager();
        const bool fd_active = dm->is_dispatch_firmware_active();
        const bool rt_done = dm->is_rt_profiler_device_init_complete(device.id());
        // Scope the service bypass to this device
        const bool service_active = !metal_ctx.get_service_core_manager().claimed_cores(device.id()).empty();
        // DRAM-only programs (e.g. the persistent tensor-prefetcher DRISC senders) run on the DRAM
        // programmable cores, which are disjoint from the FD worker grid and dispatch column. Launching
        // them via slow dispatch does not touch FD-owned cores or the FD pipeline, so it is safe to mix
        // with an active FD session regardless of profiler init state.
        const bool dram_only = program_targets_only_dram_cores(program, metal_ctx.hal());
        TT_ASSERT(
            !(fd_active && rt_done) || service_active || dram_only,
            "Cannot force slow dispatch while fast dispatch firmware is active and real-time profiler init has "
            "completed on this device.");
    }

#ifdef TT_METAL_USE_EMULE
    if (metal_ctx.get_cluster().get_target_device_type() != tt::TargetDevice::Emule)
#endif
    {
        program.impl().compile(&device);
    }
    program.impl().finalize_dataflow_buffer_configs();
    if (!program.impl().is_finalized()) {
        program.impl().finalize_offsets(&device);
    }

    // First configure (allocate buffers + write configs/binaries), then write runtime args.
    // This allows us to allocate ephemeral scratchpad buffers, and pass their locations as implicit CRTAs.
    ConfigureDeviceWithProgram(device, program, force_slow_dispatch);
    WriteRuntimeArgsToDevice(device, program, force_slow_dispatch);

#ifdef TT_METAL_USE_EMULE
    if (metal_ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Emule) {
        // Emulated mode always executes synchronously (slow dispatch only): all kernels complete before this
        // function returns.
        emule::execute_program_emulated(&device, program);
        return;
    }
#endif

    auto device_id = device.id();
    metal_ctx.get_cluster().dram_barrier(device_id);

    // Note: the l1_barrier below is needed to be sure writes to cores that
    // don't get the GO mailbox (eg, storage cores) have all landed
    metal_ctx.get_cluster().l1_barrier(device_id);

    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = program.impl().logical_cores();
    const auto& hal = metal_ctx.hal();
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        HalProgrammableCoreType programmable_core_type = hal.get_programmable_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            auto* kg = program.impl().kernels_on_core(logical_core, programmable_core_type_index);
            // Raw runtime id matches Tracy / fast dispatch; profiler ingest encodes with device_id once.
            kg->launch_msg.view().kernel_config().host_assigned_id() = program.get_runtime_id();

            auto physical_core = device.virtual_core_from_logical_core(logical_core, core_type);
            if (force_slow_dispatch) {
                tt::llrt::send_reset_go_signal(MetalEnvAccessor(metal_ctx.get_env()).impl(), device_id, physical_core);
            }

            tt::llrt::write_launch_msg_to_core(
                MetalEnvAccessor(metal_ctx.get_env()).impl(),
                device_id,
                physical_core,
                kg->launch_msg.view(),
                kg->go_msg.view(),
                hal.get_dev_addr(programmable_core_type, HalL1MemAddrType::LAUNCH));
        }
    }
}

}  // namespace tt::tt_metal::slow_dispatch
