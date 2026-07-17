// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "debug_tools_fixture.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
// Do we really want to expose Hal like this?
// This looks like an API level test
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher NOC sanitization.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

enum watcher_features_t {
    SanitizeNOCAddress,
    SanitizeNOCAlignmentL1Write,
    SanitizeNOCAlignmentL1Read,
    SanitizeNOCZeroL1Write,
    SanitizeNOCMailboxWrite,
    SanitizeNOCMailboxWriteUncachedAlias,
    SanitizeNOCInlineWriteDram,
    SanitizeNOCLinkedTransaction,
    SanitizeL1Overflow,
    SanitizeL1OverflowStraddle,
    SanitizeEthSrcL1Overflow,
    SanitizeEthDestL1Overflow,
    SanitizeNOCMulticastInvalidRange,
    SanitizeNOCWriteWithStateBadCoord,
    SanitizeNOCInlineWriteFromState,
    SanitizeNOCInlineWriteWithState,
};

tt::tt_metal::HalMemType get_buffer_mem_type_for_test(watcher_features_t feature) {
    return feature == watcher_features_t::SanitizeNOCInlineWriteDram ? tt_metal::HalMemType::DRAM
                                                                     : tt_metal::HalMemType::L1;
}

tt::tt_metal::BufferType get_buffer_type_for_test(watcher_features_t feature) {
    return feature == watcher_features_t::SanitizeNOCInlineWriteDram ? tt_metal::BufferType::DRAM
                                                                     : tt_metal::BufferType::L1;
}

uint32_t get_address_for_test(bool use_eth_core, tt::tt_metal::HalL1MemAddrType type, bool high_address = false) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (use_eth_core) {
        const auto active_eth_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, type);
        const auto idle_eth_addr = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, type);
        if (high_address) {
            return std::max(active_eth_addr, idle_eth_addr);
        }
        return std::min(active_eth_addr, idle_eth_addr);
    }
    return hal.get_dev_addr(HalProgrammableCoreType::TENSIX, type);
}

CoreCoord get_core_coord_for_test(const std::shared_ptr<distributed::MeshBuffer>& buffer) {
    if (buffer->device_local_config().buffer_type == tt_metal::BufferType::L1) {
        return buffer->device()->worker_core_from_logical_core(
            buffer->get_backing_buffer()->allocator()->get_logical_core_from_bank_id(0));
    }
    auto logical_dram_core = buffer->device()->logical_core_from_dram_channel(0);
    return buffer->device()->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
}

void RunTestOnCore(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    CoreCoord& core,
    bool is_eth_core,
    watcher_features_t feature,
    bool use_ncrisc = false,
    bool is_idle_eth_core = false,
    bool multi_dm_race = false) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;

    if (tt::tt_metal::MetalContext::instance().rtoptions().watcher_noc_sanitize_disabled()) {
        GTEST_SKIP();
    }

    // Non-IDLE_ETH cores (TENSIX/ACTIVE_ETH, all archs) run under both fast and slow dispatch.
    // IDLE_ETH cores only support slow dispatch (FD not yet implemented for them).
    if (is_idle_eth_core && !fixture->IsSlowDispatch()) {
        GTEST_SKIP() << "IDLE_ETH core tests only run under Slow Dispatch";
    }
    if (multi_dm_race && !is_quasar) {
        GTEST_SKIP() << "Multi-DM race test only runs on Quasar";
    }
    // Both exercise Quasar's uncached L1 alias, which only exists on Quasar DM cores.
    if ((feature == SanitizeNOCMailboxWriteUncachedAlias || feature == SanitizeL1OverflowStraddle) &&
        (!is_quasar || is_eth_core)) {
        GTEST_SKIP() << "Uncached-alias tests only apply to Quasar DM cores";
    }

    // TENSIX cores use the Metal 2.0 variant; ETH cores stay on the legacy kernel/API.
    const std::string kernel_legacy = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_to_noc_coord.cpp";
    const std::string kernel_metal2 = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_to_noc_coord_2_0.cpp";
    // On Quasar, DM0/DM1 are reserved for internal use; map brisc/ncrisc onto the first two user DMs.
    uint32_t dm_id = 0;
    if (is_quasar) {
        dm_id = use_ncrisc ? 3 : 2;
    } else {
        dm_id = use_ncrisc ? 1 : 0;
    }

    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();

    CoreCoord virtual_core;
    if (is_eth_core) {
        virtual_core = device->ethernet_core_from_logical_core(core);
    } else {
        virtual_core = device->worker_core_from_logical_core(core);
    }
    log_info(
        LogTest,
        "Running test on device {} {} core {} (virtual core {})...",
        device->id(),
        (is_eth_core) ? "eth" : "worker",
        core.str(),
        virtual_core.str());

    // Set up dram buffers
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 50;
    auto buffer_mem_type = get_buffer_mem_type_for_test(feature);
    uint32_t buffer_size = single_tile_size * num_tiles;
    auto config_buffer_type = get_buffer_type_for_test(feature);

    distributed::DeviceLocalBufferConfig scratch_local_config{
        .page_size = buffer_size, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig scratch_buffer_config{.size = buffer_size};

    auto local_scratch_buffer =
        distributed::MeshBuffer::create(scratch_buffer_config, scratch_local_config, mesh_device.get());
    uint32_t buffer_addr = local_scratch_buffer->address();
    // For ethernet core, need to have smaller buffer and force buffer to be at a different address
    if (is_eth_core) {
        buffer_size = 1024;
        buffer_addr = get_address_for_test(true, HalL1MemAddrType::UNRESERVED, true);
    }

    distributed::DeviceLocalBufferConfig local_config{.page_size = buffer_size, .buffer_type = config_buffer_type};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto input_buffer = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    uint32_t input_buffer_addr = input_buffer->address();

    auto output_buffer = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    uint32_t output_buffer_addr = output_buffer->address();

    auto input_buf_noc_xy = get_core_coord_for_test(input_buffer);
    auto output_buf_noc_xy = get_core_coord_for_test(output_buffer);
    log_info(tt::LogTest, "Input/Output Buffer mem type: {}", enchantum::to_string(buffer_mem_type));
    log_info(tt::LogTest, "Input Buffer NOC XY: {}", input_buf_noc_xy);
    log_info(tt::LogTest, "Output Buffer NOC XY: {}", output_buf_noc_xy);
    log_info(tt::LogTest, "Local scratch buffer addr: {:#x}", buffer_addr);

    // A copy kernel, we'll feed it incorrect inputs to test sanitization.
    KernelHandle dram_copy_kernel = 0;
    int noc = 0;
    const experimental::KernelSpecName DRAM_COPY_KERNEL_NAME{"dram_copy"};
    if (is_eth_core) {
        // ETH cores: invoke the original (legacy) kernel via the legacy host API.
        tt_metal::EthernetConfig config = {.noc = tt_metal::NOC::NOC_0};
        if (is_idle_eth_core) {
            config.eth_mode = Eth::IDLE;
        }
        eth_test_common::set_arch_specific_eth_config(config);
        noc = static_cast<int>(config.noc);
        dram_copy_kernel = tt_metal::CreateKernel(program, kernel_legacy, core, config);
    } else {
        // TENSIX kernel is launched via Metal 2.0 on both gen1 (WH/BH) and gen2 (Quasar).
        // On Quasar, user DMs (DM2..DM7) run the kernel; multi_dm_race syncs them to race, else only dm_id executes.
        // On WH/BH, BRISC or NCRISC (selected by use_ncrisc) runs the kernel.
        experimental::KernelSpec::CompileTimeArgs cta_bindings;
        experimental::KernelSpec::CompilerOptions::Defines defines;
        if (is_quasar && multi_dm_race) {
            constexpr uint32_t num_dms = 6;
            constexpr uint32_t multi_dm_base_addr = 0xFFFF0000;
            constexpr uint32_t multi_dm_base_size = 0x1000;
            // Allocate dedicated L1 region for the DM barrier counter (avoid overlap with scratch buffer)
            distributed::ReplicatedBufferConfig sync_cfg{.size = 32};
            distributed::DeviceLocalBufferConfig sync_lcl{.page_size = 32, .buffer_type = tt::tt_metal::BufferType::L1};
            auto sync_buf = distributed::MeshBuffer::create(sync_cfg, sync_lcl, mesh_device.get());
            uint32_t l1_sync_addr = sync_buf->address();
            std::vector<uint32_t> init{0, 0};  // 8 bytes: Quasar barrier uses 64-bit atomics
            tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_sync_addr, init);
            cta_bindings = {
                {"num_dms", num_dms},
                {"multi_dm_base_addr", multi_dm_base_addr},
                {"multi_dm_base_size", multi_dm_base_size},
                {"l1_sync_addr", l1_sync_addr},
            };
            defines = {{"TEST_MULTI_DM_SANITIZE_RACE", "1"}};
        } else if (is_quasar) {
            cta_bindings = {{"dm_id", dm_id}};
        }
        // (gen1 path: no CTA bindings needed; the kernel runs on exactly one DM processor.)

        // Select the DM config variant matching the current architecture (Gen2 on Quasar, Gen1 otherwise).
        auto gen1_processor =
            use_ncrisc ? tt::tt_metal::DataMovementProcessor::RISCV_1 : tt::tt_metal::DataMovementProcessor::RISCV_0;
        auto gen1_noc = use_ncrisc ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default;
        experimental::DataMovementHardwareConfig dm_cfg;
        if (is_quasar) {
            dm_cfg = experimental::DataMovementGen2Config{};
        } else {
            dm_cfg = experimental::DataMovementGen1Config{.processor = gen1_processor, .noc = gen1_noc};
        }
        uint32_t num_threads = is_quasar ? 6u : 1u;
        if (!is_quasar) {
            noc = static_cast<int>(gen1_noc);
        }
        experimental::KernelSpec dm_spec{
            .unique_id = DRAM_COPY_KERNEL_NAME,
            .source = kernel_metal2,
            .num_threads = num_threads,
            .compiler_options = {.defines = defines},
            .compile_time_args = cta_bindings,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"local_buffer_addr",
                      "buffer_src_addr",
                      "src_noc_x",
                      "src_noc_y",
                      "buffer_dst_addr",
                      "dst_noc_x",
                      "dst_noc_y",
                      "buffer_size",
                      "use_inline_dw_write",
                      "bad_linked_transaction",
                      "l1_overflow_addr",
                      "eth_src_overflow_addr",
                      "eth_dest_overflow_addr",
                      "use_multicast_semaphore_inc",
                      "mcast_dst_end_x",
                      "mcast_dst_end_y",
                      "use_write_with_state",
                      "use_inline_dw_write_from_state",
                      "use_inline_dw_write_with_state"}},
            .hw_config = dm_cfg,
        };
        experimental::WorkUnitSpec wu{
            .name = "main",
            .kernels = {DRAM_COPY_KERNEL_NAME},
            .target_nodes = experimental::NodeCoord{core},
        };
        experimental::ProgramSpec spec{
            .name = "watcher_sanitize",
            .kernels = {dm_spec},
            .work_units = {wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device, spec);
        if (is_quasar) {
            // Quasar SD does not yet expose a NOC index in the same way as legacy DMs; the watcher
            // log emits "noc0" for Metal 2.0 DM kernels. Match that so expected strings line up.
            noc = 0;
        }
    }

    // Write to the input buffer
    std::vector<uint32_t> input_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    distributed::WriteShard(cq, input_buffer, input_vec, zero_coord);

    // Write runtime args - update to a core that doesn't exist or an improperly aligned address,
    // depending on the flags passed in.
    bool use_inline_dw_write = false;
    bool bad_linked_transaction = false;
    uint32_t l1_overflow_addr = 0;
    uint32_t eth_src_overflow_addr_words = 0;
    uint32_t eth_dest_overflow_addr_words = 0;
    bool use_multicast_semaphore_inc = false;
    uint32_t mcast_dst_end_x = 0;
    uint32_t mcast_dst_end_y = 0;
    bool use_write_with_state = false;
    bool use_inline_dw_write_from_state = false;
    bool use_inline_dw_write_with_state = false;
    switch (feature) {
        case SanitizeNOCAddress:
            output_buf_noc_xy.x = 26;
            output_buf_noc_xy.y = 18;
            break;
        case SanitizeNOCAlignmentL1Write:
            output_buffer_addr++;  // This is illegal because reading DRAM->L1 needs DRAM alignment
                                   // requirements (32 byte aligned).
            buffer_size--;
            break;
        case SanitizeNOCAlignmentL1Read:
            input_buffer_addr++;
            buffer_size--;
            break;
        case SanitizeNOCZeroL1Write: output_buffer_addr = 0; break;
        case SanitizeNOCMailboxWrite:
            // This is illegal because we'd be writing to the mailbox memory
            buffer_addr = get_address_for_test(is_eth_core, HalL1MemAddrType::MAILBOX);
            break;
        case SanitizeNOCMailboxWriteUncachedAlias:
            // Quasar-only (skipped at the top otherwise): the mailbox is aliased into the uncached L1
            // window (upper 4 MB, same physical memory as the cached view). Target the alias so a NoC
            // access landing in the mailbox through it is still flagged. Complements
            // SanitizeNOCMailboxWrite, which covers the cached view.
            buffer_addr = get_address_for_test(is_eth_core, HalL1MemAddrType::MAILBOX) +
                          hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
            break;
        case SanitizeNOCInlineWriteDram: use_inline_dw_write = true; break;
        case SanitizeNOCLinkedTransaction: bad_linked_transaction = true; break;
        case SanitizeL1Overflow: l1_overflow_addr = 0xDDDDDDDD; break;
        case SanitizeL1OverflowStraddle:
            // 4-byte access starting 2 bytes below the top of L1, straddling the cached/uncached-alias seam.
            l1_overflow_addr = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - 2;
            break;
        case SanitizeEthSrcL1Overflow: eth_src_overflow_addr_words = 0xAAAAAAAA; break;
        case SanitizeEthDestL1Overflow: eth_dest_overflow_addr_words = 0xBBBBBBBB; break;
        case SanitizeNOCMulticastInvalidRange: {
            // This test requires at least 2 DRAM channels to create an invalid multicast range
            if (device->num_dram_channels() < 2) {
                log_info(
                    LogTest,
                    "Skipping SanitizeNOCMulticastInvalidRange: requires at least 2 DRAM channels, device has {}",
                    device->num_dram_channels());
                GTEST_SKIP();
            }
            // Use invalid multicast range with actual DRAM cores: start > end
            // Wrap-around is only allowed for Tensix cores, not DRAM
            use_multicast_semaphore_inc = true;

            // Get actual DRAM NOC coordinates
            auto dram_logical_0 = device->logical_core_from_dram_channel(0);
            auto dram_logical_1 = device->logical_core_from_dram_channel(1);
            auto dram_noc_0 = device->virtual_core_from_logical_core(dram_logical_0, CoreType::DRAM);
            auto dram_noc_1 = device->virtual_core_from_logical_core(dram_logical_1, CoreType::DRAM);

            // Ensure start > end to trigger wrap-around check (which should fail for DRAM)
            if (dram_noc_0.x > dram_noc_1.x || dram_noc_0.y > dram_noc_1.y) {
                output_buf_noc_xy = dram_noc_0;
                mcast_dst_end_x = dram_noc_1.x;
                mcast_dst_end_y = dram_noc_1.y;
            } else {
                output_buf_noc_xy = dram_noc_1;
                mcast_dst_end_x = dram_noc_0.x;
                mcast_dst_end_y = dram_noc_0.y;
            }
            break;
        }
        case SanitizeNOCWriteWithStateBadCoord:
            // Stateful write to a non-existent core. The destination coordinate lives in NOC_RET_ADDR; a
            // sanitizer that mistakenly read NOC_TARG_ADDR would instead see the sender's own (valid) coordinate
            // and fail to flag the bad target. The zero destination offset keeps the failure deterministic
            // either way (it never silently succeeds), and the small size forces the one-packet write path.
            output_buf_noc_xy.x = 26;
            output_buf_noc_xy.y = 18;
            output_buffer_addr = 0;
            buffer_size = 32;
            use_write_with_state = true;
            break;
        case SanitizeNOCInlineWriteFromState:
            // Bad destination coordinate, but keep the (nonzero) destination offset: this exercises
            // DEBUG_SANITIZE_NOC_ADDR_FROM_STATE after noc_inline_dw_write_set_state. The reported offset
            // discriminates low-bits bugs: dropping NOC_TARG_ADDR_LO on tt-1xx or NOC_RET_ADDR_LO on tt-2xx
            // would report offset 0 instead of the real destination offset.
            output_buf_noc_xy.x = 26;
            output_buf_noc_xy.y = 18;
            use_inline_dw_write_from_state = true;
            break;
        case SanitizeNOCInlineWriteWithState:
            if (hal.get_arch() == tt::ARCH::BLACKHOLE) {
                GTEST_SKIP() << "cq_noc_inline_dw_write_with_state-style helper is not exposed on Blackhole";
            }
            output_buf_noc_xy.x = 26;
            output_buf_noc_xy.y = 18;
            use_inline_dw_write_with_state = true;
            break;
        default:
            log_warning(LogTest, "Unrecognized feature to test ({}), skipping...", feature);
            GTEST_SKIP();
            break;
    }

    const std::vector<uint32_t> rta_values = {
        buffer_addr,
        input_buffer_addr,
        input_buf_noc_xy.x,
        input_buf_noc_xy.y,
        output_buffer_addr,
        output_buf_noc_xy.x,
        output_buf_noc_xy.y,
        buffer_size,
        use_inline_dw_write,
        bad_linked_transaction,
        l1_overflow_addr,
        eth_src_overflow_addr_words,
        eth_dest_overflow_addr_words,
        use_multicast_semaphore_inc,
        mcast_dst_end_x,
        mcast_dst_end_y,
        use_write_with_state,
        use_inline_dw_write_from_state,
        use_inline_dw_write_with_state};

    if (is_eth_core) {
        // ETH cores still go through the legacy API.
        tt_metal::SetRuntimeArgs(program, dram_copy_kernel, core, rta_values);
    } else {
        const experimental::NodeCoord node{core};
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DRAM_COPY_KERNEL_NAME,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"local_buffer_addr", buffer_addr},
                 {"buffer_src_addr", input_buffer_addr},
                 {"src_noc_x", input_buf_noc_xy.x},
                 {"src_noc_y", input_buf_noc_xy.y},
                 {"buffer_dst_addr", output_buffer_addr},
                 {"dst_noc_x", output_buf_noc_xy.x},
                 {"dst_noc_y", output_buf_noc_xy.y},
                 {"buffer_size", buffer_size},
                 {"use_inline_dw_write", use_inline_dw_write},
                 {"bad_linked_transaction", bad_linked_transaction},
                 {"l1_overflow_addr", l1_overflow_addr},
                 {"eth_src_overflow_addr", eth_src_overflow_addr_words},
                 {"eth_dest_overflow_addr", eth_dest_overflow_addr_words},
                 {"use_multicast_semaphore_inc", use_multicast_semaphore_inc},
                 {"mcast_dst_end_x", mcast_dst_end_x},
                 {"mcast_dst_end_y", mcast_dst_end_y},
                 {"use_write_with_state", use_write_with_state},
                 {"use_inline_dw_write_from_state", use_inline_dw_write_from_state},
                 {"use_inline_dw_write_with_state", use_inline_dw_write_with_state}}),
        }};
        experimental::SetProgramRunArgs(program, params);
    }
    workload.add_program(device_range, std::move(program));

    // Run the kernel; its illegal NoC transaction trips watcher test mode. Whether that reaches the
    // host as an exception here is a race with the watcher poll (fires on the slow Quasar sim via
    // #48842's fast-dispatch rethrow; usually not on fast HW), so this catch is best-effort. The
    // watcher-log check below always runs (regardless of this catch) and is the real verification.
    try {
        fixture->RunProgram(mesh_device, workload);
    } catch (std::runtime_error& e) {
        const std::string error = std::string(e.what());
        log_info(tt::LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find("Aborting wait due to watcher error") != std::string::npos) << error;
    }

    // We should be able to find the expected watcher error in the log as well.
    std::string expected;
    CoreCoord input_core_virtual_coords = device->virtual_noc0_coordinate(noc, input_buf_noc_xy);
    CoreCoord output_core_virtual_coords = device->virtual_noc0_coordinate(noc, output_buf_noc_xy);
    // TODO: replace ierisc and erisc with hal.get_processor_class_name() after
    // unifying all tests + watcher_device_reader::get_riscv_name() with same method
    std::string risc_name;
    if (is_eth_core) {
        risc_name = is_idle_eth_core ? "ierisc" : "erisc";
    } else {
        risc_name = hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, dm_id, false);
    }
    const char* core_name = "worker";
    if (is_eth_core) {
        core_name = is_idle_eth_core ? "idleth" : "acteth";
    }
    // Note: for multi_dm_race, expected string is built but not used - verification uses regex instead
    switch (feature) {
        // Stateful write to a bad coordinate reports the same "did not map to any known core" error as a plain
        // bad-coordinate write; the destination coordinate is reconstructed from NOC_RET_ADDR state registers.
        case SanitizeNOCWriteWithStateBadCoord:
        case SanitizeNOCAddress:
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast write {} "
                "bytes from local L1[{:#08x}] to Unknown core w/ virtual coords {} [addr=0x{:08x}] (NOC target "
                "address did not map to any known Tensix/Ethernet/DRAM/PCIE core).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                output_buf_noc_xy.str(),
                output_buffer_addr);
            break;
        case SanitizeNOCAlignmentL1Write: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast write {} "
                "bytes from local L1[{:#08x}] to Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (invalid address "
                "alignment in NOC transaction).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                input_core_virtual_coords,
                output_buffer_addr);
            break;
        }
        case SanitizeNOCAlignmentL1Read: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast read {} "
                "bytes to local L1[{:#08x}] from Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (invalid address "
                "alignment in NOC transaction).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                input_core_virtual_coords,
                input_buffer_addr);
        } break;
        case SanitizeNOCZeroL1Write: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast write {} "
                "bytes from local L1[{:#08x}] to Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (NOC target "
                "overwrites mailboxes).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                input_core_virtual_coords,
                output_buffer_addr);
        } break;
        case SanitizeNOCMailboxWriteUncachedAlias:
        case SanitizeNOCMailboxWrite: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast read {} "
                "bytes to local L1[{:#08x}] from Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (Local L1 "
                "overwrites mailboxes).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                input_buf_noc_xy.str(),
                input_buffer_addr);
        } break;
        case SanitizeNOCInlineWriteDram: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc0 tried to unicast write 4 bytes "
                "from local L1[{:#08x}] to DRAM core w/ virtual coords {} DRAM[addr=0x{:08x}] (inline dw writes do not "
                "support DRAM destination addresses).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                0,
                output_core_virtual_coords.str(),
                output_buffer_addr);
        } break;
        case SanitizeNOCLinkedTransaction: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast write {} "
                "bytes from local L1[{:#08x}] to Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (submitting a "
                "non-mcast transaction when there's a linked transaction).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                output_core_virtual_coords.str(),
                output_buffer_addr);
        } break;
        case SanitizeL1OverflowStraddle:
        case SanitizeL1Overflow: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} core overflowed L1 with access to {:#x} "
                "of length {} (read or write past the end of local memory).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                l1_overflow_addr,
                sizeof(std::uint32_t));
        } break;
        case SanitizeEthSrcL1Overflow: {
            expected = fmt::format(
                "Device {} acteth core(x={:2},y={:2}) virtual(x={:2},y={:2}): erisc core overflowed L1 with access to "
                "{:#x} "
                "of length 64 (ethernet send with L1 source overflow).",
                device->id(),
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                (eth_src_overflow_addr_words << 4));
        } break;
        case SanitizeEthDestL1Overflow: {
            expected = fmt::format(
                "Device {} acteth core(x={:2},y={:2}) virtual(x={:2},y={:2}): erisc core overflowed L1 with access to "
                "{:#x} "
                "of length 64 (ethernet send to core with L1 destination overflow).",
                device->id(),
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                (eth_dest_overflow_addr_words << 4));
        } break;
        case SanitizeNOCInlineWriteFromState:
        case SanitizeNOCInlineWriteWithState:
            // Inline dw write sanitized straight from the command-buffer state (DEBUG_SANITIZE_NOC_ADDR_FROM_STATE
            // uses read semantics with l1_addr 0). The destination coordinate is invalid; [addr=...] is the
            // reconstructed destination offset, which must be the real offset rather than 0.
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast read 4 "
                "bytes to local L1[{:#08x}] from Unknown core w/ virtual coords {} [addr=0x{:08x}] (NOC target "
                "address did not map to any known Tensix/Ethernet/DRAM/PCIE core).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                0,  // l1_addr is 0 for address-only (FROM_STATE) sanitization
                output_buf_noc_xy.str(),
                output_buffer_addr);
            break;
        case SanitizeNOCMulticastInvalidRange: {
            // The watcher device reader formats multicast coords using CoreCoord::str() +
            // "-" + CoreCoord::str(), which (since UMD bump) produces "X1-Y1-X2-Y2".
            // Build the expected string the same way to stay format-agnostic.
            CoreCoord mcast_start_coord = output_buf_noc_xy;
            CoreCoord mcast_end_coord = {mcast_dst_end_x, mcast_dst_end_y};
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to multicast write 4 "
                "bytes from local L1[{:#08x}] to DRAM core range w/ virtual coords {}-{} "
                "DRAM[addr=0x{:08x}] (multicast invalid range).",
                device->id(),
                core_name,
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                0,  // l1_addr is 0 for address-only sanitization
                mcast_start_coord.str(),
                mcast_end_coord.str(),
                output_buffer_addr);
        } break;
        default:
            log_warning(LogTest, "Unrecognized feature to test ({}), skipping...", feature);
            GTEST_SKIP();
            break;
    }

    if (!multi_dm_race) {
        log_info(LogTest, "Expected error: {}", expected);
    }
    std::string exception;
    do {
        exception = MetalContext::instance().watcher_server()->exception_message();
    } while (exception.empty());
    log_info(LogTest, "Reported error: {}", exception);

    if (multi_dm_race) {
        // Verify CAS atomicity: addr and size low bits must match (same DM wrote both)
        std::regex addr_regex("addr=0x([0-9a-fA-F]+)");
        std::regex size_regex("write ([0-9]+) bytes");
        std::smatch addr_match, size_match;

        ASSERT_TRUE(std::regex_search(exception, addr_match, addr_regex)) << "Could not find addr in error";
        ASSERT_TRUE(std::regex_search(exception, size_match, size_regex)) << "Could not find size in error";

        uint32_t reported_addr = std::stoul(addr_match[1].str(), nullptr, 16);
        uint32_t reported_size = std::stoul(size_match[1].str());
        uint32_t addr_dm_id = reported_addr & 0xF;
        uint32_t size_dm_id = reported_size & 0xF;

        EXPECT_EQ(addr_dm_id, size_dm_id)
            << "CAS race corruption: addr dm_id=" << addr_dm_id << " but size dm_id=" << size_dm_id;
        log_info(
            LogTest, "Multi-DM race: DM{} won CAS with addr=0x{:x} size={}", addr_dm_id, reported_addr, reported_size);
    } else {
        EXPECT_EQ(exception, expected);
    }
}

void RunTestEth(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    watcher_features_t feature) {
    auto* device = mesh_device->get_devices()[0];
    if (fixture->IsSlowDispatch()) {
        GTEST_SKIP();
    }
    // Run on the first ethernet core (if there are any).
    if (device->get_active_ethernet_cores(true).empty()) {
        log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
        GTEST_SKIP();
    }
    CoreCoord core = *(device->get_active_ethernet_cores(true).begin());
    RunTestOnCore(fixture, mesh_device, core, true, feature);
}

void RunTestIEth(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    watcher_features_t feature) {
    auto* device = mesh_device->get_devices()[0];
    // Run on the first ethernet core (if there are any).
    if (device->get_inactive_ethernet_cores().empty()) {
        log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
        GTEST_SKIP();
    }
    CoreCoord core = *(device->get_inactive_ethernet_cores().begin());
    RunTestOnCore(fixture, mesh_device, core, true, feature, false /*use_ncrisc*/, true /*is_idle_eth_core*/);
}

// Run tests for host-side sanitization (uses functions that are from watcher_server.hpp).
void CheckHostSanitization(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = mesh_device->get_devices()[0];
    // Try reading from a core that doesn't exist
    constexpr CoreCoord core = {99, 99};
    uint64_t addr = 0;
    uint32_t sz_bytes = 4;
    try {
        [[maybe_unused]] auto data =
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(device->id(), core, addr, sz_bytes);
    } catch (std::runtime_error& e) {
        const std::string expected = fmt::format("Host watcher: bad {} NOC coord {}\n", "read", core.str());
        const std::string error = std::string(e.what());
        log_info(tt::LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find(expected) != std::string::npos);
    }
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitize) {
    CheckHostSanitization(this->devices_[0]);

    // Only run on device 0 because this test takes down the watcher server.
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCAddress);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCAlignmentL1Write) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCAlignmentL1Write);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCAlignmentL1Read) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCAlignmentL1Read);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCAlignmentL1ReadNCrisc) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCAlignmentL1Read, true);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCZeroL1Write) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCZeroL1Write);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCMailboxWrite) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCMailboxWrite);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCMailboxWriteUncachedAlias) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCMailboxWriteUncachedAlias);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCInlineWriteDram) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCInlineWriteDram);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherSanitizeEth) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestEth(fixture, mesh_device, SanitizeNOCAddress);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherSanitizeNOCMailboxWrite) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestEth(fixture, mesh_device, SanitizeNOCMailboxWrite);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherSanitizeNOCInlineWriteDram) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestEth(fixture, mesh_device, SanitizeNOCInlineWriteDram);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, IdleEthTestWatcherSanitizeIEth) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestIEth(fixture, mesh_device, SanitizeNOCAddress);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, IdleEthTestWatcherSanitizeNOCInlineWriteDram) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestIEth(fixture, mesh_device, SanitizeNOCInlineWriteDram);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, DISABLED_SanitizeNOCLinkedTransaction) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCLinkedTransaction);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeL1Overflow) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeL1Overflow);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeL1OverflowStraddle) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeL1OverflowStraddle);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherSanitizeL1Overflow) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestEth(fixture, mesh_device, SanitizeL1Overflow);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherSanitizeEthSrcL1Overflow) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestEth(fixture, mesh_device, SanitizeEthSrcL1Overflow);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, ActiveEthTestWatcherSanitizeEthDestL1Overflow) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestEth(fixture, mesh_device, SanitizeEthDestL1Overflow);
        },
        this->devices_[0]);
}

TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeMulticastSemaphoreInc) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCMulticastInvalidRange);
        },
        this->devices_[0]);
}

// Regression test for the stateful-write NOC sanitizer: a write issued via set_async_write_state +
// async_write_with_state must be sanitized against the destination coordinate held in NOC_RET_ADDR. A
// sanitizer that reads NOC_TARG_ADDR instead would see the sender's own (valid) coordinate and report the
// wrong error (or none), so this test fails unless the destination is reconstructed from NOC_RET_ADDR.
TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCWriteWithState) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCWriteWithStateBadCoord);
        },
        this->devices_[0]);
}

// Regression test for the inline-dw-write NOC sanitizer, exercised through noc_inline_dw_write_set_state:
// the destination is programmed into the inline command buffer and then sanitized via
// DEBUG_SANITIZE_NOC_ADDR_FROM_STATE.
TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCInlineWriteFromState) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCInlineWriteFromState);
        },
        this->devices_[0]);
}

// Regression test for the inline-dw-write NOC sanitizer, exercised the way cq_noc_inline_dw_write_with_state
// does: the destination is programmed into the WR_REG command buffer and then sanitized via
// DEBUG_SANITIZE_NOC_ADDR_FROM_STATE.
TEST_F(MeshWatcherFixture, TensixTestWatcherSanitizeNOCInlineWriteWithState) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, mesh_device, core, false, SanitizeNOCInlineWriteWithState);
        },
        this->devices_[0]);
}

// Quasar multi-DM race test: all DMs sync then race to trigger sanitize error
// Each DM uses unique identifiable data (addr/size). Verifies CAS ensures consistent error reporting
TEST_F(MeshWatcherFixture, QuasarTestWatcherSanitizeMultiDMRace) {
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            CoreCoord core{0, 0};
            RunTestOnCore(
                fixture,
                mesh_device,
                core,
                false,
                SanitizeNOCAddress,
                false,
                false /*is_idle_eth_core*/,
                true /*multi_dm_race*/);
        },
        this->devices_[0]);
}
