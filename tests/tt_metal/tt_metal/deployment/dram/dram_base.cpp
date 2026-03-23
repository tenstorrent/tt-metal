#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "dram_base.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "command_queue_fixture.hpp"
#include "kernels/common_dram.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;

bool run_dram_base_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    auto* const device = mesh_device->get_devices()[0];

    TT_FATAL(cfg.bank_id < 8, "bank_id must not exceed the total number of controllers");
    TT_FATAL(cfg.total_bytes <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes must be under (4GB-16MB)");
    TT_FATAL(cfg.chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(cfg.total_bytes % sizeof(uint32_t) == 0, "total_bytes must be word aligned");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    MetalContext::instance().get_cluster().write_core(
        device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    auto kernel = tt_metal::CreateKernel(
        program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

    DramTestParameters params{
        .bank_id = cfg.bank_id,
        .bank_offset_lo = (uint32_t)(cfg.bank_offset & 0xFFFFFFFFull),
        .bank_offset_hi = (uint32_t)(cfg.bank_offset >> 32),
        .total_bytes = cfg.total_bytes,
        .chunk_bytes = cfg.chunk_bytes,
        .pattern_id = cfg.pattern_id,
        .seed = seed,
        .pass_index = pass_index,
        .repeat_index = repeat_index,
        .result_l1_addr = result_l1_address,
        .expect_l1_addr = expect_l1_address,
        .observe_l1_addr = observe_l1_address,
        .write_noc = cfg.write_noc,
        .read_noc = cfg.read_noc,
        .max_burst_len = cfg.max_burst_len,
        .transfer_len_mode = cfg.transfer_len_mode,
        .skip_writes = cfg.skip_writes,
        .skip_reads = cfg.skip_reads,
    };

    tt_metal::SetRuntimeArgs(
        program,
        kernel,
        core,
        {
            params.bank_id,
            params.bank_offset_lo,
            params.bank_offset_hi,
            params.total_bytes,
            params.chunk_bytes,
            params.pattern_id,
            params.seed,
            params.pass_index,
            params.repeat_index,
            params.result_l1_addr,
            params.expect_l1_addr,
            params.observe_l1_addr,
            params.write_noc,
            params.read_noc,
            params.max_burst_len,
            params.transfer_len_mode,
            params.skip_writes,
            params.skip_reads,
        });

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    auto raw_result = MetalContext::instance().get_cluster().read_core(
        device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

    const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

    const double tensix_freq_hz = 1.35e9;
    const double ms_per_tick = 1.0e3 / tensix_freq_hz;
    double prepare_time_ms = result->prepare_ticks * ms_per_tick;
    double write_time_ms = result->write_ticks * ms_per_tick;
    double read_time_ms = result->read_ticks * ms_per_tick;

    log_info(
        tt::LogTest,
        "{} pattern done on core {}: repeat={}, pass={}, bank={}, transfers={}",
        pattern_name(result->pattern_id),
        core,
        result->repeat_index,
        result->pass_index,
        result->bank_id,
        result->transfers);

    log_info(
        tt::LogTest,
        "{} pattern timing on core {}: prepare_time={:.3f}ms, write_time={:.3f}ms, read_time={:.3f}ms",
        pattern_name(result->pattern_id),
        core,
        prepare_time_ms,
        write_time_ms,
        read_time_ms);

    bool pass_test = !result->failures;
    if (!pass_test) {
        log_info(
            tt::LogTest,
            "Mismatch at core {}: failures={}, first_fail_addr=0x{:08x}, expected=0x{:08x}, observed=0x{:08x}",
            core,
            result->failures,
            result->first_fail_addr,
            result->first_expected,
            result->first_observed);
    }

    return pass_test;
}

bool run_dram_multi_core_single_controller_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    auto* const device = mesh_device->get_devices()[0];

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(cfg.bank_id < 8, "bank_id must not exceed the total number of controllers");
    TT_FATAL(cfg.total_bytes <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes must be under (4GB-16MB)");
    TT_FATAL(cfg.chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(cfg.total_bytes % 4096u == 0, "total_bytes must be 4KB aligned for multi-core controller mode");

    const uint64_t total_bytes = cfg.total_bytes;
    const uint64_t bytes_per_core = (total_bytes / cores.size()) & ~0xFFFULL;

    TT_FATAL(bytes_per_core >= cfg.chunk_bytes, "bytes_per_core too small");
    TT_FATAL(bytes_per_core <= std::numeric_limits<uint32_t>::max(), "bytes_per_core must fit into uint32_t");

    struct l1_allocator alloc = new_tensix_allocator();

    // Same L1 offsets reused on every core; L1 is core-local.
    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    for (size_t i = 0; i < cores.size(); ++i) {
        const CoreCoord core = cores[i];
        const uint64_t bank_offset = cfg.bank_offset + i * bytes_per_core;

        auto kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                cfg.bank_id,
                (uint32_t)bank_offset,
                (uint32_t)(bank_offset >> 32),
                (uint32_t)bytes_per_core,
                cfg.chunk_bytes,
                cfg.pattern_id,
                seed,
                pass_index,
                repeat_index,
                result_l1_address,
                expect_l1_address,
                observe_l1_address,
                cfg.write_noc,
                cfg.read_noc,
                cfg.max_burst_len,
                cfg.transfer_len_mode,
                cfg.skip_writes,
                cfg.skip_reads,
            });
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    bool all_pass = true;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

        const double tensix_freq_hz = 1.35e9;
        const double ms_per_tick = 1.0e3 / tensix_freq_hz;
        double prepare_time_ms = result->prepare_ticks * ms_per_tick;
        double write_time_ms = result->write_ticks * ms_per_tick;
        double read_time_ms = result->read_ticks * ms_per_tick;

        log_info(
            tt::LogTest,
            "{} pattern done on core {}: repeat={}, pass={}, bank={}, transfers={}",
            pattern_name(result->pattern_id),
            core,
            result->repeat_index,
            result->pass_index,
            result->bank_id,
            result->transfers);

        log_info(
            tt::LogTest,
            "{} pattern timing on core {}: prepare_time={:.3f}ms, write_time={:.3f}ms, read_time={:.3f}ms",
            pattern_name(result->pattern_id),
            core,
            prepare_time_ms,
            write_time_ms,
            read_time_ms);

        bool pass_test = !result->failures;
        all_pass &= pass_test;

        if (!pass_test) {
            log_info(
                tt::LogTest,
                "Mismatch at core {}: failures={}, first_fail_addr=0x{:08x}, expected=0x{:08x}, observed=0x{:08x}",
                core,
                result->failures,
                result->first_fail_addr,
                result->first_expected,
                result->first_observed);
        }
    }

    return all_pass;
}

bool run_dram_multi_core_all_controllers_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    auto* const device = mesh_device->get_devices()[0];

    constexpr uint32_t num_controllers = 8u;

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(
        total_bytes_per_controller <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes_per_controller must be under (4GB-16MB)");
    TT_FATAL(chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(total_bytes_per_controller % 4096u == 0, "total_bytes_per_controller must be 4KB aligned");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(&alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(&alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    // Split cores into 8 contiguous groups, one per controller.
    const size_t total_cores = cores.size();
    const size_t base_cores_per_controller = total_cores / num_controllers;
    const size_t remainder = total_cores % num_controllers;

    size_t core_begin = 0;

    for (uint32_t bank_id = 0; bank_id < num_controllers; ++bank_id) {
        const size_t cores_in_this_controller = base_cores_per_controller + (bank_id < remainder ? 1 : 0);

        if (cores_in_this_controller == 0) {
            continue;
        }

        const uint64_t bytes_per_core = ((uint64_t)total_bytes_per_controller / cores_in_this_controller) & ~0xFFFULL;

        TT_FATAL(bytes_per_core >= chunk_bytes, "bytes_per_core too small");
        TT_FATAL(bytes_per_core <= std::numeric_limits<uint32_t>::max(), "bytes_per_core must fit into uint32_t");

        for (size_t local_idx = 0; local_idx < cores_in_this_controller; ++local_idx) {
            const size_t global_idx = core_begin + local_idx;
            const CoreCoord core = cores[global_idx];
            const uint64_t bank_offset = local_idx * bytes_per_core;

            auto kernel = tt_metal::CreateKernel(
                program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

            tt_metal::SetRuntimeArgs(
                program,
                kernel,
                core,
                {
                    bank_id,
                    (uint32_t)bank_offset,
                    (uint32_t)(bank_offset >> 32),
                    (uint32_t)bytes_per_core,
                    chunk_bytes,
                    pattern_id,
                    seed,
                    pass_index,
                    repeat_index,
                    result_l1_address,
                    expect_l1_address,
                    observe_l1_address,
                    write_noc,
                    read_noc,
                    max_burst_len,
                    transfer_len_mode,
                    skip_writes,
                    skip_reads,
                });
        }

        core_begin += cores_in_this_controller;
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    bool all_pass = true;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

        const double tensix_freq_hz = 1.35e9;
        const double ms_per_tick = 1.0e3 / tensix_freq_hz;
        double prepare_time_ms = result->prepare_ticks * ms_per_tick;
        double write_time_ms = result->write_ticks * ms_per_tick;
        double read_time_ms = result->read_ticks * ms_per_tick;

        log_info(
            tt::LogTest,
            "{} pattern done on core {}: repeat={}, pass={}, bank={}, transfers={}",
            pattern_name(result->pattern_id),
            core,
            result->repeat_index,
            result->pass_index,
            result->bank_id,
            result->transfers);

        log_info(
            tt::LogTest,
            "{} pattern timing on core {}: prepare_time={:.3f}ms, write_time={:.3f}ms, read_time={:.3f}ms",
            pattern_name(result->pattern_id),
            core,
            prepare_time_ms,
            write_time_ms,
            read_time_ms);

        bool pass_test = !result->failures;
        all_pass &= pass_test;

        if (!pass_test) {
            log_info(
                tt::LogTest,
                "Mismatch at core {}: failures={}, first_fail_addr=0x{:08x}, expected=0x{:08x}, observed=0x{:08x}",
                core,
                result->failures,
                result->first_fail_addr,
                result->first_expected,
                result->first_observed);
        }
    }

    return all_pass;
}

}  // namespace tt::tt_metal
