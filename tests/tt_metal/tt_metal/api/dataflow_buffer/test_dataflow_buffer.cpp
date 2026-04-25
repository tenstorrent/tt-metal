// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

enum class DFBPorCType : uint8_t { DM, TENSIX };

class DFBImplicitSyncParamFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<bool> {};

static std::string ImplicitSyncParamName(const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "ImplicitSyncTrue" : "ImplicitSyncFalse";
}

// expected_output, when non-null, is compared against the device output instead of input.
// This is used for Tensix→DM ring-pressure tests where the device cycles through fewer
// unique ring slots than entries_per_core, so output != input by design.
void execute_program_and_verify(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    Program& program,
    const std::shared_ptr<distributed::MeshBuffer>& in_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& out_buffer,
    distributed::MeshCoordinate& zero_coord,
    std::vector<uint32_t>& input,
    bool verify_output = true,
    const std::vector<uint32_t>* expected_output = nullptr) {
    distributed::WriteShard(mesh_device->mesh_command_queue(), in_buffer, input, zero_coord, true);

    if (mesh_device->get_devices()[0]->arch() == ARCH::QUASAR) {
        // TODO #38042: Need to wait for data to be written, the barrier needs to be uplifted for Quasar
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::vector<uint32_t> rdback_dram;
        distributed::ReadShard(mesh_device->mesh_command_queue(), rdback_dram, in_buffer, zero_coord, true);

        tt_driver_atomics::mfence();

        EXPECT_EQ(rdback_dram, input);
    }

    // Execute using slow dispatch (DFBs not yet supported in MeshWorkload path)
    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    std::vector<uint32_t> output;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output, out_buffer, zero_coord, true);

    if (verify_output) {
        const std::vector<uint32_t>& expected = expected_output ? *expected_output : input;
        if (expected != output) {
            log_info(tt::LogTest, "Printing expected");
            for (auto i : expected) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
            log_info(tt::LogTest, "Printing output");
            for (auto i : output) {
                std::cout << i << " ";
            }
        }
        EXPECT_EQ(expected, output);
    }
}

// Runs a single DFB program on one or more cores and verifies output == input.
//
// When core_range_set contains N > 1 cores the global DRAM buffers are sized
// N x entries_per_core x entry_size and each core receives a unique
// chunk_offset (= core_idx * entries_per_core) so it accesses a disjoint
// slice of the buffer.  Multi-core use requires DM producer and consumer.
void run_single_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dfb_config,
    DFBPorCType producer_type,
    DFBPorCType consumer_type,
    const CoreRangeSet& core_range_set = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))),
    std::optional<uint32_t> num_entries_in_buffer = std::nullopt) {

    TT_FATAL(
        !(producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::TENSIX),
        "Both producer and consumer cannot be Tensix. At least one must be a DM kernel for NOC transfers.");
    TT_FATAL(
        core_range_set.num_cores() == 1 ||
            (producer_type == DFBPorCType::DM && consumer_type == DFBPorCType::DM),
        "Multi-core DFB programs only support DM producer and consumer.");

    const auto arch = mesh_device->get_devices()[0]->arch();
    const bool is_quasar = (arch == ARCH::QUASAR);

    if (!is_quasar) {
        // WH/BH DM: one BRISC (RISCV_0) as producer and one NCRISC (RISCV_1) as consumer.
        // Configs with num_producers > 1 or num_consumers > 1 require multi-threaded DM
        // which is not available on WH/BH.
        if (dfb_config.num_producers > 1 || dfb_config.num_consumers > 1) {
            GTEST_SKIP() << "WH/BH DFB supports only 1 DM producer (BRISC) and 1 DM consumer (NCRISC)";
        }
        // read_in / write_out are Quasar-only; the device-side kernel would fail to compile
        // if enable_implicit_sync=true is propagated as a compile-time arg.
        dfb_config.enable_implicit_sync = false;
    }

    Program program = CreateProgram();
    auto zero_coord = distributed::MeshCoordinate(0, 0);

    const uint32_t num_cores = core_range_set.num_cores();
    const uint32_t entries_per_core = num_entries_in_buffer.has_value() ? num_entries_in_buffer.value() : dfb_config.num_entries;
    const uint32_t entry_size = dfb_config.entry_size;
    // page_size = entry_size makes every entry independently addressable by page_id.
    const uint32_t total_buffer_size = num_cores * entries_per_core * entry_size;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = entry_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer:  [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    // Ceiling division so every producer gets a loop bound that covers the largest slice.
    // Producers whose page_id would exceed entries_per_core use the runtime bounds
    // check in the kernel to skip the out-of-range iteration.
    uint32_t num_entries_per_producer =
        (entries_per_core + dfb_config.num_producers - 1) / dfb_config.num_producers;
    const bool is_blocked = (dfb_config.cap == dfb::AccessPattern::BLOCKED);
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        num_entries_per_producer,
        (uint32_t)dfb_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    KernelHandle producer_kernel;
    if (producer_type == DFBPorCType::DM) {
        const std::string dm_producer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp";
        if (is_quasar) {
            producer_kernel = experimental::quasar::CreateKernel(
                program,
                dm_producer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
        } else {
            producer_kernel = CreateKernel(
                program,
                dm_producer_kernel_path,
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .compile_args = producer_cta});
        }
    } else {
        const std::string t6_producer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp";
        if (is_quasar) {
            producer_kernel = CreateKernel(
                program,
                t6_producer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarComputeConfig{
                    .num_threads_per_cluster = dfb_config.num_producers, .compile_args = producer_cta});
        } else {
            producer_kernel = CreateKernel(
                program, t6_producer_kernel_path, core_range_set, ComputeConfig{.compile_args = producer_cta});
        }
    }

    uint32_t num_entries_per_consumer = is_blocked
                                            ? entries_per_core
                                            : (entries_per_core + dfb_config.num_consumers - 1) / dfb_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)is_blocked,
        (uint32_t)dfb_config.enable_implicit_sync};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);

    KernelHandle consumer_kernel;
    if (consumer_type == DFBPorCType::DM) {
        const std::string dm_consumer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp";
        if (is_quasar) {
            consumer_kernel = experimental::quasar::CreateKernel(
                program,
                dm_consumer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
        } else {
            consumer_kernel = CreateKernel(
                program,
                dm_consumer_kernel_path,
                core_range_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .compile_args = consumer_cta});
        }
    } else {
        const std::string t6_consumer_kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp";
        if (is_quasar) {
            consumer_kernel = CreateKernel(
                program,
                t6_consumer_kernel_path,
                core_range_set,
                experimental::quasar::QuasarComputeConfig{
                    .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});
        } else {
            consumer_kernel = CreateKernel(
                program, t6_consumer_kernel_path, core_range_set, ComputeConfig{.compile_args = consumer_cta});
        }
    }

    auto logical_dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program, logical_dfb_id, producer_kernel, consumer_kernel);

    auto dfb = program.impl().get_dataflow_buffer(logical_dfb_id);
    const uint32_t producer_mask = dfb->config.producer_risc_mask;
    const uint32_t consumer_mask = dfb->config.consumer_risc_mask;

    // Build a per-core chunk-offset map (used for both runtime args and L1 pre-fill/verify).
    std::map<CoreCoord, uint32_t> core_to_chunk_offset;
    uint32_t core_idx = 0;
    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
            for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                core_to_chunk_offset[CoreCoord(x, y)] = core_idx++ * entries_per_core;
            }
        }
    }

    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
            for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                const CoreCoord core(x, y);
                const uint32_t chunk_offset = core_to_chunk_offset.at(core);
                SetRuntimeArgs(program, producer_kernel, core, {producer_mask, chunk_offset, entries_per_core});
                SetRuntimeArgs(
                    program, consumer_kernel, core,
                    {consumer_mask, (uint32_t)logical_dfb_id, chunk_offset, entries_per_core});
            }
        }
    }

    // Generate input once; shared by in_buffer write, L1 pre-fill, and verification.
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_buffer_size / sizeof(uint32_t));

    IDevice* device = mesh_device->get_devices()[0];
    // const uint32_t words_per_core = entries_per_core * entry_size / sizeof(uint32_t);

    // For Tensix → DM: pre-fill each core's DFB L1 with its input chunk so the
    // Tensix producer kernel can read from L1 while DM consumer drains to DRAM.
    //
    // l1_by_core addresses are not populated until allocate_dataflow_buffers() runs
    // during program compilation. Since this is a single-DFB test it is always placed at the L1 base allocator address.
    //
    // IMPORTANT: the slice written to L1 must be exactly the physical ring size
    // (dfb->total_size() bytes = num_entries * entry_size). Writing more than the ring
    // size would corrupt L1 beyond the ring (kernel stack, config structures, etc.).
    // For ring-pressure tests (entries_per_core > num_entries) only the first
    // num_entries slots are filled; the producer kernel cycles through those same
    // slots repeatedly, which is the expected behaviour.
    if (producer_type == DFBPorCType::TENSIX) {
        const uint32_t dfb_l1_addr =
            static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        const uint32_t ring_words = dfb->total_size() / sizeof(uint32_t);
        for (const CoreRange& cr : core_range_set.ranges()) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                    const CoreCoord core(x, y);
                    const uint32_t co = core_to_chunk_offset.at(core);
                    const uint32_t wpe = entry_size / sizeof(uint32_t);
                    std::vector<uint32_t> slice(ring_words, 0);
                    for (uint32_t p = 0; p < dfb_config.num_producers; p++) {
                        for (uint32_t e = 0; e < num_entries_per_producer; e++) {
                            const uint32_t page_id = co + e * dfb_config.num_producers + p;
                            if (page_id >= co + entries_per_core) {
                                break;
                            }
                            // Ring layout depends on stride_in_entries, which is set by the
                            // consumer access pattern:
                            //   STRIDED: stride = num_producers → interleaved (slot = e*P + p)
                            //   BLOCKED: stride = 1 → TC-first   (slot = p*E + e)
                            const uint32_t dst_slot =
                                (dfb_config.cap == dfb::AccessPattern::BLOCKED)
                                    ? (p * num_entries_per_producer + e)
                                    : (e * dfb_config.num_producers + p);

                            // Stop once all physical ring slots are filled; for ring-pressure
                            // tests the remaining iterations would alias back to already-filled
                            // slots, so there is nothing new to write.
                            if (dst_slot >= dfb_config.num_entries) {
                                break;
                            }

                            std::copy(
                                input.begin() + page_id * wpe,
                                input.begin() + page_id * wpe + wpe,
                                slice.begin() + dst_slot * wpe);
                        }
                    }
                    detail::WriteToDeviceL1(device, core, dfb_l1_addr, slice);
                }
            }
        }
    }

    // For Tensix → DM ring-pressure tests (entries_per_core > num_entries), the
    // Tensix producer cycles through the same num_entries ring slots indefinitely.
    // Each STRIDED consumer c always reads ring slot (c % num_entries), which was
    // pre-filled with input page c.  The expected out_buffer page p therefore
    // contains the data from ring slot (p % num_consumers) % num_entries, not
    // input[p].  Build the corrected expected vector so the verification is sound.
    std::optional<std::vector<uint32_t>> tensix_dm_expected;
    if (producer_type == DFBPorCType::TENSIX && consumer_type == DFBPorCType::DM &&
        entries_per_core > dfb_config.num_entries &&
        dfb_config.cap == dfb::AccessPattern::STRIDED) {
        const uint32_t wpe = entry_size / sizeof(uint32_t);
        tensix_dm_expected.emplace(num_cores * entries_per_core * wpe, 0u);
        for (const CoreRange& cr : core_range_set.ranges()) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                    const CoreCoord core(x, y);
                    const uint32_t co = core_to_chunk_offset.at(core);
                    for (uint32_t p = 0; p < entries_per_core; p++) {
                        // Consumer c = p % num_consumers always reads the ring slot it
                        // was assigned (slot = c % num_entries), which holds input[co + c].
                        const uint32_t ring_slot =
                            (p % dfb_config.num_consumers) % dfb_config.num_entries;
                        std::copy(
                            input.begin() + (co + ring_slot) * wpe,
                            input.begin() + (co + ring_slot + 1) * wpe,
                            tensix_dm_expected->begin() + (co + p) * wpe);
                    }
                }
            }
        }
    }

    // Launch program; verify out_buffer only for DM → DM paths (Tensix consumer
    // does not write to DRAM, so out_buffer verification is skipped there).
    execute_program_and_verify(
        mesh_device, program, in_buffer, out_buffer, zero_coord,
        input,
        /*verify_output=*/(consumer_type == DFBPorCType::DM),
        tensix_dm_expected ? &*tensix_dm_expected : nullptr);

    // For DM → Tensix: verify each core's DFB L1 against the expected input chunk.
    if (consumer_type == DFBPorCType::TENSIX) {
        for (const auto& group : dfb->groups) {
            for (const auto& [core, alloc_addr] : group.l1_by_core) {
                const uint32_t co = core_to_chunk_offset.at(core);
                std::vector<uint32_t> l1_data;
                detail::ReadFromDeviceL1(device, core, alloc_addr, dfb->total_size(), l1_data);
                const uint32_t wpe_v = entry_size / sizeof(uint32_t);
                // Physical ring holds dfb_config.num_entries entries; for ring-pressure
                // tests (entries_per_core > dfb_config.num_entries) the ring wraps and
                // only the last ring_capacity writes per producer survive in L1.
                // Size expected to l1_data so the comparison is against what is actually there.
                const uint32_t total_ring_words = dfb->total_size() / sizeof(uint32_t);
                std::vector<uint32_t> expected(total_ring_words, 0);
                if (dfb_config.cap == dfb::AccessPattern::BLOCKED) {
                    // BLOCKED consumer: ring is TC-first (stride_in_entries=1).
                    // Each producer p has ring_capacity consecutive ring slots.
                    // After wrapping, only the last ring_capacity entries from each
                    // producer survive: e in [num_entries_per_producer - ring_capacity, ...).
                    const uint32_t ring_capacity = dfb_config.num_entries / dfb_config.num_producers;
                    const uint32_t last_e_base = num_entries_per_producer - ring_capacity;
                    for (uint32_t p = 0; p < dfb_config.num_producers; p++) {
                        for (uint32_t c = 0; c < ring_capacity; c++) {
                            const uint32_t ring_slot = p * ring_capacity + c;
                            const uint32_t e = last_e_base + c;
                            const uint32_t page_id = co + e * dfb_config.num_producers + p;
                            if (page_id >= co + entries_per_core) {
                                break;
                            }
                            std::copy(
                                input.begin() + page_id * wpe_v,
                                input.begin() + page_id * wpe_v + wpe_v,
                                expected.begin() + ring_slot * wpe_v);
                        }
                    }
                } else {
                    // STRIDED consumer: ring is interleaved, matching sequential input order.
                    // For ring-pressure tests (entries_per_core > dfb_config.num_entries) only
                    // the last dfb_config.num_entries entries survive in L1; copy that suffix.
                    const uint32_t ring_start_page = co + entries_per_core - dfb_config.num_entries;
                    std::copy(
                        input.begin() + ring_start_page * wpe_v,
                        input.begin() + ring_start_page * wpe_v + total_ring_words,
                        expected.begin());
                }
                if (expected != l1_data) {
                    std::cout << "expected: ";
                    for (const auto& e : expected) {
                        std::cout << e << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "l1_data: ";
                    for (const auto& l : l1_data) {
                        std::cout << l << " ";
                    }
                    std::cout << std::endl;
                }
                EXPECT_EQ(expected, l1_data)
                    << "DFB L1 mismatch on core (" << core.x << "," << core.y << ")";
            }
        }
    }
}

void run_in_dfb_out_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    experimental::dfb::DataflowBufferConfig& dm2tensix_config,
    experimental::dfb::DataflowBufferConfig& tensix2dm_config) {
    TT_FATAL(
        dm2tensix_config.num_entries == tensix2dm_config.num_entries,
        "Num entries must be the same for in and out DFBs");
    TT_FATAL(
        dm2tensix_config.entry_size == tensix2dm_config.entry_size, "Entry size must be the same for in and out DFBs");

    Program program = CreateProgram();

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    uint32_t buffer_size = dm2tensix_config.entry_size * dm2tensix_config.num_entries;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    log_info(tt::LogTest, "In Buffer: [address: {} B, size: {} B]", in_buffer->address(), in_buffer->size());
    log_info(tt::LogTest, "Out Buffer: [address: {} B, size: {} B]", out_buffer->address(), out_buffer->size());

    CoreCoord logical_core = CoreCoord(0, 0);

    uint32_t num_entries_per_producer = dm2tensix_config.num_entries / dm2tensix_config.num_producers;
    const bool in_is_blocked = (dm2tensix_config.cap == dfb::AccessPattern::BLOCKED);
    std::vector<uint32_t> producer_cta = {
        (uint32_t)in_buffer->address(),
        num_entries_per_producer,
        0 /*implicit_sync=false*/,
        (uint32_t)in_is_blocked};
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(producer_cta);

    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = dm2tensix_config.num_producers, .compile_args = producer_cta});

    uint32_t num_entries_per_unpacker = dm2tensix_config.num_entries / dm2tensix_config.num_consumers;
    uint32_t num_entries_per_packer = tensix2dm_config.num_entries / tensix2dm_config.num_producers;
    TT_FATAL(
        num_entries_per_unpacker == num_entries_per_packer, "Num entries per unpacker and packer must be the same");
    std::vector<uint32_t> compute_cta = {num_entries_per_unpacker};
    auto compute_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6.cpp",
        logical_core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 1, .compile_args = compute_cta});

    const bool out_is_blocked = (tensix2dm_config.cap == dfb::AccessPattern::BLOCKED);
    uint32_t num_entries_per_consumer = out_is_blocked ? tensix2dm_config.num_entries : tensix2dm_config.num_entries / tensix2dm_config.num_consumers;
    std::vector<uint32_t> consumer_cta = {
        (uint32_t)out_buffer->address(),
        num_entries_per_consumer,
        (uint32_t)out_is_blocked,
        0 /*implicit_sync=false*/};
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(consumer_cta);
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        logical_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = tensix2dm_config.num_consumers, .compile_args = consumer_cta});

    auto input_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, dm2tensix_config);
    auto output_dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, tensix2dm_config);

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, input_dfb_id, producer_kernel, compute_kernel);
    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, output_dfb_id, compute_kernel, consumer_kernel);

    auto input_dfb = program.impl().get_dataflow_buffer(input_dfb_id);
    auto output_dfb = program.impl().get_dataflow_buffer(output_dfb_id);

    SetRuntimeArgs(program, producer_kernel, logical_core, {(uint32_t)input_dfb->config.producer_risc_mask, 0u});
    SetRuntimeArgs(
        program,
        compute_kernel,
        logical_core,
        {(uint32_t)input_dfb_id, (uint32_t)output_dfb_id});
    SetRuntimeArgs(program, consumer_kernel, logical_core, {(uint32_t)output_dfb->config.consumer_risc_mask, (uint32_t)output_dfb_id, 0u});

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, buffer_size / sizeof(uint32_t));
    execute_program_and_verify(mesh_device, program, in_buffer, out_buffer, zero_coord, input);
}

// =====================================================================================
// Single-DFB test macros
//
// WH/BH supports only 1 DM producer (BRISC) + 1 DM consumer (NCRISC) and no implicit_sync,
// so 1x1 configurations may run there with implicit_sync=false; everything else is Quasar-only.
// =====================================================================================

#define DFB_SKIP_IF_UNSUPPORTED(num_p, num_c)                                                           \
    if (devices_.at(0)->arch() != ARCH::QUASAR && (GetParam() || (num_p) > 1 || (num_c) > 1)) {        \
        GTEST_SKIP();                                                                                   \
    }

// DM -> blocked DM is unsupported with implicit_sync today.
#define DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC                                                            \
    if (GetParam()) {                                                                                   \
        GTEST_SKIP() << "Skipping DM to blocked DM with implicit sync until support is added";          \
    }

#define DFB_NO_EXTRA_SKIP ((void)0)

constexpr uint32_t dfb_gcd(uint32_t a, uint32_t b) { return b == 0 ? a : dfb_gcd(b, a % b); }

constexpr uint32_t dfb_default_num_entries(uint32_t num_p, uint32_t num_c) {
    const uint32_t m = (num_p / dfb_gcd(num_p, num_c)) * num_c;
    return ((16u + m - 1u) / m) * m;
}

// Single-DFB test on a single core with default ring derived from (num_p, num_c)
// and entry_size=1024.
//   prefix       e.g. DM, DMTensix, TensixDM
//   suffix       e.g. 3Sx1S, 6Sx2B
//   p_kind/c_kind   DM | TENSIX
//   num_p/num_c     1..6 (DM); 1..4 (TENSIX)
//   pap_kind/cap_kind   STRIDED | BLOCKED
//   extra_skip      DFB_NO_EXTRA_SKIP or DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC
#define DFB_TEST(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip)          \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                    \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                      \
        extra_skip;                                                                                     \
        experimental::dfb::DataflowBufferConfig config{                                                 \
            .entry_size = 1024,                                                                         \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                   \
            .num_producers = (num_p),                                                                   \
            .pap = dfb::AccessPattern::pap_kind,                                                        \
            .num_consumers = (num_c),                                                                   \
            .cap = dfb::AccessPattern::cap_kind,                                                        \
            .enable_implicit_sync = GetParam()};                                                        \
        run_single_dfb_program(                                                                         \
            this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind);                    \
    }

// Variant for DM->DM tests that pass an explicit num_entries_in_buffer (forces wraparound
// when the requested total exceeds the ring size).
#define DFB_TEST_BUF(prefix, suffix, p_kind, c_kind, num_p, pap_kind, num_c, cap_kind, extra_skip, n_buf) \
    TEST_P(DFBImplicitSyncParamFixture, prefix##Test1xDFB##suffix) {                                    \
        DFB_SKIP_IF_UNSUPPORTED((num_p), (num_c));                                                      \
        extra_skip;                                                                                     \
        experimental::dfb::DataflowBufferConfig config{                                                 \
            .entry_size = 1024,                                                                         \
            .num_entries = dfb_default_num_entries((num_p), (num_c)),                                   \
            .num_producers = (num_p), .pap = dfb::AccessPattern::pap_kind,                              \
            .num_consumers = (num_c), .cap = dfb::AccessPattern::cap_kind,                              \
            .enable_implicit_sync = GetParam()};                                                        \
        CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));                       \
        run_single_dfb_program(                                                                         \
            this->devices_.at(0), config, DFBPorCType::p_kind, DFBPorCType::c_kind,                     \
            core_range_set, (n_buf));                                                                   \
    }

// =====================================================================================
// Strided
// =====================================================================================

// 1x1 (DM->DM uses num_entries_in_buffer=18 to exercise wraparound)
DFB_TEST_BUF(DM,       1Sx1S, DM,     DM,     1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP, 18)
DFB_TEST    (DMTensix, 1Sx1S, DM,     TENSIX, 1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx1S, TENSIX, DM,     1, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

// TEST_F(MeshDeviceFixture, DMTensixDMTest2xDFB1Sx1S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//             .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 2,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 2,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

// TEST_F(MeshDeviceFixture, DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S) {
//     if (devices_.at(0)->arch() != ARCH::QUASAR) {
//         GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
//     }
//     experimental::dfb::DataflowBufferConfig dm2tensix_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 4,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 1,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     experimental::dfb::DataflowBufferConfig tensix2dm_config{
//         .entry_size = 1024,
//         .num_entries = 16,
//         .num_producers = 1,
//         .pap = dfb::AccessPattern::STRIDED,
//         .num_consumers = 4,
//         .cap = dfb::AccessPattern::STRIDED,
//         .enable_implicit_sync = false};

//     run_in_dfb_out_dfb_program(this->devices_.at(0), dm2tensix_config, tensix2dm_config);
// }

DFB_TEST    (DM,       1Sx4S, DM,     DM,     1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 1Sx4S, DM,     TENSIX, 1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx4S, TENSIX, DM,     1, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx1S, DM,     DM,     4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 4Sx1S, DM,     TENSIX, 4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx1S, TENSIX, DM,     4, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST_BUF(DM,       4Sx4S, DM,     DM,     4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP, 29)
DFB_TEST    (DMTensix, 4Sx4S, DM,     TENSIX, 4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx4S, TENSIX, DM,     4, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST_BUF(DM,       2Sx4S, DM,     DM,     2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP, 21)
DFB_TEST    (DMTensix, 2Sx4S, DM,     TENSIX, 2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx4S, TENSIX, DM,     2, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx2S, DM,     DM,     4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 4Sx2S, DM,     TENSIX, 4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx2S, TENSIX, DM,     4, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->DM strided: power-of-2 (1Sx2S, 2Sx1S, 2Sx2S)
DFB_TEST    (DM,       1Sx2S, DM,     DM,     1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       2Sx1S, DM,     DM,     2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       2Sx2S, DM,     DM,     2, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->DM strided: 3-DM consumer column
DFB_TEST    (DM,       1Sx3S, DM,     DM,     1, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DM,       2Sx3S, DM,     DM,     2, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP) // needs contiguous access pattern

DFB_TEST    (DM,       3Sx1S, DM,     DM,     3, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DM,       3Sx2S, DM,     DM,     3, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP) // needs contiguous access pattern
DFB_TEST    (DM,       3Sx3S, DM,     DM,     3, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       1Sx5S, DM,     DM,     1, STRIDED, 5, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DM,       5Sx1S, DM,     DM,     5, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)

// DM->Tensix strided (Tensix consumers limited to {1,2,4})
// Power-of-2 gaps
DFB_TEST    (DMTensix, 1Sx2S, DM,     TENSIX, 1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 2Sx1S, DM,     TENSIX, 2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// 3-DM producer
DFB_TEST    (DMTensix, 3Sx1S, DM,     TENSIX, 3, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DMTensix, 3Sx2S, DM,     TENSIX, 3, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP) // needs contiguous access pattern
// DFB_TEST    (DMTensix, 3Sx4S, DM,     TENSIX, 3, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP) // needs contiguous access pattern
// 6-DM producer
DFB_TEST    (DMTensix, 6Sx1S, DM,     TENSIX, 6, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx2S, DM,     TENSIX, 6, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (DMTensix, 6Sx4S, DM,     TENSIX, 6, STRIDED, 4, STRIDED, DFB_NO_EXTRA_SKIP) // needs contiguous access pattern

// Tensix->DM strided (Tensix producers limited to {1,2,4})
// Power-of-2 gaps
DFB_TEST    (TensixDM, 2Sx1S, TENSIX, DM,     2, STRIDED, 1, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx2S, TENSIX, DM,     1, STRIDED, 2, STRIDED, DFB_NO_EXTRA_SKIP)
// 3-DM consumer
DFB_TEST    (TensixDM, 1Sx3S, TENSIX, DM,     1, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx3S, TENSIX, DM,     2, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx3S, TENSIX, DM,     4, STRIDED, 3, STRIDED, DFB_NO_EXTRA_SKIP) // needs contiguous access pattern
// 6-DM consumer
DFB_TEST    (TensixDM, 1Sx6S, TENSIX, DM,     1, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx6S, TENSIX, DM,     2, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx6S, TENSIX, DM,     4, STRIDED, 6, STRIDED, DFB_NO_EXTRA_SKIP) // needs contiguous access pattern

// =====================================================================================
// Blocked
// =====================================================================================

DFB_TEST    (DM,       1Sx4B, DM,     DM,     1, STRIDED, 4, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 1Sx4B, DM,     TENSIX, 1, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 1Sx4B, TENSIX, DM,     1, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx1B, DM,     DM,     4, STRIDED, 1, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx1B, DM,     TENSIX, 4, STRIDED, 1, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx1B, TENSIX, DM,     4, STRIDED, 1, BLOCKED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx4B, DM,     DM,     4, STRIDED, 4, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx4B, DM,     TENSIX, 4, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx4B, TENSIX, DM,     4, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       4Sx2B, DM,     DM,     4, STRIDED, 2, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 4Sx2B, DM,     TENSIX, 4, STRIDED, 2, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 4Sx2B, TENSIX, DM,     4, STRIDED, 2, BLOCKED, DFB_NO_EXTRA_SKIP)

DFB_TEST    (DM,       2Sx4B, DM,     DM,     2, STRIDED, 4, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DMTensix, 2Sx4B, DM,     TENSIX, 2, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (TensixDM, 2Sx4B, TENSIX, DM,     2, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)

// DM->DM blocked: 3-DM producer
DFB_TEST    (DM,       3Sx1B, DM,     DM,     3, STRIDED, 1, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DM,       3Sx2B, DM,     DM,     3, STRIDED, 2, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DM,       3Sx3B, DM,     DM,     3, STRIDED, 3, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)

// DM->DM blocked: 3-DM consumer
DFB_TEST    (DM,       1Sx3B, DM,     DM,     1, STRIDED, 3, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)
DFB_TEST    (DM,       2Sx3B, DM,     DM,     2, STRIDED, 3, BLOCKED, DFB_SKIP_DM_DM_BLOCKED_IMPLICIT_SYNC)

// DM->Tensix blocked (Tensix consumers limited to {1,2,4})
DFB_TEST    (DMTensix, 3Sx1B, DM,     TENSIX, 3, STRIDED, 1, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 3Sx2B, DM,     TENSIX, 3, STRIDED, 2, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 3Sx4B, DM,     TENSIX, 3, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx1B, DM,     TENSIX, 6, STRIDED, 1, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx2B, DM,     TENSIX, 6, STRIDED, 2, BLOCKED, DFB_NO_EXTRA_SKIP)
DFB_TEST    (DMTensix, 6Sx4B, DM,     TENSIX, 6, STRIDED, 4, BLOCKED, DFB_NO_EXTRA_SKIP)

// Tensix->DM blocked (Tensix producers limited to {1,2,4}).
// DFB_TEST    (TensixDM, 1Sx3B, TENSIX, DM,     1, STRIDED, 3, BLOCKED, DFB_NO_EXTRA_SKIP) // revisit the 3B Tensix consumer
// DFB_TEST    (TensixDM, 2Sx3B, TENSIX, DM,     2, STRIDED, 3, BLOCKED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx3B, TENSIX, DM,     4, STRIDED, 3, BLOCKED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 1Sx6B, TENSIX, DM,     1, STRIDED, 6, BLOCKED, DFB_NO_EXTRA_SKIP) // revisit more than 4 blocked consumers
// DFB_TEST    (TensixDM, 2Sx6B, TENSIX, DM,     2, STRIDED, 6, BLOCKED, DFB_NO_EXTRA_SKIP)
// DFB_TEST    (TensixDM, 4Sx6B, TENSIX, DM,     4, STRIDED, 6, BLOCKED, DFB_NO_EXTRA_SKIP)

// 1 strided DM producer, 1 strided DM consumer, num_entries=4.
// Ring wraps 16x; baseline to confirm wraparound logic with minimum participants.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_RingPressure_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(1, 1);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

// 4 strided DM producers, 4 strided DM consumers, num_entries=4 -> capacity=1.
// Each producer stalls after every push; ring wraps 64x per producer.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_RingPressure_4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(4, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

// 4 DM producers (STRIDED), 4 Tensix consumers (BLOCKED), num_entries=4 -> capacity=1.
// Every push stalls until all 4 blocked Tensix consumers ack; ring wraps 64x per producer.
// Exercises remapper fan-out on the DM->Tensix path under maximum ring pressure.
TEST_P(DFBImplicitSyncParamFixture, DMTensixTest1xDFB_RingPressure_4Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(4, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,  // tight ring: capacity = num_entries / num_producers = 1
        .num_producers = 4,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::TENSIX,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

// 2 Tensix producers (STRIDED), 4 DM consumers (STRIDED), num_entries=4 -> capacity=1.
// Ring wraps 64x per producer; exercises the Tensix->DM path with asymmetric P:C ratio
// under tight ring pressure (num_consumers > num_producers).
TEST_P(DFBImplicitSyncParamFixture, TensixDMTest1xDFB_RingPressure_2Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB ring-pressure test for WH/BH until DFB is backported";
    }
    DFB_SKIP_IF_UNSUPPORTED(2, 4);
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,  // tight ring: capacity = num_entries / max(num_p, num_c) = 1
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};
    run_single_dfb_program(
        this->devices_.at(0), config, DFBPorCType::TENSIX, DFBPorCType::DM,
        CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), /*num_entries_in_buffer=*/64);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx1S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_2Sx2S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

TEST_P(DFBImplicitSyncParamFixture, MultiCoreDMTest2Core_1Sx4B) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::BLOCKED,
        .enable_implicit_sync = GetParam()};

    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));
    run_single_dfb_program(this->devices_.at(0), config, DFBPorCType::DM, DFBPorCType::DM, core_range_set);
}

INSTANTIATE_TEST_SUITE_P(
    ImplicitSync,
    DFBImplicitSyncParamFixture,
    ::testing::Bool(),
    ImplicitSyncParamName);

}  // end namespace tt::tt_metal
