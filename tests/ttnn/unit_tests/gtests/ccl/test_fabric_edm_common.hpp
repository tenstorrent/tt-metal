// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "tt_metal/fabric/ccl/ccl_common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/reshape/view.hpp"
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tile.hpp>

#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "gtest/gtest.h"
#include "common/tt_backend_api_types.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_set>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

enum TwoInputReaderKernelWriteMode { LOCAL_WRITEBACK, FABRIC_UNICAST, FABRIC_MULTICAST };

static constexpr size_t TEST_WORKERS_SUBDEVICE_INDEX = 0;
static constexpr size_t TEST_EDM_FABRIC_SUBDEVICE_INDEX = 1;

using tt::tt_metal::distributed::MeshContainer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshDeviceView;
using tt::tt_metal::distributed::MeshShape;
using tt::tt_metal::distributed::SystemMesh;

class BaseFabricFixture {
protected:
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::size_t num_devices_{};
    bool device_open = false;

    // Common constants for both fixtures
    static constexpr size_t TG_NUM_DEVICES = 36;
    static constexpr size_t GALAXY_6U_NUM_DEVICES = 32;

    // Gets the appropriate mesh shape based on device configuration
    MeshShape GetDeterminedMeshShape() const {
        if (num_devices_ == TG_NUM_DEVICES || num_devices_ == GALAXY_6U_NUM_DEVICES) {
            return MeshShape{8, 4};
        }
        if (num_devices_ == 4) {
            return MeshShape{1, 4};
        }
        return MeshShape{2, 4};
    }

    // Validates environment and hardware for tests
    void ValidateEnvironment() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

        switch (arch_) {
            case tt::ARCH::WORMHOLE_B0:
                if (!(num_devices_ >= 8 && (tt::tt_metal::GetNumPCIeDevices() == 4 ||
                                            tt::tt_metal::GetNumPCIeDevices() == GALAXY_6U_NUM_DEVICES))) {
                    TT_THROW("This suite can only be run on T3000 or TG Wormhole devices");
                }
                break;

            case tt::ARCH::BLACKHOLE:
                if (num_devices_ != 4) {
                    TT_THROW("This suite can only be run on LLMBox");
                }
                break;

            default: TT_THROW("Only Wormhole or Blackhole devices are supported in this test suite");
        };
    }

public:
    BaseFabricFixture() = default;

    BaseFabricFixture(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {
        tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode);
    }

    virtual ~BaseFabricFixture() { tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED); }

    virtual void SetupDevices() = 0;
    virtual void TearDown() = 0;
};

class Fabric1DDeviceInitFixture {
public:
    tt::ARCH arch_{tt::ARCH::Invalid};
    std::size_t num_devices_{};
    bool device_open = false;

    // Common constants for both fixtures
    static constexpr size_t TG_NUM_DEVICES = 36;
    static constexpr size_t GALAXY_6U_NUM_DEVICES = 32;

    std::shared_ptr<MeshDevice> mesh_device_;

    // Gets the appropriate mesh shape based on device configuration
    MeshShape GetDeterminedMeshShape() const { return SystemMesh::instance().shape(); }

    // Validates environment and hardware for tests
    void ValidateEnvironment() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

        if (!(arch_ == tt::ARCH::WORMHOLE_B0 && num_devices_ >= 8 &&
              (tt::tt_metal::GetNumPCIeDevices() == 4 || tt::tt_metal::GetNumPCIeDevices() == GALAXY_6U_NUM_DEVICES))) {
            TT_THROW("This suite can only be run on T3000 or TG Wormhole devices");
        }
    }

    void SetupDevices() {
        ValidateEnvironment();

        const MeshShape cluster_shape = GetDeterminedMeshShape();

        mesh_device_ = MeshDevice::create(MeshDeviceConfig(cluster_shape));
        device_open = true;
    }

    void TearDown() {
        if (device_open) {
            mesh_device_->close();
            device_open = false;
        }
    }

    Fabric1DDeviceInitFixture() { this->SetupDevices(); }

    Fabric1DDeviceInitFixture(
        tt::tt_fabric::FabricConfig fabric_config,
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {
        tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode);
        this->SetupDevices();
    }

    ~Fabric1DDeviceInitFixture() {
        TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

class MeshFabric1DFixture : public BaseFabricFixture {
public:
    std::shared_ptr<MeshDevice> mesh_device_;

    void SetupDevices() override {
        ValidateEnvironment();
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(GetDeterminedMeshShape()));
        device_open = true;
    }

    void TearDown() override {
        if (device_open) {
            mesh_device_->close();
            device_open = false;
        }
    }

    MeshFabric1DFixture() { this->SetupDevices(); }

    MeshFabric1DFixture(tt::tt_fabric::FabricConfig fabric_config) : BaseFabricFixture(fabric_config) {
        this->SetupDevices();
    }

    ~MeshFabric1DFixture() override {
        if (device_open) {
            TearDown();
        }
    }
};

class Fabric1DRingStrictDeviceInitFixture : public Fabric1DDeviceInitFixture {
public:
    Fabric1DRingStrictDeviceInitFixture() :
        Fabric1DDeviceInitFixture(
            tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {}
};

class Fabric1DRingRelaxedDeviceInitFixture : public Fabric1DDeviceInitFixture {
public:
    Fabric1DRingRelaxedDeviceInitFixture() :
        Fabric1DDeviceInitFixture(
            tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
            tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE) {}
};

class MeshFabric1DLineDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DLineDeviceInitFixture() : MeshFabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D) {}
};

class MeshFabric1DRingDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DRingDeviceInitFixture() : MeshFabric1DFixture(tt::tt_fabric::FabricConfig::FABRIC_1D_RING) {}
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    BufferType input_buffer_type;
    BufferType output_buffer_type;
    tt::DataFormat l1_data_format;
};

struct KernelXY {
    uint16_t x;
    uint16_t y;

    uint32_t to_uint32() const { return y << 16 | x; }
};

enum Correctness { Correct, Incorrect };

template <typename CONTAINER_T>
Correctness run_output_check(CONTAINER_T const& inputs, CONTAINER_T output_buffer) {
    constexpr bool debug_mode = true;

    log_info(tt::LogTest, "Checking outputs");
    bool pass = true;

    std::size_t num_printed_mismatches = 0;
    for (size_t i = 0; i < inputs.size() && num_printed_mismatches < 64; i++) {
        if (output_buffer[i] != inputs[i]) {
            if (debug_mode) {
                if (pass) {
                    log_error(tt::LogTest, "Output mismatch");
                }
                log_error(tt::LogTest, "[{}]: expected {} got {}", i, inputs[i], output_buffer[i]);
                num_printed_mismatches++;
            }
            pass = false;
        }
    }
    if (num_printed_mismatches > 0) {
        log_error(tt::LogTest, "... (remaining mismatches omitted)");
    }

    log_info(tt::LogTest, "Output check: {}", pass ? "PASS" : "FAIL");
    return pass ? Correctness::Correct : Correctness::Incorrect;
};

inline void run_workloads(
    std::vector<tt::tt_metal::distributed::MeshWorkload>& workloads,
    const std::vector<std::shared_ptr<MeshDevice>>& devices) {
    EXPECT_EQ(workloads.size(), devices.size());
    const size_t num_workloads = workloads.size();

    log_info(tt::LogTest, "Running...");

    log_debug(tt::LogTest, "Calling Finish");
    std::vector<std::thread> threads;
    threads.reserve(num_workloads);
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        for (size_t i = 0; i < num_workloads; i++) {
            threads.emplace_back(std::thread([&] {
                tt_metal::distributed::EnqueueMeshWorkload(devices.at(i)->mesh_command_queue(), workloads.at(i), true);
            }));
        }

        std::ranges::for_each(threads, [](std::thread& t) { t.join(); });
    } else {
        for (size_t i = 0; i < num_workloads; i++) {
            tt_metal::distributed::EnqueueMeshWorkload(devices.at(i)->mesh_command_queue(), workloads.at(i), false);
        }

        for (size_t i = 0; i < num_workloads; i++) {
            tt_metal::distributed::Finish(devices.at(i)->mesh_command_queue());
        }
    }
}

struct EthLinkHop {
    CoreCoord hop_src;
    CoreCoord hop_dest;
};

struct ChipConnection {
    std::vector<EthLinkHop> links;
};

struct unicast_send {
    size_t distance;
};
struct mcast_send {
    size_t distance;
    size_t range;
};

using mode_variant_t = std::variant<mcast_send, unicast_send>;

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////               LOCAL CHIP TENSOR READ?WRITE (2 INPUT)
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

inline ttnn::ccl::Shape4D<uint32_t> shape_to_shape_in_tiles(const ttnn::Shape& shape) {
    auto logical_shape = shape;
    logical_shape[-2] /= tt::constants::TILE_HEIGHT;
    logical_shape[-1] /= tt::constants::TILE_WIDTH;
    EXPECT_TRUE(logical_shape.size() == 4);
    ttnn::ccl::Shape4D<uint32_t> shape_in_tiles = {
        logical_shape[0], logical_shape[1], logical_shape[2], logical_shape[3]};
    return shape_in_tiles;
}

bool RunPipelinedWorkersTest(

    ttnn::Shape tensor_shape,
    const size_t split_dim,

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    const size_t num_stages,
    std::vector<size_t> num_workers_per_stage,
    const size_t slices_per_stage,
    const tt::DataFormat data_format,
    const size_t page_size_bytes,
    const size_t cb_packet_size_in_pages,
    const size_t /*num_packets_per_cb*/,
    auto layout,

    std::vector<std::vector<size_t>> worker_chunk_read_order,
    std::vector<MemoryConfig> mem_configs) {
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return true;
    }

    const auto cb_index = tt::CB::c_in0;

    Program program = tt::tt_metal::CreateProgram();

    MeshFabric1DFixture test_fixture;
    auto full_mesh_device = test_fixture.mesh_device_;

    std::shared_ptr<distributed::MeshDevice> mesh_device =
        full_mesh_device->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));

    // General setup is as follows:
    // Worker 1 reads input tensor as a sequence of slices - it forwards to an output tensor and after each slice, it
    // writes a semaphore increment to some known semaphore address on the destination worker so the destination worker
    // knows it's safe to read that slice.
    // HOWEVER. the reader will be programmed to read the chunks in a different order than they were written, this way
    // we can identify synchronization related bugs (e.g. if sender semaphore increments before writes flush)

    TT_FATAL(num_workers_per_stage.size() == num_stages, "Must have a read order for each stage");
    TT_FATAL(worker_chunk_read_order.size() == num_stages, "Must have a read order for each stage");
    for (size_t i = 0; i < num_stages; ++i) {
        TT_FATAL(worker_chunk_read_order[i].size() == slices_per_stage, "Must have a read order for each slice");
    }

    // Validate the test setup
    TT_FATAL(num_stages > 1, "Must have at least 2 stages");
    TT_FATAL(num_stages < 8, "Must have at most 8 stages");
    for (size_t i = 0; i < num_stages; ++i) {
        TT_FATAL(num_workers_per_stage[i] > 0, "Must have at least 1 worker per stage");
        TT_FATAL(num_workers_per_stage[i] < 8, "Must have at most 8 workers per stage");
    }

    std::vector<TensorSpec> tensor_specs;
    tensor_specs.reserve(num_stages + 1);
    for (size_t i = 0; i < num_stages + 1; ++i) {
        tensor_specs.push_back(TensorSpec(
            tensor_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), mem_configs[i])));
    }

    // Allocate the tensors - pull to function
    const size_t num_tensors = num_stages + 1;
    std::vector<Tensor> host_tensors;
    std::vector<Tensor> device_tensors;
    host_tensors.reserve(num_tensors);
    device_tensors.reserve(num_tensors);
    auto num_elems = std::reduce(tensor_shape.cbegin(), tensor_shape.cend(), 1, std::multiplies<uint32_t>());
    host_tensors.push_back(
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to_layout(layout));
    for (size_t i = 1; i < num_tensors; ++i) {
        host_tensors.push_back(
            ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape));
    }
    TT_FATAL(mem_configs.size() == num_tensors, "Must have a memory config for each tensor");
    for (size_t i = 0; i < num_tensors; i++) {
        device_tensors.push_back(host_tensors[i].to_device(mesh_device.get(), mem_configs[i]));
        log_info(tt::LogTest, "Tensor[{}] allocated starting at address {}", i, device_tensors[i].buffer()->address());
    }
    TT_FATAL(
        device_tensors.size() == num_tensors,
        "device_tensors size {} does not match expected {}",
        device_tensors.size(),
        num_tensors);
    TT_FATAL(
        device_tensors.size() == host_tensors.size(),
        "device_tensors size {} does not match host_tensors size {}",
        device_tensors.size(),
        host_tensors.size());

    // MAIN STUFF

    // Initial setup like worker core assignment, chunk read order, etc.

    std::vector<CoreRangeSet> pipeline_stage_worker_cores = {};
    pipeline_stage_worker_cores.reserve(num_stages);
    for (size_t i = 0; i < num_stages; ++i) {
        pipeline_stage_worker_cores.push_back(
            CoreRangeSet(CoreRange(CoreCoord(0, i), CoreCoord(num_workers_per_stage[i] - 1, i))));
    }
    CoreRangeSet all_workers_cores = CoreRangeSet();
    for (size_t i = 0; i < num_stages; ++i) {
    }

    // Create circular buffers
    for (size_t stage = 0; stage < num_stages; stage++) {
        const size_t cb_packet_size_in_pages = 4;
        const size_t num_packets_per_cb = 4;
        tt_metal::CircularBufferConfig cb_config =
            tt_metal::CircularBufferConfig(
                cb_packet_size_in_pages * num_packets_per_cb * page_size_bytes, {{cb_index, data_format}})
                .set_page_size(cb_index, page_size_bytes);
        CreateCircularBuffer(program, pipeline_stage_worker_cores[stage], cb_config);
    }

    // Generate the reader semaphores
    std::vector<std::vector<uint32_t>> input_tensor_semaphores;
    input_tensor_semaphores.reserve(num_stages);
    for (size_t stage = 0; stage < num_stages; stage++) {
        input_tensor_semaphores.push_back({});
        for (size_t j = 0; j < slices_per_stage; j++) {
            input_tensor_semaphores[stage].push_back(CreateSemaphore(program, pipeline_stage_worker_cores[stage], 0));
        }
    }

    constexpr size_t num_command_streams = 1;
    std::vector<KernelHandle> reader_kernels;
    std::vector<KernelHandle> writer_kernels;
    // Create the kernel handles for each pipeline stage
    for (size_t stage = 0; stage < num_stages; stage++) {
        auto reader_kernel = ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {tt::CB::c_in0},
            {&device_tensors[stage]},
            pipeline_stage_worker_cores[stage],
            tt_metal::ReaderDataMovementConfig{},
            num_command_streams);
        reader_kernels.push_back(reader_kernel);
        auto writer_kernel = ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
            program,
            {tt::CB::c_in0},
            {&device_tensors[stage + 1]},
            pipeline_stage_worker_cores[stage],
            tt_metal::WriterDataMovementConfig{},
            num_command_streams);
        writer_kernels.push_back(writer_kernel);
    }

    // Generate the tensor slices for each tensor/worker
    std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> tensor_slices;
    tensor_slices.reserve(num_stages + 1);
    for (size_t t = 0; t < num_tensors; t++) {
        tensor_slices.push_back(
            ttnn::ccl::cmd::builder::generate_tensor_slices(slices_per_stage, device_tensors[t], split_dim));
    }
    std::vector<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>> per_stage_worker_reader_tensor_slices;
    std::vector<std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>> per_stage_worker_writer_tensor_slices;
    per_stage_worker_reader_tensor_slices.reserve(num_tensors);
    per_stage_worker_writer_tensor_slices.reserve(num_tensors);
    for (size_t stage = 0; stage < num_stages; stage++) {
        per_stage_worker_reader_tensor_slices.push_back(
            ttnn::ccl::cmd::builder::split_tensor_slices_across_workers_page_aligned(
                num_workers_per_stage[stage], tensor_slices[stage]));
        // We could compute this once and reuse it but I am generating it twice so I can have size mismatches
        per_stage_worker_writer_tensor_slices.push_back(
            ttnn::ccl::cmd::builder::split_tensor_slices_across_workers_page_aligned(
                num_workers_per_stage[stage], tensor_slices[stage + 1]));
        TT_FATAL(
            per_stage_worker_reader_tensor_slices.back().size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices. Got {} but expected {}",
            per_stage_worker_reader_tensor_slices.back().size(),
            num_workers_per_stage[stage]);
        TT_FATAL(
            per_stage_worker_writer_tensor_slices.back().size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices. Got {} but expected {}",
            per_stage_worker_writer_tensor_slices.back().size(),
            num_workers_per_stage[stage]);
    }

    // Build the command stream for each stage/worker
    // Seminc example
    // - local_core_semaphore_inc(second_command_stream_done_semaphore_id, 1);
    // semwait example
    // - local_semaphore_wait(second_command_stream_done_semaphore_id, 1)
    // read tensor slice to cb example
    // - read_tensor_slice_to_cb(in0_command_tensor_slice, cb_indices.at(0))
    // write tensor slice to cb example
    // - build_write_tensor_slice_to_cb(out0_command_tensor_slice, cb_indices.at(0))
    TT_FATAL(per_stage_worker_reader_tensor_slices.size() == num_stages, "Mismatch in tensor slices");
    for (size_t stage = 0; stage < num_stages; stage++) {
        bool last_stage = stage == num_stages - 1;
        bool first_stage = stage == 0;

        const auto worker_cores = corerange_to_cores(pipeline_stage_worker_cores[stage]);
        TT_FATAL(worker_cores.size() == num_workers_per_stage[stage], "Mismatch in worker cores");
        std::optional<std::vector<CoreCoord>> next_worker_cores =
            !last_stage ? corerange_to_cores(pipeline_stage_worker_cores[stage + 1])
                        : std::optional<std::vector<CoreCoord>>(std::nullopt);

        TT_FATAL(
            per_stage_worker_reader_tensor_slices[stage].size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices");
        TT_FATAL(
            per_stage_worker_writer_tensor_slices[stage].size() == num_workers_per_stage[stage],
            "Mismatch in tensor slices");
        for (size_t worker = 0; worker < num_workers_per_stage[stage]; worker++) {
            std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> reader_cmd_stream;
            std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> writer_cmd_stream;
            TT_FATAL(
                per_stage_worker_reader_tensor_slices[stage][worker].size() == slices_per_stage,
                "Mismatch in tensor slices");
            TT_FATAL(
                per_stage_worker_writer_tensor_slices[stage][worker].size() == slices_per_stage,
                "Mismatch in tensor slices");
            for (size_t slice_logical = 0; slice_logical < slices_per_stage; slice_logical++) {
                const auto slice_actual = worker_chunk_read_order[stage][slice_logical];
                // reader
                if (!first_stage) {
                    reader_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_semaphore_wait(
                        input_tensor_semaphores[stage][slice_actual], num_workers_per_stage[stage - 1]));
                }
                reader_cmd_stream.push_back(ttnn::ccl::cmd::uops::read_tensor_slice_to_cb(
                    per_stage_worker_reader_tensor_slices[stage][worker][slice_actual], cb_index));
                log_info(tt::LogTest, "Worker {} reading/writing slice {}", worker, slice_actual);

                // writer
                writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_write_cb_to_tensor_slice(
                    per_stage_worker_writer_tensor_slices[stage][worker][slice_actual], cb_index));
                if (not last_stage) {
                    for (auto next_worker_xy : next_worker_cores.value()) {
                        log_info(
                            tt::LogTest,
                            "Stage {} Worker {} noc seminc to core (logical) x={},y={}",
                            stage,
                            worker,
                            next_worker_xy.x,
                            next_worker_xy.y);
                        writer_cmd_stream.push_back(ttnn::ccl::cmd::uops::local_chip_noc_semaphore_inc(
                            mesh_device->worker_core_from_logical_core(next_worker_xy).x,
                            mesh_device->worker_core_from_logical_core(next_worker_xy).y,
                            input_tensor_semaphores[stage + 1][slice_actual],
                            1));
                    }
                }
            }
            ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
                program,
                reader_kernels[stage],
                {&device_tensors[stage]},
                {page_size_bytes},
                mesh_device->get_devices()[0],
                0,  // link = 0, don't care, since we aren't specifying connections
                cb_packet_size_in_pages,
                {worker_cores.at(worker)},
                reader_cmd_stream,
                std::nullopt,
                std::nullopt,
                std::nullopt);
            ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
                program,
                writer_kernels[stage],
                {&device_tensors[stage + 1]},
                {page_size_bytes},
                mesh_device->get_devices()[0],
                0,  // link = 0, don't care, since we aren't specifying connections
                cb_packet_size_in_pages,
                {worker_cores.at(worker)},
                writer_cmd_stream,
                std::nullopt,
                std::nullopt,
                std::nullopt);
        }
    }
    std::vector<tt::tt_metal::distributed::MeshWorkload> mesh_workloads(1);
    mesh_workloads[0].add_program(tt::tt_fabric::MeshCoordinateRange({0, 0}, {0, 0}), std::move(program));

    run_workloads(mesh_workloads, {mesh_device});

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        log_info(tt::LogTest, "Reading back outputs");
        auto input_cpu = device_tensors[0].cpu();
        auto final_out_cpu = device_tensors.back().cpu();

        auto in_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(input_cpu);
        auto out_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(final_out_cpu);

        auto in_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(host_tensors[0]);

        bool input_copyback_check_passed = run_output_check(in_tensor_data, in_tensor_copyback) == Correctness::Correct;
        TT_FATAL(input_copyback_check_passed, "Input 0 copyback check failed");

        log_info(tt::LogTest, "Comparing outputs");

        pass &= run_output_check(in_tensor_data, out_tensor_copyback) == Correctness::Correct;
        if (pass) {
            log_info(tt::LogTest, "Output check passed for output 0");
        } else {
            log_error(tt::LogTest, "Output check failed for output 0");
        }
    }

    return pass;
}
