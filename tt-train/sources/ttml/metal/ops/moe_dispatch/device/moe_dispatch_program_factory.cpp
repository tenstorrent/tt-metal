// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_dispatch_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/types.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace {
constexpr uint32_t TILE_H = 32;
constexpr uint32_t BLOCK_SIZE = 4;
constexpr uint32_t TILES_PER_BATCH = BLOCK_SIZE * BLOCK_SIZE;
constexpr uint32_t NUM_PACKET_HEADERS = 8;
const std::string kDir = "tt-train/sources/ttml/metal/ops/moe_dispatch/device/kernels/";
}  // namespace

namespace ttml::metal::ops::moe_dispatch {

using namespace tt::tt_metal;

MoeDispatchMeshWorkloadFactory::cached_mesh_workload_t MoeDispatchMeshWorkloadFactory::create_mesh_workload(
    const MoeDispatchParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MoeDispatchTensorArgs& tensor_args,
    std::vector<ttnn::Tensor>& output) {
    distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached = create_at(attrs, coord, tensor_args, output, tensor_coords);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached.program));
        shared_vars.emplace(coord, std::move(cached.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_vars));
}

ttnn::device_operation::CachedProgram<MoeDispatchMeshWorkloadFactory::shared_variables_t>
MoeDispatchMeshWorkloadFactory::create_at(
    const MoeDispatchParams& attrs,
    const ttnn::MeshCoordinate& mesh_coord,
    const MoeDispatchTensorArgs& tensor_args,
    std::vector<ttnn::Tensor>& output_tensors,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
    const auto& sorted_hidden = tensor_args.sorted_hidden;
    const auto& w_up = tensor_args.w_up;
    auto& output = output_tensors[0];
    auto& dispatch_buf = output_tensors[1];
    auto* mesh_device = sorted_hidden.device();
    IDevice* target_device = mesh_device->get_device(mesh_coord);

    const uint32_t D = sorted_hidden.padded_shape()[3];
    const uint32_t K_t = D / TILE_H;
    const uint32_t E_local = attrs.E_local;
    const uint32_t EP = static_cast<uint32_t>(attrs.expert_counts_per_device.size());
    const uint32_t E = EP > 0 ? static_cast<uint32_t>(attrs.expert_counts_per_device[0].size()) : 0;
    const uint32_t ffn_dim = w_up.padded_shape()[3];
    const uint32_t N_t = ffn_dim / TILE_H;
    const uint32_t N_t_rounded = ((N_t + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    const uint32_t tile_bytes = 2 * TILE_H * TILE_H;
    const uint32_t row_bytes = K_t * tile_bytes;

    auto mesh_shape = mesh_device->shape();
    uint32_t num_devices = mesh_shape[attrs.cluster_axis];
    uint32_t dispatch_axis_index = mesh_coord[attrs.cluster_axis];

    TT_FATAL(dispatch_axis_index < EP, "dispatch_axis_index out of range");
    const auto& local_counts = attrs.expert_counts_per_device[dispatch_axis_index];
    const auto& local_offsets = attrs.expert_offsets_per_device[dispatch_axis_index];

    uint32_t my_first_expert_global = dispatch_axis_index * E_local;

    // ---- Dispatch_buf layout: expert-major, device-minor ----
    // For each local expert, tokens from device 0, then device 1, etc.
    // Aggregated counts and base offsets.
    std::vector<uint32_t> agg_counts_tilerows(E_local, 0);
    std::vector<uint32_t> expert_base_tilerows(E_local, 0);
    for (uint32_t e = 0; e < E_local; e++) {
        uint32_t ge = my_first_expert_global + e;
        if (ge >= E)
            break;
        for (uint32_t d = 0; d < EP; d++) {
            agg_counts_tilerows[e] += attrs.expert_counts_per_device[d][ge] / TILE_H;
        }
    }
    for (uint32_t e = 1; e < E_local; e++) {
        expert_base_tilerows[e] = expert_base_tilerows[e - 1] + agg_counts_tilerows[e - 1];
    }

    // Per-expert write offset for THIS sender device
    std::vector<uint32_t> sender_dst_row(E, 0);
    for (uint32_t ge = 0; ge < E; ge++) {
        uint32_t owner = ge / E_local;
        uint32_t e_local_on_owner = ge - owner * E_local;
        uint32_t owner_base = 0;
        for (uint32_t prev = 0; prev < e_local_on_owner; prev++) {
            uint32_t prev_ge = owner * E_local + prev;
            for (uint32_t d = 0; d < EP; d++) {
                owner_base += attrs.expert_counts_per_device[d][prev_ge] / TILE_H;
            }
        }
        uint32_t dev_offset = 0;
        for (uint32_t d = 0; d < dispatch_axis_index; d++) {
            dev_offset += attrs.expert_counts_per_device[d][ge] / TILE_H;
        }
        sender_dst_row[ge] = owner_base + dev_offset;
    }

    auto device_grid = target_device->compute_with_storage_grid_size();
    auto worker_cores = corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {device_grid.x - 1, device_grid.y - 1})));
    TT_FATAL(worker_cores.size() >= 2, "Need at least 2 worker cores");

    Program program{};

    CoreCoord sender_core = worker_cores[0];
    CoreCoord receiver_core = worker_cores[1];
    CoreRange sender_range{sender_core, sender_core};
    CoreRange receiver_range{receiver_core, receiver_core};

    auto receiver_phys = target_device->worker_core_from_logical_core(receiver_core);
    auto sender_phys = target_device->worker_core_from_logical_core(sender_core);

    // ---- CBs ----
    CreateCircularBuffer(
        program,
        sender_range,
        CircularBufferConfig(2 * row_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, row_bytes));
    auto pkt_hdr_cb_index = tt::CBIndex::c_3;
    auto pkt_hdr_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    CreateCircularBuffer(
        program,
        sender_range,
        CircularBufferConfig(NUM_PACKET_HEADERS * pkt_hdr_size * 2, {{pkt_hdr_cb_index, tt::DataFormat::RawUInt32}})
            .set_page_size(pkt_hdr_cb_index, pkt_hdr_size));

    CreateCircularBuffer(
        program,
        receiver_range,
        CircularBufferConfig(BLOCK_SIZE * tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_bytes));
    CreateCircularBuffer(
        program,
        receiver_range,
        CircularBufferConfig(TILES_PER_BATCH * tile_bytes, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, tile_bytes));
    CreateCircularBuffer(
        program,
        receiver_range,
        CircularBufferConfig(N_t_rounded * tile_bytes, {{tt::CBIndex::c_10, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_10, tile_bytes));

    // ---- Semaphores ----
    auto tiles_ready_sem_id = CreateSemaphore(program, receiver_range, 0);
    uint32_t go_sem_init = (dispatch_axis_index == 0) ? E : 0;
    auto go_sem_id = CreateSemaphore(program, sender_range, go_sem_init);

    bool is_last = (dispatch_axis_index == num_devices - 1);

    // ---- Sender kernel ----
    auto input_ta_ct = TensorAccessorArgs(*sorted_hidden.buffer()).get_compile_time_args();
    auto dispatch_ta_ct = TensorAccessorArgs(*dispatch_buf.buffer()).get_compile_time_args();

    std::vector<uint32_t> sender_ct = {
        static_cast<uint32_t>(tt::CBIndex::c_0),
        static_cast<uint32_t>(pkt_hdr_cb_index),
        tile_bytes,
        K_t,
        E,
        E_local,
        dispatch_axis_index,
        num_devices,
    };
    sender_ct.insert(sender_ct.end(), input_ta_ct.begin(), input_ta_ct.end());
    sender_ct.insert(sender_ct.end(), dispatch_ta_ct.begin(), dispatch_ta_ct.end());

    auto sender_kid = CreateKernel(
        program,
        kDir + "dataflow/sender.cpp",
        sender_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = sender_ct});

    std::vector<uint32_t> sender_rt = {
        sorted_hidden.buffer()->address(),
        dispatch_buf.buffer()->address(),
        static_cast<uint32_t>(receiver_phys.x),
        static_cast<uint32_t>(receiver_phys.y),
        tiles_ready_sem_id,
        go_sem_id,
        static_cast<uint32_t>(sender_phys.x),
        static_cast<uint32_t>(sender_phys.y),
        is_last ? 1u : 0u,
    };
    for (auto c : local_counts) sender_rt.push_back(c / TILE_H);
    for (auto o : local_offsets) sender_rt.push_back(o / TILE_H);
    for (uint32_t ge = 0; ge < E; ge++) sender_rt.push_back(sender_dst_row[ge]);

    auto forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        sorted_hidden, mesh_coord, 1, ttnn::ccl::Topology::Linear, attrs.cluster_axis);
    auto backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        sorted_hidden, mesh_coord, -1, ttnn::ccl::Topology::Linear, attrs.cluster_axis);

    sender_rt.push_back(forward_coord.has_value() ? 1 : 0);
    if (forward_coord.has_value()) {
        auto src_id = mesh_device->get_fabric_node_id(mesh_coord);
        auto dst_id = mesh_device->get_fabric_node_id(forward_coord.value());
        tt::tt_fabric::append_fabric_connection_rt_args(src_id, dst_id, 0, program, sender_core, sender_rt);
    }
    sender_rt.push_back(backward_coord.has_value() ? 1 : 0);
    if (backward_coord.has_value()) {
        auto src_id = mesh_device->get_fabric_node_id(mesh_coord);
        auto dst_id = mesh_device->get_fabric_node_id(backward_coord.value());
        tt::tt_fabric::append_fabric_connection_rt_args(src_id, dst_id, 0, program, sender_core, sender_rt);
    }
    SetRuntimeArgs(program, sender_kid, sender_core, sender_rt);

    // ---- Receiver reader kernel ----
    auto w_ta_ct = TensorAccessorArgs(*w_up.buffer()).get_compile_time_args();
    std::vector<uint32_t> recv_ct = {K_t, N_t, BLOCK_SIZE, E_local, num_devices, 0u /*my_first_expert_weight*/};
    recv_ct.insert(recv_ct.end(), dispatch_ta_ct.begin(), dispatch_ta_ct.end());
    recv_ct.insert(recv_ct.end(), w_ta_ct.begin(), w_ta_ct.end());

    auto recv_reader_kid =
        CreateKernel(program, kDir + "dataflow/receiver_reader.cpp", receiver_range, ReaderDataMovementConfig(recv_ct));

    std::vector<uint32_t> recv_rt = {
        dispatch_buf.buffer()->address(),
        w_up.buffer()->address(),
        tiles_ready_sem_id,
    };
    // Per (device, local_expert) counts
    for (uint32_t d = 0; d < EP; d++) {
        for (uint32_t e = 0; e < E_local; e++) {
            uint32_t ge = my_first_expert_global + e;
            recv_rt.push_back(ge < E ? attrs.expert_counts_per_device[d][ge] / TILE_H : 0);
        }
    }
    SetRuntimeArgs(program, recv_reader_kid, receiver_core, recv_rt);

    // ---- Compute kernel ----
    std::vector<uint32_t> compute_ct = {K_t, N_t, BLOCK_SIZE, E_local};
    auto compute_kid = CreateKernel(
        program,
        kDir + "compute/expert_matmul.cpp",
        receiver_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct});

    std::vector<uint32_t> compute_rt;
    for (uint32_t e = 0; e < E_local; e++) compute_rt.push_back(agg_counts_tilerows[e]);
    SetRuntimeArgs(program, compute_kid, receiver_core, compute_rt);

    // ---- Writer kernel ----
    auto output_ta_ct = TensorAccessorArgs(*output.buffer()).get_compile_time_args();
    std::vector<uint32_t> writer_ct = {N_t, BLOCK_SIZE, E_local};
    writer_ct.insert(writer_ct.end(), output_ta_ct.begin(), output_ta_ct.end());

    auto writer_kid = CreateKernel(
        program, kDir + "dataflow/receiver_writer.cpp", receiver_range, WriterDataMovementConfig(writer_ct));

    std::vector<uint32_t> writer_rt = {output.buffer()->address()};
    for (uint32_t e = 0; e < E_local; e++) writer_rt.push_back(agg_counts_tilerows[e]);
    for (uint32_t e = 0; e < E_local; e++) writer_rt.push_back(expert_base_tilerows[e]);
    SetRuntimeArgs(program, writer_kid, receiver_core, writer_rt);

    return {std::move(program), {sender_kid, recv_reader_kid, compute_kid, writer_kid}};
}

void MoeDispatchMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t&, const MoeDispatchParams&, const MoeDispatchTensorArgs&, std::vector<ttnn::Tensor>&) {
}

}  // namespace ttml::metal::ops::moe_dispatch
