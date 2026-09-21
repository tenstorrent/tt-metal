// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_shift_fused_program_factory.hpp"

#include <algorithm>
#include <set>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttml::metal::ops::ring_shift_fused {

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshSocket;
using tt::tt_metal::distributed::SocketEndpoint;

constexpr uint32_t kDataCb = tt::CBIndex::c_0;
constexpr uint32_t kHeaderCb = tt::CBIndex::c_1;
constexpr uint32_t kHandshakeCb = tt::CBIndex::c_2;
constexpr uint32_t kDataCbPages = 4U;

const char* kReaderPath =
    "tt-train/sources/ttml/metal/ops/ring_shift_fused/device/kernels/dataflow/ring_shift_fused_reader.cpp";
const char* kWriterPath =
    "tt-train/sources/ttml/metal/ops/ring_shift_fused/device/kernels/dataflow/ring_shift_fused_writer.cpp";
const char* kReceiverPath =
    "tt-train/sources/ttml/metal/ops/ring_shift_fused/device/kernels/dataflow/ring_shift_fused_receiver.cpp";

// A tensor as the kernels see it: one page size, a page count, and how the
// pages are packed into fabric packets.
struct TensorPlan {
    uint32_t page_size{};
    uint32_t pages{};
    uint32_t pages_per_packet{};
    uint32_t packing{};
};

// This core's contiguous share of a tensor's pages: the first `remainder`
// cores take one more.
std::pair<uint32_t, uint32_t> share_of(uint32_t pages, uint32_t core, uint32_t cores) {
    const uint32_t per_core = pages / cores;
    const uint32_t remainder = pages % cores;
    const uint32_t count = per_core + (core < remainder ? 1U : 0U);
    const uint32_t start = core * per_core + std::min(core, remainder);
    return {start, count};
}

struct ChipRoles {
    // Sender cores on this chip; per core, the fabric node of the chip its
    // connection reaches and the config address of its socket.
    std::vector<CoreCoord> sender_cores;
    std::vector<tt::tt_fabric::FabricNodeId> sender_peers;
    std::vector<uint32_t> sender_configs;
    // Receiver cores on this chip, the fabric node that sends to each, and
    // the config address of its socket.
    std::vector<CoreCoord> receiver_cores;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_peers;
    std::vector<uint32_t> receiver_configs;
};

ChipRoles roles_of(const operation_attributes_t& attrs, const ttnn::MeshCoordinate& coord) {
    ChipRoles roles;
    for (size_t i = 0; i < attrs.send_sockets.size(); ++i) {
        const auto& send_socket = attrs.send_sockets[i];
        const auto& recv_socket = attrs.recv_sockets[i];
        for (const auto& connection : send_socket.get_config().socket_connection_config) {
            if (connection.sender_core.device_coord == coord) {
                roles.sender_cores.push_back(connection.sender_core.core_coord);
                roles.sender_peers.push_back(
                    send_socket.get_fabric_node_id(SocketEndpoint::RECEIVER, connection.receiver_core.device_coord));
                roles.sender_configs.push_back(static_cast<uint32_t>(send_socket.get_config_buffer()->address()));
            }
        }
        for (const auto& connection : recv_socket.get_config().socket_connection_config) {
            if (connection.receiver_core.device_coord == coord) {
                roles.receiver_cores.push_back(connection.receiver_core.core_coord);
                roles.receiver_peers.push_back(
                    recv_socket.get_fabric_node_id(SocketEndpoint::SENDER, connection.sender_core.device_coord));
                roles.receiver_configs.push_back(static_cast<uint32_t>(recv_socket.get_config_buffer()->address()));
            }
        }
    }
    return roles;
}

struct ChipProgram {
    tt::tt_metal::Program program;
    RingShiftFusedProgramFactory::shared_variables_t shared;
};

ChipProgram build_chip_program(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs,
    const ttnn::MeshCoordinate& coord) {
    const auto& inputs = tensor_args.inputs;
    auto* mesh_device = inputs.front().device();
    auto* device = mesh_device->get_device(coord);
    const auto self_node = mesh_device->get_fabric_node_id(coord);
    const auto roles = roles_of(attrs, coord);
    const uint32_t num_tensors = static_cast<uint32_t>(inputs.size());
    const uint32_t links = std::max<uint32_t>(1U, static_cast<uint32_t>(roles.sender_cores.size()));

    // ---- the plans: pages into packets, and the packet buffer's size.
    uint32_t max_alignment = device->allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    for (const auto& tensor : inputs) {
        max_alignment = std::max(max_alignment, static_cast<uint32_t>(tensor.buffer()->alignment()));
    }
    const uint32_t payload =
        tt::round_down(static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()), max_alignment);
    const uint32_t num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    std::vector<TensorPlan> plans(num_tensors);
    uint32_t data_page_bytes = payload;
    for (uint32_t t = 0; t < num_tensors; ++t) {
        auto* buffer = inputs[t].buffer();
        TensorPlan& plan = plans[t];
        plan.page_size = static_cast<uint32_t>(buffer->aligned_page_size());
        plan.pages = static_cast<uint32_t>(buffer->num_pages());
        TT_FATAL(
            plan.page_size % max_alignment == 0 && plan.page_size <= payload,
            "ring_shift_fused: tensor {} has a {}-byte page; pages must be aligned to {} and fit a {}-byte fabric packet",
            t, plan.page_size, max_alignment, payload);
        plan.pages_per_packet = payload / plan.page_size;
        plan.packing = (plan.pages_per_packet > 1U && num_banks > 1U) ? 1U : 0U;
        if (plan.packing != 0U) {
            data_page_bytes = std::max(data_page_bytes, num_banks * plan.pages_per_packet * plan.page_size);
        }
    }
    const uint32_t handshake_page_size = tt::align(std::max(64U, 4U * (num_tensors + 1U)), max_alignment);
    const uint32_t header_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    std::set<tt::tt_metal::CoreRange> sender_ranges;
    for (const auto& core : roles.sender_cores) {
        sender_ranges.insert(tt::tt_metal::CoreRange(core));
    }
    std::set<tt::tt_metal::CoreRange> receiver_ranges;
    for (const auto& core : roles.receiver_cores) {
        receiver_ranges.insert(tt::tt_metal::CoreRange(core));
    }
    const tt::tt_metal::CoreRangeSet sender_set(sender_ranges);
    const tt::tt_metal::CoreRangeSet receiver_set(receiver_ranges);

    // ---- buffers: the packets in flight, the fabric headers, the handshake page.
    const bool sends = !roles.sender_cores.empty();
    const bool receives = !roles.receiver_cores.empty();
    if (sends) {
        tt::tt_metal::CreateCircularBuffer(
            program, sender_set,
            tt::tt_metal::CircularBufferConfig(kDataCbPages * data_page_bytes, {{kDataCb, tt::DataFormat::UInt32}})
                .set_page_size(kDataCb, data_page_bytes));
        tt::tt_metal::CreateCircularBuffer(
            program, sender_set,
            tt::tt_metal::CircularBufferConfig(2U * handshake_page_size, {{kHandshakeCb, tt::DataFormat::UInt32}})
                .set_page_size(kHandshakeCb, handshake_page_size));
    }
    tt::tt_metal::CreateCircularBuffer(
        program, sender_set.merge(receiver_set),
        tt::tt_metal::CircularBufferConfig(2U * header_bytes, {{kHeaderCb, tt::DataFormat::UInt32}})
            .set_page_size(kHeaderCb, header_bytes));

    // ---- kernels.
    tt::tt_metal::KernelHandle reader{}, writer{}, receiver{};
    if (sends) {
        reader = tt::tt_metal::CreateKernel(
            program, kReaderPath, sender_set,
            tt::tt_metal::ReaderDataMovementConfig({kDataCb, num_banks, num_tensors}));
        writer = tt::tt_metal::CreateKernel(
            program, kWriterPath, sender_set,
            tt::tt_metal::WriterDataMovementConfig(
                {kDataCb, kHeaderCb, kHandshakeCb, handshake_page_size, num_banks, num_tensors}));
    }
    if (receives) {
        receiver = tt::tt_metal::CreateKernel(
            program, kReceiverPath, receiver_set,
            tt::tt_metal::WriterDataMovementConfig({kHeaderCb, handshake_page_size, num_tensors}));
    }

    for (uint32_t c = 0; c < roles.sender_cores.size(); ++c) {
        const auto& core = roles.sender_cores[c];
        std::vector<uint32_t> reader_args;
        std::vector<uint32_t> writer_args{roles.sender_configs[c]};
        for (uint32_t t = 0; t < num_tensors; ++t) {
            const auto [start, count] = share_of(plans[t].pages, c, links);
            reader_args.insert(
                reader_args.end(),
                {static_cast<uint32_t>(inputs[t].buffer()->address()), plans[t].page_size, count, start,
                 plans[t].pages_per_packet, plans[t].packing});
            writer_args.insert(
                writer_args.end(), {plans[t].page_size, count, start, plans[t].pages_per_packet, plans[t].packing});
        }
        const auto peer = roles.sender_peers[c];
        const auto link_indices = tt::tt_fabric::get_forwarding_link_indices(self_node, peer);
        TT_FATAL(!link_indices.empty(), "ring_shift_fused: no fabric link from chip {} to its neighbour", coord);
        tt::tt_fabric::append_fabric_connection_rt_args(
            self_node, peer, link_indices[c % link_indices.size()], program, core, writer_args);
        tt::tt_metal::SetRuntimeArgs(program, reader, core, reader_args);
        tt::tt_metal::SetRuntimeArgs(program, writer, core, writer_args);
    }
    for (uint32_t c = 0; c < roles.receiver_cores.size(); ++c) {
        const auto& core = roles.receiver_cores[c];
        std::vector<uint32_t> receiver_args{roles.receiver_configs[c]};
        for (uint32_t t = 0; t < num_tensors; ++t) {
            receiver_args.push_back(static_cast<uint32_t>(outputs[t].buffer()->address()));
        }
        const auto peer = roles.receiver_peers[c];
        const auto link_indices = tt::tt_fabric::get_forwarding_link_indices(self_node, peer);
        TT_FATAL(!link_indices.empty(), "ring_shift_fused: no fabric link from chip {} back to its sender", coord);
        tt::tt_fabric::append_fabric_connection_rt_args(
            self_node, peer, link_indices[c % link_indices.size()], program, core, receiver_args);
        tt::tt_metal::SetRuntimeArgs(program, receiver, core, receiver_args);
    }

    return ChipProgram{
        .program = std::move(program),
        .shared = RingShiftFusedProgramFactory::shared_variables_t{
            .reader_kernel = reader,
            .writer_kernel = writer,
            .receiver_kernel = receiver,
            .sender_cores = roles.sender_cores,
            .receiver_cores = roles.receiver_cores,
            .num_tensors = num_tensors}};
}

}  // namespace

RingShiftFusedProgramFactory::cached_mesh_workload_t RingShiftFusedProgramFactory::create_mesh_workload(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto* mesh_device = tensor_args.inputs.front().device();
    TT_FATAL(mesh_device != nullptr, "ring_shift_fused: the tensors must be on a mesh device");
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinateRange, shared_variables_t> shared_vars;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_device->shape())) {
        const auto roles = roles_of(attrs, coord);
        if (roles.sender_cores.empty() && roles.receiver_cores.empty()) {
            continue;  // a chip the sockets do not touch
        }
        auto chip = build_chip_program(attrs, tensor_args, tensor_return_value, coord);
        ttnn::MeshCoordinateRange single{coord};
        shared_vars[single] = std::move(chip.shared);
        mesh_workload.add_program(single, std::move(chip.program));
    }
    return cached_mesh_workload_t(std::move(mesh_workload), std::move(shared_vars));
}

void RingShiftFusedProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(coord_range);
        const auto roles = roles_of(attrs, coord_range.start_coord());
        for (uint32_t c = 0; c < shared.sender_cores.size(); ++c) {
            const auto& core = shared.sender_cores[c];
            auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel, core);
            for (uint32_t t = 0; t < shared.num_tensors; ++t) {
                reader_args[6U * t] = static_cast<uint32_t>(tensor_args.inputs[t].buffer()->address());
            }
            tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel, core)[0] = roles.sender_configs[c];
        }
        for (uint32_t c = 0; c < shared.receiver_cores.size(); ++c) {
            const auto& core = shared.receiver_cores[c];
            auto& receiver_args = tt::tt_metal::GetRuntimeArgs(program, shared.receiver_kernel, core);
            receiver_args[0] = roles.receiver_configs[c];
            for (uint32_t t = 0; t < shared.num_tensors; ++t) {
                receiver_args[1U + t] = static_cast<uint32_t>(tensor_return_value[t].buffer()->address());
            }
        }
    }
}

}  // namespace ttml::metal::ops::ring_shift_fused
