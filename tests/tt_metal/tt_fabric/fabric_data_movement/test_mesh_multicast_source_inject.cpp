// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/mcast_reverse_tree.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_mesh_multicast_source_inject_common.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

namespace source_inject = tt::tt_fabric::fabric_router_tests::source_inject;

struct SourceInjectExtents {
    uint32_t n = 0;
    uint32_t s = 0;
    uint32_t e = 0;
    uint32_t w = 0;
};

struct SourceInjectBranch {
    RoutingDirection primary_output;
    SourceInjectExtents extents;
};

struct SourceInjectCandidate {
    FabricNodeId source;
    ChipId source_physical_id;
    uint32_t source_y;
    uint32_t source_x;
    RoutingDirection primary_output;
    SourceInjectExtents extents;
    std::vector<RoutingDirection> root_outputs;
    std::set<ChipId> target_physical_ids;
    bool express;
};

struct SourceInjectProgram {
    ChipId physical_id;
    std::shared_ptr<tt_metal::distributed::MeshDevice> device;
    tt_metal::Program program;
    bool is_target;
};

template <typename Visitor>
bool visit_candidate_branches(const tt_metal::distributed::MeshShape& mesh_shape, Visitor&& visitor) {
    const uint32_t y_size = mesh_shape[0];
    const uint32_t x_size = mesh_shape[1];

    // Preserve the production client's four-way decomposition. E and W cover the source row
    // independently; N and S are separate trunks whose target rows may carry both E and W teeth.
    // Stop as soon as the visitor finds enough coverage; large meshes need not materialize or scan
    // every possible branch.
    for (uint32_t east = 1; east < x_size; ++east) {
        if (visitor(SourceInjectBranch{RoutingDirection::E, SourceInjectExtents{.e = east}})) {
            return true;
        }
    }
    for (uint32_t west = 1; west < x_size; ++west) {
        if (visitor(SourceInjectBranch{RoutingDirection::W, SourceInjectExtents{.w = west}})) {
            return true;
        }
    }
    for (uint32_t east = 0; east < x_size; ++east) {
        for (uint32_t west = 0; west < x_size; ++west) {
            if (east + west >= x_size) {
                continue;
            }
            for (uint32_t north = 1; north < y_size; ++north) {
                if (visitor(SourceInjectBranch{
                        RoutingDirection::N, SourceInjectExtents{.n = north, .e = east, .w = west}})) {
                    return true;
                }
            }
            for (uint32_t south = 1; south < y_size; ++south) {
                if (visitor(SourceInjectBranch{
                        RoutingDirection::S, SourceInjectExtents{.s = south, .e = east, .w = west}})) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool candidate_contains_z(const SourceInjectCandidate& candidate) {
    return std::find(candidate.root_outputs.begin(), candidate.root_outputs.end(), RoutingDirection::Z) !=
           candidate.root_outputs.end();
}

bool candidate_exercises_z_fanout(const SourceInjectCandidate& candidate) {
    return candidate_contains_z(candidate) && candidate.root_outputs.size() > 1;
}

bool outputs_match_branch(const std::vector<RoutingDirection>& root_outputs, RoutingDirection primary_output) {
    if (root_outputs.empty()) {
        return false;
    }
    return std::all_of(root_outputs.begin(), root_outputs.end(), [&](RoutingDirection output) {
        if (output == primary_output) {
            return true;
        }
        return output == RoutingDirection::Z &&
               (primary_output == RoutingDirection::N || primary_output == RoutingDirection::S);
    });
}

const char* branch_name(RoutingDirection primary_output) {
    switch (primary_output) {
        case RoutingDirection::E: return "E";
        case RoutingDirection::W: return "W";
        case RoutingDirection::N: return "N";
        case RoutingDirection::S: return "S";
        default: return "invalid";
    }
}

bool direction_is_connectable(
    const ControlPlane& control_plane, const FabricNodeId& source, RoutingDirection direction) {
    const auto neighbors = control_plane.get_intra_chip_neighbors(source, direction);
    if (neighbors.empty()) {
        return false;
    }

    const FabricNodeId first_hop(source.mesh_id, neighbors.front());
    const auto forwarding_direction = control_plane.get_forwarding_direction(source, first_hop);
    return forwarding_direction.has_value() && forwarding_direction.value() == direction &&
           !get_forwarding_link_indices(source, first_hop).empty();
}

std::optional<std::set<ChipId>> target_devices(
    const ControlPlane& control_plane,
    const FabricNodeId& source,
    uint32_t root_y,
    uint32_t root_x,
    const SourceInjectExtents& extents,
    const tt_metal::distributed::MeshShape& mesh_shape,
    const std::unordered_set<ChipId>& local_physical_ids) {
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const uint32_t y_size = mesh_shape[0];
    const uint32_t x_size = mesh_shape[1];
    std::set<uint32_t> target_rows;
    std::set<uint32_t> target_columns{root_x};

    // Match encode_2d_mcast_maps exactly. In particular, a non-zero N/S range excludes the source
    // row, while the source column is always part of every selected row.
    if (extents.n == 0 && extents.s == 0) {
        target_rows.insert(root_y);
    } else {
        for (uint32_t hop = 1; hop <= extents.n; ++hop) {
            target_rows.insert((root_y + y_size - (hop % y_size)) % y_size);
        }
        for (uint32_t hop = 1; hop <= extents.s; ++hop) {
            target_rows.insert((root_y + hop) % y_size);
        }
    }
    for (uint32_t hop = 1; hop <= extents.e; ++hop) {
        target_columns.insert((root_x + hop) % x_size);
    }
    for (uint32_t hop = 1; hop <= extents.w; ++hop) {
        target_columns.insert((root_x + x_size - (hop % x_size)) % x_size);
    }

    std::set<ChipId> target_physical_ids;
    for (const uint32_t y : target_rows) {
        for (const uint32_t x : target_columns) {
            const ChipId chip_id =
                mesh_graph.coordinate_to_chip(source.mesh_id, tt_metal::distributed::MeshCoordinate({y, x}));
            if (chip_id == source.chip_id) {
                continue;
            }
            const auto physical_id =
                control_plane.try_get_physical_chip_id_from_fabric_node_id(FabricNodeId(source.mesh_id, chip_id));
            if (!physical_id.has_value() || !local_physical_ids.contains(physical_id.value())) {
                return std::nullopt;
            }
            target_physical_ids.insert(physical_id.value());
        }
    }
    return target_physical_ids;
}

bool is_better_candidate(const SourceInjectCandidate& candidate, const std::optional<SourceInjectCandidate>& current) {
    if (!current.has_value()) {
        return true;
    }
    if (candidate_exercises_z_fanout(candidate) != candidate_exercises_z_fanout(current.value())) {
        return candidate_exercises_z_fanout(candidate);
    }
    if (candidate_contains_z(candidate) != candidate_contains_z(current.value())) {
        return candidate_contains_z(candidate);
    }
    if (candidate.express != current->express) {
        return candidate.express;
    }
    if (candidate.root_outputs.size() != current->root_outputs.size()) {
        return candidate.root_outputs.size() > current->root_outputs.size();
    }
    if (candidate.target_physical_ids.size() != current->target_physical_ids.size()) {
        return candidate.target_physical_ids.size() > current->target_physical_ids.size();
    }
    return std::tie(
               candidate.source.mesh_id,
               candidate.source.chip_id,
               candidate.primary_output,
               candidate.extents.n,
               candidate.extents.s,
               candidate.extents.e,
               candidate.extents.w) <
           std::tie(
               current->source.mesh_id,
               current->source.chip_id,
               current->primary_output,
               current->extents.n,
               current->extents.s,
               current->extents.e,
               current->extents.w);
}

std::optional<SourceInjectCandidate> select_candidate(BaseFabricFixture* fixture) {
    auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    std::unordered_set<ChipId> local_physical_ids;
    for (const auto& device : fixture->get_devices()) {
        local_physical_ids.insert(device->get_device_ids()[0]);
    }

    std::optional<SourceInjectCandidate> best;
    auto user_mesh_ids = control_plane.get_user_physical_mesh_ids();
    std::sort(user_mesh_ids.begin(), user_mesh_ids.end(), [&](MeshId lhs, MeshId rhs) {
        const bool lhs_express = control_plane.express_routing_enabled(lhs);
        const bool rhs_express = control_plane.express_routing_enabled(rhs);
        return lhs_express != rhs_express ? lhs_express > rhs_express : lhs < rhs;
    });

    for (const MeshId mesh_id : user_mesh_ids) {
        const auto mesh_shape = control_plane.get_physical_mesh_shape(mesh_id);
        if (mesh_shape[0] * mesh_shape[1] < 3) {
            continue;
        }

        // Every target and non-target is checked. Skip a mesh split across hosts rather than silently
        // leaving part of its multicast range unobservable.
        bool full_mesh_is_local = true;
        for (uint32_t y = 0; y < mesh_shape[0] && full_mesh_is_local; ++y) {
            for (uint32_t x = 0; x < mesh_shape[1]; ++x) {
                const ChipId chip_id =
                    mesh_graph.coordinate_to_chip(mesh_id, tt_metal::distributed::MeshCoordinate({y, x}));
                const auto physical_id =
                    control_plane.try_get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh_id, chip_id));
                if (!physical_id.has_value() || !local_physical_ids.contains(physical_id.value())) {
                    full_mesh_is_local = false;
                    break;
                }
            }
        }
        if (!full_mesh_is_local) {
            continue;
        }

        const auto* y_topology = control_plane.axis_topology(mesh_id, 0);
        const auto* x_topology = control_plane.axis_topology(mesh_id, 1);
        TT_FATAL(y_topology != nullptr && x_topology != nullptr, "Missing 2D axis topology for mesh {}", mesh_id);

        const bool express = control_plane.express_routing_enabled(mesh_id);
        if (!express && best.has_value() && candidate_contains_z(best.value())) {
            return best;
        }

        for (uint32_t root_y = 0; root_y < mesh_shape[0]; ++root_y) {
            for (uint32_t root_x = 0; root_x < mesh_shape[1]; ++root_x) {
                const ChipId source_chip_id =
                    mesh_graph.coordinate_to_chip(mesh_id, tt_metal::distributed::MeshCoordinate({root_y, root_x}));
                const FabricNodeId source(mesh_id, source_chip_id);
                const ChipId source_physical_id = control_plane.get_physical_chip_id_from_fabric_node_id(source);

                const bool found_preferred_branch = visit_candidate_branches(mesh_shape, [&](const auto& branch) {
                    const auto& extents = branch.extents;
                    std::string failure;
                    // Run the production reverse-tree encoder for one client branch. Geometry alone
                    // cannot reveal an express Z output selected for this exact source and target set.
                    const auto root_outputs = mcast_root_output_directions(
                        mesh_graph,
                        mesh_id,
                        *y_topology,
                        *x_topology,
                        root_y,
                        root_x,
                        extents.n,
                        extents.s,
                        extents.e,
                        extents.w,
                        &failure);
                    TT_FATAL(
                        failure.empty(),
                        "Failed to derive multicast outputs for mesh {} chip {}: {}",
                        mesh_id,
                        source_chip_id,
                        failure);
                    // A legal branch may leave on its cardinal output and, for N/S under express
                    // routing, Z. Reject wrapped/combined candidates that change the logical branch.
                    if (!outputs_match_branch(root_outputs, branch.primary_output)) {
                        return false;
                    }

                    const bool all_outputs_connectable =
                        std::all_of(root_outputs.begin(), root_outputs.end(), [&](RoutingDirection output) {
                            return direction_is_connectable(control_plane, source, output);
                        });
                    if (!all_outputs_connectable) {
                        return false;
                    }

                    auto targets =
                        target_devices(control_plane, source, root_y, root_x, extents, mesh_shape, local_physical_ids);
                    if (!targets.has_value() || targets->empty()) {
                        return false;
                    }

                    SourceInjectCandidate candidate{
                        source,
                        source_physical_id,
                        root_y,
                        root_x,
                        branch.primary_output,
                        extents,
                        root_outputs,
                        std::move(targets.value()),
                        express};
                    const bool exercises_z_fanout = candidate_exercises_z_fanout(candidate);
                    const bool has_multiple_targets = candidate.target_physical_ids.size() >= 2;
                    if (is_better_candidate(candidate, best)) {
                        best = std::move(candidate);
                    }

                    // Cardinal plus Z is the widest legal branch root and directly exercises the
                    // express behavior this API adds over the existing single-connection path.
                    if (express && exercises_z_fanout) {
                        return true;
                    }
                    // On a non-express mesh the fanout degenerates to one cardinal connection. Prefer
                    // a branch with at least two targets so the test still exercises multicast delivery.
                    if (!express && has_multiple_targets) {
                        return true;
                    }
                    return false;
                });
                if (found_preferred_branch) {
                    return best;
                }
            }
        }
    }
    return best;
}

std::vector<eth_chan_directions> attempted_directions(const SourceInjectCandidate& candidate) {
    auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    const uint32_t max_connections =
        candidate.express && tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE ? 5 : 4;
    std::vector<eth_chan_directions> attempted;
    attempted.reserve(max_connections);

    const auto append_if_absent = [&](RoutingDirection direction) {
        const auto eth_direction = control_plane.routing_direction_to_eth_direction(direction);
        if (std::find(attempted.begin(), attempted.end(), eth_direction) == attempted.end()) {
            attempted.push_back(eth_direction);
        }
    };
    for (const RoutingDirection output : candidate.root_outputs) {
        append_if_absent(output);
    }
    for (const RoutingDirection direction :
         {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
        if (attempted.size() == max_connections) {
            break;
        }
        if ((!candidate.express && direction == RoutingDirection::Z) ||
            !direction_is_connectable(control_plane, candidate.source, direction)) {
            continue;
        }
        append_if_absent(direction);
    }
    return attempted;
}

void run_source_inject_test(BaseFabricFixture* fixture) {
    const auto candidate = select_candidate(fixture);
    if (!candidate.has_value()) {
        GTEST_SKIP() << "No host-local source has a legal multicast branch with connectable root outputs";
    }

    auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    const auto source_device = fixture->get_device(candidate->source_physical_id);
    const tt_metal::CoreCoord sender_logical_core{0, 0};
    const tt_metal::CoreCoord receiver_logical_core{1, 0};
    const auto source_grid = source_device->compute_with_storage_grid_size();
    ASSERT_GT(source_grid.x, receiver_logical_core.x);
    ASSERT_GT(source_grid.y, receiver_logical_core.y);
    const auto receiver_virtual_core = source_device->worker_core_from_logical_core(receiver_logical_core);
    const auto worker_mem_map = fixture->generate_worker_mem_map(source_device, topology);
    const uint32_t data_base = worker_mem_map.target_address;
    const uint32_t payload_size = worker_mem_map.packet_payload_size_bytes;
    const uint32_t validation_data_size = source_inject::data_size(payload_size);

    ASSERT_GE(payload_size, 2 * source_inject::PAYLOAD_ALIGNMENT);
    ASSERT_EQ(payload_size % source_inject::PAYLOAD_ALIGNMENT, 0u);
    ASSERT_LE(validation_data_size, worker_mem_map.test_results_address - data_base);

    std::vector<std::pair<ChipId, std::shared_ptr<tt_metal::distributed::MeshDevice>>> mesh_devices;
    for (const auto& device : fixture->get_devices()) {
        const ChipId physical_id = device->get_device_ids()[0];
        const auto node = control_plane.get_fabric_node_id_from_physical_chip_id(physical_id);
        if (node.mesh_id != candidate->source.mesh_id) {
            continue;
        }

        const auto grid = device->compute_with_storage_grid_size();
        ASSERT_GT(grid.x, receiver_logical_core.x);
        ASSERT_GT(grid.y, receiver_logical_core.y);
        ASSERT_EQ(device->worker_core_from_logical_core(receiver_logical_core), receiver_virtual_core)
            << "Fixed multicast destination core must have one common virtual NOC coordinate";
        const auto mem_map = fixture->generate_worker_mem_map(device, topology);
        ASSERT_EQ(mem_map.target_address, data_base);
        ASSERT_EQ(mem_map.packet_payload_size_bytes, payload_size);
        ASSERT_EQ(mem_map.test_results_address, worker_mem_map.test_results_address);
        ASSERT_EQ(mem_map.notification_mailbox_address, worker_mem_map.notification_mailbox_address);
        mesh_devices.emplace_back(physical_id, device);
    }
    ASSERT_FALSE(mesh_devices.empty());

    std::vector<uint32_t> initial_data(validation_data_size / sizeof(uint32_t), source_inject::SENTINEL);
    const uint32_t counter_word_offset =
        (source_inject::counter_base(data_base, payload_size) - data_base) / sizeof(uint32_t);
    std::fill_n(
        initial_data.begin() + counter_word_offset, source_inject::ATOMIC_COUNTER_COUNT, static_cast<uint32_t>(0));
    std::vector<uint32_t> zero_results(worker_mem_map.test_results_size_bytes / sizeof(uint32_t), 0);
    std::vector<uint32_t> stop_value(1, 0);

    for (const auto& [physical_id, device] : mesh_devices) {
        tt_metal::slow_dispatch::WriteToL1(*device, receiver_logical_core, data_base, initial_data, CoreType::WORKER);
        tt_metal::slow_dispatch::WriteToL1(
            *device, receiver_logical_core, worker_mem_map.test_results_address, zero_results, CoreType::WORKER);
        tt_metal::slow_dispatch::WriteToL1(
            *device, receiver_logical_core, worker_mem_map.notification_mailbox_address, stop_value, CoreType::WORKER);
        tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_id);
    }
    // The sender status is live-polled while the source's combined sender/checker program runs.
    // Clear its core independently so a stale PASS cannot release the receiver hold protocol early.
    tt_metal::slow_dispatch::WriteToL1(
        *source_device, sender_logical_core, worker_mem_map.test_results_address, zero_results, CoreType::WORKER);
    tt_metal::MetalContext::instance().get_cluster().l1_barrier(candidate->source_physical_id);

    const auto connection_directions = attempted_directions(candidate.value());
    ASSERT_GE(connection_directions.size(), candidate->root_outputs.size());
    for (const RoutingDirection output : candidate->root_outputs) {
        const auto eth_direction = control_plane.routing_direction_to_eth_direction(output);
        ASSERT_NE(
            std::find(connection_directions.begin(), connection_directions.end(), eth_direction),
            connection_directions.end());
    }

    log_info(
        tt::LogTest,
        "Source-inject test mesh {} chip {} ({},{}), {} branch N{} S{} E{} W{}, root outputs {}, targets {}",
        candidate->source.mesh_id,
        candidate->source.chip_id,
        candidate->source_y,
        candidate->source_x,
        branch_name(candidate->primary_output),
        candidate->extents.n,
        candidate->extents.s,
        candidate->extents.e,
        candidate->extents.w,
        candidate->root_outputs.size(),
        candidate->target_physical_ids.size());

    std::vector<SourceInjectProgram> programs;
    programs.reserve(mesh_devices.size());
    constexpr uint32_t QUIESCENCE_TIME_US = 10'000;
    for (const auto& [physical_id, device] : mesh_devices) {
        const bool is_target = candidate->target_physical_ids.contains(physical_id);
        // Device AICLK is reported in MHz, i.e. cycles per microsecond.
        const uint64_t quiescence_cycles =
            static_cast<uint64_t>(tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(physical_id)) *
            QUIESCENCE_TIME_US;
        ASSERT_LE(quiescence_cycles, std::numeric_limits<uint32_t>::max());

        auto program = tt_metal::CreateProgram();
        const auto receiver_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/"
            "test_mesh_multicast_source_inject_receiver.cpp",
            {receiver_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {
                    worker_mem_map.test_results_address,
                    worker_mem_map.test_results_size_bytes,
                    worker_mem_map.notification_mailbox_address,
                    static_cast<uint32_t>(quiescence_cycles)}});
        tt_metal::SetRuntimeArgs(program, receiver_kernel, receiver_logical_core, {data_base, payload_size, is_target});

        if (physical_id == candidate->source_physical_id) {
            auto sender_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/"
                "test_mesh_multicast_source_inject_sender.cpp",
                {sender_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = {worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes}});

            std::vector<uint32_t> sender_runtime_args = {
                data_base,
                payload_size,
                receiver_virtual_core.x,
                receiver_virtual_core.y,
                candidate->extents.e,
                candidate->extents.w,
                candidate->extents.n,
                candidate->extents.s,
                0};
            const size_t connection_count_index = sender_runtime_args.size() - 1;
            const uint32_t connection_count = append_routing_plane_connection_manager_rt_args(
                candidate->source,
                connection_directions,
                {},
                program,
                sender_kernel,
                sender_logical_core,
                sender_runtime_args,
                FabricApiType::Mesh,
                CoreType::WORKER);
            ASSERT_EQ(connection_count, connection_directions.size());
            sender_runtime_args[connection_count_index] = connection_count;
            tt_metal::SetRuntimeArgs(program, sender_kernel, sender_logical_core, sender_runtime_args);
        }

        programs.push_back(SourceInjectProgram{physical_id, device, std::move(program), is_target});
    }

    // Start every remote observer before the source program. The source program contains both its
    // sender and its non-target checker because two independent programs cannot share one unit-mesh CQ.
    for (auto& record : programs) {
        if (record.physical_id != candidate->source_physical_id) {
            fixture->RunProgramNonblocking(record.device, std::move(record.program));
        }
    }
    for (auto& record : programs) {
        if (record.physical_id == candidate->source_physical_id) {
            fixture->RunProgramNonblocking(record.device, std::move(record.program));
            break;
        }
    }

    const auto read_status = [&](const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
                                 const tt_metal::CoreCoord& core) {
        std::vector<uint32_t> status;
        tt_metal::slow_dispatch::ReadFromL1(
            *device, core, worker_mem_map.test_results_address, sizeof(uint32_t), status, CoreType::WORKER);
        return status[TT_FABRIC_STATUS_INDEX];
    };

    // READY_TO_STOP is published before receivers enter their hold loop. The host reads that live
    // status rather than waiting for program exit, then releases every checker only after the sender
    // has closed its manager and every target has completed its initial validation.
    bool sender_done = false;
    bool targets_ready = false;
    bool host_timed_out = false;
    const auto validation_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    do {
        sender_done = read_status(source_device, sender_logical_core) == TT_FABRIC_STATUS_PASS;
        targets_ready = true;
        for (const auto& record : programs) {
            if (!record.is_target) {
                continue;
            }
            const uint32_t status = read_status(record.device, receiver_logical_core);
            targets_ready &= status == source_inject::STATUS_READY_TO_STOP ||
                             status == TT_FABRIC_STATUS_DATA_MISMATCH || status == TT_FABRIC_STATUS_TIMEOUT;
        }
        if (std::chrono::steady_clock::now() >= validation_deadline) {
            host_timed_out = true;
            break;
        }
    } while (!sender_done || !targets_ready);

    stop_value[0] = 1;
    for (const auto& record : programs) {
        tt_metal::slow_dispatch::WriteToL1(
            *record.device,
            receiver_logical_core,
            worker_mem_map.notification_mailbox_address,
            stop_value,
            CoreType::WORKER);
        tt_metal::MetalContext::instance().get_cluster().l1_barrier(record.physical_id);
    }
    for (auto& record : programs) {
        fixture->WaitForSingleProgramDone(record.device);
    }

    EXPECT_FALSE(host_timed_out) << "Timed out waiting for sender completion and target validation";
    EXPECT_EQ(read_status(source_device, sender_logical_core), TT_FABRIC_STATUS_PASS);
    for (const auto& record : programs) {
        std::vector<uint32_t> results;
        tt_metal::slow_dispatch::ReadFromL1(
            *record.device,
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            results,
            CoreType::WORKER);
        EXPECT_EQ(results[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS)
            << "Receiver failure on physical chip " << record.physical_id << ", phase " << results[TT_FABRIC_MISC_INDEX]
            << ", packet " << results[TT_FABRIC_MISC_INDEX + 1] << ", address 0x" << std::hex
            << results[TT_FABRIC_MISC_INDEX + 2] << ", expected 0x" << results[TT_FABRIC_MISC_INDEX + 3]
            << ", actual 0x" << results[TT_FABRIC_MISC_INDEX + 4];
    }
}

}  // namespace

TEST_F(Fabric2DFixture, TestMeshMulticastSourceInjectApis) { run_source_inject_test(this); }

}  // namespace tt::tt_fabric::fabric_router_tests
