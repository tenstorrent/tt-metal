// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "harness_run.hpp"

#include <algorithm>
#include <functional>
#include <thread>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/program/slow_dispatch.hpp"

namespace tt::tt_fabric::vcluster {

namespace {

using namespace tt::tt_metal;

constexpr auto kPollInterval = std::chrono::milliseconds(200);
constexpr uint32_t kHostWrite = 1;

IDevice& device_of(distributed::MeshDevice& mesh) { return slow_dispatch::physical_device_from_unit_mesh(mesh); }

CoreCoord cce0_of_tile(IDevice& device, uint32_t tile) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device.id());
    for (const auto& core : soc_desc.get_arch_descriptor().get_keraunos().cores) {
        if (core.hsio == tile && core.cce == 0) {
            const CoreCoord logical = soc_desc.get_logical_ethernet_core_from_physical({core.coord.x, core.coord.y});
            TT_FATAL(
                device.ethernet_cores().contains(logical),
                "CCE0 of tile {} ({}) is not an Ethernet core of device {}",
                tile,
                logical.str(),
                device.id());
            return logical;
        }
    }
    TT_FATAL(false, "The SoC descriptor of device {} lists no CCE0 of tile {}", device.id(), tile);
    return {};
}

std::vector<uint32_t> read_words(distributed::MeshDevice& mesh, CoreCoord core, uint64_t address, uint32_t count) {
    std::vector<uint32_t> words;
    slow_dispatch::ReadFromL1(mesh, core, address, count * sizeof(uint32_t), words, CoreType::ETH);
    TT_FATAL(
        words.size() == count, "Read of {} words at {:#x} on {} returned {}", count, address, core.str(), words.size());
    return words;
}

uint32_t read_control(distributed::MeshDevice& mesh, CoreCoord core, const PlannedNode& node, vc_node::ControlWord w) {
    return read_words(mesh, core, node.region_base + node.layout.control + w * sizeof(uint32_t), 1)[0];
}

void write_control(
    distributed::MeshDevice& mesh, CoreCoord core, const PlannedNode& node, vc_node::ControlWord w, uint32_t value) {
    std::vector<uint32_t> words{value};
    slow_dispatch::WriteToL1(mesh, core, node.region_base + node.layout.control + w * sizeof(uint32_t), words, CoreType::ETH);
}

uint64_t read_u64(distributed::MeshDevice& mesh, CoreCoord core, uint64_t address) {
    const auto words = read_words(mesh, core, address, 2);
    return words[0] | (uint64_t{words[1]} << 32);
}

// Polls until @p done holds or @p deadline passes. True when it held.
bool wait_until(const std::function<bool()>& done, std::chrono::steady_clock::time_point deadline) {
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(kPollInterval);
    }
    return true;
}

NodeReadout read_node(distributed::MeshDevice& mesh, CoreCoord core, const PlannedNode& node) {
    const auto control =
        read_words(mesh, core, node.region_base + node.layout.control, vc_node::kControlBytes / sizeof(uint32_t));
    NodeReadout out{
        .status = control[vc_node::kCtlStatus],
        .postcode = control[vc_node::kCtlPostcode],
        .detail = control[vc_node::kCtlDetail],
        .originated = control[vc_node::kCtlOriginated],
        .delivered = control[vc_node::kCtlDelivered],
        .bad = control[vc_node::kCtlBad]};
    for (uint32_t p = 0; p < node.ports.size(); p++) {
        const uint64_t block = node.region_base + node.layout.port_block(p);
        out.sent.push_back(read_u64(mesh, core, block + vc_node::kPortSent));
        out.received.push_back(read_u64(mesh, core, block + vc_node::kPortReceived));
    }
    return out;
}

// SPEC-0012 § The harness, Verdict.
std::vector<std::string> verdict(const HarnessPlan& plan, const std::vector<NodeReadout>& nodes) {
    std::vector<std::string> failures;
    const uint64_t want = uint64_t{plan.shape.packets_per_flow} * (plan.nodes.size() - 1);
    for (std::size_t d = 0; d < nodes.size(); d++) {
        const PlannedNode& node = plan.nodes[d];
        const NodeReadout& r = nodes[d];
        if (r.status != vc_node::kStatusDone) {
            failures.push_back(fmt::format(
                "device {}: status {:#x}, want {:#x} (postcode {:#x}, detail {})",
                node.name,
                r.status,
                static_cast<uint32_t>(vc_node::kStatusDone),
                r.postcode,
                r.detail));
        }
        if (r.delivered != want || r.bad != 0) {
            failures.push_back(
                fmt::format("device {}: delivered {} bad {}, want {} and 0", node.name, r.delivered, r.bad, want));
        }
        if (r.originated != want) {
            failures.push_back(fmt::format("device {}: originated {}, want {}", node.name, r.originated, want));
        }
        for (std::size_t p = 0; p < node.ports.size(); p++) {
            const uint64_t want_sent = node.predicted_sent[p];
            const uint64_t want_received = node.predicted_received[p];
            if (r.sent[p] != want_sent || r.received[p] != want_received || want_sent < plan.shape.packets_per_flow ||
                want_received < plan.shape.packets_per_flow) {
                failures.push_back(fmt::format(
                    "device {} port {}: sent {} received {}, predicted {} and {}, each at least {}",
                    node.name,
                    p,
                    r.sent[p],
                    r.received[p],
                    want_sent,
                    want_received,
                    plan.shape.packets_per_flow));
            }
        }
    }
    return failures;
}

std::string findings_text(const std::vector<ManifestFinding>& rows, const std::vector<HarnessFinding>& h_rows) {
    std::vector<std::string> lines;
    for (const auto& f : rows) {
        lines.push_back(fmt::format("row {}: {}", static_cast<uint32_t>(f.check), f.detail));
    }
    for (const auto& f : h_rows) {
        lines.push_back(fmt::format("H{}: {}", static_cast<uint32_t>(f.check), f.detail));
    }
    return fmt::format("{}", fmt::join(lines, "; "));
}

}  // namespace

Program build_harness_program(distributed::MeshDevice& mesh, const HarnessPlan& plan, uint32_t nonce) {
    IDevice& device = device_of(mesh);
    Program program = CreateProgram();
    for (const PlannedNode& node : plan.nodes) {
        const CoreCoord core = cce0_of_tile(device, node.tile);
        const KernelHandle kernel = CreateKernel(
            program,
            kNodeKernel,
            core,
            EthernetConfig{
                .eth_mode = Eth::IDLE,
                .noc = node.hart == 0 ? NOC::NOC_0 : NOC::NOC_1,
                .processor = static_cast<DataMovementProcessor>(node.hart)});
        SetRuntimeArgs(program, kernel, core, node_runtime_args(plan, node.device, nonce));
    }
    program.impl().compile(&device);
    return program;
}

HarnessResult run_harness(
    distributed::MeshDevice& mesh,
    const Manifest& manifest,
    const PackageFacts& facts,
    const RunShape& shape,
    std::chrono::seconds timeout) {
    // 1.
    const auto rows = check_manifest(manifest, facts);
    const auto h_rows = rows.empty() ? check_harness_manifest(manifest, facts, shape) : std::vector<HarnessFinding>{};
    TT_FATAL(rows.empty() && h_rows.empty(), "The manifest is refused: {}", findings_text(rows, h_rows));
    // 2, 3.
    const HarnessPlan plan = plan_harness(manifest, shape);
    const uint32_t nonce =
        static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1u;
    IDevice& device = device_of(mesh);
    std::vector<CoreCoord> cores;
    for (const PlannedNode& node : plan.nodes) {
        cores.push_back(cce0_of_tile(device, node.tile));
    }
    // 4.
    for (std::size_t d = 0; d < plan.nodes.size(); d++) {
        const PlannedNode& node = plan.nodes[d];
        std::vector<uint32_t> zeros(node.layout.total / sizeof(uint32_t), 0);
        slow_dispatch::WriteToL1(mesh, cores[d], node.region_base, zeros, CoreType::ETH);
    }
    Program program = build_harness_program(mesh, plan, nonce);
    log_info(tt::LogTest, "Harness: {} nodes, ring_slots {}, nonce {:#x}", plan.nodes.size(), plan.ring_slots, nonce);
    slow_dispatch::LaunchProgramAsync(device, program, /*force_slow_dispatch=*/false);

    HarnessResult result;
    const auto status_of = [&](std::size_t d) { return read_control(mesh, cores[d], plan.nodes[d], vc_node::kCtlStatus); };
    // 5.
    bool ready = true;
    for (std::size_t d = 0; d < plan.nodes.size() && ready; d++) {
        uint32_t status = vc_node::kStatusNotStarted;
        ready = wait_until(
                    [&] {
                        status = status_of(d);
                        return status == vc_node::kStatusWaitingForGo || status == vc_node::kStatusHalted;
                    },
                    std::chrono::steady_clock::now() + timeout) &&
                status == vc_node::kStatusWaitingForGo;
    }
    if (ready) {
        for (std::size_t d = 0; d < plan.nodes.size(); d++) {
            write_control(mesh, cores[d], plan.nodes[d], vc_node::kCtlGo, kHostWrite);
        }
        // 6.
        const uint32_t want = shape.packets_per_flow * (plan.nodes.size() - 1);
        result.timed_out = !wait_until(
            [&] {
                bool all_delivered = true;
                for (std::size_t d = 0; d < plan.nodes.size(); d++) {
                    if (status_of(d) == vc_node::kStatusHalted) {
                        return true;
                    }
                    all_delivered &= read_control(mesh, cores[d], plan.nodes[d], vc_node::kCtlDelivered) == want;
                }
                return all_delivered;
            },
            std::chrono::steady_clock::now() + timeout);
    } else {
        result.failures.push_back("a node halted or did not reach waiting-for-go in time; go was not written");
    }
    // 7.
    for (std::size_t d = 0; d < plan.nodes.size(); d++) {
        write_control(mesh, cores[d], plan.nodes[d], vc_node::kCtlStop, kHostWrite);
    }
    slow_dispatch::WaitProgramDone(device, program);
    // 8.
    for (std::size_t d = 0; d < plan.nodes.size(); d++) {
        result.nodes.push_back(read_node(mesh, cores[d], plan.nodes[d]));
    }
    auto failures = verdict(plan, result.nodes);
    result.failures.insert(result.failures.end(), failures.begin(), failures.end());
    return result;
}

}  // namespace tt::tt_fabric::vcluster
