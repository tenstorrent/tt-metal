// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <distributed/mesh_device_view_impl.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <internal/inter_domain.hpp>

#include <internal/cluster.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::inter_domain_test {
namespace {

using namespace tt::tt_metal;
using namespace tt::tt_fabric::internal;
namespace fs = std::filesystem;

constexpr uint32_t kNumPackets = 16;
constexpr uint32_t kPacketSize = 1024;
constexpr uint32_t kBytes = kNumPackets * kPacketSize;
using RouteEntry = std::remove_extent_t<decltype(HybridMeshPacketHeader::route_buffer)>;
constexpr size_t kEntriesPerArg = sizeof(uint32_t) / sizeof(RouteEntry);
constexpr size_t kEntryBits = sizeof(RouteEntry) * CHAR_BIT;
constexpr uint32_t kHeaderOffset = 0;
constexpr uint32_t kSrcOffset = 4096;
constexpr uint32_t kDstOffset = kSrcOffset + kBytes;
const CoreCoord kCore{0, 0};

std::optional<uint32_t> env_u32(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::optional<uint32_t>(std::stoul(v)) : std::nullopt;
}

struct PortInfo {
    uint64_t asic = 0;
    uint32_t chan = 0;
    uint64_t peer_asic = 0;
    uint32_t peer_chan = 0;
    InterDomainMetadata md;
};
struct RxInfo {
    uint32_t x = 0, y = 0, addr = 0;
    uint32_t mesh = 0, chip = 0;
};
// one process's side: its ports, its chips, and what it sent (from, to)
struct Side {
    std::vector<PortInfo> ports;
    std::map<uint64_t, RxInfo> rx;
    std::set<std::pair<uint64_t, uint64_t>> sent;
};

void write_file(const fs::path& path, const std::string& text) {
    const fs::path tmp = path.string() + ".tmp";
    std::ofstream(tmp) << text;
    fs::rename(tmp, path);
}

void wait_file(const fs::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (!fs::exists(path)) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "no " << path;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

std::string to_text(const Side& s) {
    std::ostringstream o;
    for (const auto& p : s.ports) {
        o << "port " << p.asic << " " << p.chan << " " << p.peer_asic << " " << p.peer_chan << " " << p.md.domain_id
          << " " << p.md.vc2_receiver_channel_addr << " " << p.md.vc2_receiver_channel_slots << " "
          << p.md.max_packet_size_bytes << " " << p.md.layout_fingerprint << "\n";
    }
    for (const auto& [asic, r] : s.rx) {
        o << "rx " << asic << " " << r.x << " " << r.y << " " << r.addr << " " << r.mesh << " " << r.chip << "\n";
    }
    for (const auto& [from, to] : s.sent) {
        o << "sent " << from << " " << to << "\n";
    }
    return o.str();
}

void from_text(const fs::path& path, Side& s) {
    std::ifstream in(path);
    std::string kind;
    while (in >> kind) {
        if (kind == "port") {
            PortInfo p;
            in >> p.asic >> p.chan >> p.peer_asic >> p.peer_chan >> p.md.domain_id >> p.md.vc2_receiver_channel_addr >>
                p.md.vc2_receiver_channel_slots >> p.md.max_packet_size_bytes >> p.md.layout_fingerprint;
            s.ports.push_back(p);
        } else if (kind == "rx") {
            uint64_t asic = 0;
            RxInfo r;
            in >> asic >> r.x >> r.y >> r.addr >> r.mesh >> r.chip;
            s.rx[asic] = r;
        } else {
            uint64_t from = 0, to = 0;
            in >> from >> to;
            s.sent.emplace(from, to);
        }
    }
}

uint32_t pattern(uint32_t domain, uint64_t asic, uint32_t round, uint32_t i) {
    return (domain << 24) ^ (round << 20) ^ static_cast<uint32_t>(asic * 2654435761u) ^ (i * 0x9E3779B9u);
}

// a chip's slot: its index among its domain's chips
uint32_t slot_of(const Side& domain, uint64_t asic) {
    return static_cast<uint32_t>(std::distance(domain.rx.begin(), domain.rx.find(asic)));
}

// Each domain is one process per mesh of the MGD; processes meet through files in TT_INTER_DOMAIN_DIR. Each domain
// sends to every chip of the other: from each exit chip over its cable (landing there, then VC2 across the peer's
// domain), and from the exit chip's mesh behind it (VC0 to the exit chip, then the cable).
TEST(InterDomain, Sanity) {
    const auto domain = env_u32("TT_INTER_DOMAIN_ID");
    const auto peer_domain = env_u32("TT_INTER_DOMAIN_PEER_ID");
    const char* dir_env = std::getenv("TT_INTER_DOMAIN_DIR");
    if (!domain || !peer_domain || dir_env == nullptr) {
        GTEST_SKIP() << "needs a peer process: see run_inter_domain_sanity.sh";
    }
    ASSERT_TRUE(*domain < 256 && *peer_domain < 256) << "pattern() keeps a domain id in one byte";
    const fs::path dir(dir_env);

    SetFabricConfig(FabricConfig::FABRIC_2D);
    auto mesh =
        distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()));
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto meshes = mesh_graph.get_mesh_ids();
    const uint32_t my_mesh = *mesh->get_fabric_node_id(distributed::MeshCoordinate(0, 0)).mesh_id;
    auto file = [&](uint32_t d, uint32_t m, const std::string& what) {
        return dir / ("d" + std::to_string(d) + ".m" + std::to_string(m) + "." + what);
    };
    auto mine = [&](const std::string& what) { return file(*domain, my_mesh, what); };
    // every process of domain d: each mesh's file
    auto gather = [&](uint32_t d, const std::string& what) {
        Side all;
        for (MeshId m : meshes) {
            wait_file(file(d, *m, what));
            from_text(file(d, *m, what), all);
        }
        return all;
    };

    std::map<uint64_t, distributed::MeshCoordinate> coord_of;
    std::map<uint64_t, IDevice*> device_of;
    std::map<uint64_t, tt::tt_fabric::FabricNodeId> node_of;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh->shape())) {
        const auto node = mesh->get_fabric_node_id(coord);
        const uint64_t asic = *tt::tt_metal::internal::get_chip_unique_id_from_fabric_node_id(*mesh, node);
        coord_of.emplace(asic, coord);
        device_of[asic] = mesh->get_view().impl().get_device(coord);
        node_of.emplace(asic, node);
    }

    Side me;
    std::vector<InterDomainPort> ports;
    for (const auto& port : get_inter_domain_ports(*mesh)) {
        if (port.has_router) {
            ports.push_back(port);
        }
    }
    for (const auto& port : ports) {
        me.ports.push_back(PortInfo{
            .asic = get_inter_domain_port_asic(port),
            .chan = port.eth_chan,
            .peer_asic = port.peer_asic,
            .peer_chan = port.peer_chan,
            .md = get_inter_domain_metadata(port, *domain)});
        EXPECT_EQ(get_inter_domain_state(port), InterDomainState::DRAIN);
    }
    for (const auto& [asic, dev] : device_of) {
        const uint32_t base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
        const auto core = dev->worker_core_from_logical_core(kCore);
        const auto& node = node_of.at(asic);
        me.rx[asic] = RxInfo{
            static_cast<uint32_t>(core.x),
            static_cast<uint32_t>(core.y),
            base + kDstOffset,
            *node.mesh_id,
            node.chip_id};
    }
    write_file(mine("meta"), to_text(me));
    const Side mine_all = gather(*domain, "meta");
    const Side peer = gather(*peer_domain, "meta");

    struct Cable {
        InterDomainPort port;
        PortInfo peer;
        uint32_t rank = 0;
        std::pair<uint64_t, uint64_t> chips;
    };
    std::map<std::pair<uint64_t, uint64_t>, Cable> cables;
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto& p = me.ports[i];
        for (const auto& q : peer.ports) {
            if (q.asic != p.peer_asic || q.chan != p.peer_chan || q.peer_asic != p.asic || q.peer_chan != p.chan) {
                continue;
            }
            const bool lower = *domain < *peer_domain;
            const uint32_t rank = lower ? p.chan : q.chan;
            const auto key = lower ? std::pair{p.asic, q.asic} : std::pair{q.asic, p.asic};
            auto it = cables.find(key);
            if (it == cables.end() || rank < it->second.rank) {
                cables[key] = Cable{ports[i], q, rank, {p.asic, q.asic}};
            }
        }
    }
    ASSERT_FALSE(cables.empty() && !me.ports.empty()) << "no cable to domain " << *peer_domain;

    const uint32_t region = static_cast<uint32_t>(peer.rx.size()) * kBytes;
    const uint32_t rounds = env_u32("TT_INTER_DOMAIN_ROUNDS").value_or(2);
    for (uint32_t round = 0; round < rounds; ++round) {
        auto step = [&](const std::string& what) {
            log_info(tt::LogTest, "domain {} mesh {} round {}: {}", *domain, my_mesh, round, what);
            write_file(mine(std::to_string(round) + "." + what), "");
            for (MeshId m : meshes) {
                wait_file(file(*peer_domain, *m, std::to_string(round) + "." + what));
            }
        };
        for (const auto& [asic, dev] : device_of) {
            std::vector<uint32_t> zeros(region / 4, 0);
            detail::WriteToDeviceL1(dev, kCore, me.rx.at(asic).addr, zeros);
        }
        step("ready");

        for (const auto& [_, cable] : cables) {
            connect_inter_domain(cable.port, *domain, cable.peer.md);
            EXPECT_EQ(get_inter_domain_state(cable.port), InterDomainState::SERVICE_TRAFFIC);
        }
        step("connected");

        struct Send {
            uint64_t from;
            uint64_t to;
            std::vector<RouteEntry> route;
            std::function<void(Program&, std::vector<uint32_t>&)> connect;
        };
        std::vector<Send> sends;
        for (const auto& [_, cable] : cables) {
            const auto [exit_chip, peer_chip] = cable.chips;
            const auto port = cable.port;
            auto over_cable = [port](Program& program, std::vector<uint32_t>& args) {
                append_inter_domain_connection_rt_args(port, program, kCore, args);
            };
            for (const auto& [to, _] : peer.rx) {
                sends.push_back(Send{exit_chip, to, {}, over_cable});
            }
            const auto exit = node_of.at(exit_chip);
            for (const auto& [asic, dev] : device_of) {
                const auto from = node_of.at(asic);
                const auto out = control_plane.get_forwarding_direction(from, exit);
                if (asic == exit_chip || !out || control_plane.get_chip_neighbors(from, *out).empty()) {
                    continue;
                }
                const auto shape = mesh_graph.get_mesh_shape(exit.mesh_id);
                const auto at = mesh_graph.chip_to_coordinate(exit.mesh_id, exit.chip_id);
                const auto ingress = get_opposite_direction(*out);
                std::vector<RouteEntry> route(shape[0] + shape[1], 0);
                const bool x_axis = ingress == RoutingDirection::E || ingress == RoutingDirection::W;
                route[x_axis ? shape[0] + at[1] : at[0]] =
                    1u << control_plane.routing_direction_to_eth_direction(port.direction);
                sends.push_back(Send{
                    asic,
                    peer_chip,
                    route,
                    [&mesh, from, exit, plane = port.plane](Program& program, std::vector<uint32_t>& args) {
                        append_fabric_connection_rt_args(*mesh, from, exit, plane, program, kCore, args);
                    }});
            }
        }
        Side sent;
        for (auto& send : sends) {
            IDevice* dev = device_of.at(send.from);
            const uint32_t base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
            std::vector<uint32_t> data(kBytes / 4);
            for (uint32_t i = 0; i < data.size(); ++i) {
                data[i] = pattern(*domain, send.from, round, i);
            }
            detail::WriteToDeviceL1(dev, kCore, base + kSrcOffset, data);
            Program program = CreateProgram();
            const auto kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_fabric/inter_domain/kernels/inter_domain_sender.cpp",
                kCore,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .defines = {{"FABRIC_2D", "1"}}});
            const auto& rx = peer.rx.at(send.to);
            std::vector<uint32_t> args = {
                rx.x,
                rx.y,
                rx.addr + slot_of(mine_all, send.from) * kBytes,
                base + kSrcOffset,
                base + kHeaderOffset,
                kNumPackets,
                kPacketSize,
                rx.mesh,
                rx.chip,
                static_cast<uint32_t>(send.route.size())};
            for (size_t b = 0; b < send.route.size(); b += kEntriesPerArg) {
                uint32_t word = 0;
                for (size_t k = 0; k < kEntriesPerArg && b + k < send.route.size(); ++k) {
                    word |= uint32_t{send.route[b + k]} << (kEntryBits * k);
                }
                args.push_back(word);
            }
            send.connect(program, args);
            SetRuntimeArgs(program, kernel, kCore, args);
            distributed::MeshWorkload workload;
            workload.add_program(distributed::MeshCoordinateRange(coord_of.at(send.from)), std::move(program));
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
            sent.sent.emplace(send.from, send.to);
        }
        log_info(tt::LogTest, "domain {} mesh {} round {}: {} sends", *domain, my_mesh, round, sends.size());
        write_file(mine(std::to_string(round) + ".sent"), to_text(sent));
        const Side expected = gather(*peer_domain, std::to_string(round) + ".sent");

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        std::map<std::pair<uint64_t, uint64_t>, uint32_t> bad;
        for (const auto& [from, to] : expected.sent) {
            if (!device_of.contains(to)) {
                continue;
            }
            while (true) {
                std::vector<uint32_t> got;
                detail::ReadFromDeviceL1(
                    device_of.at(to), kCore, me.rx.at(to).addr + slot_of(peer, from) * kBytes, kBytes, got);
                uint32_t wrong = 0;
                for (uint32_t i = 0; i < kBytes / 4; ++i) {
                    wrong += got[i] != pattern(*peer_domain, from, round, i);
                }
                bad[std::pair(from, to)] = wrong;
                if (wrong == 0 || std::chrono::steady_clock::now() > deadline) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            EXPECT_EQ(bad[std::pair(from, to)], 0u) << "round " << round << ": chip " << to << " from asic " << from;
        }
        log_info(tt::LogTest, "domain {} mesh {} round {}: checked {} sends", *domain, my_mesh, round, bad.size());
        step("received");

        for (const auto& [_, cable] : cables) {
            disconnect_inter_domain(cable.port);
            EXPECT_EQ(get_inter_domain_state(cable.port), InterDomainState::DRAIN);
        }
        step("done");
    }
    mesh->close();
    SetFabricConfig(FabricConfig::DISABLED);
}

}  // namespace
}  // namespace tt::tt_fabric::inter_domain_test
