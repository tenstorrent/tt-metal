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
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/mesh/allocator.hpp>
#include <tt-metalium/mesh/mesh_device.hpp>
#include <tt-metalium/mesh/mesh_device_view.hpp>
#include <distributed/mesh_device_view_impl.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <internal/fabric/inter_domain.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>

#include <internal/cluster.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
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
// one process's side: its ports, its chips, and what it sent (from, the chip it landed on, to)
struct Side {
    std::vector<PortInfo> ports;
    std::map<uint64_t, RxInfo> rx;
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> sent;
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
    for (const auto& [from, via, to] : s.sent) {
        o << "sent " << from << " " << via << " " << to << "\n";
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
            uint64_t from = 0, via = 0, to = 0;
            in >> from >> via >> to;
            s.sent.emplace(from, via, to);
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

// Packets each of this process's routers sent, by (chip, eth channel).
std::map<std::pair<FabricNodeId, chan_id_t>, uint64_t> packets_sent(
    const distributed::MeshDevice& mesh, const std::map<uint64_t, FabricNodeId>& node_of) {
    std::map<std::pair<FabricNodeId, chan_id_t>, uint64_t> sent;
    for (const auto& [_, node] : node_of) {
        for (const auto& sample : read_fabric_telemetry(mesh, node)) {
            if (sample.snapshot.dynamic_info) {
                sent[{node, sample.channel_id}] = sample.snapshot.dynamic_info->tx_bandwidth.packets_sent;
            }
        }
    }
    return sent;
}

// Each domain is one process per mesh of the MGD; processes meet through files in TT_INTER_DOMAIN_DIR. Each domain
// sends to every chip of the other from each exit chip over its cable, landing there, then VC2 across the peer's
// domain. One send at a time, plus sends from the exit chip's mesh behind it (VC0 to the exit chip, then the cable); or
// (all_at_once) every cable's sends together, then the packets each router sent checked against VC2's routes over the
// domain's mesh (TT_METAL_FABRIC_TELEMETRY=1).
void run(bool all_at_once) {
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
    const bool check_routes = all_at_once && MetalContext::instance().rtoptions().get_enable_fabric_telemetry();

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
            tt::tt_metal::detail::WriteToDeviceL1(dev, kCore, me.rx.at(asic).addr, zeros);
        }
        step("ready");

        for (const auto& [_, cable] : cables) {
            log_info(
                tt::LogTest,
                "domain {}: cable {} chan {} {} plane {} to asic {} chan {}",
                *domain,
                cable.port.node,
                cable.port.eth_chan,
                cable.port.direction,
                cable.port.plane,
                cable.peer.asic,
                cable.peer.chan);
            connect_inter_domain(cable.port, *domain, cable.peer.md);
            EXPECT_EQ(get_inter_domain_state(cable.port), InterDomainState::SERVICE_TRAFFIC);
        }
        const auto before = check_routes ? packets_sent(*mesh, node_of) : decltype(packets_sent(*mesh, node_of)){};
        step("connected");

        // one worker per connection, sending to each of `to` in turn
        struct Send {
            uint64_t from;
            uint64_t via;
            std::vector<uint64_t> to;
            std::vector<RouteEntry> route;
            std::function<void(Program&, const CoreCoord&, std::vector<uint32_t>&)> connect;
        };
        std::vector<Send> sends;
        for (const auto& [_, cable] : cables) {
            const auto [exit_chip, peer_chip] = cable.chips;
            const auto port = cable.port;
            auto over_cable = [port](Program& program, const CoreCoord& core, std::vector<uint32_t>& args) {
                append_inter_domain_connection_rt_args(port, program, core, args);
            };
            std::vector<uint64_t> all;
            for (const auto& [to, _] : peer.rx) {
                all.push_back(to);
            }
            if (all_at_once) {
                sends.push_back(Send{exit_chip, peer_chip, all, {}, over_cable});
                continue;
            }
            for (uint64_t to : all) {
                sends.push_back(Send{exit_chip, peer_chip, {to}, {}, over_cable});
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
                    {peer_chip},
                    route,
                    [&mesh, from, exit, plane = port.plane](
                        Program& program, const CoreCoord& core, std::vector<uint32_t>& args) {
                        append_fabric_connection_rt_args(*mesh, from, exit, plane, program, core, args);
                    }});
            }
        }
        Side sent;
        std::map<uint64_t, Program> programs;  // all at once: every chip's sends in one program, a core each
        std::map<uint64_t, uint32_t> cores_used;
        for (auto& send : sends) {
            IDevice* dev = device_of.at(send.from);
            const uint32_t base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
            const uint32_t k = all_at_once ? cores_used[send.from]++ : 0;
            const auto grid = dev->compute_with_storage_grid_size();
            const CoreCoord core{k % grid.x, k / grid.x};
            std::vector<uint32_t> data(kBytes / 4);
            for (uint32_t i = 0; i < data.size(); ++i) {
                data[i] = pattern(*domain, send.from, round, i);
            }
            tt::tt_metal::detail::WriteToDeviceL1(dev, core, base + kSrcOffset, data);
            Program single = CreateProgram();
            Program& program = all_at_once ? programs[send.from] : single;
            const auto kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_fabric/inter_domain/kernels/inter_domain_sender.cpp",
                core,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .defines = {{"FABRIC_2D", "1"}}});
            std::vector<uint32_t> args = {
                base + kSrcOffset,
                base + kHeaderOffset,
                kNumPackets,
                kPacketSize,
                static_cast<uint32_t>(send.route.size())};
            for (size_t b = 0; b < send.route.size(); b += kEntriesPerArg) {
                uint32_t word = 0;
                for (size_t e = 0; e < kEntriesPerArg && b + e < send.route.size(); ++e) {
                    word |= uint32_t{send.route[b + e]} << (kEntryBits * e);
                }
                args.push_back(word);
            }
            args.push_back(static_cast<uint32_t>(send.to.size()));
            for (uint64_t to : send.to) {
                const auto& rx = peer.rx.at(to);
                args.insert(
                    args.end(), {rx.x, rx.y, rx.addr + slot_of(mine_all, send.from) * kBytes, rx.mesh, rx.chip});
                sent.sent.emplace(send.from, send.via, to);
            }
            send.connect(program, core, args);
            SetRuntimeArgs(program, kernel, core, args);
            if (!all_at_once) {
                distributed::MeshWorkload workload;
                workload.add_program(distributed::MeshCoordinateRange(coord_of.at(send.from)), std::move(single));
                distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
            }
        }
        if (all_at_once) {
            distributed::MeshWorkload workload;
            for (auto& [asic, program] : programs) {
                workload.add_program(distributed::MeshCoordinateRange(coord_of.at(asic)), std::move(program));
            }
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
        }
        log_info(tt::LogTest, "domain {} mesh {} round {}: {} sends", *domain, my_mesh, round, sent.sent.size());
        write_file(mine(std::to_string(round) + ".sent"), to_text(sent));
        const Side expected = gather(*peer_domain, std::to_string(round) + ".sent");

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        uint32_t checked = 0;
        for (const auto& [from, via, to] : expected.sent) {
            if (!device_of.contains(to)) {
                continue;
            }
            uint32_t wrong = 0;
            while (true) {
                std::vector<uint32_t> got;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device_of.at(to), kCore, me.rx.at(to).addr + slot_of(peer, from) * kBytes, kBytes, got);
                wrong = 0;
                for (uint32_t i = 0; i < kBytes / 4; ++i) {
                    wrong += got[i] != pattern(*peer_domain, from, round, i);
                }
                if (wrong == 0 || std::chrono::steady_clock::now() > deadline) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ++checked;
            EXPECT_EQ(wrong, 0u) << "round " << round << ": chip " << to << " from asic " << from;
        }
        log_info(tt::LogTest, "domain {} mesh {} round {}: checked {} sends", *domain, my_mesh, round, checked);

        if (check_routes) {
            // VC2 over the domain's mesh, Y then X from the landing chip: the packets each of this process's routers
            // forwards to a neighbour in the domain, counted, against what each one sent
            const auto& dm = control_plane.get_fabric_context().get_inter_domain_mesh();
            const uint32_t cols = dm.shape()[1];
            std::map<ChipId, FabricNodeId> at_domain_chip;
            for (const auto& [node, chip] : dm.domain_chip) {
                at_domain_chip.emplace(chip, node);
            }
            std::map<std::pair<FabricNodeId, FabricNodeId>, uint64_t> predicted;
            uint32_t longest = 0;
            for (const auto& [from, via, to] : expected.sent) {
                const auto a = dm.domain_chip.at(control_plane.get_fabric_node_id_from_asic_id(via));
                const auto b = dm.domain_chip.at(control_plane.get_fabric_node_id_from_asic_id(to));
                int y = a / cols, x = a % cols;
                const int dy = b / cols, dx = b % cols;
                longest = std::max<uint32_t>(longest, std::abs(y - dy) + std::abs(x - dx));
                while (y != dy || x != dx) {
                    const int ny = y != dy ? y + (dy > y ? 1 : -1) : y;
                    const int nx = y != dy ? x : x + (dx > x ? 1 : -1);
                    predicted[{at_domain_chip.at(y * cols + x), at_domain_chip.at(ny * cols + nx)}] += kNumPackets;
                    y = ny;
                    x = nx;
                }
            }
            const auto after = packets_sent(*mesh, node_of);
            std::map<std::pair<FabricNodeId, FabricNodeId>, uint64_t> measured;
            for (const auto& [key, count] : after) {
                const auto& [node, chan] = key;
                std::optional<FabricNodeId> peer_node;
                if (const auto live = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan)) {
                    peer_node = live->first;
                }
                for (const auto& l : dm.vc2_links) {
                    if (l.a == node && l.a_chan == chan) {
                        peer_node = l.b;
                    } else if (l.b == node && l.b_chan == chan) {
                        peer_node = l.a;
                    }
                }
                if (peer_node && dm.domain_chip.contains(*peer_node)) {
                    const auto was = before.find(key);
                    measured[{node, *peer_node}] += count - (was == before.end() ? 0 : was->second);
                }
            }
            uint32_t links = 0;
            for (const auto& [link, count] : measured) {
                const auto it = predicted.find(link);
                EXPECT_EQ(count, it == predicted.end() ? 0 : it->second)
                    << "round " << round << ": " << link.first << " -> " << link.second;
                ++links;
            }
            log_info(
                tt::LogTest,
                "domain {} mesh {} round {}: {} links match VC2's routes over the {}x{} domain, longest {} hops",
                *domain,
                my_mesh,
                round,
                links,
                dm.shape()[0],
                cols,
                longest);
        }
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

TEST(InterDomain, Sanity) { run(/*all_at_once=*/false); }

TEST(InterDomain, AllAtOnce) { run(/*all_at_once=*/true); }

}  // namespace
}  // namespace tt::tt_fabric::inter_domain_test
