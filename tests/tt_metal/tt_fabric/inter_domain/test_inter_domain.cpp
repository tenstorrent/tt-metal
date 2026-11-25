// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <enchantum/enchantum.hpp>
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
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>
#include <internal/inter_domain.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>

#include <internal/cluster.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
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
        if (port.has_router && port.link_trained) {  // a router whose link is down relays and never connects
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
            std::optional<FabricNodeId> behind;  // over VC0 to this exit chip first
            uint32_t out_cable = 0;
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
                sends.push_back(Send{exit_chip, peer_chip, all, std::nullopt, 0, over_cable});
                continue;
            }
            for (uint64_t to : all) {
                sends.push_back(Send{exit_chip, peer_chip, {to}, std::nullopt, 0, over_cable});
            }
            const auto exit = node_of.at(exit_chip);
            for (const auto& [asic, dev] : device_of) {
                const auto from = node_of.at(asic);
                if (asic == exit_chip || from.mesh_id != exit.mesh_id) {
                    continue;
                }
                sends.push_back(Send{
                    asic,
                    peer_chip,
                    {peer_chip},
                    exit,
                    1u << control_plane.routing_direction_to_eth_direction(port.direction),
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
                send.behind.has_value(),
                send.behind ? send.behind->chip_id : 0,
                send.behind ? *send.behind->mesh_id : 0,
                send.out_cable};
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
                log_info(
                    tt::LogTest,
                    "domain {}: send {} -> {} via {}{}",
                    *domain,
                    send.from,
                    send.to.front(),
                    send.via,
                    send.behind ? " behind" : "");
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
            auto node_at = [&](uint64_t asic) {
                const auto& rx = mine_all.rx.at(asic);
                return FabricNodeId(MeshId{rx.mesh}, rx.chip);
            };
            for (const auto& [from, via, to] : expected.sent) {
                const auto a = dm.domain_chip.at(node_at(via));
                const auto b = dm.domain_chip.at(node_at(to));
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

// ---------------------------------------------------------------------------------------------------------------------
// Routing plane mismatch, scenario 1 (#65596): the domain routes on more planes than it has live links to the peer.
// TT_METAL_FABRIC_INTER_DOMAIN_DOWN_PLANES (the same on both sides) takes those planes' links down: their routers stay
// and relay into a live link's router on the same side. Every routed port, live or relaying, carries a worker's
// traffic (into the port's router) and traffic from another chip of the domain on the port's plane (through the
// domain's router on that plane, into the port's router); every byte must land at the peer domain.
// ---------------------------------------------------------------------------------------------------------------------

constexpr uint32_t kRelayPackets = 32;  // more than a channel has slots: every ring wraps
constexpr uint32_t kRelayBytes = kRelayPackets * kPacketSize;
constexpr uint32_t kRelaySrcOffset = 4096;
constexpr uint32_t kRelayDstOffset = kRelaySrcOffset + kRelayBytes;
constexpr uint32_t kRelayMaxSlots = 24;

struct RoutedPort {
    uint64_t asic = 0;
    uint32_t chan = 0;
    uint64_t peer_asic = 0;
    uint32_t peer_chan = 0;
    uint32_t plane = 0;
    bool live = true;
    InterDomainMetadata md;
};

// One stream a side sends: to its peer's chip `to`, into slot `slot` there.
struct Stream {
    uint64_t to = 0;
    uint32_t slot = 0;
};

// A router's init postcode (EDMStatus) and inter-domain state, for a router that has not settled.
void log_router(const InterDomainPort& port) {
    auto& context = MetalContext::instance();
    const auto& control_plane = context.get_control_plane();
    const auto& config = control_plane.get_fabric_context().get_builder_context().get_fabric_router_config();
    const ChipId chip = control_plane.get_physical_chip_id_from_fabric_node_id(port.node);
    const tt_cxy_pair core(chip, context.get_cluster().get_virtual_eth_core_from_channel(chip, port.eth_chan));
    uint32_t status = 0;
    context.get_cluster().read_core(&status, sizeof(status), core, config.edm_status_address);
    std::string semaphores;
    for (size_t ch = 0; ch < builder_config::num_max_sender_channels; ++ch) {
        uint32_t value = 0;
        context.get_cluster().read_core(
            &value, sizeof(value), core, config.sender_channels_connection_semaphore_address[ch]);
        semaphores += " " + std::to_string(value);
    }
    log_warning(
        tt::LogTest,
        "{} chan {} plane {} ({}): state {}, edm status {:#x}, sender connection semaphores{}",
        port.node,
        int(port.eth_chan),
        port.plane,
        port.link_trained ? "live" : "relaying",
        enchantum::to_string(get_inter_domain_state(port)),
        status,
        semaphores);
}

// Waits for every routed port's router to settle: DRAIN where its link is up, RELAY where it relays.
bool wait_settled(const std::vector<InterDomainPort>& ports) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto settled = [](const InterDomainPort& p) {
        return get_inter_domain_state(p) == (p.link_trained ? InterDomainState::DRAIN : InterDomainState::RELAY);
    };
    while (!std::all_of(ports.begin(), ports.end(), settled)) {
        if (std::chrono::steady_clock::now() > deadline) {
            for (const auto& p : ports) {
                log_router(p);
            }
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

uint32_t relay_pattern(uint32_t domain, uint32_t slot, uint32_t round, uint32_t i) {
    return (domain << 24) ^ (slot << 16) ^ (round << 12) ^ (i * 0x9E3779B9u);
}

// Each fabric router's packets sent over its link so far, by (asic, eth channel); empty without telemetry.
std::map<std::pair<uint64_t, uint32_t>, uint64_t> packets_sent_by_asic(
    const distributed::MeshDevice& mesh, const std::map<uint64_t, tt::tt_fabric::FabricNodeId>& node_of) {
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> sent;
    for (const auto& [asic, node] : node_of) {
        for (const auto& sample : read_fabric_telemetry(mesh, node)) {
            if (sample.snapshot.dynamic_info &&
                telemetry_stat_enabled(
                    sample.snapshot.static_info.supported_stats, FabricTelemetryStatistic::Bandwidth)) {
                sent[{asic, sample.channel_id}] = sample.snapshot.dynamic_info->tx_bandwidth.packets_sent;
            }
        }
    }
    return sent;
}

// Scenario 2 (#65597): what lands on a live port for another chip of this domain crosses the domain on the port's own
// plane when that chip's id is even, and on its scatter partner's plane (the plane of the down port relaying into it)
// when odd. In the far-only round nothing else crosses this domain, so a partner plane's router toward the other chip
// sends packets exactly when that chip's id is odd.
void check_scatter(
    const distributed::MeshDevice& mesh,
    const std::map<uint64_t, tt::tt_fabric::FabricNodeId>& node_of,
    const std::vector<InterDomainPort>& ports,
    const std::map<std::pair<uint64_t, uint32_t>, uint64_t>& before,
    const std::map<std::pair<uint64_t, uint32_t>, uint64_t>& after) {
    if (after.empty() || !MetalContext::instance().rtoptions().get_enable_fabric_telemetry()) {
        log_warning(
            tt::LogTest,
            "no fabric telemetry (TT_METAL_FABRIC_TELEMETRY, on the domain's own links): the scatter check is skipped");
        return;
    }
    const auto& control_plane = MetalContext::instance().get_control_plane();
    // a router counts only where telemetry runs on it (TT_METAL_FABRIC_TELEMETRY's chips and channels)
    auto delta = [&](uint64_t asic, uint32_t chan) -> std::optional<uint64_t> {
        const auto a = after.find({asic, chan});
        const auto b = before.find({asic, chan});
        if (a == after.end() || b == before.end()) {
            return std::nullopt;
        }
        return a->second - b->second;
    };
    // the down planes in order relay into the live ones in order: a live plane's partner is the down plane at its rank
    std::map<std::pair<uint64_t, RoutingDirection>, std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> sides;
    for (const auto& port : ports) {
        auto& [live, down] = sides[{get_inter_domain_port_asic(port), port.direction}];
        (port.link_trained ? live : down).push_back(port.plane);
    }
    size_t checked = 0;
    for (auto& [side, planes] : sides) {
        auto& [live, down] = planes;
        std::sort(live.begin(), live.end());
        std::sort(down.begin(), down.end());
        const auto& landing = node_of.at(side.first);
        for (size_t k = 0; k < std::min(live.size(), down.size()); ++k) {
            for (const auto& [other_asic, other] : node_of) {
                const auto out = control_plane.get_forwarding_direction(landing, other);
                if (other_asic == side.first || !out) {
                    continue;
                }
                const auto chans = control_plane.get_active_fabric_eth_routing_planes_in_direction(landing, *out);
                if (chans.size() <= std::max(live[k], down[k])) {
                    continue;
                }
                const auto partner_count = delta(side.first, chans[down[k]]);
                const auto own_count = delta(side.first, chans[live[k]]);
                if (!partner_count || !own_count) {
                    continue;
                }
                const uint64_t on_partner = *partner_count;
                const uint64_t on_own = *own_count;
                const bool odd = (other.chip_id & 1u) != 0;
                log_info(
                    tt::LogTest,
                    "{} toward chip {}: plane {} (live) sent {}, plane {} (its partner) sent {}",
                    landing,
                    other.chip_id,
                    live[k],
                    on_own,
                    down[k],
                    on_partner);
                if (odd) {
                    EXPECT_GT(on_partner, 0u) << landing << " plane " << down[k] << ": nothing scattered onto it";
                    EXPECT_EQ(on_own, 0u) << landing << " plane " << live[k] << ": an odd chip's traffic on it";
                } else {
                    EXPECT_EQ(on_partner, 0u) << landing << " plane " << down[k] << ": an even chip's traffic on it";
                    EXPECT_GT(on_own, 0u) << landing << " plane " << live[k];
                }
                ++checked;
            }
        }
    }
    if (checked == 0) {
        log_warning(tt::LogTest, "no telemetry on this domain's routers toward a scatter partner: nothing checked");
    }
}

TEST(InterDomain, RelayingPlanes) {
    const auto domain = env_u32("TT_INTER_DOMAIN_ID");
    const auto peer_domain = env_u32("TT_INTER_DOMAIN_PEER_ID");
    const char* dir_env = std::getenv("TT_INTER_DOMAIN_DIR");
    if (!domain || !peer_domain || dir_env == nullptr) {
        GTEST_SKIP() << "needs a peer process: see run_inter_domain_sanity.sh";
    }
    if (MetalContext::instance().rtoptions().get_fabric_inter_domain_down_planes().empty()) {
        GTEST_SKIP() << "needs TT_METAL_FABRIC_INTER_DOMAIN_DOWN_PLANES";
    }
    const fs::path dir(dir_env);
    auto mine = [&](const std::string& what) { return dir / ("d" + std::to_string(*domain) + "." + what); };
    auto theirs = [&](const std::string& what) { return dir / ("d" + std::to_string(*peer_domain) + "." + what); };

    SetFabricConfig(FabricConfig::FABRIC_2D);
    auto mesh =
        distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()));
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

    // routed ports, in (asic, chan) order: a stream's slot is its port's index
    std::vector<InterDomainPort> ports;
    std::vector<RoutedPort> mine_routed;
    for (const auto& port : get_inter_domain_ports(*mesh)) {
        if (port.has_router) {
            ports.push_back(port);
        }
    }
    std::sort(ports.begin(), ports.end(), [](const auto& a, const auto& b) {
        return std::pair{get_inter_domain_port_asic(a), a.eth_chan} <
               std::pair{get_inter_domain_port_asic(b), b.eth_chan};
    });
    ASSERT_LE(ports.size() * 3, kRelayMaxSlots);
    ASSERT_TRUE(wait_settled(ports)) << "a router did not reach DRAIN (live) or RELAY (relaying)";
    size_t relaying = 0;
    for (const auto& port : ports) {
        mine_routed.push_back(RoutedPort{
            .asic = get_inter_domain_port_asic(port),
            .chan = port.eth_chan,
            .peer_asic = port.peer_asic,
            .peer_chan = port.peer_chan,
            .plane = port.plane,
            .live = port.link_trained,
            .md = get_inter_domain_metadata(port, *domain)});
        EXPECT_EQ(get_inter_domain_state(port), port.link_trained ? InterDomainState::DRAIN : InterDomainState::RELAY)
            << port.node << " chan " << int(port.eth_chan);
        relaying += !port.link_trained;
    }
    ASSERT_GT(relaying, 0u) << "no relaying router: is a plane down?";
    log_info(tt::LogTest, "domain {}: {} routed ports, {} relaying", *domain, ports.size(), relaying);

    std::map<uint64_t, RxInfo> rx;
    for (const auto& [asic, dev] : device_of) {
        const uint32_t base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
        const auto core = dev->worker_core_from_logical_core(kCore);
        const auto& node = node_of.at(asic);
        rx[asic] = RxInfo{
            static_cast<uint32_t>(core.x),
            static_cast<uint32_t>(core.y),
            base + kRelayDstOffset,
            *node.mesh_id,
            node.chip_id};
    }
    {
        std::ostringstream o;
        for (const auto& p : mine_routed) {
            o << "port " << p.asic << " " << p.chan << " " << p.peer_asic << " " << p.peer_chan << " " << p.plane << " "
              << p.live << " " << p.md.domain_id << " " << p.md.vc2_receiver_channel_addr << " "
              << p.md.vc2_receiver_channel_slots << " " << p.md.max_packet_size_bytes << " " << p.md.layout_fingerprint
              << "\n";
        }
        for (const auto& [asic, r] : rx) {
            o << "rx " << asic << " " << r.x << " " << r.y << " " << r.addr << " " << r.mesh << " " << r.chip << "\n";
        }
        write_file(mine("relay.meta"), o.str());
    }
    wait_file(theirs("relay.meta"));
    std::vector<RoutedPort> peer_routed;
    std::map<uint64_t, RxInfo> peer_rx;
    {
        std::ifstream in(theirs("relay.meta"));
        std::string kind;
        while (in >> kind) {
            if (kind == "port") {
                RoutedPort p;
                in >> p.asic >> p.chan >> p.peer_asic >> p.peer_chan >> p.plane >> p.live >> p.md.domain_id >>
                    p.md.vc2_receiver_channel_addr >> p.md.vc2_receiver_channel_slots >> p.md.max_packet_size_bytes >>
                    p.md.layout_fingerprint;
                peer_routed.push_back(p);
            } else {
                uint64_t asic = 0;
                RxInfo r;
                in >> asic >> r.x >> r.y >> r.addr >> r.mesh >> r.chip;
                peer_rx[asic] = r;
            }
        }
    }

    // every live link, matched to its far end; both ends must agree it is live
    std::vector<std::pair<InterDomainPort, RoutedPort>> cables;
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto& p = mine_routed[i];
        for (const auto& q : peer_routed) {
            if (q.asic == p.peer_asic && q.chan == p.peer_chan && q.peer_asic == p.asic && q.peer_chan == p.chan) {
                ASSERT_EQ(q.live, p.live) << "the two ends of chan " << p.chan << " disagree on whether it is down";
                if (p.live) {
                    cables.emplace_back(ports[i], q);
                }
            }
        }
    }
    ASSERT_FALSE(cables.empty());

    const auto& control_plane = MetalContext::instance().get_control_plane();
    const uint32_t rounds = env_u32("TT_INTER_DOMAIN_ROUNDS").value_or(2);
    // the last round sends only into the peer's other chips (kind 2), so this domain's routers carry only what lands
    // here for its other chips: the scatter check (scenario 2) reads their packet counts around it
    std::optional<std::map<std::pair<uint64_t, uint32_t>, uint64_t>> sent_before_far_round;
    for (uint32_t round = 0; round <= rounds; ++round) {
        const bool far_only = round == rounds;
        auto step = [&](const std::string& what) {
            log_info(tt::LogTest, "domain {} round {}: {}", *domain, round, what);
            write_file(mine(std::to_string(round) + ".relay." + what), "");
            wait_file(theirs(std::to_string(round) + ".relay." + what));
        };
        for (const auto& [asic, dev] : device_of) {
            std::vector<uint32_t> zeros(kRelayMaxSlots * kRelayBytes / 4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(dev, kCore, rx.at(asic).addr, zeros);
        }
        step("ready");
        for (const auto& [port, q] : cables) {
            connect_inter_domain(port, *domain, q.md);
            EXPECT_EQ(get_inter_domain_state(port), InterDomainState::SERVICE_TRAFFIC);
        }
        if (far_only) {
            // before this side says it is connected: the peer sends only after both have
            sent_before_far_round = packets_sent_by_asic(*mesh, node_of);
        }
        step("connected");

        // per port: slot 3i from a worker on its chip into the chip at the port's far end, slot 3i+1 from another chip
        // of the domain on its plane into the same chip, slot 3i+2 from a worker on its chip into another chip of the
        // peer domain (it crosses the peer domain on the plane it lands on, or its scatter partner's)
        // a port's worker channel takes one worker at a time: its far stream (3i+2) goes in a second program, after
        std::map<std::pair<uint32_t, uint64_t>, Program> programs;  // (wave, asic)
        std::vector<Stream> streams;
        for (size_t i = 0; i < ports.size(); ++i) {
            const auto port = ports[i];
            if (!peer_rx.contains(port.peer_asic)) {
                continue;  // its side leads to a chip in neither domain
            }
            const uint64_t exit_asic = mine_routed[i].asic;
            const auto exit = node_of.at(exit_asic);
            auto add = [&](uint64_t asic,
                           uint64_t to_asic,
                           uint32_t slot,
                           std::optional<FabricNodeId> behind,  // over VC0 to this exit chip first, then out the cable
                           uint32_t out_cable,
                           const std::function<void(Program&, const CoreCoord&, std::vector<uint32_t>&)>& connect) {
                if (far_only && slot % 3 != 2) {
                    return;
                }
                const auto& to = peer_rx.at(to_asic);
                IDevice* dev = device_of.at(asic);
                const CoreCoord core{static_cast<uint32_t>(slot % 8), 1 + static_cast<uint32_t>(slot / 8)};
                const uint32_t base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
                std::vector<uint32_t> data(kRelayBytes / 4);
                for (uint32_t w = 0; w < data.size(); ++w) {
                    data[w] = relay_pattern(*domain, slot, round, w);
                }
                tt::tt_metal::detail::WriteToDeviceL1(dev, core, base + kRelaySrcOffset, data);
                auto& program = programs.try_emplace({slot % 3 == 2 ? 1u : 0u, asic}, CreateProgram()).first->second;
                const auto kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_fabric/inter_domain/kernels/inter_domain_sender.cpp",
                    core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .defines = {{"FABRIC_2D", "1"}}});
                std::vector<uint32_t> args = {
                    base + kRelaySrcOffset,
                    base + kHeaderOffset,
                    kRelayPackets,
                    kPacketSize,
                    behind.has_value(),
                    behind ? behind->chip_id : 0,
                    behind ? *behind->mesh_id : 0,
                    out_cable,
                    1,  // one destination
                    to.x,
                    to.y,
                    to.addr + slot * kRelayBytes,
                    to.mesh,
                    to.chip};
                connect(program, core, args);
                SetRuntimeArgs(program, kernel, core, args);
                streams.push_back(Stream{to_asic, slot});
                log_info(
                    tt::LogTest,
                    "domain {}: asic {} sends slot {} through chan {} plane {} ({})",
                    *domain,
                    asic,
                    slot,
                    int(port.eth_chan),
                    port.plane,
                    port.link_trained ? "live" : "relaying");
            };
            const auto through_port = [port](Program& program, const CoreCoord& core, std::vector<uint32_t>& args) {
                append_inter_domain_connection_rt_args(port, program, core, args);
            };
            add(exit_asic, port.peer_asic, 3 * i, std::nullopt, 0, through_port);
            for (const auto& [far, _] : peer_rx) {
                if (far != port.peer_asic) {
                    add(exit_asic, far, 3 * i + 2, std::nullopt, 0, through_port);
                    break;  // one is enough
                }
            }
            for (const auto& [asic, dev] : device_of) {
                const auto from = node_of.at(asic);
                const auto out = control_plane.get_forwarding_direction(from, exit);
                if (asic == exit_asic || !out || control_plane.get_chip_neighbors(from, *out).empty()) {
                    continue;
                }
                add(asic,
                    port.peer_asic,
                    3 * i + 1,
                    exit,
                    1u << control_plane.routing_direction_to_eth_direction(port.direction),
                    [&mesh, from, exit, plane = port.plane](
                        Program& program, const CoreCoord& core, std::vector<uint32_t>& args) {
                        append_fabric_connection_rt_args(*mesh, from, exit, plane, program, core, args);
                    });
                break;  // one chip behind is enough
            }
        }
        {
            std::ostringstream o;
            for (const auto& st : streams) {
                o << st.to << " " << st.slot << "\n";
            }
            write_file(mine(std::to_string(round) + ".relay.streams"), o.str());
        }
        for (auto& [wave_asic, program] : programs) {
            const uint64_t asic = wave_asic.second;
            distributed::MeshWorkload workload;
            workload.add_program(distributed::MeshCoordinateRange(coord_of.at(asic)), std::move(program));
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
        }
        step("sent");

        // what the peer sent here
        std::map<uint64_t, std::vector<uint32_t>> expected;  // my chip -> slots
        {
            std::ifstream in(theirs(std::to_string(round) + ".relay.streams"));
            uint64_t to = 0;
            uint32_t slot = 0;
            while (in >> to >> slot) {
                expected[to].push_back(slot);
            }
        }
        size_t checked = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (const auto& [asic, slots] : expected) {
            ASSERT_TRUE(device_of.contains(asic)) << "the peer sent to asic " << asic << ", not in this domain";
            std::map<uint32_t, uint32_t> bad;
            while (true) {
                std::vector<uint32_t> got;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device_of.at(asic), kCore, rx.at(asic).addr, kRelayMaxSlots * kRelayBytes, got);
                bad.clear();
                for (uint32_t slot : slots) {
                    for (uint32_t w = 0; w < kRelayBytes / 4; ++w) {
                        bad[slot] += got[slot * kRelayBytes / 4 + w] != relay_pattern(*peer_domain, slot, round, w);
                    }
                }
                if (std::all_of(bad.begin(), bad.end(), [](const auto& b) { return b.second == 0; }) ||
                    std::chrono::steady_clock::now() > deadline) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            for (uint32_t slot : slots) {
                EXPECT_EQ(bad[slot], 0u) << "round " << round << " asic " << asic << " slot " << slot << ": "
                                         << bad[slot] << " bad words";
                ++checked;
            }
        }
        if (!far_only) {
            // a worker's stream per peer port into this domain at least, and more where either domain has two chips
            const auto into_here = std::count_if(peer_routed.begin(), peer_routed.end(), [&](const RoutedPort& q) {
                return device_of.contains(q.peer_asic);
            });
            EXPECT_GE(checked, static_cast<size_t>(into_here));
            EXPECT_GT(checked, 0u);
        } else if (sent_before_far_round && device_of.size() > 1) {
            check_scatter(*mesh, node_of, ports, *sent_before_far_round, packets_sent_by_asic(*mesh, node_of));
        }
        step("received");

        for (const auto& [port, q] : cables) {
            disconnect_inter_domain(port);
            EXPECT_EQ(get_inter_domain_state(port), InterDomainState::DRAIN);
        }
        for (const auto& port : ports) {
            if (!port.link_trained) {
                EXPECT_EQ(get_inter_domain_state(port), InterDomainState::RELAY);
            }
        }
        step("done");
    }
    mesh->close();
    SetFabricConfig(FabricConfig::DISABLED);
}

}  // namespace
}  // namespace tt::tt_fabric::inter_domain_test
