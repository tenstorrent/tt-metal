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

#include <fmt/ranges.h>
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
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"

namespace tt::tt_fabric::inter_domain_test {
namespace {

using namespace tt::tt_metal;
using namespace tt::tt_fabric::internal;
namespace fs = std::filesystem;

constexpr uint32_t kNumPackets = 4;  // per pair
constexpr uint32_t kPacketSize = 1024;
constexpr uint32_t kBytes = kNumPackets * kPacketSize;
constexpr uint32_t kHeaderOffset = 0;
constexpr uint32_t kSrcOffset = 4096;  // a sender's data, a pair after another
constexpr uint32_t kDstOffset = 4096;  // on kCore, a slot per sending chip
const CoreCoord kCore{0, 0};           // receives; the senders are on the cores after it

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
    uint32_t mesh = 0, chip = 0, plane = 0;
    uint32_t action = 0;  // its link's direction bit
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
          << p.md.max_packet_size_bytes << " " << p.md.layout_fingerprint << " " << p.mesh << " " << p.chip << " "
          << p.plane << " " << p.action << "\n";
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
                p.md.vc2_receiver_channel_slots >> p.md.max_packet_size_bytes >> p.md.layout_fingerprint >> p.mesh >>
                p.chip >> p.plane >> p.action;
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

// what `from` sends `to`
uint32_t pattern(uint32_t domain, uint64_t from, uint64_t to, uint32_t round, uint32_t i) {
    return (domain << 24) ^ (round << 20) ^ static_cast<uint32_t>(from * 2654435761u) ^
           static_cast<uint32_t>(to * 0x85EBCA77u) ^ (i * 0x9E3779B9u);
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
        try {
            for (const auto& sample : read_fabric_telemetry(mesh, node)) {
                if (sample.snapshot.dynamic_info) {
                    sent[{node, sample.channel_id}] = sample.snapshot.dynamic_info->tx_bandwidth.packets_sent;
                }
            }
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "{}: no telemetry: {}", node, e.what());
        }
    }
    return sent;
}

// Each domain is one process per mesh, or per host of a mesh, of the MGD; processes meet through files in
// TT_INTER_DOMAIN_DIR.
//   - Sanity, AllPairs: every chip of each domain sends to every chip of the other, over a cable by the pair (so every
//     cable carries some): from the exit chip straight onto it, from its mesh VC0 to the exit chip, from another mesh
//     VC0 then VC1 across the meshes; landing on the peer's chip, then VC2 across its domain. A sending chip and cable
//     at a time (Sanity), or all together (AllPairs).
//   - Routes: from each exit chip to every chip of the other domain, all together, then the packets each router sent
//     checked against VC2's routes over the domain's mesh (TT_METAL_FABRIC_TELEMETRY=1).
enum class Mode { Sanity, AllPairs, Routes };
void run(Mode mode) {
    const bool all_at_once = mode != Mode::Sanity;
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
    const auto my_rank = control_plane.get_local_host_rank_id_binding();
    // a process: its domain, mesh and host rank
    using Process = std::tuple<uint32_t, uint32_t, uint32_t>;
    auto file = [&](const Process& p, const std::string& what) {
        const auto& [d, m, r] = p;
        return dir / fmt::format("d{}.m{}.r{}.{}", d, m, r, what);
    };
    auto mine = [&](const std::string& what) { return file({*domain, my_mesh, *my_rank}, what); };
    auto processes = [&](uint32_t d) {
        std::set<Process> all;
        for (MeshId m : meshes) {
            for (const auto& r : mesh_graph.get_host_ranks(m).values()) {
                all.emplace(d, *m, *r);
            }
        }
        return all;
    };
    // every process of domain d: each one's file
    auto gather = [&](uint32_t d, const std::string& what) {
        Side all;
        for (const auto& p : processes(d)) {
            wait_file(file(p, what));
            from_text(file(p, what), all);
        }
        return all;
    };

    std::map<uint64_t, distributed::MeshCoordinate> coord_of;
    std::map<uint64_t, IDevice*> device_of;
    std::map<uint64_t, tt::tt_fabric::FabricNodeId> node_of;
    for (const auto& coord : mesh_graph.get_coord_range(MeshId{my_mesh}, my_rank)) {
        const auto node = mesh->get_fabric_node_id(coord);
        const uint64_t asic = *tt::tt_metal::internal::get_chip_unique_id_from_fabric_node_id(*mesh, node);
        coord_of.emplace(asic, coord);
        device_of[asic] = mesh->get_view().impl().get_device(coord);
        node_of.emplace(asic, node);
        log_info(tt::LogTest, "domain {}: {} is device {}, asic {}", *domain, node, device_of[asic]->id(), asic);
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
            .md = get_inter_domain_metadata(port, *domain),
            .mesh = *port.node.mesh_id,
            .chip = port.node.chip_id,
            .plane = port.plane,
            .action = 1u << control_plane.routing_direction_to_eth_direction(port.direction)});
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

    // every cable of this domain (the lower domain's lowest channel where chips have several), and the port where
    // this process has it
    struct Cable {
        PortInfo exit;
        PortInfo peer;
        std::optional<InterDomainPort> port;
        uint32_t rank = 0;
    };
    std::map<std::pair<uint64_t, uint64_t>, Cable> cables;
    for (const auto& p : mine_all.ports) {
        for (const auto& q : peer.ports) {
            if (q.asic != p.peer_asic || q.chan != p.peer_chan || q.peer_asic != p.asic || q.peer_chan != p.chan) {
                continue;
            }
            const bool lower = *domain < *peer_domain;
            const uint32_t rank = lower ? p.chan : q.chan;
            auto it = cables.find({p.asic, q.asic});
            if (it == cables.end() || rank < it->second.rank) {
                cables[{p.asic, q.asic}] = Cable{p, q, std::nullopt, rank};
            }
        }
    }
    for (auto& [_, cable] : cables) {
        for (size_t i = 0; i < ports.size(); ++i) {
            if (me.ports[i].asic == cable.exit.asic && me.ports[i].chan == cable.exit.chan) {
                cable.port = ports[i];
            }
        }
    }
    ASSERT_FALSE(cables.empty()) << "no cable to domain " << *peer_domain;
    const bool check_routes =
        mode == Mode::Routes && MetalContext::instance().rtoptions().get_enable_fabric_telemetry();

    const uint32_t region = static_cast<uint32_t>(peer.rx.size()) * kBytes;  // a slot per peer chip
    const uint32_t rounds = env_u32("TT_INTER_DOMAIN_ROUNDS").value_or(2);
    for (uint32_t round = 0; round < rounds; ++round) {
        // both domains' processes: a mesh can send over another's cable
        auto step = [&](const std::string& what) {
            log_info(tt::LogTest, "domain {} mesh {} round {}: {}", *domain, my_mesh, round, what);
            write_file(mine(std::to_string(round) + "." + what), "");
            for (uint32_t d : {*domain, *peer_domain}) {
                for (const auto& p : processes(d)) {
                    wait_file(file(p, std::to_string(round) + "." + what));
                }
            }
        };
        for (const auto& [asic, dev] : device_of) {
            std::vector<uint32_t> zeros(region / 4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(dev, kCore, me.rx.at(asic).addr, zeros);
        }
        step("ready");

        for (const auto& [_, cable] : cables) {
            if (!cable.port) {
                continue;
            }
            log_info(
                tt::LogTest,
                "domain {}: cable {} chan {} {} plane {} to asic {} chan {}",
                *domain,
                cable.port->node,
                cable.port->eth_chan,
                cable.port->direction,
                cable.port->plane,
                cable.peer.asic,
                cable.peer.chan);
            connect_inter_domain(*cable.port, *domain, cable.peer.md);
            EXPECT_EQ(get_inter_domain_state(*cable.port), InterDomainState::SERVICE_TRAFFIC);
        }
        // a cable router's worker credits and first stream registers (its receivers', then credits): idle now, and
        // again once its traffic has drained
        const auto at_rest = [&](const InterDomainPort& port) {
            const auto& cluster = MetalContext::instance().get_cluster();
            const auto it =
                std::find_if(node_of.begin(), node_of.end(), [&](const auto& kv) { return kv.second == port.node; });
            const auto chip = device_of.at(it->first)->id();
            std::vector<uint32_t> counts;
            for (uint32_t stream :
                 {tt::tt_fabric::connection_interface::sender_channel_0_free_slots_stream_id, 4u, 5u, 6u}) {
                uint32_t value = 0;
                cluster.read_reg(
                    &value,
                    tt_cxy_pair(chip, cluster.get_virtual_eth_core_from_channel(chip, port.eth_chan)),
                    0xFFB40000 + 0x1000 * stream + 4 * 297);
                counts.push_back(value & ((1u << 17) - 1));
            }
            return counts;
        };
        std::map<std::pair<uint64_t, uint64_t>, std::vector<uint32_t>> idle;
        for (const auto& [key, cable] : cables) {
            if (cable.port) {
                idle[key] = at_rest(*cable.port);
            }
        }
        const auto before = check_routes ? packets_sent(*mesh, node_of) : decltype(packets_sent(*mesh, node_of)){};
        step("connected");

        // one worker per sending chip and cable, sending to each of `to` in turn
        struct Send {
            uint64_t from;
            const Cable* cable;
            std::vector<uint64_t> to;
        };
        std::vector<Send> sends;
        std::vector<const Cable*> by_index;
        for (const auto& [_, cable] : cables) {
            by_index.push_back(&cable);
        }
        if (mode == Mode::Routes) {
            for (const auto& [_, cable] : cables) {
                if (cable.port) {
                    Send send{cable.exit.asic, &cable, {}};
                    for (const auto& [to, _] : peer.rx) {
                        send.to.push_back(to);
                    }
                    sends.push_back(send);
                }
            }
        } else if (!by_index.empty()) {
            for (const auto& [from, _] : device_of) {
                std::map<const Cable*, Send> by_cable;
                uint32_t d = 0;
                for (const auto& [to, _] : peer.rx) {
                    const Cable* cable = by_index[(slot_of(mine_all, from) + d++) % by_index.size()];
                    by_cable.try_emplace(cable, Send{from, cable, {}}).first->second.to.push_back(to);
                }
                for (auto& [_, send] : by_cable) {
                    sends.push_back(std::move(send));
                }
            }
        }
        Side sent;
        std::map<uint64_t, Program> programs;  // all at once: every chip's sends in one program, a core each
        std::map<uint64_t, uint32_t> cores_used;
        for (auto& send : sends) {
            IDevice* dev = device_of.at(send.from);
            const uint32_t base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
            const uint32_t k = 1 + (all_at_once ? cores_used[send.from]++ : 0);
            const auto grid = dev->compute_with_storage_grid_size();
            const CoreCoord core{k % grid.x, k / grid.x};
            std::vector<uint32_t> data;
            for (uint64_t to : send.to) {
                for (uint32_t i = 0; i < kBytes / 4; ++i) {
                    data.push_back(pattern(*domain, send.from, to, round, i));
                }
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
            const auto& exit = send.cable->exit;
            // onto the cable here; to the exit chip, on this mesh; or toward the exit's mesh
            const uint32_t route = send.from == exit.asic ? 0 : exit.mesh == my_mesh ? 1 : 2;
            std::vector<uint32_t> args = {
                base + kSrcOffset,
                base + kHeaderOffset,
                kNumPackets,
                kPacketSize,
                route,
                exit.chip,
                exit.mesh,
                exit.action};
            args.push_back(static_cast<uint32_t>(send.to.size()));
            for (uint64_t to : send.to) {
                const auto& rx = peer.rx.at(to);
                args.insert(
                    args.end(), {rx.x, rx.y, rx.addr + slot_of(mine_all, send.from) * kBytes, rx.mesh, rx.chip});
                sent.sent.emplace(send.from, send.cable->peer.asic, to);
            }
            if (route == 0) {
                append_inter_domain_connection_rt_args(*send.cable->port, program, core, args);
            } else {
                append_fabric_connection_rt_args(
                    *mesh,
                    node_of.at(send.from),
                    FabricNodeId(MeshId{exit.mesh}, exit.chip),
                    exit.plane,
                    program,
                    core,
                    args);
            }
            SetRuntimeArgs(program, kernel, core, args);
            if (!all_at_once) {
                log_info(
                    tt::LogTest,
                    "domain {}: {} sends {} via {} ({})",
                    *domain,
                    send.from,
                    send.to.size(),
                    send.cable->peer.asic,
                    route == 0   ? "on the exit chip"
                    : route == 1 ? "this mesh"
                                 : "another mesh");
                distributed::MeshWorkload workload;
                workload.add_program(distributed::MeshCoordinateRange(coord_of.at(send.from)), std::move(single));
                distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
            }
        }
        if (all_at_once && !programs.empty()) {
            distributed::MeshWorkload workload;
            for (auto& [asic, program] : programs) {
                workload.add_program(distributed::MeshCoordinateRange(coord_of.at(asic)), std::move(program));
            }
            // not waited on until the checks: where a send stalls, the routers' counts show the link
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
        }
        log_info(tt::LogTest, "domain {} mesh {} round {}: {} sends", *domain, my_mesh, round, sent.sent.size());
        write_file(mine(std::to_string(round) + ".sent"), to_text(sent));
        const Side expected = gather(*peer_domain, std::to_string(round) + ".sent");

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(env_u32("TT_INTER_DOMAIN_DEADLINE_S").value_or(30));
        uint32_t checked = 0;
        for (const auto& [from, via, to] : expected.sent) {
            if (!device_of.contains(to)) {
                continue;
            }
            uint32_t wrong = 0;
            while (true) {
                std::vector<uint32_t> got;
                try {
                    tt::tt_metal::detail::ReadFromDeviceL1(
                        device_of.at(to), kCore, me.rx.at(to).addr + slot_of(peer, from) * kBytes, kBytes, got);
                } catch (const std::exception& e) {
                    log_error(tt::LogTest, "{}: unreadable: {}", node_of.at(to), e.what());
                    got.assign(kBytes / 4, 0);
                }
                wrong = 0;
                for (uint32_t i = 0; i < kBytes / 4; ++i) {
                    wrong += got[i] != pattern(*peer_domain, from, to, round, i);
                }
                if (wrong == 0) {
                    break;
                }
                if (std::chrono::steady_clock::now() > deadline) {
                    // each wrong packet: stale (zero or last round's), or whose data it holds
                    std::map<std::string, uint32_t> kinds;
                    for (uint32_t p = 0; p < kBytes / 4; p += 256) {
                        if (got[p] == pattern(*peer_domain, from, to, round, p)) {
                            continue;
                        }
                        std::string kind = "other";
                        if (got[p] == 0 || (round > 0 && got[p] == pattern(*peer_domain, from, to, round - 1, p))) {
                            kind = "stale";
                        }
                        for (const auto& [asic, _] : peer.rx) {
                            for (const auto& [dst, _] : mine_all.rx) {
                                for (uint32_t j = 0; j < kBytes / 4; j += 256) {
                                    if (got[p] == pattern(*peer_domain, asic, dst, round, j)) {
                                        kind = fmt::format("asic {} to asic {} packet {}", asic, dst, j / 256);
                                    }
                                }
                            }
                        }
                        kinds[fmt::format("packet {}: {}", p / 256, kind)]++;
                    }
                    for (const auto& [kind, n] : kinds) {
                        log_error(tt::LogTest, "chip {} from asic {}: {}", node_of.at(to), from, kind);
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ++checked;
            EXPECT_EQ(wrong, 0u) << "round " << round << ": chip " << to << " " << node_of.at(to) << " from asic "
                                 << from << " landing on " << via << ": " << wrong << " of " << kBytes / 4
                                 << " words wrong";
        }
        log_info(tt::LogTest, "domain {} mesh {} round {}: checked {} sends", *domain, my_mesh, round, checked);
        if (mode != Mode::Routes) {
            EXPECT_EQ(checked, device_of.size() * peer.rx.size())
                << "round " << round << ": every peer chip to each here";
        }

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
        if (all_at_once) {
            distributed::Finish(mesh->mesh_command_queue());
        }
        step("received");

        // no error handling yet: each cable's router drains, its worker credited back and nothing left to land, first
        for (const auto& [key, cable] : cables) {
            if (!cable.port) {
                continue;
            }
            while (at_rest(*cable.port) != idle.at(key)) {
                ASSERT_LT(std::chrono::steady_clock::now(), deadline)
                    << "cable " << cable.port->node << " never drained: ["
                    << fmt::format("{}", fmt::join(at_rest(*cable.port), ",")) << "], at rest ["
                    << fmt::format("{}", fmt::join(idle.at(key), ",")) << "]";
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        step("drained");

        for (const auto& [_, cable] : cables) {
            if (cable.port) {
                disconnect_inter_domain(*cable.port);
                EXPECT_EQ(get_inter_domain_state(*cable.port), InterDomainState::DRAIN);
            }
        }
        step("done");
    }
    mesh->close();
    SetFabricConfig(FabricConfig::DISABLED);
}

TEST(InterDomain, Sanity) { run(Mode::Sanity); }

TEST(InterDomain, AllPairs) { run(Mode::AllPairs); }

TEST(InterDomain, AllAtOnce) { run(Mode::Routes); }

}  // namespace
}  // namespace tt::tt_fabric::inter_domain_test
