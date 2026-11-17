// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Inter-domain sanity: one process per fabric domain. Each side opens its chips with
// TT_METAL_FABRIC_INTER_DOMAIN=1, swaps router metadata with the peer domain through files, connects
// one cable per linked chip pair, sends a buffer across each, checks what the peer sent, and
// disconnects. run_inter_domain_sanity.sh starts both sides.
//
// Env: TT_INTER_DOMAIN_ID (this domain), TT_INTER_DOMAIN_PEER_ID, TT_INTER_DOMAIN_DIR (shared dir).

#include <gtest/gtest.h>

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
#include <tt-metalium/experimental/fabric/inter_domain.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

namespace tt::tt_fabric::inter_domain_test {
namespace {

using namespace tt::tt_metal;
using namespace tt::tt_fabric::experimental;
namespace fs = std::filesystem;

constexpr uint32_t kNumPackets = 64;
constexpr uint32_t kPacketSize = 1024;
constexpr uint32_t kBytes = kNumPackets * kPacketSize;
constexpr uint32_t kHeaderOffset = 0;
constexpr uint32_t kSrcOffset = 4096;
constexpr uint32_t kDstOffset = kSrcOffset + kBytes;  // kSlots receive slots
const CoreCoord kCore{0, 0};
const CoreCoord kBehindCore{1, 0};
const CoreCoord kFarCore{2, 0};
constexpr uint32_t kSlots = 3;

std::optional<uint32_t> env_u32(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::optional<uint32_t>(std::stoul(v)) : std::nullopt;
}

// what one side publishes: its routers' metadata and where each chip takes the peer's data
struct PortInfo {
    uint64_t asic = 0;
    uint32_t chan = 0;
    uint64_t peer_asic = 0;
    uint32_t peer_chan = 0;
    InterDomainMetadata md;
};
struct RxInfo {
    uint32_t x = 0, y = 0, addr = 0;
    uint32_t mesh = 0, chip = 0;  // the chip in its own domain
};
struct Side {
    std::vector<PortInfo> ports;
    std::map<uint64_t, RxInfo> rx;  // by asic
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
          << p.md.max_packet_size_bytes << "\n";
    }
    for (const auto& [asic, r] : s.rx) {
        o << "rx " << asic << " " << r.x << " " << r.y << " " << r.addr << " " << r.mesh << " " << r.chip << "\n";
    }
    return o.str();
}

Side from_text(const fs::path& path) {
    Side s;
    std::ifstream in(path);
    std::string kind;
    while (in >> kind) {
        if (kind == "port") {
            PortInfo p;
            in >> p.asic >> p.chan >> p.peer_asic >> p.peer_chan >> p.md.domain_id >> p.md.vc2_receiver_channel_addr >>
                p.md.vc2_receiver_channel_slots >> p.md.max_packet_size_bytes;
            s.ports.push_back(p);
        } else {
            uint64_t asic = 0;
            RxInfo r;
            in >> asic >> r.x >> r.y >> r.addr >> r.mesh >> r.chip;
            s.rx[asic] = r;
        }
    }
    return s;
}

uint32_t pattern(uint32_t domain, uint64_t asic, uint32_t round, uint32_t i) {
    return (domain << 24) ^ (round << 20) ^ static_cast<uint32_t>(asic * 2654435761u) ^ (i * 0x9E3779B9u);
}

TEST(InterDomain, Sanity) {
    const auto domain = env_u32("TT_INTER_DOMAIN_ID");
    const auto peer_domain = env_u32("TT_INTER_DOMAIN_PEER_ID");
    const char* dir_env = std::getenv("TT_INTER_DOMAIN_DIR");
    if (!domain || !peer_domain || dir_env == nullptr) {
        GTEST_SKIP() << "needs a peer process: see run_inter_domain_sanity.sh";
    }
    const fs::path dir(dir_env);
    auto mine = [&](const std::string& what) { return dir / ("d" + std::to_string(*domain) + "." + what); };
    auto theirs = [&](const std::string& what) { return dir / ("d" + std::to_string(*peer_domain) + "." + what); };

    SetFabricConfig(FabricConfig::FABRIC_2D);
    auto mesh =
        distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()));

    // every chip in this mesh, by asic
    std::map<uint64_t, distributed::MeshCoordinate> coord_of;
    std::map<uint64_t, IDevice*> device_of;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh->shape())) {
        IDevice* dev = mesh->get_view().impl().get_device(coord);
        const uint64_t asic = MetalContext::instance().get_cluster().get_unique_chip_ids().at(dev->id());
        coord_of.emplace(asic, coord);
        device_of[asic] = dev;
    }

    // publish ours: metadata per port, a receive buffer per chip
    Side me;
    std::vector<InterDomainPort> ports;
    for (const auto& port : get_inter_domain_ports()) {
        if (port.has_router) {
            ports.push_back(port);
        }
    }
    ASSERT_FALSE(ports.empty()) << "no inter-domain routers";
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
        const auto node =
            MetalContext::instance().get_control_plane().get_fabric_node_id_from_physical_chip_id(dev->id());
        me.rx[asic] = RxInfo{
            static_cast<uint32_t>(core.x),
            static_cast<uint32_t>(core.y),
            base + kDstOffset,
            *node.mesh_id,
            node.chip_id};
    }
    write_file(mine("meta"), to_text(me));
    wait_file(theirs("meta"));
    const Side peer = from_text(theirs("meta"));

    // one cable per linked chip pair, picked the same way on both sides: lowest chan on the lower domain's end
    struct Cable {
        InterDomainPort port;
        PortInfo peer;
        uint32_t rank = 0;
    };
    std::map<std::pair<uint64_t, uint64_t>, Cable> cables;  // (our asic, their asic)
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto& p = me.ports[i];
        for (const auto& q : peer.ports) {
            if (q.asic != p.peer_asic || q.chan != p.peer_chan || q.peer_asic != p.asic || q.peer_chan != p.chan) {
                continue;
            }
            const uint32_t rank = *domain < *peer_domain ? p.chan : q.chan;
            auto it = cables.find({p.asic, q.asic});
            if (it == cables.end() || rank < it->second.rank) {
                cables[{p.asic, q.asic}] = Cable{ports[i], q, rank};
            }
        }
    }
    ASSERT_FALSE(cables.empty()) << "no cable to domain " << *peer_domain;

    // Rounds of connect, traffic, disconnect: the second and later rounds reconnect the same routers.
    const uint32_t rounds = env_u32("TT_INTER_DOMAIN_ROUNDS").value_or(2);
    for (uint32_t round = 0; round < rounds; ++round) {
        auto step = [&](const std::string& what) {
            write_file(mine(std::to_string(round) + "." + what), "");
            wait_file(theirs(std::to_string(round) + "." + what));
        };
        // slot 0: from across the cable; 1: from behind the chip across it; 2: through the chip across from another
        for (const auto& [asic, dev] : device_of) {
            std::vector<uint32_t> zeros(kSlots * kBytes / 4, 0);
            detail::WriteToDeviceL1(dev, kCore, me.rx.at(asic).addr, zeros);
        }
        step("ready");

        for (const auto& [chips, cable] : cables) {
            const auto& port = cable.port;
            const auto& q = cable.peer;
            log_info(
                tt::LogTest,
                "domain {} round {}: connecting {} chan {} to asic {} chan {}",
                *domain,
                round,
                port.node,
                port.eth_chan,
                q.asic,
                q.chan);
            connect_inter_domain(port, *domain, q.md);
            EXPECT_EQ(get_inter_domain_state(port), InterDomainState::SERVICE_TRAFFIC);
        }
        step("connected");

        // Each chip with a cable sends straight across it (into slot 0 there), and across it to every other chip of the
        // peer domain (slot 2: the peer's inter-domain router lands it and forwards it on VC2). Each other chip of this
        // domain next to it sends through it (slot 1): VC0 to the exit chip, whose route leaves by the cable's
        // inter-domain router.
        const auto& control_plane = MetalContext::instance().get_control_plane();
        const auto& mesh_graph = control_plane.get_mesh_graph();
        auto node_of = [&](uint64_t asic) {
            return control_plane.get_fabric_node_id_from_physical_chip_id(device_of.at(asic)->id());
        };
        struct Send {
            CoreCoord core;
            uint64_t to;
            uint32_t slot;
            std::vector<uint8_t> route;
            std::function<void(Program&, std::vector<uint32_t>&)> connect;
        };
        std::map<uint64_t, std::vector<Send>> sends;  // by sending chip
        for (const auto& [chips, cable] : cables) {
            const auto port = cable.port;
            sends[chips.first].push_back(
                Send{kCore, chips.second, 0, {}, [port](Program& program, std::vector<uint32_t>& args) {
                         append_inter_domain_connection_rt_args(port, program, kCore, args);
                     }});
            for (const auto& [far, rx] : peer.rx) {
                if (far != chips.second) {
                    sends[chips.first].push_back(
                        Send{kFarCore, far, 2, {}, [port](Program& program, std::vector<uint32_t>& args) {
                                 append_inter_domain_connection_rt_args(port, program, kFarCore, args);
                             }});
                }
            }
            const auto exit = node_of(chips.first);
            for (const auto& [asic, dev] : device_of) {
                const auto from = node_of(asic);
                const auto out = control_plane.get_forwarding_direction(from, exit);
                if (asic == chips.first || !out || control_plane.get_chip_neighbors(from, *out).empty()) {
                    continue;
                }
                // the exit chip's ingress router reads one byte: X if it faces E/W, else Y
                const auto shape = mesh_graph.get_mesh_shape(exit.mesh_id);
                const auto at = mesh_graph.chip_to_coordinate(exit.mesh_id, exit.chip_id);
                const auto ingress = get_opposite_direction(*out);
                std::vector<uint8_t> route(shape[0] + shape[1], 0);
                const bool x_axis = ingress == RoutingDirection::E || ingress == RoutingDirection::W;
                route[x_axis ? shape[0] + at[1] : at[0]] =
                    1u << control_plane.routing_direction_to_eth_direction(port.direction);
                sends[asic].push_back(Send{
                    kBehindCore,
                    chips.second,
                    1,
                    route,
                    [&mesh, from, exit](Program& program, std::vector<uint32_t>& args) {
                        append_fabric_connection_rt_args(*mesh, from, exit, 0, program, kBehindCore, args);
                    }});
            }
        }
        for (auto& [asic, chip_sends] : sends) {
            IDevice* dev = device_of.at(asic);
            const uint32_t base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
            std::vector<uint32_t> data(kBytes / 4);
            for (uint32_t i = 0; i < data.size(); ++i) {
                data[i] = pattern(*domain, asic, round, i);
            }
            Program program = CreateProgram();
            for (auto& send : chip_sends) {
                detail::WriteToDeviceL1(dev, send.core, base + kSrcOffset, data);
                const auto kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_fabric/inter_domain/kernels/inter_domain_sender.cpp",
                    send.core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .defines = {{"FABRIC_2D", "1"}}});
                const auto& rx = peer.rx.at(send.to);
                std::vector<uint32_t> args = {
                    rx.x,
                    rx.y,
                    rx.addr + send.slot * kBytes,
                    base + kSrcOffset,
                    base + kHeaderOffset,
                    kNumPackets,
                    kPacketSize,
                    rx.mesh,
                    rx.chip,
                    static_cast<uint32_t>(send.route.size())};
                for (size_t b = 0; b < send.route.size(); b += 4) {
                    uint32_t word = 0;
                    for (size_t k = 0; k < 4 && b + k < send.route.size(); ++k) {
                        word |= uint32_t{send.route[b + k]} << (8 * k);
                    }
                    args.push_back(word);
                }
                send.connect(program, args);
                SetRuntimeArgs(program, kernel, send.core, args);
                log_info(
                    tt::LogTest,
                    "domain {}: asic {} core {} sends to asic {} slot {}",
                    *domain,
                    asic,
                    send.core.str(),
                    send.to,
                    send.slot);
            }
            distributed::MeshWorkload workload;
            workload.add_program(distributed::MeshCoordinateRange(coord_of.at(asic)), std::move(program));
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
        }
        step("sent");

        // what the peer sent each of our chips with a cable: slot 0 from across it, slots 1 and 2 from the chip behind
        // that
        for (const auto& [chips, cable] : cables) {
            IDevice* dev = device_of.at(chips.first);
            std::vector<uint32_t> got;
            detail::ReadFromDeviceL1(dev, kCore, me.rx.at(chips.first).addr, kSlots * kBytes, got);
            std::vector<std::pair<uint32_t, uint64_t>> expected = {{0, chips.second}};
            for (const auto& [asic, rx] : peer.rx) {
                if (asic != chips.second) {
                    expected.push_back({1, asic});
                    expected.push_back({2, asic});
                }
            }
            for (const auto& [slot, from] : expected) {
                uint32_t bad = 0;
                for (uint32_t i = 0; i < kBytes / 4; ++i) {
                    bad += got[slot * kBytes / 4 + i] != pattern(*peer_domain, from, round, i);
                }
                EXPECT_EQ(bad, 0u) << "round " << round << " chip " << chips.first << " slot " << slot << ": " << bad
                                   << " bad words from asic " << from;
            }
        }

        for (const auto& [chips, cable] : cables) {
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
