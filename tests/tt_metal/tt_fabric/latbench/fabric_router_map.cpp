// Dumps, for one fabric config, which ethernet router each (src, dst, link) uses and where every
// worker and ethernet core sits in NOC0 physical coordinates, so a YAML generator can place each
// sender on the worker nearest its router and pick the shorter NOC.
//
// Usage: fabric_router_map <2d|ne> <out.json>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <optional>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"

using namespace tt::tt_fabric;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;

static const char* dir_str(RoutingDirection d) {
    switch (d) {
        case RoutingDirection::N: return "N";
        case RoutingDirection::E: return "E";
        case RoutingDirection::S: return "S";
        case RoutingDirection::W: return "W";
        case RoutingDirection::Z: return "Z";
        case RoutingDirection::C: return "C";
        default: return "NONE";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <2d|ne> <out.json>\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const std::string out_path = argv[2];
    FabricConfig fabric_config = FabricConfig::FABRIC_2D;
    if (mode == "ne") {
        fabric_config = FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE;
    } else if (mode != "2d") {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }

    SetFabricConfig(fabric_config);
    auto& ctx = tt::tt_metal::MetalContext::instance();
    ctx.initialize_control_plane();
    auto& cp = ctx.get_control_plane();
    const auto user_meshes = cp.get_user_physical_mesh_ids();
    if (user_meshes.size() != 1) {
        std::fprintf(stderr, "expected one user mesh, got %zu\n", user_meshes.size());
        return 1;
    }
    const MeshId mesh_id = user_meshes[0];
    const auto mesh_shape = cp.get_physical_mesh_shape(mesh_id, MeshScope::GLOBAL);
    const auto& mesh_graph = cp.get_mesh_graph();

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(mesh_shape));
    const auto& cluster = ctx.get_cluster();

    const uint32_t rows = mesh_shape[0];
    const uint32_t cols = mesh_shape[1];
    std::vector<FabricNodeId> nodes;
    std::vector<MeshCoordinate> coords;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            MeshCoordinate coord(r, c);
            coords.push_back(coord);
            nodes.emplace_back(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, coord));
        }
    }

    std::ofstream out(out_path);
    out << "{\n\"fabric\": \"" << mode << "\",\n";
    out << "\"mesh_shape\": [" << rows << ", " << cols << "],\n";
    out << "\"nodes\": [\n";
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        const auto phys_chip = cp.get_physical_chip_id_from_fabric_node_id(node);
        const auto& soc = cluster.get_soc_desc(phys_chip);
        auto* device = mesh_device->get_device(coords[i][0], coords[i][1]);
        const auto grid = device->compute_with_storage_grid_size();
        out << (i ? ",\n" : "") << "{\"chip\": " << node.chip_id << ", \"coord\": [" << coords[i][0] << ", "
            << coords[i][1] << "], \"phys_chip\": " << phys_chip << ", \"aiclk_mhz\": " << cluster.get_device_aiclk(phys_chip)
            << ", \"noc_grid\": [" << soc.grid_size.x << ", " << soc.grid_size.y << "],\n";
        out << " \"workers\": [";
        bool first = true;
        for (uint32_t y = 0; y < grid.y; ++y) {
            for (uint32_t x = 0; x < grid.x; ++x) {
                tt::tt_metal::CoreCoord logical(x, y);
                auto phys = soc.get_physical_tensix_core_from_logical(logical);
                auto virt = device->worker_core_from_logical_core(logical);
                out << (first ? "" : ", ") << "[" << x << ", " << y << ", " << phys.x << ", " << phys.y << ", " << virt.x
                    << ", " << virt.y << "]";
                first = false;
            }
        }
        out << "],\n \"eth\": {";
        first = true;
        for (const auto& [chan, dir] : cp.get_active_fabric_eth_channels(node)) {
            auto phys = soc.get_eth_core_for_channel(chan, tt::CoordSystem::NOC0);
            auto virt = cluster.get_virtual_eth_core_from_channel(phys_chip, chan);
            out << (first ? "" : ", ") << "\"" << static_cast<int>(chan) << "\": {\"phys\": [" << phys.x << ", " << phys.y
                << "], \"virt\": [" << virt.x << ", " << virt.y << "], \"dir\": \"" << static_cast<int>(dir) << "\"}";
            first = false;
        }
        out << "},\n \"dirs\": {";
        first = true;
        for (auto d : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
            auto chans = cp.get_active_fabric_eth_channels_in_direction(node, d);
            if (chans.empty()) {
                continue;
            }
            out << (first ? "" : ", ") << "\"" << dir_str(d) << "\": [";
            for (size_t k = 0; k < chans.size(); ++k) {
                auto peer = cp.get_connected_mesh_chip_chan_ids(node, chans[k]);
                out << (k ? ", " : "") << "{\"link_idx\": " << k << ", \"chan\": " << static_cast<int>(chans[k])
                    << ", \"peer_chip\": " << peer.first.chip_id << ", \"peer_chan\": " << static_cast<int>(peer.second)
                    << "}";
            }
            out << "]";
            first = false;
        }
        out << "}}";
    }
    out << "\n],\n\"routes\": [\n";
    bool first_route = true;
    for (const auto& src : nodes) {
        for (const auto& dst : nodes) {
            if (src == dst) {
                continue;
            }
            // Walk the route hop by hop. Each hop records the full link set in that direction and which of
            // those links forward to dst, plus the channel the next chip receives on.
            std::vector<std::string> hops;
            FabricNodeId cur = src;
            bool ok = true;
            for (int guard = 0; cur != dst && guard < 32; ++guard) {
                auto dir = cp.get_forwarding_direction(cur, dst);
                if (!dir.has_value() || dir.value() == RoutingDirection::NONE || dir.value() == RoutingDirection::C) {
                    ok = false;
                    break;
                }
                auto chans = cp.get_active_fabric_eth_channels_in_direction(cur, dir.value());
                std::vector<uint32_t> links;
                try {
                    links = get_forwarding_link_indices_in_direction(cp, cur, dst, dir.value());
                } catch (...) {
                    ok = false;
                    break;
                }
                if (links.empty() || chans.empty()) {
                    ok = false;
                    break;
                }
                std::string hop = "{\"node\": " + std::to_string(cur.chip_id) + ", \"dir\": \"" + dir_str(dir.value()) +
                                  "\", \"links\": [";
                FabricNodeId next = cur;
                for (size_t k = 0; k < links.size(); ++k) {
                    auto peer = cp.get_connected_mesh_chip_chan_ids(cur, chans[links[k]]);
                    if (k == 0) {
                        next = peer.first;
                    }
                    hop += (k ? ", " : "") + std::string("{\"link_idx\": ") + std::to_string(links[k]) +
                           ", \"chan\": " + std::to_string(static_cast<int>(chans[links[k]])) +
                           ", \"next_chip\": " + std::to_string(peer.first.chip_id) +
                           ", \"next_chan\": " + std::to_string(static_cast<int>(peer.second)) + "}";
                }
                hop += "]}";
                hops.push_back(hop);
                if (next == cur) {
                    ok = false;
                    break;
                }
                cur = next;
            }
            if (!ok || cur != dst) {
                continue;
            }
            out << (first_route ? "" : ",\n") << "{\"src\": " << src.chip_id << ", \"dst\": " << dst.chip_id
                << ", \"hops\": " << hops.size() << ", \"path\": [";
            for (size_t h = 0; h < hops.size(); ++h) {
                out << (h ? ", " : "") << hops[h];
            }
            out << "]}";
            first_route = false;
        }
    }
    out << "\n]\n}\n";
    out.close();

    mesh_device->close();
    SetFabricConfig(FabricConfig::DISABLED);
    std::printf("wrote %s\n", out_path.c_str());
    return 0;
}
