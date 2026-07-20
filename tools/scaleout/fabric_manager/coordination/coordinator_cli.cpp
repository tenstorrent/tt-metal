// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/fabric_manager/coordination/coordinator_cli.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cstddef>
#include <functional>
#include <map>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/system_coordinator.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/cluster_descriptor.hpp>
#include <enchantum/enchantum.hpp>

#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "tt_metal/llrt/tt_target_device.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include "tools/scaleout/fabric_manager/coordination/controller.hpp"
#include "tools/scaleout/fabric_manager/coordination/in_process_transport.hpp"
#include "tools/scaleout/fabric_manager/coordination/service_coordinator.hpp"
#include "tools/scaleout/fabric_manager/coordination/tcp_transport.hpp"

namespace tt::scaleout_tools::fabric_manager {

namespace {

using tt::tt_fabric::MeshId;
using Scope = tt::tt_fabric::coordination::Scope;
using Bytes = tt::tt_fabric::coordination::Bytes;

int require_int_option(const std::vector<std::string>& args, const std::string& name) {
    TT_FATAL(test_args::has_command_option(args, name), "Missing required option {}", name);
    return std::stoi(test_args::get_command_option(args, name));
}

// Parses "meshid:index/count[,meshid:index/count...]" into the agent's mesh membership.
void parse_mesh_membership(const std::string& spec, AgentIdentity& identity) {
    std::size_t pos = 0;
    while (pos < spec.size()) {
        auto comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        auto colon = tok.find(':');
        auto slash = tok.find('/');
        TT_FATAL(
            colon != std::string::npos && slash != std::string::npos && slash > colon,
            "Invalid --mesh-membership token '{}'. Expected meshid:index/count",
            tok);
        auto mesh_id = static_cast<uint32_t>(std::stoul(tok.substr(0, colon)));
        int index = std::stoi(tok.substr(colon + 1, slash - colon - 1));
        int count = std::stoi(tok.substr(slash + 1));
        identity.mesh_membership[mesh_id] = MeshMembership{index, count};
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
}

// Splits "host:port" on the last ':'.
std::pair<std::string, uint16_t> parse_endpoint(const std::string& endpoint) {
    auto colon = endpoint.rfind(':');
    TT_FATAL(colon != std::string::npos, "Invalid --controller endpoint '{}'. Expected host:port", endpoint);
    std::string host = endpoint.substr(0, colon);
    auto port = static_cast<uint16_t>(std::stoi(endpoint.substr(colon + 1)));
    return {host, port};
}

// Builds an agent's no-MPI ServiceCoordinator from the shared agent CLI args:
//   --world-index I --world-size N --controller HOST:PORT [--mesh-membership meshid:index/count,...]
// If --mesh-membership is omitted, a default single-mesh (mesh 0) membership mirroring the world
// identity is synthesized so mesh-scoped collectives work out of the box for the common case.
// Connects to the controller (which must already be listening) before returning.
std::shared_ptr<ServiceCoordinator> build_agent_coordinator(const std::vector<std::string>& args) {
    AgentIdentity identity;
    identity.world_index = require_int_option(args, "--world-index");
    identity.world_size = require_int_option(args, "--world-size");
    if (test_args::has_command_option(args, "--mesh-membership")) {
        parse_mesh_membership(test_args::get_command_option(args, "--mesh-membership"), identity);
    } else {
        identity.mesh_membership[0] = MeshMembership{identity.world_index, identity.world_size};
    }

    TT_FATAL(test_args::has_command_option(args, "--controller"), "agent role requires --controller <host:port>");
    auto [host, port] = parse_endpoint(test_args::get_command_option(args, "--controller"));

    auto transport = std::make_shared<TcpTransport>(host, port);
    auto coordinator = std::make_shared<ServiceCoordinator>(std::move(identity), std::move(transport));
    std::cout << "[fabric-manager agent] connected to controller " << host << ":" << port
              << " (world_index=" << coordinator->local_index(Scope::world())
              << ", world_size=" << coordinator->participant_count(Scope::world()) << ")" << std::endl;
    return coordinator;
}

}  // namespace

Role parse_role(const std::vector<std::string>& args) {
    if (!test_args::has_command_option(args, "--role")) {
        return Role::Standalone;
    }
    const std::string role = test_args::get_command_option(args, "--role");
    if (role == "standalone") {
        return Role::Standalone;
    }
    if (role == "controller") {
        return Role::Controller;
    }
    if (role == "agent") {
        return Role::Agent;
    }
    if (role == "selftest") {
        return Role::SelfTest;
    }
    if (role == "discover-psd") {
        return Role::DiscoverPsd;
    }
    if (role == "routing-bringup") {
        return Role::RoutingBringup;
    }
    TT_FATAL(
        false,
        "Unknown --role '{}'. Expected one of: standalone, controller, agent, selftest, discover-psd, routing-bringup",
        role);
    return Role::Standalone;
}

int run_controller(const std::vector<std::string>& args) {
    int world_size = require_int_option(args, "--world-size");
    uint16_t port = test_args::has_command_option(args, "--port")
                        ? static_cast<uint16_t>(std::stoi(test_args::get_command_option(args, "--port")))
                        : static_cast<uint16_t>(7777);

    auto controller = std::make_shared<Controller>();
    TcpControllerServer server(port, controller);
    std::cout << "[fabric-manager controller] listening on port " << server.port() << ", expecting " << world_size
              << " agent(s)" << std::endl;
    server.serve(world_size);
    std::cout << "[fabric-manager controller] all " << world_size << " agent(s) completed; shutting down" << std::endl;
    return 0;
}

int run_selftest(const std::vector<std::string>& args) {
    int n = test_args::has_command_option(args, "--world-size")
                ? std::stoi(test_args::get_command_option(args, "--world-size"))
                : 4;
    TT_FATAL(n >= 1, "--world-size must be >= 1");

    const bool use_tcp = test_args::has_command_option(args, "--transport") &&
                         test_args::get_command_option(args, "--transport") == "tcp";

    auto controller = std::make_shared<Controller>();

    // For the TCP variant, stand up a real controller server on an ephemeral loopback port
    // and have each agent thread connect over sockets -- exercising the full client+server.
    std::shared_ptr<TcpControllerServer> server;
    std::thread server_thread;
    uint16_t tcp_port = 0;
    if (use_tcp) {
        server = std::make_shared<TcpControllerServer>(/*port=*/0, controller);
        tcp_port = server->port();
        server_thread = std::thread([server, n]() { server->serve(n); });
    }

    std::cout << "[fabric-manager selftest] running " << (use_tcp ? "TCP" : "in-process") << " rendezvous with " << n
              << " agents" << (use_tcp ? (" on loopback port " + std::to_string(tcp_port)) : std::string{})
              << std::endl;

    auto make_transport = [&](int /*index*/) -> std::shared_ptr<ControllerTransport> {
        if (use_tcp) {
            return std::make_shared<TcpTransport>("127.0.0.1", tcp_port);
        }
        return std::make_shared<InProcessTransport>(controller);
    };

    std::atomic<bool> ok{true};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        threads.emplace_back([i, n, &make_transport, &ok, &failures]() {
            try {
                auto transport = make_transport(i);
                AgentIdentity identity;
                identity.world_index = i;
                identity.world_size = n;
                identity.mesh_membership[0] = MeshMembership{i, n};
                ServiceCoordinator sc(identity, transport);

                TT_FATAL(sc.participant_count(Scope::world()) == n, "world participant_count mismatch");
                TT_FATAL(sc.local_index(Scope::world()) == i, "world local_index mismatch");

                sc.barrier(Scope::world());

                Bytes mine{static_cast<uint8_t>(i)};
                auto gathered = sc.all_gather(mine, Scope::world());
                TT_FATAL(static_cast<int>(gathered.size()) == n, "world all_gather size {} != {}", gathered.size(), n);
                for (int k = 0; k < n; ++k) {
                    TT_FATAL(
                        gathered[k] == Bytes{static_cast<uint8_t>(k)},
                        "world all_gather slot {} wrong (order not preserved?)",
                        k);
                }

                auto bcast = sc.broadcast(i == 0 ? Bytes{42} : Bytes{}, /*root_index=*/0, Scope::world());
                TT_FATAL(bcast == Bytes{42}, "world broadcast payload mismatch on agent {}", i);

                // Mesh-scoped collective (proves Scope::mesh path).
                auto mesh_gathered = sc.all_gather(mine, Scope::mesh(MeshId{0}));
                TT_FATAL(
                    static_cast<int>(mesh_gathered.size()) == n,
                    "mesh all_gather size {} != {}",
                    mesh_gathered.size(),
                    n);
            } catch (const std::exception& e) {
                ok = false;
                failures.fetch_add(1);
                std::cerr << "[fabric-manager selftest] agent " << i << " FAILED: " << e.what() << std::endl;
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    if (server_thread.joinable()) {
        server_thread.join();
    }

    if (ok) {
        std::cout << "[fabric-manager selftest] PASS (" << n << " agents, world + mesh scopes, "
                  << (use_tcp ? "TCP" : "in-process") << ")" << std::endl;
        return 0;
    }
    std::cout << "[fabric-manager selftest] FAIL (" << failures.load() << "/" << n << " agents)" << std::endl;
    return 1;
}

void register_agent_coordinator(const std::vector<std::string>& args) {
    // Optionally propagate this agent's mesh binding to the control plane, which reads it from env.
    if (test_args::has_command_option(args, "--mesh-id")) {
        setenv("TT_MESH_ID", test_args::get_command_option(args, "--mesh-id").c_str(), 1);
    }
    if (test_args::has_command_option(args, "--mesh-host-rank")) {
        setenv("TT_MESH_HOST_RANK", test_args::get_command_option(args, "--mesh-host-rank").c_str(), 1);
    }

    // Connect now (controller must already be up) and cache the coordinator; the factory
    // hands the same instance to the control plane when it is constructed.
    auto coordinator = build_agent_coordinator(args);
    tt::tt_fabric::coordination::set_system_coordinator_factory(
        [coordinator]() -> std::shared_ptr<tt::tt_fabric::coordination::SystemCoordinator> { return coordinator; });
}

int run_discovery_psd(const std::vector<std::string>& args) {
    // Multi-process, no-MPI bring-up check for the coordinator-routed global-PSD path: each agent
    // loads its own mock cluster descriptor, then runs physical_system_discovery with the
    // ServiceCoordinator so the gather -> merge -> scatter goes through the controller (over TCP)
    // instead of MPI. Every agent must end up with the same merged global PSD (all N hosts).
    TT_FATAL(
        test_args::has_command_option(args, "--mock-cluster-desc"),
        "discover-psd role requires --mock-cluster-desc <path to a mock cluster descriptor yaml>");
    const std::string mock_desc_path = test_args::get_command_option(args, "--mock-cluster-desc");
    setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", mock_desc_path.c_str(), 1);

    auto coordinator = build_agent_coordinator(args);
    const int world_index = coordinator->local_index(Scope::world());
    const int world_size = coordinator->participant_count(Scope::world());

    auto cluster_desc = tt::umd::ClusterDescriptor::create_from_yaml(mock_desc_path);
    TT_FATAL(cluster_desc != nullptr, "Failed to load mock cluster descriptor from '{}'", mock_desc_path);

    // distributed_context is unused on the coordinator path (see run_physical_system_discovery);
    // pass an empty handle to prove no DistributedContext/MPI is required for a no-MPI agent.
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> no_context;
    auto psd = tt::tt_metal::run_physical_system_discovery(
        *cluster_desc,
        no_context,
        tt::TargetDevice::Mock,
        /*run_global_discovery=*/true,
        /*run_live_discovery=*/false,
        coordinator);

    auto hostnames = psd.get_all_hostnames();
    std::sort(hostnames.begin(), hostnames.end());
    const auto& host_to_rank = psd.get_host_to_rank_map();

    // Emit a canonical, machine-checkable fingerprint so the driver can assert every agent
    // converged on the identical global view.
    std::ostringstream fp;
    fp << "hosts=" << hostnames.size() << " [";
    for (std::size_t i = 0; i < hostnames.size(); ++i) {
        if (i != 0) {
            fp << ",";
        }
        auto it = host_to_rank.find(hostnames[i]);
        fp << hostnames[i] << ":" << (it != host_to_rank.end() ? it->second : -1);
    }
    fp << "]";

    const bool ok = static_cast<int>(hostnames.size()) == world_size;
    std::cout << "[fabric-manager discover-psd] agent " << world_index << "/" << world_size << " "
              << (ok ? "PSD_OK" : "PSD_FAIL") << " " << fp.str() << std::endl;
    if (!ok) {
        std::cerr << "[fabric-manager discover-psd] agent " << world_index << " expected " << world_size
                  << " hosts in merged PSD but saw " << hostnames.size() << std::endl;
        return 1;
    }
    return 0;
}

int run_routing_bringup(const std::vector<std::string>& args) {
    // Multi-process, no-MPI routing bring-up check. Each agent loads its own mock cluster descriptor
    // and builds a full ControlPlane (physical discovery + topology mapping + routing-table
    // configuration) with all cross-host exchanges routed through the coordinator (over TCP) instead
    // of MPI. The final fabric-node -> ASIC mapping is a global quantity, identical on every agent, so
    // the driver can assert that every agent converges on the same routing bring-up.
    TT_FATAL(
        test_args::has_command_option(args, "--mock-cluster-desc"),
        "routing-bringup role requires --mock-cluster-desc <path to a mock cluster descriptor yaml>");
    TT_FATAL(
        test_args::has_command_option(args, "--mesh-graph-desc"),
        "routing-bringup role requires --mesh-graph-desc <path to a mesh graph descriptor textproto>");

    const std::string mock_desc_path = test_args::get_command_option(args, "--mock-cluster-desc");
    const std::string mesh_graph_desc = test_args::get_command_option(args, "--mesh-graph-desc");

    // The control plane reads the physical topology from the mock cluster descriptor and the agent's
    // mesh binding from TT_MESH_ID / TT_MESH_HOST_RANK. Fabric config/routing must run in slow-dispatch.
    setenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", mock_desc_path.c_str(), 1);
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
    TT_FATAL(test_args::has_command_option(args, "--mesh-id"), "routing-bringup role requires --mesh-id <mesh id>");
    setenv("TT_MESH_ID", test_args::get_command_option(args, "--mesh-id").c_str(), 1);
    TT_FATAL(
        test_args::has_command_option(args, "--mesh-host-rank"),
        "routing-bringup role requires --mesh-host-rank <host rank>");
    setenv("TT_MESH_HOST_RANK", test_args::get_command_option(args, "--mesh-host-rank").c_str(), 1);

    auto coordinator = build_agent_coordinator(args);
    const int world_index = coordinator->local_index(Scope::world());
    const int world_size = coordinator->participant_count(Scope::world());

    auto fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D;
    if (test_args::has_command_option(args, "--fabric-config")) {
        auto parsed =
            enchantum::cast<tt::tt_fabric::FabricConfig>(test_args::get_command_option(args, "--fabric-config"));
        TT_FATAL(parsed.has_value(), "Invalid --fabric-config value");
        fabric_config = parsed.value();
    }

    // Build the control plane directly (mirroring the multi-host gtest harness) but inject the
    // no-MPI coordinator so its construction-time exchanges (discovery, topology mapping, intermesh,
    // router-port-directions) all resolve through the controller. The local DistributedContext is
    // size-1; identity and cross-host collectives come from the coordinator.
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();

    tt::tt_fabric::ControlPlane control_plane(
        cluster,
        rtoptions,
        hal,
        distributed_context,
        mesh_graph_desc,
        fabric_config,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        tt::tt_fabric::FabricTensixConfig::DISABLED,
        tt::tt_fabric::FabricUDMMode::DISABLED,
        tt::tt_fabric::FabricRouterConfig{},
        tt::tt_fabric::FabricManagerMode::DEFAULT,
        coordinator);

    // Configure routing tables (CPU-only on a mock cluster). This also exercises the coordinator-routed
    // router-port-directions exchange across hosts.
    control_plane.configure_routing_tables_for_fabric_ethernet_channels();

    // Build a canonical, global fabric-mapping fingerprint. get_asic_id_from_fabric_node_id resolves
    // every mapped node on every agent (the solved mapping is broadcast to all agents), so a correct
    // no-MPI bring-up yields the identical fingerprint everywhere.
    const auto& mesh_graph = control_plane.get_mesh_graph();
    auto mesh_ids = mesh_graph.get_all_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end(), [](const auto& a, const auto& b) { return *a < *b; });

    std::ostringstream detail;
    std::size_t mapped_nodes = 0;
    for (const auto& mesh_id : mesh_ids) {
        const auto shape = control_plane.get_physical_mesh_shape(mesh_id, tt::tt_fabric::MeshScope::GLOBAL);
        detail << "M" << *mesh_id << "=[";
        for (std::size_t d = 0; d < shape.dims(); ++d) {
            if (d != 0) {
                detail << "x";
            }
            detail << shape[d];
        }
        detail << "]{";
        bool first = true;
        for (const auto& [coord, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            const tt::tt_fabric::FabricNodeId fabric_node_id(mesh_id, chip_id);
            std::uint64_t asic = 0;
            bool mapped = true;
            try {
                asic = *control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
            } catch (const std::exception&) {
                mapped = false;
            }
            if (!first) {
                detail << ",";
            }
            first = false;
            detail << chip_id << ":";
            if (mapped) {
                detail << asic;
                ++mapped_nodes;
            } else {
                detail << "UNMAPPED";
            }
        }
        detail << "}";
    }

    // Collapse the (possibly large) detail into a stable hash for the primary fingerprint, but also
    // print the mesh count / mapped-node count so a mismatch is easy to triage.
    const std::size_t fp_hash = std::hash<std::string>{}(detail.str());
    std::ostringstream fp;
    fp << "meshes=" << mesh_ids.size() << " mapped_nodes=" << mapped_nodes << " map_hash=0x" << std::hex << fp_hash;

    const bool ok = mapped_nodes > 0;
    std::cout << "[fabric-manager routing-bringup] agent " << world_index << "/" << world_size << " "
              << (ok ? "ROUTING_OK" : "ROUTING_FAIL") << " " << fp.str() << std::endl;
    if (!ok) {
        std::cerr << "[fabric-manager routing-bringup] agent " << world_index << " produced an empty fabric mapping"
                  << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace tt::scaleout_tools::fabric_manager
