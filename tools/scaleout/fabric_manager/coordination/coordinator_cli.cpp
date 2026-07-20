// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/fabric_manager/coordination/coordinator_cli.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/system_coordinator.hpp>

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
    TT_FATAL(false, "Unknown --role '{}'. Expected one of: standalone, controller, agent, selftest", role);
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
    AgentIdentity identity;
    identity.world_index = require_int_option(args, "--world-index");
    identity.world_size = require_int_option(args, "--world-size");
    if (test_args::has_command_option(args, "--mesh-membership")) {
        parse_mesh_membership(test_args::get_command_option(args, "--mesh-membership"), identity);
    }

    TT_FATAL(test_args::has_command_option(args, "--controller"), "--role agent requires --controller <host:port>");
    auto [host, port] = parse_endpoint(test_args::get_command_option(args, "--controller"));

    // Optionally propagate this agent's mesh binding to the control plane, which reads it from env.
    if (test_args::has_command_option(args, "--mesh-id")) {
        setenv("TT_MESH_ID", test_args::get_command_option(args, "--mesh-id").c_str(), 1);
    }
    if (test_args::has_command_option(args, "--mesh-host-rank")) {
        setenv("TT_MESH_HOST_RANK", test_args::get_command_option(args, "--mesh-host-rank").c_str(), 1);
    }

    // Connect now (controller must already be up) and cache the coordinator; the factory
    // hands the same instance to the control plane when it is constructed.
    auto transport = std::make_shared<TcpTransport>(host, port);
    auto coordinator = std::make_shared<ServiceCoordinator>(std::move(identity), std::move(transport));
    std::cout << "[fabric-manager agent] connected to controller " << host << ":" << port
              << " (world_index=" << coordinator->local_index(Scope::world())
              << ", world_size=" << coordinator->participant_count(Scope::world()) << ")" << std::endl;

    tt::tt_fabric::coordination::set_system_coordinator_factory(
        [coordinator]() -> std::shared_ptr<tt::tt_fabric::coordination::SystemCoordinator> { return coordinator; });
}

}  // namespace tt::scaleout_tools::fabric_manager
