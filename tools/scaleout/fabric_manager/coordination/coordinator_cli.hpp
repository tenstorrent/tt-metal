// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// CLI glue for the fabric-manager coordinator roles (Option (a)).
//
// run_fabric_manager gains a --role selector:
//   * standalone (default) - unchanged: single-process / MPI path, no coordinator.
//   * controller           - run the central rendezvous server; no hardware touched.
//   * agent                - register an external ServiceCoordinator (no MPI) so the
//                            control plane routes its cross-host exchanges through the
//                            controller, then run the normal fabric bring-up.
//   * selftest             - spin N in-process agents through the coordinator to verify
//                            the rendezvous logic locally (no hardware, CI-friendly).
//   * discover-psd         - no-MPI multi-process bring-up check: run physical-system
//                            discovery over the controller (TCP) against a per-agent mock
//                            cluster descriptor and assert the merged global PSD converges.
//
// This whole unit is compiled only when FABRIC_MANAGER_WITH_SERVICE_COORDINATOR is
// defined (CMake option), giving compile-time backend selection with runtime (--role)
// flexibility for testing.
//

#include <cstdint>
#include <string>
#include <vector>

namespace tt::scaleout_tools::fabric_manager {

enum class Role : uint8_t {
    Standalone,
    Controller,
    Agent,
    SelfTest,
    DiscoverPsd,
};

// Parses --role (default Standalone). Errors on an unknown value.
Role parse_role(const std::vector<std::string>& args);

// Runs the controller rendezvous server until all agents finish. Returns process exit code.
int run_controller(const std::vector<std::string>& args);

// Runs the in-process rendezvous self-test. Returns process exit code (0 == PASS).
int run_selftest(const std::vector<std::string>& args);

// Builds an external ServiceCoordinator from the agent args and registers it as the
// tt_metal coordinator factory, so it is injected when the control plane is constructed.
// Call BEFORE triggering fabric bring-up.
void register_agent_coordinator(const std::vector<std::string>& args);

// Runs the no-MPI multi-process global-PSD bring-up check for one agent. Returns process exit
// code (0 == the merged global PSD converged to the expected host count).
int run_discovery_psd(const std::vector<std::string>& args);

}  // namespace tt::scaleout_tools::fabric_manager
