// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The harness of SPEC-0012 on a device: lays out each node's region, launches one node kernel a device, starts them
// together and returns what they counted.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>

#include "tt_metal/tt_fabric/keraunos/vcluster/harness.hpp"

namespace tt::tt_fabric::vcluster {

// The node kernel's source, relative to TT_METAL_HOME.
inline constexpr const char* kNodeKernel = "tests/tt_metal/tt_fabric/keraunos/vcluster/kernels/vc_node.cpp";

// One node kernel a device of `plan`, with its runtime arguments for a run tagged `nonce`, built for `mesh`'s one
// device. Each node is an idle Ethernet kernel on its slot's hart of the CCE0 of its slot's tile. Compiles the
// program; does not write the device. Throws std::runtime_error (TT_FATAL) when the package's SoC descriptor lists no
// CCE0 for a node's tile, or that core is not an Ethernet core of the device.
tt::tt_metal::Program build_harness_program(
    tt::tt_metal::distributed::MeshDevice& mesh, const HarnessPlan& plan, uint32_t nonce);

// What one node reported at the end of a run.
struct NodeReadout {
    uint32_t status = 0;
    uint32_t postcode = 0;
    uint32_t detail = 0;
    uint32_t originated = 0;
    uint32_t delivered = 0;
    uint32_t bad = 0;
    std::vector<uint64_t> sent;      // index = port
    std::vector<uint64_t> received;  // index = port
};

struct HarnessResult {
    std::vector<NodeReadout> nodes;  // index = fabric_chip
    bool timed_out = false;          // step 6 ended on the timeout, not on the counts or a halt
    // Every verdict clause of SPEC-0012 § The harness that failed, one line each, naming the device or port and the
    // value read against the value wanted. Empty means the run passed.
    std::vector<std::string> failures;
};

// SPEC-0012 § The harness, steps 1 to 8, on `mesh`, a one-chip open of the package with fabric disabled and slow
// dispatch. Throws std::runtime_error (TT_FATAL) before any device access when check_manifest() or
// check_harness_manifest() reports a finding, naming every one. Otherwise zeroes each region, launches the nodes,
// waits up to `timeout` for every node to report kStatusWaitingForGo and writes go, waits up to `timeout` for every
// node's delivered count, a halt, or the timeout, writes stop, waits for the program, and reads every node's counters.
// If a node never reaches kStatusWaitingForGo, go is not written; stop is written and the program awaited all the
// same, so a node that never launched leaves this call blocked in the wait for the program.
HarnessResult run_harness(
    tt::tt_metal::distributed::MeshDevice& mesh,
    const Manifest& manifest,
    const PackageFacts& facts,
    const RunShape& shape,
    std::chrono::seconds timeout);

}  // namespace tt::tt_fabric::vcluster
