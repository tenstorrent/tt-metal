// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace tt::tt_metal {
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental::program_fingerprint {

/// Describes what is fixed when `program` is created, one line per element, in a deterministic order:
/// each kernel's source, cores, processor, compile-time arguments, named compile-time arguments, defines and
/// build settings; each circular buffer's cores, size and per-index page size, format and tile geometry; each
/// dataflow buffer's configuration; each semaphore. Runtime arguments, buffer addresses and what the program fills in
/// when it is finalized are left out, since they change between dispatches of one program or are absent from a
/// program that was never dispatched.
/// Two programs with equal descriptions build to the same binaries and layout.
std::vector<std::string> describe(Program& program);

/// `describe` for every program of `workload`, each line prefixed by the program's mesh coordinate range,
/// ranges in sorted order.
std::vector<std::string> describe(distributed::MeshWorkload& workload);

}  // namespace tt::tt_metal::experimental::program_fingerprint
