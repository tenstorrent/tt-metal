// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::tt_metal {

class Program;

namespace experimental {

// Everything a program asks the device to run, as one line of JSON: per kernel its source (path and a hash of its
// text), core type, processor class and config, cores, compile-time and named compile-time args, defines, and the
// per-core and common runtime args; per circular buffer its cores, total size, address, whether it is globally
// allocated, and each buffer index's page size and data format; per semaphore its cores, id, initial value and core
// type. Kernels come in handle order and cores, buffer indices and map keys sorted, so two builds of the same
// program give the same text. For tools that compare what two builds of an operation ask the device to run (the
// native migration's mesh scale check); nothing in the runtime reads it.
std::string program_fingerprint_json(Program& program);

}  // namespace experimental
}  // namespace tt::tt_metal
