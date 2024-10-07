// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/program/program_pool.hpp"
#include <fmt/core.h>

namespace tt::tt_metal {

void ProgramPool::initialize() {
    programs.clear();
}

ProgramHandle ProgramPool::create_program() {
    std::unique_lock lock(mutex_);
    ProgramKey key = programs.emplace();
    return ProgramHandle{key.raw()};
}

void ProgramPool::release_program(ProgramHandle handle) {
    std::unique_lock lock(mutex_);
    ProgramKey key(handle.key);
    programs.remove(ProgramKey(handle.key));
}

tt_metal::Program* ProgramPool::get_program(ProgramHandle handle) {
    std::shared_lock lock(mutex_);
    Program* ptr = programs.get(ProgramKey(handle.key));
    TT_FATAL(ptr, "ProgramPool::get_program: Program not found");
    return ptr;
}

}  // namespace tt::tt_metal
