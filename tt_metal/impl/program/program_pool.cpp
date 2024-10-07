// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program_pool.hpp"

namespace tt::tt_metal {

void ProgramPool::initialize() { programs.clear(); }

ProgramHandle ProgramPool::create_program() {
    std::unique_lock lock(mutex_);
    ProgramKey key = programs.emplace();
    return ProgramHandle{key.raw()};
}

void ProgramPool::release_program(ProgramHandle handle) {
    if (not handle.is_valid()) {
        return;
    }

    std::unique_lock lock(mutex_);
    ProgramKey key(handle.key);
    programs.remove(key);
}

tt_metal::Program* ProgramPool::get_program(ProgramHandle handle) {
    std::shared_lock lock(mutex_);
    ProgramKey key(handle.key);
    Program* ptr = programs.get(key);
    TT_FATAL(ptr, "ProgramPool::get_program: Program not found");
    return ptr;
}

ScopedProgramHandle CreateScopedProgram() { return ScopedProgramHandle(CreateProgram()); }

ScopedProgramHandle::ScopedProgramHandle(ScopedProgramHandle&& other) noexcept :
    handle_(std::exchange(other.handle_, ProgramHandle{})) {}

ScopedProgramHandle& ScopedProgramHandle::operator=(ScopedProgramHandle&& other) noexcept {
    if (this != &other) {
        this->~ScopedProgramHandle();
        new (this) ScopedProgramHandle(std::move(other));
    }
    return *this;
}

ScopedProgramHandle::~ScopedProgramHandle() { CloseProgram(this->handle_); }

void swap(ScopedProgramHandle& lhs, ScopedProgramHandle& rhs) noexcept {
    using std::swap;
    swap(lhs.handle_, rhs.handle_);
}

}  // namespace tt::tt_metal
