// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/program/program_handle.hpp"
#include "tt_stl/slotmap.hpp"

#include <shared_mutex>

namespace tt::tt_metal {

class ProgramPool {
   public:
    static ProgramPool& instance() {
        static ProgramPool pool;
        return pool;
    }

    void initialize();

    ProgramHandle create_program();
    void release_program(ProgramHandle handle);
    tt_metal::Program* get_program(ProgramHandle handle);

   private:
    MAKE_SLOTMAP_KEY(ProgramKey, uint32_t, 16);

    ProgramPool() = default;
    ~ProgramPool() = default;

    std::shared_mutex mutex_;
    tt::stl::SlotMap<ProgramKey, tt_metal::Program> programs;
};

class ScopedProgramHandle {
   public:
    ScopedProgramHandle() = default;
    ScopedProgramHandle(ProgramHandle handle) : handle_(handle) {}

    ScopedProgramHandle(const ScopedProgramHandle&) = delete;
    ScopedProgramHandle& operator=(const ScopedProgramHandle&) = delete;

    ScopedProgramHandle(ScopedProgramHandle&& other) noexcept;
    ScopedProgramHandle& operator=(ScopedProgramHandle&& other) noexcept;

    ~ScopedProgramHandle();

    operator ProgramHandle() const { return this->handle_; }

    friend void swap(ScopedProgramHandle& lhs, ScopedProgramHandle& rhs) noexcept;

   private:
    ProgramHandle handle_;
};

ScopedProgramHandle CreateScopedProgram();

}  // namespace tt
