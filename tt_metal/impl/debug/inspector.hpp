// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

namespace inspector {
class Data;
}

class Inspector {
public:
    static bool is_enabled();

    static std::unique_ptr<inspector::Data> initialize();

    static void program_created(
        const detail::ProgramImpl* program) noexcept;
    static void program_destroyed(
        const detail::ProgramImpl* program) noexcept;
    static void program_set_binary_status(
        const detail::ProgramImpl* program,
        std::size_t device_id,
        ProgramBinaryStatus status) noexcept;
    static void program_compile_started(
        const detail::ProgramImpl* program,
        const IDevice* device,
        uint32_t build_key) noexcept;
    static void program_compile_already_exists(
        const detail::ProgramImpl* program,
        const IDevice* device,
        uint32_t build_key) noexcept;
    static void program_kernel_compile_finished(
        const detail::ProgramImpl* program,
        const IDevice* device,
        const std::shared_ptr<Kernel>& kernel,
        const tt::tt_metal::JitBuildOptions& build_options) noexcept;
    static void program_compile_finished(
        const detail::ProgramImpl* program,
        const IDevice* device,
        uint32_t build_key) noexcept;
};

}  // namespace tt::tt_metal
