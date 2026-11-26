// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tt-emule hooks that tt-metal calls on an emulated device and that have no meaning elsewhere.
//
// In an emule build (TT_METAL_USE_EMULE) this is tt-emule's emulated_program_runner.hpp. Otherwise the hooks below
// are inline stubs that throw: an emulated device cannot reach them, because the cluster only opens an emulated
// target in an emule build, so call sites branch on the target at run time without an #ifdef. Same pattern as
// host_sanitizers.hpp.

#ifdef TT_METAL_USE_EMULE

#include "emulated_program_runner.hpp"

#else

#include <cstdint>
#include <functional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {
class IDevice;
class Program;
struct DeviceBuildEnv;
}  // namespace tt::tt_metal

namespace tt::tt_metal::emule {

[[noreturn]] inline void throw_emule_not_built() {
    TT_THROW("An emulated device, but tt-metal was built without TT_METAL_USE_EMULE");
}

struct KernelTextSlot {
    uint32_t offset = 0;
    uint32_t size = 0;
};

inline KernelTextSlot kernel_text_slot(
    HalProgrammableCoreType /*core_type*/,
    HalProcessorClassType /*processor_class*/,
    int /*processor_type*/,
    uint32_t& /*config_offset*/) {
    throw_emule_not_built();
}

inline void compile_program(IDevice* /*device*/, Program& /*program*/, const DeviceBuildEnv& /*build_env*/) {
    throw_emule_not_built();
}

inline void initialize_firmware_state(
    int /*device_id*/,
    int /*context_id*/,
    const std::function<void(HalProgrammableCoreType, const CoreCoord& logical, const CoreCoord& virtual_core)>&
    /*write_core_state*/) {
    throw_emule_not_built();
}

}  // namespace tt::tt_metal::emule

#endif
