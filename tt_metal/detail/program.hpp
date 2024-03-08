// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/kernels/kernel_types.hpp"

using namespace tt::tt_metal;

namespace tt::tt_metal::detail{

    inline KernelHandle AddKernel ( Program & program, std::shared_ptr<Kernel> kernel, const CoreType &core_type)
    {
        return program.add_kernel(kernel, core_type);
    }

    inline std::shared_ptr<Kernel> GetKernel(const Program &program, KernelHandle kernel_id) {
        return program.get_kernel(kernel_id);
    }

    inline std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CBHandle id) {
        return program.get_circular_buffer(id);
    }

    // Checks that circular buffers do not grow into L1 buffer space
    inline void ValidateCircularBufferRegion(const Program &program, const Device *device) {
        program.validate_circular_buffer_region(device);
    }

}  // namespace tt::tt_metal::detail
