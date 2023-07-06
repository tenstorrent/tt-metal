#pragma once

#include "tt_metal/impl/program.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
using namespace tt::tt_metal;

namespace tt::tt_metal::detail{

    inline void AddKernel ( Program & program, Kernel * kernel)
    {
        program.add_kernel(kernel);
    }

    inline void AddSemaphore ( Program & program, const CoreRangeSet & crs, uint32_t address, uint32_t init_value)
    {
        program.add_semaphore(crs, address, init_value);
    }

    inline void AddCircularBuffer ( Program & program, CircularBuffer *circular_buffer)
    {
        program.add_circular_buffer(circular_buffer);
    }

    // Checks that circular buffers do not grow into L1 buffer space
    // If `logical_core` is supplied the check if only performed for that core, otherwise all cores with circular buffers are validated
    inline void ValidateCircularBufferRegion(const Program &program, const Device *device, std::optional<CoreCoord> logical_core = std::nullopt) {
        program.validate_circular_buffer_region(device, logical_core);
    }    

}  // namespace tt::tt_metal::detail
