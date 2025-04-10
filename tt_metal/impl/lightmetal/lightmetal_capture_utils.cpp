// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <lightmetal_capture_utils.hpp>
#include <stdint.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <cstddef>
#include <vector>

#include "lightmetal/host_api_capture_helpers.hpp"

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

void LightMetalCompareToCapture(
    CommandQueue& cq, const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer, void* dst) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();

    // If dst ptr is not provided, just allocate temp space for rd return capture/usage.
    std::vector<uint32_t> rd_data_tmp;
    if (!dst) {
        size_t buffer_size = std::holds_alternative<std::reference_wrapper<Buffer>>(buffer)
                                 ? std::get<std::reference_wrapper<Buffer>>(buffer).get().size()
                                 : std::get<std::shared_ptr<Buffer>>(buffer)->size();
        rd_data_tmp.resize(buffer_size / sizeof(uint32_t));
        dst = rd_data_tmp.data();
    }

    EnqueueReadBuffer(cq, buffer, dst, true);  // Blocking read to get golden value.
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureLightMetalCompare, cq, buffer, dst, false);
}

void LightMetalCompareToGolden(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* golden_data) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureLightMetalCompare, cq, buffer, golden_data, true);
}

}  // namespace tt::tt_metal
