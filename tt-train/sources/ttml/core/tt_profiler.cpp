// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_profiler.hpp"

#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

#if defined(TRACY_ENABLE)
constexpr bool is_tracy_enabled = true;
#else
constexpr bool is_tracy_enabled = false;
#endif

namespace ttml::core {

void TTProfiler::read_results(
    ttnn::distributed::MeshDevice* device,
    const std::string& noop_identifier,
    const size_t number_of_noops,
    tt::tt_metal::ProfilerReadState read_state) const {
    assert(device);
    if (!m_enabled) {
        return;
    }
    call_device_noop(device, number_of_noops, noop_identifier);
    tt::tt_metal::ReadMeshDeviceProfilerResults(*device, read_state);
    call_device_noop(device, number_of_noops, noop_identifier);
}

void TTProfiler::call_device_noop(
    ttnn::distributed::MeshDevice* device, size_t count, const std::string& noop_identifier) const {
    assert(device);
    if (!m_enabled) {
        return;
    }

    auto fake_tensor = ttml::core::from_vector({1.F}, ttnn::Shape({1, 1, 1, 1}), device, ttnn::Layout::ROW_MAJOR);
    for (size_t i = 0; i < count; ++i) {
        [[maybe_unused]] auto _ = ttml::metal::profiler_no_op(fake_tensor, noop_identifier);
    }
}

bool TTProfiler::is_enabled() const {
    return m_enabled;
}

void TTProfiler::enable() {
    m_enabled = true;
}

void TTProfiler::disable() {
    m_enabled = false;
}

TTProfiler::TTProfiler() : m_enabled(false) {
    if (is_tracy_enabled) {
        enable();

        tt::tt_metal::detail::ProfilerSync(tt::tt_metal::ProfilerSyncState::INIT);
    }
}

TTProfiler::~TTProfiler() {
    if (is_tracy_enabled) {
        tt::tt_metal::detail::ProfilerSync(tt::tt_metal::ProfilerSyncState::CLOSE_DEVICE);
    }
}

}  // namespace ttml::core
