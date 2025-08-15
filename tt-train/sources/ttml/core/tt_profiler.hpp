// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

namespace ttml::core {
class TTProfiler {
public:
    TTProfiler();
    ~TTProfiler();

    TTProfiler(const TTProfiler&) = delete;
    TTProfiler& operator=(const TTProfiler&) = delete;
    TTProfiler(TTProfiler&&) = delete;
    TTProfiler& operator=(TTProfiler&&) = delete;

    void read_results(
        ttnn::distributed::MeshDevice* device,
        const std::string& noop_identifier = "noop_identifier",
        const size_t number_of_noops = 5U,
        tt::tt_metal::ProfilerReadState read_state = tt::tt_metal::ProfilerReadState::NORMAL) const;

    void call_device_noop(
        ttnn::distributed::MeshDevice* device, size_t count, const std::string& noop_identifier) const;

    [[nodiscard]] bool is_enabled() const;
    void enable();
    void disable();

private:
    bool m_enabled = false;
};

}  // namespace ttml::core
