// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

#include <utility>

#include "llrt/zone_meta.hpp"

namespace tt::tt_metal::streaming_profiler {

void ZoneNameMirror::refresh() {
    std::vector<llrt::ZoneMetaEntry> delta;
    cursor_ = llrt::ZoneMetaRegistry::instance().additions_since(cursor_, delta);
    for (auto& e : delta) {
        sites_.emplace(e.zone_id, Site{std::move(e.name), std::move(e.file), e.line});
    }
}

}  // namespace tt::tt_metal::streaming_profiler
