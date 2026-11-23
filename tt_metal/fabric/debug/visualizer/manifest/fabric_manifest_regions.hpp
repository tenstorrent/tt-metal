// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Walks over the captured bytes the manifest names: every L1 region and stream register of a router, and the L1 areas
// the architecture fixes on every router core. Each comes with its path, the JSON pointer below its node in a
// PER_ROUTER manifest, so a reader of captured bytes and the manifest's JSON agree on where each region is.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tt_stl/overloaded.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::manifest {

namespace detail {

// Calls `on_region` for each field in `fields_by_erisc` whose content is an L1 region or a stream register, with its
// path: `fields_path`, then erisc<N> and the field's key.
template <typename OnRegion>
void visit_field_regions(std::string_view fields_path, const FieldsByErisc& fields_by_erisc, OnRegion& on_region) {
    for (size_t erisc_id = 0; erisc_id < fields_by_erisc.size(); ++erisc_id) {
        for (const Field& field : fields_by_erisc[erisc_id]) {
            if (!field.content.has_value()) {
                continue;
            }
            const std::string path = fmt::format("{}/erisc{}/{}", fields_path, erisc_id, field.key);
            std::visit(
                ttsl::overloaded{
                    [&](const content::L1& l1) { on_region(std::string_view(path), l1); },
                    [&](const content::StreamReg& stream) { on_region(std::string_view(path), stream); },
                    [](const auto& /*value*/) {},
                },
                *field.content);
        }
    }
}

// Calls `on_region` for `l1` at `path`, when the region exists.
template <typename OnRegion>
void visit_optional_l1(std::string_view path, const std::optional<content::L1>& l1, OnRegion& on_region) {
    if (l1.has_value()) {
        on_region(path, *l1);
    }
}

}  // namespace detail

// Calls `on_region(path, const content::L1&)` or `on_region(path, const content::StreamReg&)` for every L1 region and
// stream register `router` names, in the order the writer writes them. `path` is the region's JSON pointer below the
// router's node in a PER_ROUTER manifest, without a leading slash, such as "fields/erisc0/edm_status" or
// "channels/senders/vc0/ch1/ring_buffer". Fields that hold a number, flag or enum, or no content, aren't regions and
// are skipped. A stream register named by several fields comes once per field.
template <typename OnRegion>
void for_each_router_region(const Router& router, OnRegion&& on_region) {
    for (const auto& layout : k_credit_counter_arrays) {
        const std::string path = fmt::format("credit_counters/{}", lower_enum_name(layout.array));
        on_region(std::string_view(path), router.credit_counters.*layout.member);
    }
    for (size_t vc = 0; vc < router.channels.senders.size(); ++vc) {
        const std::vector<SenderChannel>& senders = router.channels.senders[vc];
        for (size_t channel = 0; channel < senders.size(); ++channel) {
            const std::string channel_path = fmt::format("channels/senders/vc{}/ch{}", vc, channel);
            detail::visit_optional_l1(channel_path + "/ring_buffer", senders[channel].ring_buffer, on_region);
            detail::visit_field_regions(channel_path + "/fields", senders[channel].fields_by_erisc, on_region);
        }
    }
    for (size_t vc = 0; vc < router.channels.receivers.size(); ++vc) {
        const std::vector<ReceiverChannel>& receivers = router.channels.receivers[vc];
        for (size_t channel = 0; channel < receivers.size(); ++channel) {
            const std::string channel_path = fmt::format("channels/receivers/vc{}/ch{}", vc, channel);
            detail::visit_optional_l1(channel_path + "/ring_buffer", receivers[channel].ring_buffer, on_region);
            detail::visit_field_regions(channel_path + "/fields", receivers[channel].fields_by_erisc, on_region);
        }
    }
    for (size_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
        for (const DownstreamEdge& edge : router.intra_chip_downstream_edges[vc]) {
            const std::string fields_path =
                fmt::format("intra_chip_downstream_edges/vc{}/edge{}/fields", vc, edge.edge);
            detail::visit_field_regions(fields_path, edge.fields_by_erisc, on_region);
        }
    }
    detail::visit_field_regions("fields", router.fields_by_erisc, on_region);
    detail::visit_optional_l1("leftover_l1", router.leftover_l1, on_region);
}

// Calls `on_area(name, const content::L1&)` for every ArchArea the architecture has, in ArchArea's order, which is the
// order the writer writes them. `name` is the area's key in the arch block's l1_areas, its lower_enum_name, such as
// "heartbeat" or "routing_table". The manifest describes these areas once, but every router core has its own copy.
template <typename OnArea>
void for_each_arch_area(const Arch& arch, OnArea&& on_area) {
    for (const ArchArea area : enchantum::values<ArchArea>) {
        detail::visit_optional_l1(lower_enum_name(area), arch.areas.get(area), on_area);
    }
}

}  // namespace tt::tt_fabric::manifest
