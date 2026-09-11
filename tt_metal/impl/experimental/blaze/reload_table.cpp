// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// EXPERIMENTAL: runtime binary reload — Blaze-only.

#include <tt-metalium/experimental/blaze/reload_table.hpp>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal::experimental::blaze {

std::optional<ReloadTable> merge_reload_tables(
    const std::optional<ReloadTable>& mine, const std::optional<ReloadTable>& other) {
    if (!other.has_value()) {
        return mine;
    }
    if (!mine.has_value()) {
        return other;
    }
    TT_FATAL(
        mine->address == other->address,
        "Cannot merge ProgramDescriptors with different reload_table_addr ({:#x} and {:#x})",
        mine->address,
        other->address);
    return ReloadTable{mine->address, mine->cores.merge(other->cores)};
}

std::uint64_t hash_reload_table(const std::optional<ReloadTable>& table) {
    std::uint64_t hash = 0;
    ttsl::hash::hash_combine(hash, table.has_value());
    if (table.has_value()) {
        ttsl::hash::hash_combine(hash, table->address);
        ttsl::hash::hash_combine(hash, table->cores);
    }
    return hash;
}

}  // namespace tt::tt_metal::experimental::blaze
