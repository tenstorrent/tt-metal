// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENTAL: runtime binary reload — Blaze-only. A program that reloads carries the L1 address
// of its per-core stage table into the launch message of every kernel group inside `cores`
// (kernel_config_msg_t::reload_table_addr, 0 = no table); the Blaze firmware walks that table.
// Same extension pattern as named_kernel_args.hpp: a member of the stable descriptor whose type,
// merge rule and hash live here.

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal::experimental::blaze {

struct ReloadTable {
    // L1 address of the table, the same on every core in `cores`.
    uint32_t address = 0;
    // The cores that walk the table. Every core of a kernel group shares one launch message, so
    // this must cover each group it touches in full; the program refuses a partial group.
    CoreRangeSet cores;
};

// ProgramDescriptor::merge rule: a merged program has one launch message, so at most one table.
// `other` absent leaves `mine`; two tables must agree on the address and their cores are unioned.
std::optional<ReloadTable> merge_reload_tables(
    const std::optional<ReloadTable>& mine, const std::optional<ReloadTable>& other);

// For the program cache key: the address rides in the cached program's launch message, so
// descriptors that differ only here are different programs.
std::uint64_t hash_reload_table(const std::optional<ReloadTable>& table);

}  // namespace tt::tt_metal::experimental::blaze
