// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// read_value's accepts (compute_api/sync.h). read_value is not an op on a unit: nothing is unpacked, computed or
// packed, so it has no OpConfig, Work or Plan, only the rule for what it takes. Arch-independent.
//
//   accepts   t in L1 and the unpackers' (a tensor at a buffer's front, not yet popped); the word an unsigned
//             integer of 1, 2 or 4 bytes
//
// Not ported: a tensor this kernel packed (it has no pack -> unpack hand-off).

#pragma once

#include <cstdint>
#include <type_traits>

#include "resolver/resolver.h"

namespace llk2
{

/// accepts: see the file comment. T is the word's type.
template <class T>
constexpr Accepts read_value_accepts(const Spec& t)
{
    const bool storage = t.storage == Storage::L1 && t.thread == Thread::Unpack && t.exu == Exu::Unpacker;
    const bool formats = std::is_integral_v<T> && std::is_unsigned_v<T> && !std::is_same_v<T, bool> && sizeof(T) <= sizeof(std::uint32_t);
    return {.storage = storage, .formats = formats};
}

} // namespace llk2
