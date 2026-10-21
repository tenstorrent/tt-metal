// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"

template <typename T, typename AddressType>
class CoreLocalMem;

namespace local_mem::detail {
// The only way in to the raw L1 address behind a local-memory handle (CoreLocalMem befriends this). For the runtime:
// noc_traits_t, LLK operands and raw-address helper overloads. Scratchpad and LocalTensorAccessor reach it through
// their public local_mem().
struct AddressAccess {
    template <typename T, typename AddressType>
    static AddressType address(const CoreLocalMem<T, AddressType>& mem) {
        return mem.address_;
    }
    template <typename T, typename AddressType>
    static tt_l1_ptr T* ptr(const CoreLocalMem<T, AddressType>& mem) {
        return reinterpret_cast<tt_l1_ptr T*>(mem.address_);
    }
};
}  // namespace local_mem::detail
