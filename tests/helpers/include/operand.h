// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

class Operand
{
public:
    constexpr Operand(uint32_t base, uint32_t size) : base_addr(base), tile_size(size)
    {
    }

    [[nodiscard]] constexpr uint32_t operator[](uint32_t index) const noexcept
    {
        return base_addr + index * tile_size;
    }

private:
    uint32_t base_addr;
    uint32_t tile_size;
};
