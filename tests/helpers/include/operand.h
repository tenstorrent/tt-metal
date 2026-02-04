// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

class Operand
{
public:
    constexpr Operand(std::uint32_t base, std::uint32_t size) : base_addr(base), tile_size(size)
    {
    }

    [[nodiscard]] constexpr std::uint32_t operator[](std::uint32_t index) const noexcept
    {
        return base_addr + index * tile_size;
    }

private:
    std::uint32_t base_addr;
    std::uint32_t tile_size;
};
