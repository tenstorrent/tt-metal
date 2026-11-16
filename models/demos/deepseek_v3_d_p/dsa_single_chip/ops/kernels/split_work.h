// Derive this core's contiguous work range [first, first+count) from its logical coordinates, so a program can use
// common runtime args only (no per-core unique args to dispatch). Matches ops/_program.split_work: cores are the
// column-major prefix of the grid (linear index = x * GRID_Y + y), units split evenly with the remainder first.
#pragma once
#include <cstdint>

// Set by firmware on every RISC (same symbols dataflow_api.h reads); declared here so compute kernels can use them too.
extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;

struct WorkRange {
    uint32_t first;
    uint32_t count;
};

template <uint32_t UNITS, uint32_t NCORES, uint32_t GRID_Y>
inline __attribute__((always_inline)) WorkRange my_work_range() {
    const uint32_t idx = static_cast<uint32_t>(my_logical_x_) * GRID_Y + my_logical_y_;
    constexpr uint32_t base = UNITS / NCORES;
    constexpr uint32_t extra = UNITS % NCORES;
    return {idx * base + (idx < extra ? idx : extra), base + (idx < extra ? 1u : 0u)};
}
