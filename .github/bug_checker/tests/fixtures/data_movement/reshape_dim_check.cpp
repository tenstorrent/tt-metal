// Bug checker test fixture: reshape dimension check patterns.
// This file is intentionally buggy to validate rule reshape-dim-check.

#include "ttnn/operations/data_movement/reshape/reshape.hpp"

// --- Pattern 1: logical volume mismatch ---
Tensor reshape_volume_mismatch(const Tensor& input) {
    // BUG: input is {2, 3, 4} = 24 elements, output is {2, 3, 5} = 30 elements
    auto output_shape = ttnn::Shape({2, 3, 5});
    return ttnn::reshape(input, output_shape);
}

// --- Pattern 2: row-major width not aligned to 8 ---
Tensor reshape_row_major_unaligned(const Tensor& input) {
    // BUG: last dim is 13, not divisible by 8
    auto output_shape = ttnn::Shape({1, 1, 1, 13});
    return ttnn::reshape(input, output_shape, Layout::ROW_MAJOR);
}

// --- Pattern 3: padded vs logical shape confusion ---
bool validate_reshape_bad(const Tensor& input, const ttnn::Shape& output_shape) {
    // BUG: comparing padded volume instead of logical volume
    if (input.padded_shape().volume() == output_shape.volume()) {
        return true;  // can pass even when logical volumes differ
    }
    return false;
}

// --- Pattern 4: MoE tile distribution mismatch ---
constexpr uint32_t NUM_CORES = 32;
constexpr uint32_t NUM_EXPERTS = 8;
// BUG: table is sized for 16 cores but NUM_CORES is 32
constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP[16][NUM_EXPERTS] = {};
