# Reshape and MoE Dimension Check

## Description

Reshape operations in TTNN change the logical shape of a tensor without modifying its data.
A common class of bugs occurs when the total number of elements (volume) changes
unintentionally, or when alignment constraints for the target layout are violated.

In Mixture-of-Experts (MoE) operations, tile distributions across cores and experts must
match the expected block structure. Mismatches cause silent data corruption or hangs.

## What to Look For

1. **Logical volume mismatch**: Before and after a reshape, `logical_shape.volume()` must
   equal `input_tensor.logical_volume()`. If a dimension is inferred (set to -1), verify
   the arithmetic produces an exact division.

2. **Row-major width alignment**: For `Layout::ROW_MAJOR`, the last dimension
   (`padded_shape[3]`) must be divisible by 8 (ROW_MAJOR_WIDTH) for both input and output.

3. **Tile layout alignment**: For `Layout::TILE`, both the height and width of the padded
   shape must be divisible by 32 (TILE_HEIGHT / TILE_WIDTH), and the physical volume must
   be divisible by TILE_HW (1024).

4. **Padded vs. logical shape confusion**: Code that mixes `padded_shape()` and
   `logical_shape()` when computing output dimensions can produce incorrect results.
   Padding is layout-specific and must be applied after the logical reshape.

5. **MoE tile distribution**: In MoE ring operations, precomputed lookup tables like
   `W0_W1_TILES_PER_CORE_PER_STEP` must be consistent with the actual number of cores and
   experts. Check that array dimensions match `NUM_CORES` and iteration bounds.

## Bad Code Examples

```cpp
// BUG: volume changes — 2*3*4=24 != 2*3*5=30
auto output_shape = ttnn::Shape({2, 3, 5});
auto result = ttnn::reshape(input, output_shape);  // input is {2, 3, 4}
```

```cpp
// BUG: row-major width not divisible by 8
auto output_shape = ttnn::Shape({1, 1, 1, 13});  // 13 % 8 != 0
auto result = ttnn::reshape(input, output_shape, Layout::ROW_MAJOR);
```

```cpp
// BUG: using padded_shape volume instead of logical_volume for validation
if (input.padded_shape().volume() == output_shape.volume()) {  // wrong comparison
    // This can pass even when logical volumes differ due to padding
}
```

## Good Code Examples

```cpp
// GOOD: volumes match
TT_ASSERT(output_shape.volume() == input.logical_volume());
auto result = ttnn::reshape(input, output_shape);
```

```cpp
// GOOD: row-major width is aligned
auto output_shape = ttnn::Shape({1, 1, 1, 16});  // 16 % 8 == 0
```

```cpp
// GOOD: comparing logical volumes
if (input.logical_volume() == output_shape.volume()) {
    // Correct — compares element counts without padding
}
```
