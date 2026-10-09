To fix the issue with the 2D core grid in LayerNorm and RMSNorm, the starting index calculation in the 2D reader is adjusted to correctly account for both dimensions.

```python
# ... (previous code)

    if not use_2d_core_grid:
        # existing code
    else:
        # The grid is 2D, with each core being (x, y) in the grid
        # Compute 2D start
        # The 2D grid has tiles_per_core_x cores along x, and tiles_per_core_y along y
        # The starting index is (x * tiles_per_core_y + y) * Wt_full
        start = (x * tiles_per_core_y + y) * Wt_full
        # ... (rest of the code)
```