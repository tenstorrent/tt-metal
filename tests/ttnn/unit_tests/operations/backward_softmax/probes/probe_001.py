import torch
import ttnn

from ttnn.operations.backward_softmax.backward_softmax_program_descriptor import pick_strategy_name

# Test strategy selection for various shapes.
shapes_and_dims = [
    # Strategy 1 (DB) — small Wt/Ht
    ((1, 1, 32, 32), -1),  # Wt=1
    ((1, 1, 32, 256), -1),  # Wt=8
    ((1, 1, 32, 896), -1),  # Wt=28 (boundary)
    # Strategy 2 (SB) — medium
    ((1, 1, 32, 1024), -1),  # Wt=32
    ((1, 1, 32, 1344), -1),  # Wt=42 (boundary)
    # Strategy 3 (per-tile) — large
    ((1, 1, 32, 1376), -1),  # Wt=43
    ((1, 1, 32, 2048), -1),  # Wt=64
    # Same boundaries for dim=-2 (using Ht instead)
    ((1, 1, 1024, 32), -2),
    ((1, 1, 2048, 32), -2),
]
for shape, dim in shapes_and_dims:
    t = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    name = pick_strategy_name(t, dim=dim)
    print(f"shape={shape} dim={dim:>2} -> {name}")
