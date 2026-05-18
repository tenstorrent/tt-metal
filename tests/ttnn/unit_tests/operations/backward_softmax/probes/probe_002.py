import torch
import ttnn

from ttnn.operations.backward_softmax.backward_softmax_program_descriptor import pick_strategy_name

device = ttnn.open_device(device_id=0)
try:
    shapes_and_dims = [
        ((1, 1, 32, 32), -1),
        ((1, 1, 32, 256), -1),
        ((1, 1, 32, 896), -1),
        ((1, 1, 32, 1024), -1),
        ((1, 1, 32, 1344), -1),
        ((1, 1, 32, 1376), -1),
        ((1, 1, 32, 2048), -1),
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
finally:
    ttnn.close_device(device)
