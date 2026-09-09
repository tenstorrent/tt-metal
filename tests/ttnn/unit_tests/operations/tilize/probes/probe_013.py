import torch, ttnn

# rank 0 / rank 1 TILE spec padded shapes, and RM input page geometry
for shape in ([], [64], [1, 1, 32, 50], [1, 1, 50, 50], [3, 50, 96]):
    spec = ttnn.TensorSpec(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM, ttnn.Tile([32, 32])
    )
    print("shape", shape, "-> TILE padded", list(spec.padded_shape) if hasattr(spec, "padded_shape") else "n/a")
