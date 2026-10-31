import torch, ttnn
import ttnn.operations.layer_norm_tile.layer_norm_tile as m
from ttnn.operations.layer_norm import layer_norm

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(32, 1, 32, 768).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    g = ttnn.from_torch(
        torch.ones(768).to(torch.bfloat16).reshape(1, 1, 24, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ref = ttnn.to_torch(layer_norm(x, g, g)).float()
    orig = m.create_program_descriptor

    def patched(*a, **k):
        d = orig(*a, **k)
        for kk in d.kernels:
            kk.defines = list(kk.defines) + [("LAYER_NORM_ABLATE_COMPUTE", "1")]
        print("PROBE defines:", [kk.defines for kk in d.kernels])
        return d

    m.create_program_descriptor = patched
    device.clear_program_cache()
    out = ttnn.to_torch(layer_norm(x, g, g)).float()
    print("PROBE max diff vs real op:", (out - ref).abs().max().item())
finally:
    ttnn.close_device(device)
