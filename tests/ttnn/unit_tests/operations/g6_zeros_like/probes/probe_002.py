import torch, ttnn
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    shape = (5, 1, 32, 256)
    ones = ttnn.from_torch(
        torch.ones(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ones)
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    o = ttnn.from_torch(
        torch.ones(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for name, out in (("new", ttnn.allocate_tensor_on_device(x.spec, device)), ("optional", o)):
        ttnn.generic_op([out], pd.get_program_descriptor(out))
        t = ttnn.to_torch(out).float()
        print(
            "PROBE",
            name,
            "pages",
            out.buffer_num_pages(),
            "page_size",
            out.buffer_aligned_page_size(),
            "nonzero",
            int((t != 0).sum()),
            "of",
            t.numel(),
        )
finally:
    ttnn.close_device(device)
