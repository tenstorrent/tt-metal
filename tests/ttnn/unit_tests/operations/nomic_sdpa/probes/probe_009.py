import torch, ttnn
from eval.golden_tests.nomic_sdpa.helpers import reserve_l1, make_qkv

device = ttnn.open_device(device_id=0)
filler = reserve_l1(device, 176)
q, k, v = make_qkv((8, 12, 512, 64), "real", 0)
ts = [
    ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    for t in (q, k, v)
]
out = ttnn.allocate_tensor_on_device(
    ttnn.Shape([8, 1, 512, 768]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
)
view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
live = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
base = ttnn.get_allocator_base_address(device, ttnn.BufferType.L1)
print(
    "PROBE free below live (KiB):",
    (min(live) - base) / 1024,
    "base",
    base,
    "lowest",
    min(live),
    "bank",
    view.total_bytes_per_bank,
)
ttnn.close_device(device)
