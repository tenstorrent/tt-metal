import torch, ttnn
from ttnn.operations.g6_concat import g6_concat_plan

device = ttnn.open_device(device_id=0)
try:
    mc = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
    for name, shapes, dim, bufs in [
        ("panoptic23", [[1, 32, 64, 256]] * 5, 3, ["dram"] * 4 + ["l1"]),
        ("panoptic40", [[1, 128, 256, 32], [1, 128, 256, 256]], 3, ["l1"] * 2),
    ]:
        ts = [
            ttnn.from_torch(
                torch.randn(s), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc[b]
            )
            for s, b in zip(shapes, bufs)
        ]
        p = g6_concat_plan(ts, dim, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(
            "RES",
            name,
            p["num_cores"],
            p["block_pages"],
            p["cb_output_pages_bytes"],
            p["out_l1_per_bank"],
            p["l1_total"],
            ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks,
            flush=True,
        )
finally:
    ttnn.close_device(device)
