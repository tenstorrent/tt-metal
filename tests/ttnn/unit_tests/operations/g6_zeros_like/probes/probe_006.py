import torch, ttnn, collections
from ttnn.operations.g6_zeros_like import describe_plan

device = ttnn.open_device(device_id=0)
try:
    for name, shape, dt, tdt in (
        ("fp32", (1, 12, 128, 128), ttnn.float32, torch.float32),
        ("bf8", (5, 1, 32, 256), ttnn.bfloat8_b, torch.bfloat16),
    ):
        x = ttnn.from_torch(
            torch.zeros(shape, dtype=tdt),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        plan = describe_plan(x)
        print(name, "regime", plan[0]["regime"], "cores", len(plan), "cols", sorted({e["core"][0] for e in plan}))
        by_bank = collections.defaultdict(list)
        for e in plan:
            by_bank[e["bank"]].append(e)
        for b in sorted(by_bank, key=lambda v: (v is None, v)):
            es = by_bank[b]
            print(
                f"  bank {b}: cores={len(es)} rows={sorted({e['core'][1] for e in es})} cols={sorted({e['core'][0] for e in es})} pages={sorted(e['num_pages'] for e in es)}"
            )
finally:
    ttnn.close_device(device)
