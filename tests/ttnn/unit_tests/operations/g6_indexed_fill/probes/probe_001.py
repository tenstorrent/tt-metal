import torch, ttnn
from ttnn.operations.g6_indexed_fill import describe_plan, PLAN_CAPS

device = ttnn.open_device(device_id=0)
try:

    def dev(t):
        return ttnn.from_torch(
            t.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    for shapes in [((1,), (7, 1, 1, 32), (1, 1, 1, 32)), ((2,), (5, 2, 1, 64), (2, 2, 1, 64))]:
        ids = dev(torch.zeros(shapes[0], dtype=torch.int64))
        x = dev(torch.zeros(shapes[1], dtype=torch.int64))
        s = dev(torch.zeros(shapes[2], dtype=torch.int64))
        p = describe_plan(ids, x, s)
        roles = p.pop("roles")
        print("PLAN", shapes, p)
        for r in roles:
            print("ROLE", r)
finally:
    ttnn.close_device(device)
