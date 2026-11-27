# Plan math for the four production cases + check the cached descriptor's address patch sticks.
import ttnn
from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm
from ttnn.operations.g6_zeros import g6_zeros_plan

dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    grid, nb = pdm.device_geometry(dev)
    print("grid", grid, "banks", nb)
    for shape, dt in [
        ((32831, 1, 32, 256), ttnn.bfloat8_b),
        ((16, 12, 128, 128), ttnn.float32),
        ((8, 12, 128, 128), ttnn.float32),
        ((1, 4, 256, 128), ttnn.bfloat16),
    ]:
        plan = g6_zeros_plan(shape, dtype=dt, device=dev)
        maxruns = max(len(c.runs) for c in plan.cores)
        # every page covered exactly once
        seen = set()
        for c in plan.cores:
            for r in c.runs:
                for k in range(r.first_slot, r.first_slot + r.slots):
                    pid = k * nb + r.bank
                    assert pid < plan.tensor_pages and pid not in seen
                    seen.add(pid)
        assert len(seen) == plan.tensor_pages
        print(
            shape,
            dt,
            "P",
            plan.tensor_pages,
            "cores_used",
            plan.cores_used,
            "zsrc",
            plan.zero_src_bytes,
            "max runs/core",
            maxruns,
            "core0",
            plan.cores[0].q_count,
            plan.cores[0].runs[:2],
        )
    # odd page counts on a 12-bank-like split: bank math check host-side
    for P in [1, 5, 8, 9, 13, 100, 1001]:
        for b in [8, 12]:
            runs = pdm.bank_runs(0, P, P, b)
            ids = sorted(k * b + r.bank for r in runs for k in range(r.first_slot, r.first_slot + r.slots))
            assert ids == list(range(P)), (P, b)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([8, 12, 128, 128]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = pdm.make_plan((8, 12, 128, 128), ttnn.float32, grid, nb)
    d = pdm.create_program_descriptor(out, plan)
    print("before", d.kernels[0].common_runtime_args[0], hex(out.buffer_address()))
    d.kernels[0].common_runtime_args[0] = 12345
    print("after patch", d.kernels[0].common_runtime_args[0])
finally:
    ttnn.close_mesh_device(dev)
