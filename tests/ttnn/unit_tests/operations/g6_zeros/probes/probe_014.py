import collections
import ttnn
from ttnn.operations.g6_zeros import describe_plan
from ttnn.operations.g6_zeros.g6_zeros_program_descriptor import plan_bank_runs

device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    cases = {
        "p2": ((16, 12, 128, 128), ttnn.float32),
        "p3": ((8, 12, 128, 128), ttnn.float32),
        "p1": ((32831, 1, 32, 256), ttnn.bfloat8_b),
        "p4": ((1, 4, 256, 128), ttnn.bfloat16),
    }
    for name, (shape, dtype) in cases.items():
        d = describe_plan(shape, dtype=dtype, device=device)
        writers = {k: v for k, v in d["roles"].items() if v != "idle"}
        slots = collections.Counter()
        for v in writers.values():
            for r in v:
                slots[int(r.split("core_slots=")[1].rstrip(")"))] += 1
        print(
            f"case {name}: pages={d['num_pages']} N={d['N']} wpb={d['writers_per_bank']} writers={len(writers)} idle={len(d['roles'])-len(writers)} slots_hist={dict(slots)} ranges={d['core_ranges']}"
        )
    # exact coverage of [0, num_pages) for every production case and a few ragged counts
    grid = device.compute_with_storage_grid_size()
    for num_pages in (3072, 1536, 262648, 128, 1, 5, 8, 9, 103, 105, 1000003):
        plan = plan_bank_runs(num_pages, 8, 4096, grid)
        seen = []
        for blocks in plan.blocks:
            for run in blocks:
                seen.extend(run.first_page_id + j * 8 for j in range(run.core_slots))
                assert all((run.first_page_id + j * 8) % 8 == run.bank for j in range(run.core_slots))
        ok = sorted(seen) == list(range(num_pages))
        print(("OK" if ok else "FAIL"), num_pages, "writers", len(plan.cores))
finally:
    ttnn.close_mesh_device(device)
