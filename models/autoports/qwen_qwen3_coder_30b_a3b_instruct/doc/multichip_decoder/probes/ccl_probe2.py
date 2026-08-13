# CCL per-op cost via trace-slope (removes the ~57us host dispatch floor).
import statistics
import sys
import time

import torch

import ttnn

mode = sys.argv[1]
TOPO = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}[mode]
FAB = {"ring": ttnn.FabricConfig.FABRIC_1D_RING, "linear": ttnn.FabricConfig.FABRIC_1D}[mode]

ttnn.set_fabric_config(FAB)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000, l1_small_size=32768)
N = 4
cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
sems = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(8)]
barrier = ttnn.create_global_semaphore(mesh, cores, 0)


def timed_trace(fn, reps, iters=30):
    o = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps):
        out = fn()
        ttnn.deallocate(out)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    return statistics.median(ts)


def slope(fn, r1=1, r2=17):
    a = timed_trace(fn, r1)
    b = timed_trace(fn, r2)
    return (b - a) / (r2 - r1), a, b


def make(shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.randn(*shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def run(kind, shape, nlinks, dtype=ttnn.bfloat16):
    x = make(shape, dtype)
    if kind == "ag":
        f = lambda: ttnn.experimental.all_gather_async(
            x,
            dim=3,
            multi_device_global_semaphore=sems[0:2],
            barrier_semaphore=barrier,
            num_links=nlinks,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPO,
        )
    else:
        f = lambda: ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=sems[0:3],
            barrier_semaphore=barrier,
            num_links=nlinks,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPO,
        )
    s, a, b = slope(f)
    ttnn.deallocate(x)
    return s, a, b


CASES = [
    ("b32_h2048", 32, 2048),
    ("s128_h2048", 128, 2048),
    ("s512_h2048", 512, 2048),
    ("s2048_h2048", 2048, 2048),
]
print(f"P|topology {mode}")
for name, rows, width in CASES:
    for nl in (1, 2):
        try:
            s_ag, a, b = run("ag", (1, 1, rows, width // N), nl)
            in_b = rows * (width // N) * 2
            print(f"P|AG {name} links={nl} per_dev_in={in_b}B per_op={s_ag:.2f}us r1={a:.1f} r17={b:.1f}")
        except Exception as e:
            print(f"P|AG {name} links={nl} ERR {str(e)[:150]}")
        try:
            s_rs, a, b = run("rs", (1, 1, rows, width), nl)
            in_b = rows * width * 2
            print(f"P|RS {name} links={nl} per_dev_in={in_b}B per_op={s_rs:.2f}us r1={a:.1f} r17={b:.1f}")
        except Exception as e:
            print(f"P|RS {name} links={nl} ERR {str(e)[:150]}")

ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
