#!/usr/bin/env python3
# Per-op HOST-only (no device sync): legacy ttnn.X vs migrated ttnn.experimental.quasar.X (spec).
# elapsed/N amortized, min-of-reps; small N so the CQ doesn't backpressure on the slow emulator.
import os, sys, time, torch
sys.path.insert(0, "ttnn")
import ttnn

MODE = sys.argv[1] if len(sys.argv) > 1 else "single"   # "single" | "mesh"
TAG = sys.argv[2] if len(sys.argv) > 2 else MODE
print("ttnn module:", ttnn.__file__, "MODE", MODE, flush=True)
N_WARM, N_ITER, N_REP = 5, 200, 5

def host_us(op):
    # Run inside graph-capture NO_DISPATCH: full host dispatch path executes (create_program_spec,
    # run-args, validation, output-alloc) but NOTHING is pushed to the device -> no command-queue
    # backpressure, clean for every op on any backend. Amortized elapsed/N, min over reps. The
    # constant graph-capture overhead cancels in before/after deltas.
    op()  # warm/build the program cache
    best = float("inf")
    for _ in range(N_REP):
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            op()
        dt = (time.perf_counter() - t0) / N_ITER * 1e6
        ttnn.graph.end_graph_capture()
        best = min(best, dt)
    return best

def main():
    if MODE == "mesh":
        n = ttnn.get_num_devices(); dev = ttnn.open_mesh_device(ttnn.MeshShape(1, n))
        print(f"MESH devices={n}", flush=True); kw = dict(device=dev, mesh_mapper=ttnn.ReplicateTensorToMesh(dev))
    else:
        dev = ttnn.open_device(device_id=0); kw = dict(device=dev)
    torch.manual_seed(0); q = ttnn.experimental.quasar
    def run(name, legacy, spec):
        for which, op in (("legacy", legacy), ("spec", spec)):
            try:
                print(f"RESULT\t{TAG}\t{name}\t{which}\t{host_us(op):.2f}", flush=True)
            except Exception as e:
                print(f"EXC\t{TAG}\t{name}\t{which}\t{repr(e)[:110]}", flush=True)
    try:
        t = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16)
        xT = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, **kw)
        run("transpose", lambda: ttnn.transpose(xT, -2, -1), lambda: q.transpose(xT, -2, -1))
        xR = ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, **kw)
        run("tilize", lambda: ttnn.tilize(xR), lambda: q.tilize(xR))
        tf = torch.randn(1, 32, 32, 32, dtype=torch.bfloat16)
        xf = ttnn.from_torch(tf, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, **kw)
        run("fold", lambda: ttnn.fold(xf, 2, 2), lambda: q.fold(xf, 2, 2))
        tu = torch.randn(1, 1, 40, 96, dtype=torch.bfloat16)
        xu = ttnn.from_torch(tu, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        ue = ttnn.Shape([0, 0, 39, 95])
        run("untilize", lambda: ttnn.untilize_with_unpadding(xu, ue, use_multicore=True), lambda: q.untilize_with_unpadding(xu, ue, use_multicore=True))
        tsl = torch.randn(2, 4, 64, 128, dtype=torch.bfloat16)
        xsl = ttnn.from_torch(tsl, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, **kw)
        run("slice", lambda: ttnn.slice(xsl, [0,1,8,16],[2,3,56,112],[1,1,1,1]), lambda: q.slice(xsl, [0,1,8,16],[2,3,56,112],[1,1,1,1]))
        H, W = 256, 128
        tr = torch.randn(1, 1, H, W, dtype=torch.bfloat16)
        inm = ttnn.create_sharded_memory_config(shape=(H, W), core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)
        om = ttnn.create_sharded_memory_config(shape=(H, W), core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)
        xs = ttnn.from_torch(tr, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=inm, **kw)
        run("reshard", lambda: ttnn.reshard(xs, om), lambda: q.reshard(xs, om))
        xi = ttnn.from_torch(tr, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, **kw)
        run("i2s", lambda: ttnn.to_memory_config(xi, om), lambda: q.to_memory_config(xi, om))
    finally:
        os._exit(0)

if __name__ == "__main__":
    main()
