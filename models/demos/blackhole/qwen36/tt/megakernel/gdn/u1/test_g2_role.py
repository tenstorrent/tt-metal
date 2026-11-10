"""G2 harness: the engine GDN role (g2_role.GdnRole / e1_shim.build_engine) in ONE program with an engine stand-in core
(g2_standin.cpp, the Q root's I/O job: in_ready +1 multicast per GDN layer after `gap` cycles, wait out_ready = 24 (g+1),
reset). Device-local global semaphores in_ready / attn_kv / vb0 / vb1 on the 24 GDN cores, out_ready on the stand-in.
Per step: producers -> the combined program once -> every GDN layer bit-identical vs the stock r13 op. Steps run back
to back with the same program (the semaphores must come back reset). Then trace replays of the combined program
(per-layer edge stamps from the stand-in) for the step span at gap 0 / G2_GAP_US.
env: G2_DEV (0), G2_B (8; 1 = u1), G2_NG (6 GDN layers = NL 8 with 2 attention), G2_STEPS (3; HOLD users 1,5 on step 1),
G2_GAP_US (0), G2_REPS (200)."""
import os, sys, time
import torch, ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g2_role  # noqa: E402

DEVID = int(os.environ.get("G2_DEV", "0"))
B = int(os.environ.get("G2_B", "8"))
NG = int(os.environ.get("G2_NG", "6"))
STEPS = int(os.environ.get("G2_STEPS", "3"))
GAP = float(os.environ.get("G2_GAP_US", "0"))
REPS = int(os.environ.get("G2_REPS", "200"))
HOLD = [u for u in (1, 5) if u < B]
MHZ = 1350

dev = ttnn.open_device(device_id=DEVID, trace_region_size=32 << 20)
grid = dev.compute_with_storage_grid_size()
assert grid.x >= 12 and grid.y >= 10, grid
cores = [ttnn.CoreCoord(x, y) for y in (8, 9) for x in range(12)]
ex = set((c.x, c.y) for c in cores)
QR = ttnn.CoreCoord(0, 0)   # the stand-in (engine Q root = full-grid core 0)
prod = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in ex and (x, y) != (0, 0)]
gcrs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
qcrs = ttnn.CoreRangeSet([ttnn.CoreRange(QR, QR)])
GS = dict(in_ready=ttnn.create_global_semaphore(dev, gcrs, 0), attn_kv=ttnn.create_global_semaphore(dev, gcrs, 0),
          vb0=ttnn.create_global_semaphore(dev, gcrs, 0), vb1=ttnn.create_global_semaphore(dev, gcrs, 0),
          out_ready=ttnn.create_global_semaphore(dev, qcrs, 0))
SA = {k: ttnn.get_global_semaphore_address(v) for k, v in GS.items()}
qp = dev.worker_core_from_logical_core(QR)
print(f"[g2] B {B} NG {NG} sems {SA} out target {(qp.x, qp.y)}", flush=True)

role = g2_role.GdnRole(dev, cores, NG, B, SA, (qp.x, qp.y), prod, os.path.join(os.getcwd(), f"g2gen_b{B}"))
print(f"[g2] role: {len(role.role['kernels'])} kernels, {len(role.role['cbs'])} CBs ids {role.role['info']['cb_ids'][0]}.."
      f"{role.role['info']['cb_ids'][-1]} ({role.role['info']['cb_bytes'] // 1024} KB), argblock {role.role['info']['ab_words']} words",
      flush=True)

# the GDN rects (NoC 0 coordinates; the DRAM column split cuts rows 8-9 in two)
px = [dev.worker_core_from_logical_core(ttnn.CoreCoord(i, 8)).x for i in range(12)]
split = next((i for i in range(1, 12) if px[i] != px[i - 1] + 1), 12)


def prect(x0, x1):
    if x1 < x0:
        return [0, 0, 0, 0, 0]
    a = dev.worker_core_from_logical_core(ttnn.CoreCoord(x0, 8))
    b = dev.worker_core_from_logical_core(ttnn.CoreCoord(x1, 9))
    return [a.x, a.y, b.x, b.y, (x1 - x0 + 1) * 2]


RECTS = prect(0, split - 1) + prect(split, 11)
assert RECTS[4] + RECTS[9] == 24, RECTS
stamps = ttnn.from_torch(torch.zeros(1, 2 * NG + 14, dtype=torch.int32).reshape(1, -1), dtype=ttnn.uint32,
                         layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.MemoryConfig(
                             ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(qcrs, [1, 2 * NG + 14], ttnn.ShardOrientation.ROW_MAJOR)))


def program(gap_us):
    rt = ttnn.RuntimeArgs()
    rt[QR.x][QR.y] = [NG, int(gap_us * MHZ), SA["in_ready"], SA["out_ready"], 24, stamps.buffer_address()] + RECTS
    k = ttnn.KernelDescriptor(kernel_source=f"{HERE}/g2_standin.cpp", core_ranges=qcrs, compile_time_args=[],
                              runtime_args=rt, config=ttnn.DataMovementConfigDescriptor(
                                  processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
    return ttnn.ProgramDescriptor(kernels=list(role.role["kernels"]) + [k], semaphores=[], cbs=list(role.role["cbs"]))


io = list(role.role["io"]) + [stamps]
prog0 = program(0.0)
ok_all = True
for step in range(STEPS):
    hold = HOLD if step == 1 else []
    role.prepare(hold)
    ttnn.generic_op(io, prog0)
    ttnn.synchronize_device(dev)
    ok_all = role.check(f"g2 step {step}") and ok_all
print(f"[g2] BIT-IDENTICAL B={B} over {STEPS} steps x {NG} GDN layers in one engine-style program "
      f"(in_ready / out_ready / vb sems, argblock, HOLD {HOLD} on step 1): {ok_all}", flush=True)


def edges():
    s = ttnn.to_torch(stamps).flatten()[:2 * NG].to(torch.int64) & 0xFFFFFFFF
    d = [((s[2 * g + 1] - s[2 * g]) & 0xFFFFFFFF).item() / MHZ for g in range(NG)]
    return d


print(f"[g2] in->out per GDN layer (single run, gap 0): {['%.1f' % x for x in edges()]} us", flush=True)
for gap in sorted({0.0, GAP}):
    p = prog0 if gap == 0 else program(gap)
    ttnn.generic_op(io, p)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    ttnn.generic_op(io, p)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(REPS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.time() - t0) / REPS * 1e6
    e = edges()
    print(f"[g2] gap {gap:.0f} us: {us:.1f} us per program ({NG} GDN layers), in->out per layer "
          f"{['%.1f' % x for x in e]} us (mean {sum(e) / len(e):.1f})", flush=True)
    ttnn.release_trace(dev, tid)
ttnn.close_device(dev)
print("[g2] done", flush=True)
