"""Iteration 47 scoping: can a reduce-scatter run concurrently with an independent matmul on the same chip?
Two sub-devices (compute: cores x 0-10, y 0-6 = 77 cores; collectives: y 7-9 = 33 cores) on the 1x4 mesh, the
async reduce-scatter on the collective sub-device, a matmul on the compute grid; the wall of (a) each alone,
(b) the pair issued back to back without a sub-device manager (serial), (c) the pair with the manager loaded
(the dispatcher stalls per sub-device only). 20 iterations, bf16, HiFi4 + fp32 accumulate."""
import os, time
import numpy as np
import torch
import ttnn
import ttml

mesh = ttml.Mesh((1, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)
device = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
grid = device.compute_with_storage_grid_size()
print(f"grid {grid.x}x{grid.y}", flush=True)
all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
compute_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 4))])
ccl_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, grid.y - 3), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
compute_grid = ttnn.CoreGrid(y=grid.y - 3, x=grid.x)


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def host(t, concat_dim=2):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=concat_dim)).float()


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


def run(label, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        msg = [l for l in str(e).splitlines() if "FATAL" in l or "rror" in l]
        print(f"{label}: FAILED {(msg[0] if msg else str(e))[:240]}", flush=True)
        return None


rng = np.random.default_rng(0)
x = dev(rng.normal(0, 1, (1, 1, 1024, 5120)).astype(np.float32))  # the reduce-scatter input (per chip)
a = dev(rng.normal(0, 1, (1, 1, 1024, 5120)).astype(np.float32))
w = dev(rng.normal(0, 0.02, (1, 1, 5120, 1024)).astype(np.float32))  # out dX-like [1024,5120]x[5120,1024]
a2 = dev(rng.normal(0, 1, (1, 1, 1024, 1536)).astype(np.float32))
w2 = dev(rng.normal(0, 0.02, (1, 1, 1536, 5120)).astype(np.float32))  # qkv dX-like [1024,1536]x[1536,5120]


def make_rs(crs, sub_device_id, workers):
    sems = [ttnn.create_global_semaphore(device, crs, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(device, crs, 0)

    def rs():
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            None,
            2,
            sems,
            barrier_semaphore=barrier,
            num_links=2,
            topology=ttnn.Topology.Ring,
            subdevice_id=sub_device_id,
            cluster_axis=1,
            num_workers_per_link=workers,
        )

    return rs


def mm():
    return ttnn.matmul(a, w, compute_kernel_config=cfg, core_grid=compute_grid)


def mm2():
    return ttnn.matmul(a2, w2, compute_kernel_config=cfg, core_grid=compute_grid)


# (a) / (b): no sub-device manager
rs_plain = make_rs(all_cores, None, None)
t_rs, rs_ref = run("rs plain", lambda: timed(rs_plain))
t_mm, mm_ref = timed(mm)
t_mm2, mm2_ref = timed(mm2)
t_pair, _ = run("pair plain", lambda: timed(lambda: (rs_plain(), mm(), mm2())))
print(
    f"no sub-devices: reduce-scatter {t_rs:.0f} us, matmul A (77 cores) {t_mm:.0f}, matmul B {t_mm2:.0f}, "
    f"the three back to back {t_pair:.0f} (sum {t_rs + t_mm + t_mm2:.0f})",
    flush=True,
)
rs_plain_full = make_rs(all_cores, None, None)
for workers in (4, 8):
    r = make_rs(all_cores, None, workers)
    t, _ = run(f"rs {workers} workers", lambda: timed(r))
    print(f"  reduce-scatter with {workers} workers per link, no sub-devices: {t:.0f} us", flush=True)
# (c): two sub-devices
compute_sd = ttnn.SubDevice([compute_crs])
ccl_sd = ttnn.SubDevice([ccl_crs])
manager = device.create_sub_device_manager([compute_sd, ccl_sd], 0)
device.load_sub_device_manager(manager)
device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
for workers in (4, 8):
    rs_sd = make_rs(ccl_crs, ttnn.SubDeviceId(1), workers)
    t_rs_sd, rs_out = run(f"rs on sub-device ({workers} workers)", lambda: timed(rs_sd)) or (None, None)
    if t_rs_sd is None:
        continue
    err = float((host(rs_out) - host(rs_ref)).abs().max())
    t_mm_sd, mm_out = timed(mm)
    t_pair_sd, _ = run("pair sub-devices", lambda: timed(lambda: (rs_sd(), mm(), mm2())))
    t_pair_sd2, _ = run("pair sub-devices mm first", lambda: timed(lambda: (mm(), rs_sd(), mm2())))
    print(
        f"sub-devices ({workers} workers per link): reduce-scatter {t_rs_sd:.0f} us (max|diff| vs plain {err:.3g}), "
        f"matmul A {t_mm_sd:.0f}, the three back to back {t_pair_sd:.0f} (rs first) / {t_pair_sd2:.0f} (mm first); "
        f"serial sum {t_rs_sd + t_mm_sd + t_mm2:.0f}",
        flush=True,
    )
device.reset_sub_device_stall_group()
device.clear_loaded_sub_device_manager()
ttml.autograd.AutoContext.get_instance().close_device()
