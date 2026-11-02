"""Iteration 47 scoping, part 2: the cost of switching sub-device managers inside a step. The big matmuls need all
110 cores, the overlap window needs the 77 + 33 split, so a dynamic scheme loads the split manager around each
collective and the full manager after it. Measures, on the 1x4 mesh, per iteration: (a) full-grid matmul alone;
(b) the overlap window alone (split loaded once: rs + two 77-core matmuls); (c) the dynamic sequence
[load split; rs; mm77; mm77; load full; mm110] against (d) the same ops serial on the full manager with the
reduce-scatter on all cores; also a bare switch pair with no ops."""
import time
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
all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
compute_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 4))])
ccl_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, grid.y - 3), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
compute_grid = ttnn.CoreGrid(y=grid.y - 3, x=grid.x)
full_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)


def dev(a):
    return ttnn.from_torch(
        torch.from_numpy(a),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps


rng = np.random.default_rng(0)
x = dev(rng.normal(0, 1, (1, 1, 1024, 5120)).astype(np.float32))
a = dev(rng.normal(0, 1, (1, 1, 1024, 5120)).astype(np.float32))
w = dev(rng.normal(0, 0.02, (1, 1, 5120, 1024)).astype(np.float32))
a2 = dev(rng.normal(0, 1, (1, 1, 1024, 1536)).astype(np.float32))
w2 = dev(rng.normal(0, 0.02, (1, 1, 1536, 5120)).astype(np.float32))
big_a = dev(rng.normal(0, 1, (1, 1, 1024, 5184)).astype(np.float32))
big_w = dev(rng.normal(0, 0.02, (1, 1, 5184, 16384)).astype(np.float32))  # a gate_up-size matmul on 110 cores


def make_rs(crs, sub_device_id, workers):
    sems = [ttnn.create_global_semaphore(device, crs, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(device, crs, 0)
    return lambda: ttnn.experimental.reduce_scatter_minimal_async(
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


mm77 = lambda: ttnn.matmul(a, w, compute_kernel_config=cfg, core_grid=compute_grid)
mm77b = lambda: ttnn.matmul(a2, w2, compute_kernel_config=cfg, core_grid=compute_grid)
mm110 = lambda: ttnn.matmul(big_a, big_w, compute_kernel_config=cfg, core_grid=full_grid)

rs_plain = make_rs(all_cores, None, None)
t_rs = timed(rs_plain)
t_77 = timed(mm77)
t_77b = timed(mm77b)
t_110 = timed(mm110)
t_serial = timed(lambda: (rs_plain(), mm77(), mm77b(), mm110()))
print(
    f"no manager: rs {t_rs:.0f}, mm77 {t_77:.0f}, mm77b {t_77b:.0f}, mm110 {t_110:.0f}; serial {t_serial:.0f} us",
    flush=True,
)

split = device.create_sub_device_manager([ttnn.SubDevice([compute_crs]), ttnn.SubDevice([ccl_crs])], 0)
full = device.create_sub_device_manager([ttnn.SubDevice([all_cores])], 0)


def host(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=2)).float()


rs_ref = host(rs_plain())
device.load_sub_device_manager(split)
device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
rs_sd = make_rs(ccl_crs, ttnn.SubDeviceId(1), 4)  # the semaphores created under the loaded split manager
t_rs_sd = timed(rs_sd)
t_window = timed(lambda: (rs_sd(), mm77(), mm77b()))
t_window2 = timed(lambda: (mm77(), rs_sd(), mm77b()))
err = float((host(rs_sd()) - rs_ref).abs().max())
print(
    f"split manager loaded once: rs on its sub-device {t_rs_sd:.0f} us (max|diff| {err:.3g}), window (rs + mm77 + mm77b) "
    f"{t_window:.0f} / (mm77 + rs + mm77b) {t_window2:.0f} us (serial parts {t_rs_sd + t_77 + t_77b:.0f})",
    flush=True,
)
device.reset_sub_device_stall_group()
device.load_sub_device_manager(full)
device.load_sub_device_manager(split)
device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
err2 = float((host(rs_sd()) - rs_ref).abs().max())
t_window3 = timed(lambda: (rs_sd(), mm77(), mm77b()))
print(f"after a switch cycle (full, split): rs max|diff| {err2:.3g}, window {t_window3:.0f} us", flush=True)
device.reset_sub_device_stall_group()


def dynamic():
    device.load_sub_device_manager(split)
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    rs_sd()
    mm77()
    mm77b()
    device.reset_sub_device_stall_group()
    device.load_sub_device_manager(full)
    mm110()


def switches_only():
    device.load_sub_device_manager(split)
    device.load_sub_device_manager(full)


t_dyn = timed(dynamic)
t_sw = timed(switches_only)
device.load_sub_device_manager(full)
t_full_mgr_serial = timed(lambda: (rs_plain(), mm77(), mm77b(), mm110()))
device.load_sub_device_manager(split)
device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
err3 = float((host(rs_sd()) - rs_ref).abs().max())
device.reset_sub_device_stall_group()
print(
    f"dynamic [load split; rs; mm77; mm77b; load full; mm110] {t_dyn:.0f} us against serial {t_serial:.0f} "
    f"(serial on the full manager {t_full_mgr_serial:.0f}); a bare switch pair {t_sw:.0f} us; rs after the dynamic loop max|diff| {err3:.3g}",
    flush=True,
)
device.clear_loaded_sub_device_manager()
ttml.autograd.AutoContext.get_instance().close_device()
