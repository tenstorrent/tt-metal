"""Device test for engine/argblock.py (spec v1.25 arg blocks). Standalone (no model): opens a 1x8 mesh (MK_MESH_N) or
chip 0 (MK_MESH=0), builds an 'ar' block over the full compute grid with per-device, per-core distinct payloads, runs the
probe kernel on every core of the grid, and checks every device's every row. Also checks check_cb_ids.
  python engine/test_argblock.py      (PYTHONPATH = tt-metal root; under flock + timeout)"""
import os, sys, torch, ttnn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argblock as AB

KD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
MESH = os.environ.get("MK_MESH", "1") == "1"
if MESH:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, int(os.environ.get("MK_MESH_N", "8"))))
else:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    g = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    cores = ttnn.corerange_to_cores(crs, row_wise=True)
    ndev = mesh.get_num_devices()
    NPAY = 11  # payload words (row index first) -> width 16
    per_dev = [{(c.x, c.y): [i] + [(d << 24) | (c.x << 16) | (c.y << 8) | j for j in range(1, NPAY)] for i, c in enumerate(cores)}
               for d in range(ndev)]
    blocks = AB.ArgBlocks(mesh)
    b = blocks.add("ar", crs, per_dev)
    print(f"[argblock] grid {g.x}x{g.y} ({len(cores)} cores) x {ndev} devices, width {b['width']} words, L1 addr {b['addr']}", flush=True)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    out = ttnn.from_torch(torch.zeros(len(cores), 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    rt = ttnn.RuntimeArgs()
    for c in cores:
        rt[c.x][c.y] = [out.buffer_address()]
    cta = [AB.KINDS["ar"]] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
    k = ttnn.KernelDescriptor(kernel_source=f"{KD}/mk_argblock_probe.cpp", core_ranges=crs, compile_time_args=cta,
                              runtime_args=rt, common_runtime_args=blocks.common_args(),
                              config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
    cb = ttnn.CBDescriptor(total_size=128, core_ranges=crs,
                           format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=24, data_format=ttnn.uint32, page_size=128)])
    desc = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    AB.check_cb_ids(desc)
    ttnn.generic_op([out] + blocks.tensors(), desc)
    ttnn.synchronize_device(mesh)
    bad = 0
    for d, dt in enumerate(ttnn.get_device_tensors(out)):
        got = (ttnn.to_torch(dt).to(torch.int64) & 0xFFFFFFFF).reshape(len(cores), 32)
        for i, c in enumerate(cores):
            want = [1, AB.KINDS["ar"], NPAY, 1] + per_dev[d][(c.x, c.y)]
            if got[i, :len(want)].tolist() != want:
                bad += 1
                if bad < 5: print(f"[argblock] MISMATCH dev {d} core ({c.x},{c.y}) got {got[i, :len(want)].tolist()} want {want}")
    try:
        AB.check_cb_ids(ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[ttnn.CBDescriptor(
            total_size=128, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=45, data_format=ttnn.uint32, page_size=128)])]))
        cb_ok = False
    except ValueError:
        cb_ok = True
    print(f"[argblock] {'PASS' if bad == 0 and cb_ok else 'FAIL'}: {ndev} devices x {len(cores)} cores, mismatches {bad}, cb-id guard {cb_ok}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
