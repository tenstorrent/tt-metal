# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo: fused_rms_minimal on a NON-RECTANGULAR shard grid is correct or garbage
depending ONLY on what was in L1 before the op ran. Relates to tt-metal #45175,
tt-xla #5738.

Background
----------
The fused rms_allgather stats reduction iterates over the shard grid's *bounding-box
rectangle* (num_cores_x * num_cores_y). If the shard grid is not a full rectangle, the
extra "phantom" cells in the bounding box are read by the reduction even though no shard
data was written to them. Their contribution to E[x^2] is therefore whatever happened to
be in their L1:
  * fresh device  -> phantom cells are 0    -> contribute 0 -> result CORRECT
  * non-fresh L1  -> phantom cells hold junk -> corrupt E[x^2] -> result WRONG
This L1-history dependence is exactly why the bug is non-deterministic: it hides on a
freshly reset device and only bites inside a running model (80 layers of prior ops).

This demo makes the causation explicit on ONE freshly-reset device, in one process:
  1. FRESH  : run the non-rect grid and a rectangular 8x8 grid       -> both correct
  2. PRIME  : write large partials into all 66 bbox cells (incl. the
              2 phantom cells (9,5),(10,5)) via a full-bbox norm
  3. STALE  : run the SAME two grids again
                - non-rect  -> CORRUPT  (reads the primed phantom cells)
                - rect 8x8  -> still correct (has no phantom cells)

Run on a freshly reset device (tt-smi -r first), on Blackhole (>=11-wide worker grid):
  python .../rms_allgather_stale_l1_demo.py

Grid geometry (llama-3.1-70B TP decode layout, #45175):
  non-rect: [(0,0)-(10,4)] (55) + [(0,5)-(8,5)] (9) = 64 cores; bbox 11x6=66 -> 2 phantom
  rect    : [(0,0)-(7,7)]                            = 64 cores; bbox 8x8=64  -> 0 phantom
"""

import torch
import ttnn

torch.manual_seed(1234)
NUM_DEV = 2
SEQ_LEN = 32
EPS = 9.99999974e-6


def torch_rms(x, g, eps):
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return ((xf * torch.rsqrt(v + eps)) * g.float()).to(torch.bfloat16)


def pcc(a, b):
    a = torch.nan_to_num(a.flatten().float())
    b = torch.nan_to_num(b.flatten().float())
    va, vb = a - a.mean(), b - b.mean()
    d = (va.norm() * vb.norm()).item()
    return (torch.dot(va, vb).item() / d) if d else float("nan")


def rel_l2(got, ref):
    return (torch.nan_to_num(got.float()) - ref.float()).norm().item() / ref.float().norm().item()


def run_norm(mesh, grid, hidden, grid_wh, fill=None):
    """One fused_rms_minimal on `grid` with program grid `grid_wh`. `fill`: None=random,
    'perrow'=per-row wildly different magnitude (for priming)."""
    per_dev_tiles = (hidden // NUM_DEV) // 32
    blk = per_dev_tiles // grid.num_cores()
    in_mem = ttnn.create_sharded_memory_config(
        shape=(32, blk * 32), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    ln = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_wh, subblock_w=1, block_h=1, block_w=blk, inplace=False)
    sem = ttnn.create_global_semaphore(mesh, grid, 0)
    ag = ttnn.create_sharded_memory_config(
        shape=(32, 32), core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    m2d = lambda d: ttnn.ShardTensor2dMesh(mesh, dims=(d, None), mesh_shape=list(ttnn.MeshShape(NUM_DEV, 1)))
    stats = ttnn.from_torch(torch.zeros([1, 1, 32, NUM_DEV], dtype=torch.bfloat16), device=mesh,
        layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ag, mesh_mapper=m2d(3))
    if fill == "perrow":
        scale = torch.tensor([10.0 ** (((r % 20) - 6)) for r in range(SEQ_LEN)]).reshape(1, 1, SEQ_LEN, 1)
        x = torch.randn((1, 1, SEQ_LEN, hidden)) * scale
    else:
        x = torch.randn((1, 1, SEQ_LEN, hidden))
    g = torch.randn((1, 1, 1, hidden))
    tin = ttnn.as_tensor(x, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT, memory_config=in_mem, mesh_mapper=m2d(3))
    tg = ttnn.as_tensor(g.reshape([1, 1, hidden // 32, 32]), dtype=ttnn.bfloat16, device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=m2d(2))
    out = ttnn.fused_rms_minimal(tin, ln, 0, mesh, sem, topology=ttnn.Topology.Linear, memory_config=in_mem,
        epsilon=EPS, dtype=ttnn.bfloat16, weight=tg, residual_input_tensor=None, stats=stats, use_noc1_only=False)
    ttnn.synchronize_device(mesh)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3, 0), mesh_shape=(NUM_DEV, 1)))[0].unsqueeze(0)
    return got, torch_rms(x, g, EPS)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    rows = []
    try:
        ccl = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
        sdm = mesh.create_sub_device_manager([ttnn.SubDevice([ccl])], 0)
        mesh.load_sub_device_manager(sdm)
        mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

        HID = 64 * 2 * 32 * NUM_DEV  # 64 cores, block_w=2 -> per-device 128 tiles
        nonrect = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),  # 55
            ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),   #  9  -> 64, bbox 11x6=66
        })
        rect = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})  # 64, bbox 8x8=64
        fullbbox = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 5))})  # 66 cores

        def measure(tag, grid, grid_wh):
            got, ref = run_norm(mesh, grid, HID, grid_wh)
            rows.append((tag, pcc(got, ref), rel_l2(got, ref), got.abs().max().item(), ref.abs().max().item()))

        # 1) FRESH L1 (phantom cells are 0 on a just-reset device)
        measure("fresh  | non-rect (11x6, 2 phantom)", nonrect, (11, 6))
        measure("fresh  | rect 8x8 (no phantom)     ", rect, (8, 8))

        # 2) PRIME: write junk into all 66 bbox cells (incl. the 2 phantom cells)
        run_norm(mesh, fullbbox, 66 * 2 * 32 * NUM_DEV, (11, 6), fill="perrow")

        # 3) STALE L1 (phantom cells now hold primed junk)
        measure("stale  | non-rect (11x6, 2 phantom)", nonrect, (11, 6))
        measure("stale  | rect 8x8 (no phantom)     ", rect, (8, 8))

        mesh.reset_sub_device_stall_group()
    finally:
        ttnn.close_mesh_device(mesh)

    print("\n=== fused_rms_minimal: stale-L1 dependence on a non-rectangular grid ===")
    print(f"{'condition':<38} {'PCC':>9} {'rel_l2':>9} {'out_max':>10} {'ref_max':>9}")
    for tag, p, r, dm, rm in rows:
        flag = "  <-- CORRUPT" if (p < 0.99 or r > 0.1) else ""
        print(f"{tag:<38} {p:>9.5f} {r:>9.4g} {dm:>10.4g} {rm:>9.3g}{flag}")
    print("\nRead: on FRESH L1 both grids are correct. After priming, ONLY the non-rectangular")
    print("grid breaks -- because its bounding box contains phantom cells the reduction reads.")
    print("The rectangular grid has no phantom cells, so it is immune regardless of L1 state.")


if __name__ == "__main__":
    main()
