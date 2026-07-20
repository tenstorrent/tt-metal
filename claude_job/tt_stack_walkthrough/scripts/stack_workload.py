# SPDX-License-Identifier: Apache-2.0
"""
Big-Mesh stack-proof workload: open ONE logical 1x16 MeshDevice across two cabled
T3Ks (8 chips/host, 16 total) and exercise the full TT software stack end-to-end --
compute (add, matmul) + the three CCL collectives (all_gather, all_reduce,
reduce_scatter) -- each checked against a torch golden with a PASS/FAIL PCC gate.

This is the runnable proof cited by STACK.md. Every layer below is touched by simply
running this script under tt-run:
  * tt-topology  : per-chip EthCoords flashed on the cards, read back by UMD (preflight
                   test_system_health confirms the mesh is fully connected).
  * tt-fabric    : set_fabric_config(FABRIC_2D) builds the ControlPlane + routing tables.
  * TTFM         : DEFAULT mode here (Metal owns fabric init+teardown); the ENABLED-mode
                   variant (attach to a run_fabric_manager fabric) is the live demo.
  * tt-run       : launches this same file on both hosts, rank->{mesh_id,host_rank}.
  * ttnn         : open_mesh_device + Shard/Replicate mappers + the ops below.

Launch (from the launcher box, repo root, venv active, env per prompt.md):

tt-run --tcp-interface ens18 \
    --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto \
    --hosts t3k-node-a,t3k-node-b \
    python3 /home/namvu/dual-t3k/tt-metal/claude_job/tt_stack_walkthrough/scripts/stack_workload.py

Verification model (multi-host!): every MPI rank runs this whole script, but each rank
only physically owns its local 8 chips. There is no host-side float gather across hosts,
so we verify PER LOCAL SHARD: ttnn.get_device_tensors() filtered to this rank's local
coords, each compared to the matching torch-golden slice. ANY rank that fails raises ->
non-zero exit, so a clean rank-0 "ALL PASS" + exit 0 means all 16 chips agreed with torch.

Env var to override the pass threshold for the bf16-reround collectives: STACK_PCC_MIN.
"""

import math
import os

import torch
import ttnn

TILE = 32
NCOL = 16  # mesh columns == chips along cluster_axis=1 of the (1,16) mesh

# all_gather is a pure data move -> must match exactly. add/matmul/all_reduce/
# reduce_scatter re-round in bf16 (or sum in a different order) -> expect ~0.999x.
PCC_EXACT = 1.0 - 1e-6
PCC_MIN = float(os.environ.get("STACK_PCC_MIN", "0.99"))
# Tolerances for the reported torch.allclose bonus check (bf16-appropriate). Override via env.
ATOL_CHECK = float(os.environ.get("STACK_ATOL", "2e-2"))
RTOL_CHECK = float(os.environ.get("STACK_RTOL", "2e-2"))


# --- tiny standalone PCC (avoid pulling in repo test deps) -------------------
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


def error_metrics(golden: torch.Tensor, out: torch.Tensor) -> dict:
    """Full error picture for one (golden, out) pair (flattened, fp32):
    pcc      Pearson correlation           — the shape/wiring check
    mse      mean squared error            — mean((out - golden)^2)
    atol     max ABSOLUTE error            — max|out - golden|            (the atol needed to pass allclose)
    rtol     max RELATIVE error            — max(|out - golden| / |golden|) (the rtol needed)
    allclose torch.allclose(out, golden, rtol=RTOL_CHECK, atol=ATOL_CHECK) — bonus bf16-tolerance bool
    """
    g = golden.flatten().to(torch.float32)
    o = out.flatten().to(torch.float32)
    diff = (o - g).abs()
    return {
        "pcc": pcc(g, o),
        "mse": (diff * diff).mean().item(),
        "atol": diff.max().item(),
        "rtol": (diff / g.abs().clamp_min(1e-9)).max().item(),
        "allclose": bool(torch.allclose(o, g, rtol=RTOL_CHECK, atol=ATOL_CHECK)),
    }


def local_coords_and_tensors(tt_tensor, mesh_device):
    """Return list of (linear_index_within_full_mesh, local_device_torch_tensor)."""
    coords = list(tt_tensor.tensor_topology().mesh_coords())
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_tensor)
    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [c for c in coords if view.is_local(c)]
    out = []
    for coord, dev_t in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        out.append((coord_to_index[coord], ttnn.to_torch(dev_t)))
    return out


def full_tensor_to_cpu(tt_replicated):
    """tt_replicated must be REPLICATED (every chip holds the whole tensor, e.g. after
    all_gather). Copy the first LOCAL chip back to CPU -> the full un-sharded tensor."""
    locs = local_coords_and_tensors(tt_replicated, tt_replicated.device())
    if not locs:
        raise RuntimeError("no local device tensor to read")
    return locs[0][1]


def check_full(golden_full, out_full, label, rank, threshold):
    """Compare a FULL (all_gather-replicated) tensor to its golden on rank 0. Raise on fail."""
    m = error_metrics(golden_full, out_full)
    ok = m["pcc"] >= threshold
    if rank == 0:
        status = "PASS" if ok else "FAIL"
        print(
            f"[rank 0] {status:4s} {label:13s} full{tuple(out_full.shape)}: "
            f"pcc={m['pcc']:.6f} (>= {threshold:.6f})  mse={m['mse']:.3e}  "
            f"atol(max_abs)={m['atol']:.3e}  rtol(max_rel)={m['rtol']:.3e}  "
            f"allclose(rtol={RTOL_CHECK:g},atol={ATOL_CHECK:g})={m['allclose']}",
            flush=True,
        )
    if not ok:
        raise AssertionError(f"[rank {rank}] {label} FAILED: pcc={m['pcc']:.6f} < {threshold:.6f}")


def check_local_shards(tt_tensor, golden_for_index, label, rank, threshold):
    """Verify each LOCAL device shard against golden_for_index(linear_index). Raise on any
    fail (on ANY rank -> non-zero exit). rank 0 prints the summary (min pcc over its shards)."""
    locs = local_coords_and_tensors(tt_tensor, tt_tensor.device())
    if not locs:
        raise RuntimeError(f"{label}: no local device tensors")
    worst = 1.0
    worst_idx = locs[0][0]
    goldens, outs = [], []
    for idx, dev_t in locs:
        g = golden_for_index(idx)
        p = pcc(g, dev_t)
        if p < worst:
            worst, worst_idx = p, idx
        goldens.append(g.flatten().to(torch.float32))
        outs.append(dev_t.flatten().to(torch.float32))
        if p < threshold:
            raise AssertionError(
                f"[rank {rank}] {label} FAILED at chip index {idx}: pcc={p:.6f} < {threshold:.6f} "
                f"(shard shape {tuple(dev_t.shape)} vs golden {tuple(g.shape)})"
            )
    if rank == 0:
        # min_pcc is the worst single shard; mse/atol/rtol/allclose are over ALL local shards pooled.
        m = error_metrics(torch.cat(goldens), torch.cat(outs))
        print(
            f"[rank 0] PASS {label:13s} {len(locs)} local shards, each{tuple(locs[0][1].shape)}: "
            f"min_pcc={worst:.6f} (>= {threshold:.6f}) at chip {worst_idx}  mse={m['mse']:.3e}  "
            f"atol(max_abs)={m['atol']:.3e}  rtol(max_rel)={m['rtol']:.3e}  "
            f"allclose(rtol={RTOL_CHECK:g},atol={ATOL_CHECK:g})={m['allclose']}",
            flush=True,
        )


def main():
    torch.manual_seed(0)  # identical inputs on every rank (SPMD)

    # tt-fabric layer: FABRIC_2D builds the cross-host ControlPlane + routing tables BEFORE
    # the mesh opens. FABRIC_1D cannot route the 1x16 mesh spanning two hosts.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # ttnn layer: open the single logical 1x16 mesh; each rank builds its local 8-chip view.
    mesh_shape = ttnn.MeshShape(1, NCOL)
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    # MPI_Init pins torch to 1 thread; restore for the host-side torch goldens.
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError("distributed context not initialized")
    rank = int(ttnn.distributed_context_get_rank())
    size = int(ttnn.distributed_context_get_size())
    ndev = math.prod(tuple(device.shape))
    if rank == 0:
        print(f"[rank 0] opened Big-Mesh {tuple(device.shape)} = {ndev} chips across {size} host(s)", flush=True)
        print("=" * 78, flush=True)

    lin = ttnn.Topology.Linear  # 1x16 is a line (no physical wrap link for Ring)

    # ==================== 1) elementwise ADD (sharded, no comms) ====================
    # bf16 in/out; verify per LOCAL shard: chip i holds a[...,i]+b[...,i].
    a = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    b = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    golden_add = (a + b).to(torch.bfloat16)
    ta = ttnn.from_torch(
        a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    tb = ttnn.from_torch(
        b,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    tadd = ttnn.add(ta, tb)
    ttnn.synchronize_device(device)
    check_local_shards(tadd, lambda i: golden_add[..., i * TILE : (i + 1) * TILE], "ADD", rank, PCC_MIN)
    # Free device + host-CQ resources before the next op (avoids remote SIGBUS from
    # command-queue/pinned-buffer accumulation across many un-freed tensors).
    for t in (ta, tb, tadd):
        ttnn.deallocate(t)
    ttnn.distributed_context_barrier()

    # ==================== 2) data-parallel MATMUL ====================
    # A rows sharded on dim2, B replicated -> chip i computes rows [i*32:(i+1)*32] @ B.
    A = torch.randn(1, 1, TILE * NCOL, 128).to(torch.bfloat16)
    B = torch.randn(1, 1, 128, 128).to(torch.bfloat16)
    golden_mm = (A.to(torch.float32) @ B.to(torch.float32)).to(torch.bfloat16)
    tA = ttnn.from_torch(
        A,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2),
    )
    tB = ttnn.from_torch(
        B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    tmm = ttnn.matmul(tA, tB)
    ttnn.synchronize_device(device)
    check_local_shards(tmm, lambda i: golden_mm[:, :, i * TILE : (i + 1) * TILE, :], "MATMUL", rank, PCC_MIN)
    for t in (tA, tB, tmm):
        ttnn.deallocate(t)
    ttnn.distributed_context_barrier()

    # ==================== 3) CCL: ALL_GATHER (pure move -> exact) ====================
    # shard x on dim3 -> all_gather(cluster_axis=1) replicates full x onto every chip.
    x = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    tg = ttnn.all_gather(tx, dim=3, cluster_axis=1, topology=lin)
    ttnn.synchronize_device(device)
    # every chip now holds full x; check each local shard == full x, and the rank-0 full copy.
    check_local_shards(tg, lambda i: x, "ALL_GATHER", rank, PCC_EXACT)
    check_full(x, full_tensor_to_cpu(tg), "ALL_GATHER(full)", rank, PCC_EXACT)
    for t in (tx, tg):
        ttnn.deallocate(t)
    ttnn.distributed_context_barrier()

    # ==================== 4) CCL: ALL_REDUCE (Sum across the 16 chips) ====================
    # shard r on dim3 -> chip i owns block_i (1,1,32,32). all_reduce(cluster_axis=1) sums the
    # 16 blocks elementwise -> EVERY chip holds S = sum_i block_i. golden = torch sum of blocks.
    r = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    blocks = [r[..., i * TILE : (i + 1) * TILE].to(torch.float32) for i in range(NCOL)]
    S = torch.stack(blocks, 0).sum(0).to(torch.bfloat16)  # (1,1,32,32)
    tr = ttnn.from_torch(
        r,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    tar = ttnn.all_reduce(tr, cluster_axis=1, topology=lin)
    ttnn.synchronize_device(device)
    check_local_shards(tar, lambda i: S, "ALL_REDUCE", rank, PCC_MIN)
    for t in (tr, tar):
        ttnn.deallocate(t)
    ttnn.distributed_context_barrier()

    # ==================== 5) CCL: REDUCE_SCATTER (Sum then scatter) ====================
    # replicate full f (1,1,32,32*16) onto all 16 chips -> reduce_scatter(dim=3,cluster_axis=1)
    # sums across the 16 replicas (= 16*f) then scatters dim3 -> chip i holds slice i of 16*f.
    f = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    scaled = (NCOL * f.to(torch.float32)).to(torch.bfloat16)  # 16 * f
    tf = ttnn.from_torch(
        f, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    trs = ttnn.reduce_scatter(tf, dim=3, cluster_axis=1, topology=lin)
    ttnn.synchronize_device(device)
    check_local_shards(trs, lambda i: scaled[..., i * TILE : (i + 1) * TILE], "REDUCE_SCATTER", rank, PCC_MIN)
    for t in (tf, trs):
        ttnn.deallocate(t)

    ttnn.distributed_context_barrier()

    if rank == 0:
        print("=" * 78, flush=True)
        print(
            "[rank 0] ALL PASS: add, matmul, all_gather, all_reduce, reduce_scatter "
            "verified vs torch golden across all 16 chips",
            flush=True,
        )
        print("=" * 78, flush=True)

    ttnn.distributed_context_barrier()
    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
