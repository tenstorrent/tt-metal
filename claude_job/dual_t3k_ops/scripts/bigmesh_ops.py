# SPDX-License-Identifier: Apache-2.0
"""
Big-Mesh demo: run ttnn add + matmul + all_gather across TWO cabled T3Ks as ONE
logical 1x16 MeshDevice (SPMD; 8 chips per host, 16 total).

Launch (from the launcher box, repo root, venv active):

    tt-run --tcp-interface ens18 \
      --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto \
      --hosts t3k-node-a,t3k-node-b \
      python3 claude_job/dual_t3k_ops/scripts/bigmesh_ops.py

Verification model (multi-host!): every MPI rank runs this whole script, but each rank
only physically owns its local half of the mesh, and there is NO host-side way to gather
float data across hosts here (no mpi4py; ttnn only exposes an int host all-gather). So to
compare the FULL, un-sharded tensor we use the on-device fabric all_gather to replicate the
result onto every chip, then copy ONE local chip back to CPU with ttnn.to_torch() -- that
single chip now holds the whole tensor. We then print error metrics (no pass/fail threshold).
"""

import math
import os

import torch
import ttnn

TILE = 32


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
    all_gather). Copy the first LOCAL chip back to CPU -> the full un-sharded torch tensor."""
    locs = local_coords_and_tensors(tt_replicated, tt_replicated.device())
    if not locs:
        raise RuntimeError("no local device tensor to read")
    return locs[0][1]


def report_error(golden_full, out_full, label, rank):
    """Print the error rate between two FULL (un-sharded) tensors. No threshold check."""
    g = golden_full.flatten().to(torch.float32)
    o = out_full.flatten().to(torch.float32)
    diff = (o - g).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_l2 = (diff.norm() / g.norm().clamp_min(1e-12)).item()  # relative L2 "error rate"
    print(
        f"[rank {rank}] {label} full{tuple(out_full.shape)}: "
        f"rel_L2_err={rel_l2 * 100:.4f}%  mean_abs_err={mean_abs:.4e}  "
        f"max_abs_err={max_abs:.4e}  pcc={pcc(golden_full, out_full):.6f}",
        flush=True,
    )


def main():
    torch.manual_seed(0)  # identical inputs on every rank (SPMD)

    # Fabric must be configured BEFORE opening the mesh so cross-host routing works.
    # FABRIC_1D failed to build routes on the 1x16 big mesh spanning two hosts
    # ("Could not find any forwarding direction ..."); FABRIC_2D routes the 1x16 correctly.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    mesh_shape = ttnn.MeshShape(1, 16)
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

    NCOL = 16  # mesh columns (shard factor)

    # -------------------- 1) elementwise ADD (sharded, no comms) --------------
    # Inputs and golden in bf16 (SAME dtype as the device): this makes the check about the
    # multi-device WIRING, not bf16 accuracy. Note: add still re-rounds its result, so CPU vs
    # device can differ by the last bit (~tiny) -- only all_gather (pure move) is exactly 0.
    a = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    b = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    golden_add = a + b
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
    # Gather the dim-3-sharded result onto every chip, then read one chip -> full tensor.
    tadd_full = ttnn.all_gather(tadd, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(device)
    if rank == 0:
        report_error(golden_add, full_tensor_to_cpu(tadd_full), "ADD", rank)

    # -------------------- 2) data-parallel MATMUL ----------------------------
    # A rows sharded across the mesh, B replicated -> each chip does its row block.
    # bf16 inputs (same as device). Golden mirrors Tensix matmul: bf16 in, fp32 ACCUMULATE, bf16
    # out. Matmul can't be exactly 0 error even with same dtype -- CPU vs chip sum the 128 products
    # in a different order (last-bit residual, not a wiring bug).
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
    # Output rows are sharded on dim 2 -> gather dim 2 onto every chip, read one -> full.
    tmm_full = ttnn.all_gather(tmm, dim=2, cluster_axis=1, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(device)
    if rank == 0:
        report_error(golden_mm, full_tensor_to_cpu(tmm_full), "MATMUL", rank)

    # -------------------- 3) CCL: ALL_GATHER (cross-chip comms) --------------
    # bf16 input (same as device). all_gather only MOVES data (no math), so with matching dtype
    # a correct multi-device gather should match the CPU golden EXACTLY (rel_L2_err = 0.0000%).
    x = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    golden_x = x  # after all_gather along the shard dim, every chip holds full x
    tx = ttnn.from_torch(
        x,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    tg = ttnn.all_gather(tx, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(device)
    if rank == 0:
        report_error(golden_x, full_tensor_to_cpu(tg), "ALL_GATHER", rank)

    ttnn.distributed_context_barrier()

    if rank == 0:
        print("=" * 60, flush=True)
        print("[rank 0] Big-Mesh: error rates printed above (full un-sharded tensors)", flush=True)
        print("=" * 60, flush=True)

    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
