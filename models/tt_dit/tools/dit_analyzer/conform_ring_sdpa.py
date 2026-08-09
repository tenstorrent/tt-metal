# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Conform the fused **ring-joint SDPA** expansion on hardware (phase 11b / galaxy workstream 5).

`dryrun/fused.py` declares `ring_joint_scaled_dot_product_attention` as a collective-hiding
kernel that expands into `("all_gather", "sdpa")` over `cluster_axis`. That declaration is
load-bearing for the whole analysis: the fused op is **one** ttnn call that fires **no visible
collective**, so if the shim did not expand it, the K/V gathers it performs internally would be
invisible to the analyzer and a redundant gather inside the kernel could never be flagged. Every
finding on H3's attention rests on that expansion being faithful.

Faithfulness here is not a shape check — it is an equivalence. This harness runs both sides on a
real mesh and compares them:

  A) **the fused kernel** — `ttnn.transformer.ring_joint_scaled_dot_product_attention`, called the
     way `attention_minimax_h3.py:368` calls it (persistent K/V ping-pong buffers, ring topology,
     `cluster_axis` = the SP axis, empty joint inputs, `joint_strategy="rear"`);
  B) **the shim's model of it** — an explicit `all_gather` of K and of V over the same axis
     through the real `CCLManager`, then a plain `scaled_dot_product_attention` against the
     gathered K/V.

If (A) ≈ (B) then modelling the kernel as gather-then-attend is sound, and the two hidden
all-gathers the analyzer reasons about are really there. bf16 with a different reduction order
(ring streaming vs one monolithic attend) will not be bit-exact, so this is a PCC check, unlike
the redundancy harnesses where `max|Δ|=0` is the right bar.

The harness also counts what a naive collective log would see on each path, which is the concrete
demonstration of *why* `fused.py` has to exist: the fused path logs zero collectives while moving
the same K/V across the ring.

H3's real attention geometry: 56 heads over TP, head_dim 128, packed sequence over SP.

    python3 models/tt_dit/tools/dit_analyzer/conform_ring_sdpa.py --mesh 4 8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEADS = 56  # H3 num_attention_heads
HEAD_DIM = 128  # H3 attention_head_dim
N_GLOBAL = 8192  # packed sequence; tile-aligned and divisible by SP (prod is 38400, same shape class)
PCC_BAR = 0.99


def _pcc(a, b) -> float:
    pass

    x, y = a.flatten().float(), b.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).item()
    return 1.0 if denom == 0 else float((x @ y).item() / denom)


def run(
    mesh_shape, topo: str = "ring", n_global: int = N_GLOBAL, q_chunk_size: int = 128, k_chunk_size: int = 256
) -> int:
    import torch

    import ttnn
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

    sp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
    tp_axis = 1 - sp_axis
    sp_f, tp_f = mesh_shape[sp_axis], mesh_shape[tp_axis]
    n_local_heads = HEADS // tp_f
    n_local = n_global // sp_f
    assert HEADS % tp_f == 0, "heads must divide TP"
    assert n_global % (sp_f * 32) == 0, "packed sequence must be tile-aligned per SP shard"

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING if topo == "ring" else ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    try:
        topology = ttnn.Topology.Ring if topo == "ring" else ttnn.Topology.Linear
        ccl = CCLManager(mesh_device=mesh, num_links=1, topology=topology)
        print(
            "ring-joint SDPA conformance on %s (%s): heads %d over TP=%d -> %d/device, "
            "seq %d over SP=%d -> %d/device, head_dim %d"
            % (tuple(mesh_shape), topo, HEADS, tp_f, n_local_heads, n_global, sp_f, n_local, HEAD_DIM)
        )

        # Heads fractured on TP (dim1), sequence fractured on SP (dim2) — H3's attention layout.
        torch.manual_seed(0)
        shape = (1, HEADS, n_global, HEAD_DIM)
        qt, kt, vt = (torch.randn(*shape, dtype=torch.float32) for _ in range(3))
        q = bf16_tensor_2dshard(qt, device=mesh, shard_mapping={tp_axis: 1, sp_axis: 2})
        k = bf16_tensor_2dshard(kt, device=mesh, shard_mapping={tp_axis: 1, sp_axis: 2})
        v = bf16_tensor_2dshard(vt, device=mesh, shard_mapping={tp_axis: 1, sp_axis: 2})

        full_grid = mesh.compute_with_storage_grid_size()
        worker_grid = (full_grid.x - 1, full_grid.y)  # ring SDPA reserves the last column for CCL
        compute_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
        )

        def prog_cfg(seq_len: int, ring: bool) -> ttnn.SDPAProgramConfig:
            # L1 bounds the (q_chunk, k_chunk) product, not just each side: at 14 heads x head_dim
            # 128 the generic (256, 512) rule overflows a Blackhole core's 1.5 MB of CB space
            # ("circular buffers grow to 1641408 B"). attention_minimax_h3's own config docstring
            # makes the same point for the production shapes; here we just pick a pair that fits.
            tile = ttnn.TILE_SIZE
            q_chunk = max(tile, min(q_chunk_size, (seq_len // tile) * tile))
            k_chunk = max(tile, min(k_chunk_size, (seq_len // tile) * tile))
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(
                    ttnn.CoreCoord(*worker_grid) if ring else ttnn.CoreCoord(full_grid.x, full_grid.y)
                ),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )

        # ---- count what a naive collective log sees on each path ----
        seen = {"fused": 0, "expanded": 0}
        phase = ["fused"]
        real_ag = ttnn.experimental.all_gather_async

        def counting_ag(*a, **kw):
            seen[phase[0]] += 1
            return real_ag(*a, **kw)

        ttnn.experimental.all_gather_async = counting_ag
        try:
            # ---- A) the fused kernel, called as attention_minimax_h3 calls it ----
            joint = bf16_tensor(torch.zeros((1, n_local_heads, 0, HEAD_DIM)), device=mesh)
            fused, _joint_out, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                joint,
                joint,
                joint,
                persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(k.shape, 2, sp_axis, dtype=k.get_dtype()),
                persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(v.shape, 2, sp_axis, dtype=v.get_dtype()),
                joint_strategy="rear",
                logical_n=n_global,
                program_config=prog_cfg(n_local, ring=True),
                compute_kernel_config=compute_cfg,
                dim=2,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
                num_links=ccl.num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh,
                topology=ccl.topology,
                subdevice_id=ccl.ccl_sub_device_id,
                ccl_core_grid_offset=(worker_grid[0], 0),
                use_column_major_ccl=True,
            )
            ttnn.synchronize_device(mesh)

            # ---- B) the shim's expansion: all_gather K and V over sp, then plain SDPA ----
            phase[0] = "expanded"
            k_full = ccl.all_gather_persistent_buffer(k, dim=2, mesh_axis=sp_axis)
            v_full = ccl.all_gather_persistent_buffer(v, dim=2, mesh_axis=sp_axis)
            expanded = ttnn.transformer.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                is_causal=False,
                program_config=prog_cfg(n_local, ring=False),
                compute_kernel_config=compute_cfg,
            )
            ttnn.synchronize_device(mesh)
        finally:
            ttnn.experimental.all_gather_async = real_ag

        a_dev = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(fused)]
        b_dev = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(expanded)]
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    print(
        "\ncollectives a naive log would see:  fused path = %d   expanded path = %d" % (seen["fused"], seen["expanded"])
    )
    print(
        "  => the fused call moves K and V around the ring while logging %d collectives, which is why\n"
        "     dryrun/fused.py must expand it — otherwise the analyzer never sees the gather.\n" % seen["fused"]
    )

    worst = 1.0
    degenerate = False
    for d, (a, b) in enumerate(zip(a_dev, b_dev)):
        if a.shape != b.shape:
            print("dev %2d: SHAPE MISMATCH fused %s vs expanded %s" % (d, tuple(a.shape), tuple(b.shape)))
            return 1
        p = _pcc(a, b)
        worst = min(worst, p)
        if a.std().item() == 0.0:
            degenerate = True
        if d < 4 or p < PCC_BAR:
            print("dev %2d: fused vs expanded  PCC = %.6f  %s" % (d, p, "OK" if p >= PCC_BAR else "FAIL"))
    print("... (%d devices checked)" % len(a_dev))

    print("\nSHIM BELIEVED: ring_joint_sdpa == all_gather(K,V over sp) then sdpa (dryrun/fused.py).")
    if degenerate:
        print("INCONCLUSIVE: an output shard is constant — the comparison would be vacuous.")
        return 1
    if worst >= PCC_BAR:
        print("DEVICE CONFIRMS: worst PCC %.6f >= %.2f over all %d devices." % (worst, PCC_BAR, len(a_dev)))
        print("  → the fused kernel and its modelled expansion agree; the two hidden K/V all-gathers are real,")
        print("    so a redundancy the analyzer finds inside this kernel is a redundancy the device performs.")
        return 0
    print(
        "DEVICE REFUTES: worst PCC %.6f < %.2f — the expansion in fused.py does not match the kernel."
        % (worst, PCC_BAR)
    )
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[4, 8])
    ap.add_argument("--topology", choices=["ring", "linear"], default="ring")
    ap.add_argument("--seq", type=int, default=N_GLOBAL, help="logical (unfractured) packed sequence length")
    ap.add_argument("--q-chunk", type=int, default=128, help="SDPA q chunk; L1 bounds q x k")
    ap.add_argument("--k-chunk", type=int, default=256, help="SDPA k chunk; L1 bounds q x k")
    args = ap.parse_args()
    raise SystemExit(run(args.mesh, args.topology, args.seq, args.q_chunk, args.k_chunk))


if __name__ == "__main__":
    main()
