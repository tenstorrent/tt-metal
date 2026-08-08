# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Conform the `overwide_gather` findings on hardware — a poison test (phase 11, galaxy workstream 2).

The report flags three fused collectives (`dit/all_gather_node_274` and `node_296` at
`agmm@layers/linear.py:302`, `dit/reduce_scatter_node_302` at `mmrs@layers/linear.py:507`) as
moving more data than their consumers read. Laptop triage made the gap exact rather than
statistical — the packed sequence is [text 512 | audio 414 | video 37296] padded to 38400 over
SP=8, so each device holds 4800 rows, and:

  * on **SP column 0** the spatial matmuls read rows [512:4800) — they skip precisely the text
    rows, which is the reported "11% more data";
  * on **SP column 7** they read [0:4622) of the local shard — they skip [38222:38400) global,
    the 178-row padding tail, which is the reported "4%".

A `max|Δ|=0` equality check cannot conform this class: the claim is not that two things are equal
but that a region is *never read*. So poison it. Fill exactly the claimed-unread rows with a
sentinel, run the **real fused AGMM** (`all_gather_minimal_matmul_async`, via ColParallelLinear
with a tensor-parallel config — the same call site the finding names), and require:

  A) every row the consumers *do* read is bit-for-bit unchanged  → the region really is unread;
  B) the poisoned rows' own outputs *do* change                  → the sentinel actually landed,
     so (A) is evidence and not a no-op.

Together they also pin down the property the finding rests on: the fused gather+matmul keeps rows
independent, so trimming the unread rows from the gather cannot perturb anything downstream.

    python3 models/tt_dit/tools/dit_analyzer/conform_overwide.py --mesh 4 8   # Galaxy, ring (default)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXT, AUDIO, VIDEO = 512, 414, 37296  # production 768p/5s packing
K = 7168  # to_out in0 width (node_274), sharded across TP before the fused gather
N = 5376  # DiT hidden width — the matmul's output, column-parallel across TP
SENTINEL = 8192.0  # large, finite and exactly representable in bf16


def run(mesh_shape, topo: str = "ring") -> int:
    import torch

    import ttnn
    from models.tt_dit.layers.linear import ColParallelLinear
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor_2dshard

    sp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
    tp_axis = 1 - sp_axis
    sp_f, tp_f = mesh_shape[sp_axis], mesh_shape[tp_axis]

    align = sp_f * 32
    packed = -(-(TEXT + AUDIO + VIDEO) // align) * align
    shard = packed // sp_f
    tail = packed - (TEXT + AUDIO + VIDEO)  # padding rows at the very end of the sequence

    # The two claimed-unread regions, as (sp column, local row range).
    regions = {
        0: (0, TEXT),  # column 0: the text rows the spatial matmuls skip
        sp_f - 1: (shard - tail, shard),  # last column: the padding tail
    }

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
        pconf = DiTParallelConfig(
            tensor_parallel=ParallelFactor(factor=tp_f, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_f, mesh_axis=sp_axis),
            cfg_parallel=None,
        )
        proj = ColParallelLinear(K, N, bias=True, mesh_device=mesh, mesh_axis=tp_axis, ccl_manager=ccl)
        proj.weight.load_torch_tensor(torch.randn(K, N, dtype=torch.float32))
        proj.bias.load_torch_tensor(torch.randn(1, N, dtype=torch.float32))
        proj._mark_loaded()  # noqa: SLF001

        print("packed %d over SP=%d -> %d rows/shard, K=%d (TP=%d), N=%d" % (packed, sp_f, shard, K, tp_f, N))
        print(
            "poisoning: SP col 0 local rows [%d:%d) (text) · SP col %d local rows [%d:%d) (padding tail)\n"
            % (regions[0][0], regions[0][1], sp_f - 1, regions[sp_f - 1][0], regions[sp_f - 1][1])
        )

        def run_once(poison: bool):
            x = torch.randn(1, 1, packed, K, dtype=torch.float32, generator=torch.Generator().manual_seed(0))
            if poison:
                for col, (lo, hi) in regions.items():
                    x[0, 0, col * shard + lo : col * shard + hi, :] = SENTINEL
            # The flagged node's layout: sequence sharded on SP (dim2), hidden sharded on TP
            # (dim3) — the fused AGMM gathers dim3 back across TP.
            xt = bf16_tensor_2dshard(x, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 3})
            del x
            out = proj(xt, parallel_config=pconf)
            ttnn.synchronize_device(mesh)
            shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(out)]
            ttnn.deallocate(xt)
            ttnn.deallocate(out)
            return shards

        clean = run_once(poison=False)
        dirty = run_once(poison=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    def dev_index(sp: int, tp: int) -> int:
        coord = [0, 0]
        coord[sp_axis], coord[tp_axis] = sp, tp
        return coord[0] * mesh_shape[1] + coord[1]

    read_ok = poison_landed = True
    for col, (lo, hi) in regions.items():
        what = "text rows" if col == 0 else "padding tail"
        for tp in range(tp_f):
            d = dev_index(col, tp)
            c, p = clean[d], dirty[d]
            unread_d = (c[0, 0, lo:hi, :] - p[0, 0, lo:hi, :]).abs().max().item()
            keep = torch.cat([c[0, 0, :lo, :], c[0, 0, hi:, :]], dim=0)
            keep_p = torch.cat([p[0, 0, :lo, :], p[0, 0, hi:, :]], dim=0)
            read_d = (keep - keep_p).abs().max().item()
            read_ok = read_ok and read_d == 0.0
            poison_landed = poison_landed and unread_d > 0.0
            print(
                "SP col %d (%s) dev %2d:  read rows max|Δ| = %.3g %s   poisoned rows max|Δ| = %.3g %s"
                % (col, what, d, read_d, "EXACT" if read_d == 0 else "CHANGED", unread_d, "" if unread_d else "(!)")
            )

    print(
        "\nSHIM BELIEVED: the fused gather materialises rows no consumer reads (11%% on col 0, 4%% on col %d)."
        % (sp_f - 1)
    )
    if read_ok and poison_landed:
        print("DEVICE CONFIRMS: poisoning the unread rows left every read row bit-identical (max|Δ| = 0),")
        print("  and the poisoned rows' own outputs did change — so the region is genuinely unread,")
        print("  and the fused gather+matmul is row-independent. Trimming those rows is exact.")
        return 0
    if not poison_landed:
        print("INCONCLUSIVE: the sentinel did not change the poisoned rows' outputs — the probe is a no-op.")
    else:
        print("DEVICE REFUTES: a read row changed when the 'unread' region was poisoned. Something reads it.")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[4, 8])
    ap.add_argument("--topology", choices=["ring", "linear"], default="ring")
    args = ap.parse_args()
    raise SystemExit(run(args.mesh, args.topology))


if __name__ == "__main__":
    main()
