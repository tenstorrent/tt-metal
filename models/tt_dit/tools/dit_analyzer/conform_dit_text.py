# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Conform the DiT **text-branch** `replicated_stage` findings on hardware (phase 11, galaxy workstream 2).

The whole-pipeline report flags four collectives in the DiT's text path — the token refiner and
the text attention/FFN (`attention_minimax_h3.py:336/419`, `token_refiner_minimax_h3.py:119`) —
as `replicated_stage`: each carries a 512-row tensor (the text stream), gathers along **TP**, and
is *replicated across all 8 SP positions while consumed on 1*. Together they are ~477 MiB of link
traffic per forward, of which 7/8 is discarded.

The claim rests on two physical facts, and this harness proves both:

A) **The text rows live entirely in SP shard 0.** The packed sequence is [text|audio|video] padded
   to sp_factor*32; at 4x8 that is 38400 over SP=8 = 4800 rows/shard, so text [0:512) sits inside
   the first shard. Only SP column 0's copy of the text branch is ever packed into the sequence —
   which is what makes the other seven copies dead. (Same mechanism as the audio `unused_gather`
   already conformed in `conform_dit_heads.py`, one modality over.)

B) **Every SP column computes the identical text result.** The text branch shards along TP and does
   nothing along SP, so for each TP row the 8 SP columns must be bit-for-bit equal. If they are,
   running the branch on a single SP column (rather than all 8 and discarding 7) is exact.

Weight *values* can't affect either fact — both follow from where the model shards — so the layers
carry random weights. Shapes are the real ones (text 512, hidden 5376, TP-sharded hidden).

    python3 models/tt_dit/tools/dit_analyzer/conform_dit_text.py --mesh 4 8     # Galaxy, ring (default)
    python3 models/tt_dit/tools/dit_analyzer/conform_dit_text.py --mesh 2 4 --topology linear
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXT = 512  # text stream length (l_len) — the rows the whole finding is about
AUDIO = 414  # audio rows, for reconstructing the real packed layout in check A
VIDEO = 37296  # production 768p/5s video rows
HID = 5376  # DiT hidden width
FFN = 14336  # refiner/text FFN width


def _load_random_weights(module, torch, _top=True) -> int:
    count = 0
    for _name, p in module.named_parameters():
        p.load_torch_tensor(torch.randn(tuple(p.total_shape), dtype=torch.float32))
        count += 1
    for _name, child in module.named_children():
        count += _load_random_weights(child, torch, _top=False)
    if _top:
        module._mark_loaded()  # noqa: SLF001
    return count


def run(mesh_shape, topo: str = "ring") -> int:
    import torch

    import ttnn
    from models.tt_dit.layers.linear import Linear
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor

    # SP is the larger axis (the production H3/LTX sp1tp0 layout on a 4x8).
    sp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
    tp_axis = 1 - sp_axis
    sp_f, tp_f = mesh_shape[sp_axis], mesh_shape[tp_axis]

    # A ring topology needs a ring fabric (see conform_encoder.py).
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING if topo == "ring" else ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    a_ok = b_ok = False
    try:
        topology = ttnn.Topology.Ring if topo == "ring" else ttnn.Topology.Linear
        ccl = CCLManager(mesh_device=mesh, num_links=1, topology=topology)

        # ---- A) the text rows land inside SP shard 0 of the packed sequence ----
        # Real packing: [text|audio|video] padded up to a multiple of sp_factor * TILE.
        align = sp_f * 32
        packed = -(-(TEXT + AUDIO + VIDEO) // align) * align
        shard = packed // sp_f
        # A row-identified sequence: row i carries the value i, so a shard boundary is visible.
        ident = torch.arange(packed, dtype=torch.float32).view(1, 1, packed, 1).expand(1, 1, packed, 32).contiguous()
        seq_sharded = bf16_tensor(ident, device=mesh, mesh_axis=sp_axis, shard_dim=2)
        dev0 = ttnn.to_torch(ttnn.get_device_tensors(seq_sharded)[0]).float()
        # device 0 must hold rows [0, shard) — and the text block must be inside it
        holds_prefix = dev0.shape[2] == shard and torch.equal(
            dev0[0, 0, :, 0], torch.arange(shard, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        )
        a_ok = holds_prefix and TEXT <= shard
        print(
            "A) text rows confined to SP shard 0  (SP=%d, packed %d -> %d/shard, text [0:%d))"
            % (sp_f, packed, shard, TEXT)
        )
        print(
            "   shard 0 holds rows [0:%d): %s   text ⊂ shard 0: %s"
            % (shard, "OK" if holds_prefix else "MISMATCH", TEXT <= shard)
        )
        print(
            "   => %s: only SP column 0's text result is packed into the sequence; %d of %d copies are dead\n"
            % ("CONFIRMED" if a_ok else "REFUTED", sp_f - 1, sp_f)
        )
        ttnn.deallocate(seq_sharded)

        # ---- B) every SP column computes the identical text result ----
        # The text hidden is TP-sharded on the hidden dim and replicated across SP — exactly the
        # `shard(dim3,tp), replicated(sp)` layout the flagged nodes carry. The AGMM's fused gather
        # makes it TP-whole, then the branch projects it.
        ff = Linear(HID, FFN, bias=True, mesh_device=mesh)
        _load_random_weights(ff, torch)
        text_hidden = torch.randn(1, 1, TEXT, HID, dtype=torch.float32)
        text_tp = bf16_tensor(text_hidden, device=mesh, mesh_axis=tp_axis, shard_dim=3)
        text_full = ccl.all_gather_persistent_buffer(text_tp, dim=3, mesh_axis=tp_axis)
        out = ff(text_full)
        ttnn.synchronize_device(mesh)

        per_dev = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(out)]
        assert len(per_dev) == sp_f * tp_f, "expected %d shards, got %d" % (sp_f * tp_f, len(per_dev))

        def dev_index(sp: int, tp: int) -> int:
            coord = [0, 0]
            coord[sp_axis], coord[tp_axis] = sp, tp
            return coord[0] * mesh_shape[1] + coord[1]

        spread = per_dev[0].std().item()
        worst = 0.0
        print("B) text branch replicated across SP  (TP=%d, %d -> %d)" % (tp_f, HID, FFN))
        for r in range(tp_f):
            row_worst = 0.0
            a = per_dev[dev_index(0, r)]
            for c in range(1, sp_f):
                row_worst = max(row_worst, (a - per_dev[dev_index(c, r)]).abs().max().item())
            worst = max(worst, row_worst)
            print(
                "   TP row %d:  max|Δ| across SP columns = %.3g  %s" % (r, row_worst, "EXACT" if row_worst == 0 else "")
            )
        b_ok = spread > 0 and worst == 0.0
        print(
            "   within-output std %.4g (non-degenerate)\n   => %s: %d of %d SP copies of the text branch are redundant\n"
            % (spread, "CONFIRMED" if b_ok else "REFUTED", sp_f - 1, sp_f)
        )
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    print("SHIM BELIEVED: the DiT text branch runs on all %d SP columns and is read on 1." % sp_f)
    if a_ok and b_ok:
        print("DEVICE CONFIRMS BOTH — the text rows sit in shard 0 (A) and every SP copy is bit-identical (B).")
        print("  → run the text branch on a single-SP-column submesh rather than replicating and discarding.")
        return 0
    print("DEVICE REFUTES: A=%s B=%s — investigate as a shim bug before trusting the class." % (a_ok, b_ok))
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[2, 4])
    ap.add_argument(
        "--topology",
        choices=["ring", "linear"],
        default="ring",
        help="collective topology; production H3 runs ring (the 2x4 Loudbox used linear)",
    )
    args = ap.parse_args()
    raise SystemExit(run(args.mesh, args.topology))


if __name__ == "__main__":
    main()
