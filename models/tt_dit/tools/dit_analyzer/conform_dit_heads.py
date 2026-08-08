# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Conform the DiT output-head redundancy findings on the 2x4 Loudbox (phase 11 / 10c).

The MiniMax-H3 DiT output head (transformer_minimax_h3.py:405-419) gathers the packed
[text|audio|video] hidden across TP, projects it to the two narrow heads, gathers those
across SP, and slices each modality out. ditcheck flags two redundancies there; this harness
conforms both against real ttnn collectives, on a sequence/hidden sharded as the DiT shards it.

A) node_360 -- audio `unused_gather`. The audio rows are the contiguous block
   [text_len : text_len+audio_len) = [512:926), which lives entirely in the *first* SP shard,
   and the audio decode reads device 0. So the SP all_gather's contribution from every other
   SP shard is never read by the audio consumer. Verified with a row-identified sequence.

B) node_348 -- output-head `participant_shrink`. proj_out / audio_proj_out are plain Linear
   (replicated weights, transformer_minimax_h3.py:306-307). After the TP all_gather makes the
   hidden replicated across TP, both projections compute bit-identically on every TP device, so
   only one TP row's result is ever read -- the TP all_gather delivers to 4 when 1 would do.
   Verified by running the real projections on the gathered hidden and diffing the TP rows.

    python3 models/tt_dit/tools/dit_analyzer/conform_dit_heads.py --mesh 4 8              # Galaxy, ring (default)
    python3 models/tt_dit/tools/dit_analyzer/conform_dit_heads.py --mesh 2 4 --topology linear   # 2x4 Loudbox
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXT, AUDIO = 512, 414
A_LO, A_HI = TEXT, TEXT + AUDIO  # audio rows [512, 926)
SEQ = 4096  # packed sequence for check A (text+audio+video, padded)
HID = 5376  # DiT hidden width
VIDEO_CH, AUDIO_CH = 96, 32  # proj_out / audio_proj_out output widths


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

    # SP shards the sequence (larger axis, so more shards show as wasted); TP is the other axis.
    sp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
    tp_axis = 1 - sp_axis
    sp_f, tp_f = mesh_shape[sp_axis], mesh_shape[tp_axis]

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
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
        devs = list(range(mesh_shape[0] * mesh_shape[1]))

        # ---- A) node_360: the audio SP all_gather is unused (audio ⊂ shard 0) ----
        shard = SEQ // sp_f
        assert A_HI <= shard, "audio must fit in the first SP shard"
        full = torch.arange(SEQ, dtype=torch.float32).view(1, 1, SEQ, 1).expand(1, 1, SEQ, AUDIO_CH).contiguous()
        seq_sharded = bf16_tensor(full, device=mesh, mesh_axis=sp_axis, shard_dim=2)
        pre0 = ttnn.to_torch(ttnn.get_device_tensors(seq_sharded)[0]).float()
        gathered = ccl.all_gather_persistent_buffer(seq_sharded, dim=2, mesh_axis=sp_axis)
        ttnn.synchronize_device(mesh)
        g0 = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
        ref = torch.arange(SEQ, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        gather_ok = g0.shape[2] == SEQ and torch.equal(g0[0, 0, :, 0], ref)
        d_audio = (g0[:, :, A_LO:A_HI, :] - pre0[:, :, A_LO:A_HI, :]).abs().max().item()
        a_ok = gather_ok and d_audio == 0.0 and A_HI <= shard
        print(
            "A) node_360 audio unused_gather  (SP=%d, seq %d -> %d/shard, audio [%d:%d))"
            % (sp_f, SEQ, shard, A_LO, A_HI)
        )
        print(
            "   gather reassembles: %s   audio slice == device-0 pre-gather: max|Δ|=%.3g   audio ⊂ shard 0: %s"
            % ("OK" if gather_ok else "MISMATCH", d_audio, A_HI <= shard)
        )
        print(
            "   => %s: %d of %d SP shards never read for audio\n" % ("CONFIRMED" if a_ok else "REFUTED", sp_f - 1, sp_f)
        )

        # ---- B) node_348: the output head is replicated across TP (only 1 TP row read) ----
        proj = Linear(
            HID, VIDEO_CH, bias=True, mesh_device=mesh
        )  # plain Linear == the DiT's proj_out (replicated weight)
        aproj = Linear(HID, AUDIO_CH, bias=True, mesh_device=mesh)  # == audio_proj_out
        _load_random_weights(proj, torch)
        _load_random_weights(aproj, torch)
        hidden = torch.randn(1, 1, 64, HID, dtype=torch.float32)  # small seq; replication is per-row
        hidden_tp = bf16_tensor(hidden, device=mesh, mesh_axis=tp_axis, shard_dim=3)  # TP-fractured hidden
        hidden_full = ccl.all_gather_persistent_buffer(
            hidden_tp, dim=3, mesh_axis=tp_axis
        )  # node_348: replicate over TP
        video_all = proj(hidden_full)
        audio_all = aproj(hidden_full)
        ttnn.synchronize_device(mesh)

        def rows(t):
            return [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)]

        vids, auds = rows(video_all), rows(audio_all)
        spread = vids[0].std().item()
        # every device must hold the identical projection (⊇ identical across each TP row group)
        worst = max((vids[d] - vids[0]).abs().max().item() for d in devs)
        worst = max(worst, max((auds[d] - auds[0]).abs().max().item() for d in devs))
        b_ok = spread > 0 and worst == 0.0
        print(
            "B) node_348 output-head participant_shrink  (TP=%d, proj_out %d / audio_proj_out %d)"
            % (tp_f, VIDEO_CH, AUDIO_CH)
        )
        print(
            "   within-output std %.4g (non-degenerate)   max|Δ| of projection across all devices = %.3g"
            % (spread, worst)
        )
        print(
            "   => %s: the projection is identical on every TP row; %d of %d TP copies are never read"
            % ("CONFIRMED" if b_ok else "REFUTED", tp_f - 1, tp_f)
        )
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    print("\nSHIM BELIEVED: audio SP-gather unused (A); output head TP-replicated, 1 row read (B).")
    if a_ok and b_ok:
        print(
            "DEVICE CONFIRMS BOTH. Fixes: slice audio on device 0 (A); gather-to-subset / TP-submesh for the head (B)."
        )
        return 0
    print("DEVICE REFUTES: A=%s B=%s — investigate before trusting." % (a_ok, b_ok))
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
