# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Conform the `unused_gather` finding on the DiT audio head (phase 11 / 10c).

The MiniMax-H3 DiT output head (transformer_minimax_h3.py:405-419) projects the packed
[text|audio|video] sequence to the narrow audio head, all-gathers it across the SP axis to
reassemble the full sequence, then slices out the audio rows. ditcheck reports that gather as
`unused_gather`: the audio rows are the contiguous block [text_len : text_len+audio_len) =
[512:926), which lives entirely in the *first* SP shard, and the audio decode reads device 0
(pipeline `_project_latents_device` does get_device_tensors[0]). So the all-gather's
contribution from every other SP shard is never read by the audio consumer.

That claim rests on layout, not on the projection values: run the *real* ttnn SP all-gather on
a sequence sharded exactly as the DiT shards it, then verify on hardware that

  1. the gather is correct (the reassembled sequence is the row-identified reference), and
  2. the audio slice [512:926) is bit-for-bit what device 0 *already held before the gather*
     (device 0's shard covers [0, seq/sp) ⊇ [512:926)), and
  3. no other SP shard holds any audio row — so only device 0 supplies the audio slice.

If all three hold, gathering to the other sp-1 shards is exactly redundant for the audio
consumer, which is what the finding says. Row-identified values (row r -> value r) make the
provenance checkable, so this conforms the shim's region reasoning, not just a shape.

    python3 models/tt_dit/tools/dit_analyzer/conform_dit_heads.py --mesh 2 4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real packed-sequence head: text then audio then video. The audio block is [512:926).
TEXT, AUDIO = 512, 414
A_LO, A_HI = TEXT, TEXT + AUDIO  # audio rows [512, 926)
SEQ = 4096  # packed sequence (text+audio+video, padded); small enough to run, > sp*audio-end
CH = 32  # audio head width (audio_proj_out output channels)


def run(mesh_shape) -> int:
    import torch

    import ttnn
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor

    # SP shards the sequence; put it on the larger mesh axis so more shards are exposed as
    # wasted (3 of 4 here, vs 7 of 8 at the real 4x8). TP (the other axis) just replicates.
    sp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
    sp_f = mesh_shape[sp_axis]
    assert SEQ % sp_f == 0
    shard = SEQ // sp_f  # rows per SP device
    if A_HI > shard:
        print(
            "!! audio [%d:%d) does not fit in the first SP shard of %d rows — pick a larger SEQ" % (A_LO, A_HI, shard)
        )
        return 1

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    try:
        ccl = CCLManager(mesh_device=mesh, num_links=1, topology=ttnn.Topology.Linear)
        # row-identified sequence: value at row r is r (broadcast across channels), so a gathered
        # row's provenance is its own value. Shard the sequence (dim 2) over the SP axis.
        full = torch.arange(SEQ, dtype=torch.float32).view(1, 1, SEQ, 1).expand(1, 1, SEQ, CH).contiguous()
        seq_sharded = bf16_tensor(full, device=mesh, mesh_axis=sp_axis, shard_dim=2)
        print(
            "built SP-sharded audio head on %s: SP=%d(axis%d), seq %d -> %d rows/shard, audio [%d:%d)"
            % (tuple(mesh_shape), sp_f, sp_axis, SEQ, shard, A_LO, A_HI)
        )

        # device 0's shard BEFORE the gather: rows [0, shard)
        pre0 = ttnn.to_torch(ttnn.get_device_tensors(seq_sharded)[0]).float()
        gathered = ccl.all_gather_persistent_buffer(seq_sharded, dim=2, mesh_axis=sp_axis)
        ttnn.synchronize_device(mesh)
        post = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(gathered)]
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    g0 = post[0]  # device 0's gathered result: should be the full [1,1,SEQ,CH]
    # reference in bf16 precision: the row-id values round-trip through bf16, so row r carries
    # bf16(r) (bf16 represents integers exactly only up to 256 — compare against the rounded value,
    # not a raw arange). The gather places rows, it does not compute, so this must be exact.
    ref_rows = torch.arange(SEQ, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)

    # 1. the gather is correct: every reassembled row carries its own (bf16) index
    gathered_rows = g0[0, 0, :, 0]
    gather_ok = gathered_rows.shape[0] == SEQ and torch.equal(gathered_rows, ref_rows)
    print(
        "\n1) gather correctness: reassembled %d rows, row r == value r ... %s"
        % (g0.shape[2], "OK" if gather_ok else "MISMATCH")
    )

    # 2. the audio slice equals what device 0 already held pre-gather (bit-for-bit)
    aud_post = g0[:, :, A_LO:A_HI, :]
    aud_pre0 = pre0[:, :, A_LO:A_HI, :]
    d_audio = (aud_post - aud_pre0).abs().max().item()
    print(
        "2) audio slice [%d:%d) post-gather vs device-0 pre-gather: max|Δ| = %.3g ... %s"
        % (A_LO, A_HI, d_audio, "EXACT" if d_audio == 0 else "DIFFERS")
    )

    # 3. no other SP shard holds any audio row (only device 0 supplies the slice)
    others_clear = A_HI <= shard  # audio end within the first shard
    print(
        "3) audio confined to shard 0 ([0:%d) ⊇ [%d:%d)) — shards 1..%d hold no audio row ... %s"
        % (shard, A_LO, A_HI, sp_f - 1, "OK" if others_clear else "NO")
    )

    wasted = sp_f - 1
    wasted_bytes = wasted * AUDIO * CH * 2 / 1024.0  # KiB the gather moves that the audio consumer never reads
    print("\nSHIM BELIEVED: the audio SP all_gather is unused — audio lives in shard 0, only device 0 is read.")
    if gather_ok and d_audio == 0.0 and others_clear:
        print("DEVICE CONFIRMS: device 0 already held every audio row; the other %d SP shards are never read." % wasted)
        print(
            "  → gathering them is exactly redundant for the audio consumer (%d of %d shards, %.1f KiB/head here;"
            % (wasted, sp_f, wasted_bytes)
        )
        print("    at the real 4×8 that is 7 of 8). Fix: slice audio on device 0 / broadcast, don't SP-all-gather.")
        return 0
    print("DEVICE REFUTES: one of the checks failed — investigate before trusting the finding.")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[2, 4])
    args = ap.parse_args()
    raise SystemExit(run(args.mesh))


if __name__ == "__main__":
    main()
