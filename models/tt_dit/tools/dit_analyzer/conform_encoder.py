# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Conform the `replicated_stage` encoder finding on hardware (phase 11 / 10c).

The whole-pipeline analysis reports that the MiniMax-H3 text encoder (a Qwen3-VL
decoder) is **tensor-parallel only**: it shards along the TP axis and does nothing
along the sequence-parallel (SP) axis, so its output is *replicated across the SP
axis*. The DiT handoff then reads that output back and re-shards it on SP — so on a
2x4 mesh (SP=axis0 factor 2, TP=axis1 factor 4), 1 of the 2 SP rows is a bit-for-bit
duplicate that the pipeline composes and discards. The shim BELIEVES this from the
distribution algebra; this harness proves the physical fact it rests on.

The claim reduces to one on-device check: **for each TP column j, the output shard
on device (SP=0, j) equals the shard on device (SP=1, j), bit for bit.** If they are
equal the replication is real and reading one SP row (rather than composing both and
throwing one away) is exact, not an approximation.

Weight *values* don't affect whether the output is SP-replicated — that follows from
where the model shards — so the encoder is loaded with random weights (no checkpoint,
no HF). We keep the real H3 structure that decides the sharding (hidden 5120, 40
heads, head_dim 128, 8 KV heads, mrope [16,24,24], TP=4) and shrink only the two dims
that change compute cost but not distribution: vocab and the MLP intermediate.

    python3 models/tt_dit/tools/dit_analyzer/conform_encoder.py --mesh 4 8              # Galaxy, ring (default)
    python3 models/tt_dit/tools/dit_analyzer/conform_encoder.py --mesh 2 4 --topology linear   # 2x4 Loudbox
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real MiniMax-H3 text-encoder structure that fixes the sharding (and thus the
# SP-replication). head_dim = hidden / heads = 5120 / 40 = 128; mrope [16,24,24]
# sums to 64 = head_dim/2, which is the real config's tell. vocab and intermediate
# are shrunk (marked *) — they scale compute, not the distribution under test.
VOCAB = 4096  # * (real 151936) — embedding table size doesn't change SP replication
HID = 5120
INTER = 13824  # * (real 25600) — MLP width doesn't change SP replication
HEADS = 40
KV_HEADS = 8
HEAD_DIM = HID // HEADS  # 128
MROPE = [16, 24, 24]
RMS_EPS = 1e-6
ROPE_THETA = 1e7
SEQ = 128  # short prompt — the replication is per-token, so a short sequence proves it


def _load_random_weights(module, torch, _top=True) -> int:
    """Load every Parameter from a random tensor of its declared total_shape — the same
    recursive walk load_meta_weights does, but with real data on device. Which devices
    hold identical shards is decided by the mesh mapper, not the values."""
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
    from models.tt_dit.encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import float32_tensor

    sp_axis, tp_axis = 0, 1
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
    try:
        topology = ttnn.Topology.Ring if topo == "ring" else ttnn.Topology.Linear
        ccl = CCLManager(mesh_device=mesh, num_links=1, topology=topology)
        pconf = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_f, mesh_axis=tp_axis))
        enc = Qwen3VlTextEncoder(
            vocab_size=VOCAB,
            hidden_size=HID,
            intermediate_size=INTER,
            hidden_act="silu",
            num_hidden_layers=1,
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            rms_norm_eps=RMS_EPS,
            rope_theta=ROPE_THETA,
            mrope_section=MROPE,
            device=mesh,
            parallel_config=pconf,
            ccl_manager=ccl,
        )
        n = _load_random_weights(enc, torch)
        print(
            "built H3 text encoder on %s: TP=%d(axis%d), SP=%d(axis%d), %d params (random), head_dim=%d"
            % (tuple(mesh_shape), tp_f, tp_axis, sp_f, sp_axis, n, HEAD_DIM)
        )

        # inputs are replicated across the whole mesh (same on every device): a replicated
        # int id sequence and replicated rope cos/sin. Identical inputs + identical (SP-
        # replicated) weights are what make the SP rows' outputs identical.
        ids = ttnn.from_torch(
            torch.randint(0, VOCAB, (1, SEQ), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
        )
        pos = (
            torch.arange(SEQ, dtype=torch.float32)[None, None, :, None]
            / (ROPE_THETA ** (torch.arange(HEAD_DIM, dtype=torch.float32) / HEAD_DIM))[None, None, None, :]
        )
        cos = float32_tensor(torch.cos(pos), device=mesh)  # replicated (no mesh_axis)
        sin = float32_tensor(torch.sin(pos), device=mesh)

        out = enc.forward(ids, pos_embeds=(cos, sin))[0]
        ttnn.synchronize_device(mesh)

        # -- read back every device shard; index in row-major mesh order: sp*tp_f + tp --
        shards = ttnn.get_device_tensors(out)
        assert len(shards) == sp_f * tp_f, "expected %d shards, got %d" % (sp_f * tp_f, len(shards))
        per_dev = [ttnn.to_torch(s).float() for s in shards]
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    dev_shape = tuple(per_dev[0].shape)
    spread = per_dev[0].std().item()
    print("\noutput per-device shard shape %s   (within-shard std %.4g — non-degenerate)" % (dev_shape, spread))
    if spread == 0.0:
        print("!! output is constant — equality across rows would be vacuous. ABORT.")
        return 1

    # -- THE CLAIM: for each TP column, the two SP rows are bit-identical --
    print("\nSP-row equality per TP column  (device %d vs %d ...):" % (0, tp_f))
    worst = 0.0
    for j in range(tp_f):
        a = per_dev[0 * tp_f + j]
        col_worst = 0.0
        for r in range(1, sp_f):
            d = (a - per_dev[r * tp_f + j]).abs().max().item()
            col_worst = max(col_worst, d)
        worst = max(worst, col_worst)
        print("  TP col %d:  max|Δ| across SP rows = %.3g  %s" % (j, col_worst, "EXACT" if col_worst == 0 else ""))

    elem = 1
    for d in dev_shape:
        elem *= d
    wasted_rows = sp_f - 1
    wasted_mib = wasted_rows * elem * 2 / (1024**2)  # bf16 output, one shard per wasted SP row, per TP column
    print(
        "\nSHIM BELIEVED: encoder output replicated across SP axis (%d rows) — %d of %d redundant."
        % (sp_f, wasted_rows, sp_f)
    )
    if worst == 0.0:
        print("DEVICE CONFIRMS: every SP row is bit-for-bit identical (max|Δ| = 0). The finding is REAL.")
        print(
            "  → at this config the duplicated SP rows are %.2f MiB/col of readback the handoff composes and discards;"
            % wasted_mib
        )
        print(
            "    at production seq (512) and full depth the waste scales with both. Fix: run the encoder on a submesh."
        )
        return 0
    print("DEVICE REFUTES: SP rows differ (max|Δ| = %.3g). The replication claim does NOT hold — investigate." % worst)
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
