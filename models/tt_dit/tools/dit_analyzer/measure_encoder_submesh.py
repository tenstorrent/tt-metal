# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Turn the conformed encoder finding into a measured number (galaxy workstream 4).

`conform_encoder.py` proved on silicon that the H3 text encoder's output is bit-for-bit
identical on all 8 SP columns of a 4x8 — the model shards along TP only, so 7 of 8 copies are
computed and discarded. That is a *counted* win: the report says 105-168 MiB per collective.
This measures it.

The fix the finding implies is the one T5 and CLIP already use in this tree
(`tests/encoders/t5/test_t5_full.py`: `mesh_device.create_submesh(...)`): build the encoder on a
submesh that omits the redundant SP columns instead of replicating across them. At production
`sp1tp0` on a 4x8 that submesh is **4x1** — TP=4 intact, one SP column.

Two arms, identical model, identical weights (seeded), identical input:

    full      encoder on the whole 4x8 — 8 SP copies, 7 discarded  (what the pipeline does today)
    submesh   encoder on a 4x1 submesh — 1 copy                    (what the finding proposes)

Correctness and perf come out of the same run: each arm writes its per-device output shards, so
the submesh result can be diffed bit-for-bit against the full-mesh SP-column-0 shards it is meant
to replace. A latency win that changes the output is not a win.

Device time comes from Tracy, not wall clock. Construction and warm-up sit outside the signpost
window; only the measured forwards are inside it.

    python3 -m tracy -p -r -v models/tt_dit/tools/dit_analyzer/measure_encoder_submesh.py --arm full
    python3 -m tracy -p -r -v models/tt_dit/tools/dit_analyzer/measure_encoder_submesh.py --arm submesh
    tt-perf-report --start-signpost start --end-signpost stop <csv>

Shapes are the production text-encoder config (`scouts/scout_h3_pipeline.py`) at the real prompt
length, with one deviation: vocab is shrunk 151936 -> 4096. The embedding is a local gather, not a
collective, and it is identical in both arms — it does not touch what is being compared.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VOCAB = 4096  # * shrunk from 151936 — local embedding gather, identical in both arms
HID = 5120
INTER = 25600
# 40 heads, not the scout's 64: this branch's Qwen3VlTextEncoder derives head_dim = hidden/heads
# rather than taking it as a kwarg, and mrope [16,24,24] sums to 64 = head_dim/2, so head_dim must
# be 128 => 5120/40. Same choice conform_encoder.py makes, for the same reason.
HEADS = 40
KV_HEADS = 8
HEAD_DIM = HID // HEADS  # 128
MROPE = [16, 24, 24]
RMS_EPS = 1e-6
ROPE_THETA = 1e7
SEQ = 512  # production prompt length (l_len)
SEED = 0


def _load_seeded_weights(module, torch) -> int:
    """Same walk as conform_encoder's, but seeded — both arms must get identical weights or the
    bit-for-bit output comparison means nothing."""
    torch.manual_seed(SEED)
    count = 0
    for _name, p in module.named_parameters():
        p.load_torch_tensor(torch.randn(tuple(p.total_shape), dtype=torch.float32))
        count += 1
    for _name, child in module.named_children():
        count += _load_seeded_weights_inner(child, torch)
    module._mark_loaded()  # noqa: SLF001
    return count


def _load_seeded_weights_inner(module, torch) -> int:
    count = 0
    for _name, p in module.named_parameters():
        p.load_torch_tensor(torch.randn(tuple(p.total_shape), dtype=torch.float32))
        count += 1
    for _name, child in module.named_children():
        count += _load_seeded_weights_inner(child, torch)
    return count


def run(arm: str, mesh_shape, tp_axis: int, iters: int, outdir: str, grid: str) -> int:
    import torch

    import ttnn
    from models.tt_dit.encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils import matmul as mm_utils
    from models.tt_dit.utils.tensor import float32_tensor

    # A confound, not a detail: get_matmul_core_grid clamps to 11x10 only when the mesh has
    # >= 32 devices (a BH Galaxy power constraint). A 4x1 submesh has 4, so it would silently
    # take the full 12x10 grid — 20% more cores — and inflate the submesh arm for a reason that
    # has nothing to do with the redundancy finding. `--grid clamped` pins 11x10 everywhere by
    # lowering the threshold (the function reads this global at call time, so it takes effect
    # even though linear.py bound the function by name); `--grid auto` leaves today's behaviour.
    if grid == "clamped":
        mm_utils._BH_GALAXY_MIN_DEVICES = 1  # noqa: SLF001

    try:
        from tracy import signpost
    except ImportError:  # running without the profiler wrapper

        def signpost(_name):
            return None

    sp_axis = 1 - tp_axis
    tp_f = mesh_shape[tp_axis]

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    dev = None
    try:
        # The pipeline holds the whole mesh either way; the arms differ only in what the encoder
        # is built on. A 4x1 submesh keeps TP=4 and drops the 7 redundant SP columns.
        if arm == "submesh":
            sub_shape = [0, 0]
            sub_shape[tp_axis], sub_shape[sp_axis] = tp_f, 1
            dev = parent.create_submesh(ttnn.MeshShape(*sub_shape))
        else:
            dev = parent
        print(
            "arm=%s  parent mesh %s  encoder on %s  grid=%s (%s)"
            % (arm, tuple(parent.shape), tuple(dev.shape), grid, mm_utils.get_matmul_core_grid(dev))
        )

        ccl = CCLManager(mesh_device=dev, num_links=1, topology=ttnn.Topology.Ring)
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
            device=dev,
            parallel_config=pconf,
            ccl_manager=ccl,
        )
        n = _load_seeded_weights(enc, torch)
        print("built encoder: TP=%d(axis%d), %d params (seeded), seq=%d" % (tp_f, tp_axis, n, SEQ))

        # ---- construction: weight upload + activation prep, deliberately outside the window ----
        torch.manual_seed(SEED + 1)
        ids = ttnn.from_torch(
            torch.randint(0, VOCAB, (1, SEQ), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
        )
        pos = (
            torch.arange(SEQ, dtype=torch.float32)[None, None, :, None]
            / (ROPE_THETA ** (torch.arange(HEAD_DIM, dtype=torch.float32) / HEAD_DIM))[None, None, None, :]
        )
        cos = float32_tensor(torch.cos(pos), device=dev)
        sin = float32_tensor(torch.sin(pos), device=dev)

        out = enc.forward(ids, pos_embeds=(cos, sin))[0]  # warm-up: populates the program cache
        ttnn.synchronize_device(dev)

        # ---- the measured window ----
        signpost("start")
        for _ in range(iters):
            out = enc.forward(ids, pos_embeds=(cos, sin))[0]
        ttnn.synchronize_device(dev)
        signpost("stop")
        try:
            ttnn.ReadDeviceProfiler(dev)
        except Exception:  # noqa: BLE001
            pass

        shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(out)]
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "encoder_out_%s.pt" % arm)
        torch.save({"arm": arm, "mesh": tuple(dev.shape), "shards": shards}, path)
        print("wrote %d output shards to %s" % (len(shards), path))
        print("iters in window: %d — divide the report's total device time by this for per-forward" % iters)
    finally:
        # Close the child before the parent: closing the parent first throws
        # "MeshDevice cq ID 0 is in use by child submesh ID 1", which aborts during teardown
        # after the measurement has already succeeded.
        if dev is not None and dev is not parent:
            ttnn.close_mesh_device(dev)
        ttnn.close_mesh_device(parent)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["full", "submesh"], required=True)
    ap.add_argument("--mesh", type=int, nargs=2, default=[4, 8])
    ap.add_argument("--tp-axis", type=int, choices=[0, 1], default=0, help="production sp1tp0 => TP on axis 0")
    ap.add_argument("--iters", type=int, default=2, help="forwards inside the signpost window")
    ap.add_argument("--outdir", default="/tmp/ditcheck_w4")
    ap.add_argument(
        "--grid",
        choices=["auto", "clamped"],
        default="clamped",
        help="clamped pins the 11x10 BH-Galaxy grid in both arms so the A/B isolates the finding",
    )
    args = ap.parse_args()
    raise SystemExit(run(args.arm, args.mesh, args.tp_axis, args.iters, args.outdir, args.grid))


if __name__ == "__main__":
    main()
