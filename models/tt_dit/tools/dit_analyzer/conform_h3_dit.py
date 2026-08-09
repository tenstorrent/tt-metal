# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Diff the **real H3 DiT's** collective log against the shim's (phase 11b / galaxy workstream 3).

Everything the analyzer says about MiniMax-H3 comes from running the model under a metadata-only
`ttnn`. That is only worth anything if the shim fires the same collectives, on the same axes, with
the same per-device shapes, as the real thing. `conform_block.py` established this for the SD3.5
block on a 2x4; this does it for **H3's own DiT on the Galaxy**, which is the model the findings
are actually about and the one with every fused kernel in play (AGMM, MMRS, ring-joint SDPA).

Two phases, because the shim and real ttnn cannot share a process:

    # A: what the shim believes (no device)
    cd scouts && python3 dump_dit_graph.py 4x8 dit.graph.json

    # B: what the silicon does
    python3 models/tt_dit/tools/dit_analyzer/conform_h3_dit.py --mesh 4 8 --graph scouts/dit.graph.json

The model is built from the same constructor arguments the scout uses, with **random weights**:
which collectives fire is decided by shapes and the parallel config, not by weight values -- the
same argument every conform harness here rests on, and the reason this needs no checkpoint.

The device log is reconciled with the shim's expanded view before comparing: a ring-joint SDPA is
one ttnn call on device but hides two K/V all-gathers, which the dry run emits as separate stages
(`dryrun/fused.py`, conformed by `conform_ring_sdpa.py`).

**Needs a device-capable H3 tree, which does not exist yet.** Two things block it, both found by
trying:

* `TT_METAL_HOME` pointing at the landed H3 worktree makes ttnn JIT-compile its *dispatch kernels*
  from that tree while the libraries come from the analyzer tree's build, and the two have
  diverged (`cq_prefetch.cpp: 'CMDDAT_Q_BLOCKS' was not declared`). Dry runs never noticed because
  they compile no kernels.
* Running H3's model against the *analyzer* tree's `tt_dit` instead is not a workaround: the two
  branches' shared `layers/linear.py` have diverged where it matters, and the H3 DiT dies with
  `ColParallelLinear.forward() got an unexpected keyword argument 'addcmul_a'` -- it needs the
  fused-addcmul epilogue that only exists on the H3 branch.

So this harness needs the H3 branch *built*: an independent clone (a worktree shares submodule
metadata, and initialising umd there would move the analyzer tree's pinned umd out from under its
build), `submodule update --init`, then `./build_metal.sh`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dit_analyzer.conform_block import _LOG  # noqa: E402
from dit_analyzer.conform_block import _expand_fused, _install_logger, _load_random_weights  # noqa: E402

# Must match scouts/scout_h3_pipeline.py exactly, or the diff compares two different models.
PRESETS = {  # preset -> (mesh, sp_axis, tp_axis, text, audio, video)
    "2x4": ((2, 4), 0, 1, 128, 64, 192),
    "4x8": ((4, 8), 1, 0, 512, 256, 1280),
    "prod": ((4, 8), 1, 0, 512, 414, 37296),
}
HID = 5120  # encoder hidden == DiT text_dim
HEADS, HEAD_DIM, HIDDEN, FFN = 56, 128, 5376, 14336


def run(preset: str, mesh_shape, graph_path: str, dit_layers: int, refiner_layers: int) -> int:
    import torch

    import ttnn
    from models.tt_dit.models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor, from_torch

    _mesh, sp_axis, tp_axis, L, n_audio, n_video = PRESETS[preset]
    mesh_shape = list(mesh_shape or _mesh)
    sp_f, tp_f = mesh_shape[sp_axis], mesh_shape[tp_axis]
    seq = L + n_audio + n_video
    align = sp_f * 32
    padded = -(-seq // align) * align

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    try:
        ccl = CCLManager(mesh_device=mesh, num_links=1, topology=ttnn.Topology.Ring)
        pconf = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_f),
            sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_f),
            cfg_parallel=None,
        )
        dit = MiniMaxH3Transformer3DModel(
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            hidden_size=HIDDEN,
            num_layers=dit_layers,
            num_refiner_layers=refiner_layers,
            ffn_dim=FFN,
            in_channels=24,
            audio_in_channels=32,
            patch_size=(1, 2, 2),
            text_dim=HID,
            freq_dim=256,
            time_embed_hidden_dim=HIDDEN,
            time_embed_dim=2688,
            rope_freq_dim=16,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
            mesh_device=mesh,
            ccl_manager=ccl,
            parallel_config=pconf,
            is_fsdp=False,
        )
        n = _load_random_weights(dit, torch)
        print(
            "real H3 DiT on %s (ring): layers=%d refiner=%d, text/audio/video=%d/%d/%d, "
            "packed %d (padded %d) over SP=%d, TP=%d, %d params (random)"
            % (tuple(mesh_shape), dit_layers, refiner_layers, L, n_audio, n_video, seq, padded, sp_f, tp_f, n)
        )

        torch.manual_seed(0)
        # Modality inputs are replicated: they are projected and concatenated into the packed
        # sequence before it is fractured, so every device needs all of them.
        tt_video = bf16_tensor(torch.randn(1, 1, n_video, 96), device=mesh)
        tt_audio = bf16_tensor(torch.randn(1, 1, n_audio, 32), device=mesh)
        tt_prompt = bf16_tensor(torch.randn(1, 1, L, HID), device=mesh)
        tt_timestep = from_torch(torch.rand(1, 1, 2, 1), device=mesh, dtype=ttnn.float32)
        # Per-row metadata covers the padded sequence and is sharded contiguously on SP, the way
        # the model fractures the packed sequence.
        tt_rope_cos = from_torch(
            torch.randn(1, 1, padded, HEAD_DIM), device=mesh, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]
        )
        tt_rope_sin = from_torch(
            torch.randn(1, 1, padded, HEAD_DIM), device=mesh, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]
        )
        tt_adaln = from_torch(
            torch.zeros(1, 1, 1, padded, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, sp_axis],
        )
        tt_tsi = from_torch(
            torch.zeros(1, 1, 1, padded, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, sp_axis],
        )

        _LOG.clear()
        _install_logger(ttnn)  # only the forward is logged; construction is already done
        dit(
            video_1BVC=tt_video,
            audio_1BAC=tt_audio,
            prompt_1BLP=tt_prompt,
            timestep=tt_timestep,
            adaln_indices=tt_adaln,
            timestep_indices=tt_tsi,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
        )
        ttnn.synchronize_device(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    print("\nordered collective log (real hardware), %d calls:" % len(_LOG))
    for e in _LOG[:40]:
        print("  %-12s axis=%s  in=%s  %s" % (e["op"], e["axis"], e["in"], e["src"]))
    if len(_LOG) > 40:
        print("  ... %d more" % (len(_LOG) - 40))

    device_counts = _expand_fused(Counter((e["op"], e["axis"]) for e in _LOG))

    if not graph_path:
        print("\nno --graph given: device log only, nothing diffed.")
        return 0

    from dit_analyzer.dryrun.verify import collectives as dry_collectives
    from dit_analyzer.ir import Graph

    dry = Graph.from_json(open(graph_path).read())
    dry_counts = Counter((op, ax) for op, ax, ext, shp in dry_collectives(dry))

    print("\n%-30s %-10s %s" % ("(op, mesh_axis)", "device", "dry run"))
    keys = sorted(set(device_counts) | set(dry_counts), key=lambda x: (str(x[0]), str(x[1])))
    mismatch = 0
    for key in keys:
        d, s = device_counts.get(key, 0), dry_counts.get(key, 0)
        mismatch += 0 if d == s else 1
        print("%-30s %-10d %-10d %s" % (str(key), d, s, "" if d == s else "<-- MISMATCH"))

    print(
        "\nSHIM BELIEVED: the dry run of this model fires exactly these collectives (%d distinct kinds)."
        % len(dry_counts)
    )
    if not mismatch:
        print("DEVICE CONFIRMS: every (op, mesh_axis) count matches. The scout is faithful for the DiT stage.")
        return 0
    print("DIFFER on %d kind(s) — reconcile as a SHIM BUG before treating any of it as model waste." % mismatch)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="4x8")
    ap.add_argument("--mesh", type=int, nargs=2, default=None)
    ap.add_argument("--graph", default=None, help="dry-run DiT graph JSON from scouts/dump_dit_graph.py")
    ap.add_argument("--dit-layers", type=int, default=1)
    ap.add_argument("--refiner-layers", type=int, default=1)
    args = ap.parse_args()
    raise SystemExit(run(args.preset, args.mesh, args.graph, args.dit_layers, args.refiner_layers))


if __name__ == "__main__":
    main()
