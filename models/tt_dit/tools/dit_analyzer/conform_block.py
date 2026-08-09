# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Whole-block collective log on hardware, diffed against the dry run (phase 11b).

Runs the real SD3.5-large joint `TransformerBlock` forward on the 2x4 Loudbox with
a lightweight collective logger monkeypatched over ttnn's collective calls, then
compares the collectives it actually fires -- op, mesh axis, in call order --
against the shim's dry-run graph for the same block. Weight *values* do not affect
which collectives run, so the block is loaded with random weights (no checkpoint,
no diffusers).

A fused ring-joint SDPA is one ttnn call on device but hides two K/V all-gathers
over the sp axis; the dry run expands it into those stages (see dryrun/fused.py),
so the log is reconciled the same way before the counts are compared.

    # 1) on a laptop (same box), dump the shim's graph:
    ditcheck dryrun sd35_block --preset bh_2x4 --out sd35.graph.json
    # 2) on the device broker:
    python3 .../conform_block.py --mesh 2 4 --graph sd35.graph.json
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections import Counter
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SD3.5-large, matching dryrun/targets.py sd35_block (SP=2 axis0, TP=4 axis1).
DIM, HEADS, HEAD_DIM, S, P = 2432, 38, 64, 4096, 352

_LOG: List[Dict[str, Any]] = []


def _src_line() -> str:
    for fr in reversed(traceback.extract_stack()):
        if "models/tt_dit/" in fr.filename and "dit_analyzer" not in fr.filename:
            i = fr.filename.index("models/tt_dit/")
            return "%s:%d" % (fr.filename[i:], fr.lineno)
    return "?"


def _per_device_shape(ttnn, t) -> Tuple[int, ...]:
    try:
        return tuple(int(d) for d in ttnn.get_device_tensors(t)[0].shape)
    except Exception:  # noqa: BLE001
        return tuple(int(d) for d in t.shape)


def _install_logger(ttnn):
    """Monkeypatch the ttnn collective calls to append to _LOG, then call through."""
    exp, xf = ttnn.experimental, ttnn.transformer

    def make(mod, name, canon):
        real = getattr(mod, name)

        def logged(*a, **k):
            inp = a[0] if a else k.get("input_tensor")
            _LOG.append(
                {
                    "op": canon,
                    "axis": k.get("cluster_axis", k.get("mesh_axis")),
                    "src": _src_line(),
                    "in": _per_device_shape(ttnn, inp) if inp is not None else None,
                }
            )
            return real(*a, **k)

        setattr(mod, name, logged)

    make(exp, "all_gather_async", "all_gather")
    make(exp, "reduce_scatter_minimal_async", "reduce_scatter")
    if hasattr(exp, "all_gather_minimal_matmul_async"):
        make(exp, "all_gather_minimal_matmul_async", "agmm")
    if hasattr(exp, "minimal_matmul_strided_reduce_scatter_async"):
        make(exp, "minimal_matmul_strided_reduce_scatter_async", "mmrs")
    make(xf, "ring_joint_scaled_dot_product_attention", "ring_sdpa")
    if hasattr(xf, "joint_scaled_dot_product_attention"):
        make(xf, "joint_scaled_dot_product_attention", "sdpa")


def _load_random_weights(module, torch, _top=True) -> int:
    """Load every Parameter from a random tensor of its declared total_shape.

    Same recursive traversal as the dry run's load_meta_weights (named_parameters
    is non-recursive; children are walked explicitly), but with real data on the
    device -- collectives do not depend on weight values, only on shapes.
    """
    count = 0
    for _name, p in module.named_parameters():
        p.load_torch_tensor(torch.randn(tuple(p.total_shape), dtype=torch.float32))
        count += 1
    for _name, child in module.named_children():
        count += _load_random_weights(child, torch, _top=False)
    if _top:
        module._mark_loaded()  # noqa: SLF001
    return count


def _expand_fused(counts: Counter) -> Counter:
    """Reconcile the device log with the shim's expanded view.

    A fused kernel is ONE ttnn call on device but performs a collective inside it, which the dry
    run emits as a separate stage (dryrun/fused.py). Comparing raw call names would report a
    mismatch on every fused op, so expand each into the collective it hides -- the same table the
    shim expands, kept in step with it:

        agmm       -> all_gather   over its cluster axis, before the matmul
        mmrs       -> reduce_scatter over its cluster axis, after the matmul
        ring_sdpa  -> two all_gathers (K and V) over the ring axis

    Only the collective is counted; the matmul half is not a collective and the dry-run side of
    the diff lists collectives only.
    """
    out = Counter()
    for (op, axis), n in counts.items():
        if op == "ring_sdpa":
            out[("all_gather", axis)] += 2 * n  # K and V gathered over the ring axis
        elif op == "agmm":
            out[("all_gather", axis)] += n
        elif op == "mmrs":
            out[("reduce_scatter", axis)] += n
        else:
            out[(op, axis)] += n
    return out


def run(mesh_shape, graph_path) -> int:
    import torch

    import ttnn
    from models.tt_dit.blocks.transformer_block import TransformerBlock
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.padding import PaddingConfig
    from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

    sp_axis, tp_axis = 0, 1
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
        sp_f, tp_f = mesh_shape[sp_axis], mesh_shape[tp_axis]
        ccl = CCLManager(mesh_device=mesh, num_links=1, topology=ttnn.Topology.Linear)
        pconf = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            sequence_parallel=ParallelFactor(factor=sp_f, mesh_axis=sp_axis),
            tensor_parallel=ParallelFactor(factor=tp_f, mesh_axis=tp_axis),
        )
        padding = PaddingConfig.from_tensor_parallel_factor(HEADS, HEAD_DIM, tp_f) if HEADS % tp_f else None
        block = TransformerBlock(
            dim=DIM,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            context_pre_only=False,
            mesh_device=mesh,
            ccl_manager=ccl,
            parallel_config=pconf,
            padding_config=padding,
        )
        n = _load_random_weights(block, torch)
        print("built SD3.5 TransformerBlock on %s, %d parameters loaded (random)" % (tuple(mesh_shape), n))

        spatial = bf16_tensor_2dshard(torch.randn(1, S, DIM), device=mesh, shard_mapping={sp_axis: 1, tp_axis: 2})
        prompt = bf16_tensor(torch.randn(1, P, DIM), device=mesh, mesh_axis=tp_axis, shard_dim=2)
        time_embed = bf16_tensor(torch.randn(1, 1, DIM), device=mesh)

        _install_logger(ttnn)
        block.forward(spatial, prompt, time_embed, spatial_sequence_length=S)
        ttnn.synchronize_device(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    # -- report the ordered log and diff the collective counts --
    print("\nordered collective log (real hardware):")
    for e in _LOG:
        print("  %-12s axis=%s  in=%s  %s" % (e["op"], e["axis"], e["in"], e["src"]))
    device_counts = _expand_fused(Counter((e["op"], e["axis"]) for e in _LOG))

    from dit_analyzer.dryrun.verify import collectives as dry_collectives
    from dit_analyzer.ir import Graph

    dry = Graph.from_json(open(graph_path).read())
    dry_counts = Counter((op, ax) for op, ax, ext, shp in dry_collectives(dry))

    print("\n%-28s %-10s %s" % ("(op, mesh_axis)", "device", "dry run"))
    keys = sorted(set(device_counts) | set(dry_counts), key=lambda x: (x[0], x[1]))
    mismatch = 0
    for key in keys:
        d, s = device_counts.get(key, 0), dry_counts.get(key, 0)
        mismatch += 0 if d == s else 1
        print("%-28s %-10d %-10d %s" % (str(key), d, s, "" if d == s else "<-- MISMATCH"))
    print("\ncollective counts %s" % ("MATCH" if not mismatch else "DIFFER (%d)" % mismatch))
    return mismatch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[2, 4])
    ap.add_argument("--graph", required=True, help="dry-run graph JSON to diff against")
    args = ap.parse_args()
    raise SystemExit(1 if run(args.mesh, args.graph) else 0)


if __name__ == "__main__":
    main()
