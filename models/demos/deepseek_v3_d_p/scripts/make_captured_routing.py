#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Build an ``expert_routing.safetensors`` for the dispatch/combine perf replay, and rank
which ``(layer, col)`` pairs are worth replaying.

``tests/perf/test_dispatch_combine_perf.py`` replays captured routing through
``TtDispatchModule`` -> ``TtCombineModule`` and asserts per-op device time. The routing it
replays comes from a file of Galaxy-global expert ids, one flat int32 tensor per key
``expert_ids_layer_{N}``, which ``load_captured_routing`` reads as
``view(dispatch_group_size, seq_len_per_chip, num_experts_per_tok)`` — chip-major, so a flat
tensor of ``dispatch_group_size * seq_len_per_chip * top_k`` ids per layer.

Which case a replay exercises is chosen by ``(layer, col)``, not by a separate file: one file
holds every layer, and ``TT_DS_CAPTURED_LAYER`` / ``TT_DS_CAPTURED_COL`` select the case. So
"4 worst + 4 nominal" is one file per model plus an 8-entry pick table.

Column load
-----------
On an 8x4 Galaxy with ``cluster_axis=0`` the dispatch groups run along mesh columns, so column
``c`` owns routed experts ``[c*experts_per_col, (c+1)*experts_per_col)`` with
``experts_per_col = num_routed_experts // 4``:

    DeepSeek V3   256 experts -> 64 per column, 8 per chip
    Kimi K2.6     384 experts -> 96 per column, 12 per chip

A layer/column's in-col share is the fraction of that layer's top-k picks landing in the
column's expert range. Uniform routing gives 25%; the hot cases in the current pick table run
39-43%. "Worst" = highest share (most dispatch traffic on one column), "nominal" = closest to
25%.

Sources
-------
``--source trace-routing`` reads a golden trace's ``routing/expert_ids_layer_{N}/*.safetensors``
stream (per-layer, row axis = token position). Available for Kimi code_debug
(``kimi_debug_55k_vllm``).

``--source flat`` reads an already-flat file — either an existing capture such as
``longbook_qa_eng_prefill_56320_nopad/expert_routing.safetensors``, or the output of the
on-device probe (``TT_DS_MOE_ROUTING_CAPTURE=1``, see ``tt/moe/moe_workload_probe.py``).
The DeepSeek code_debug trace has no routing stream at all — its own tensor_mapping.json
records "NO routing/ stream (expert ids/weights not saved)" — so DeepSeek code_debug routing
has to come from the on-device capture.

Examples
--------
    # Kimi K2.6, code_debug 55k, offline from the trace's routing stream
    python3 models/demos/deepseek_v3_d_p/scripts/make_captured_routing.py \
        --source trace-routing \
        --in /mnt/models/deepseek-prefill-cache/golden/structured_traces/kimi_debug_55k_vllm/routing \
        --num-routed-experts 384 --tokens 25600 \
        --out /path/kimi_k26_code_debug/expert_routing.safetensors

    # DeepSeek V3, from the on-device probe's capture
    python3 models/demos/deepseek_v3_d_p/scripts/make_captured_routing.py \
        --source flat --in /path/dsv3_columns_expert_routing.safetensors \
        --num-routed-experts 256 --tokens 25600 \
        --out /path/dsv3_code_debug/expert_routing.safetensors

    # Rank only, no file written
    python3 .../make_captured_routing.py --source flat --in <file> --num-routed-experts 256 --rank-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NUM_DISPATCH_GROUPS = 4
_LAYER_KEY = re.compile(r"^expert_ids_layer_(\d+)$")
_LAYER_CHUNK_KEY = re.compile(r"^expert_ids_layer_(\d+)_chunk_(\d+)$")


def _load_from_trace_routing(routing_dir: Path, top_k: int) -> dict[int, torch.Tensor]:
    """Read ``<routing_dir>/expert_ids_layer_{N}/*.safetensors`` -> {layer: [tokens, top_k]}."""
    out: dict[int, torch.Tensor] = {}
    for sub in sorted(routing_dir.iterdir()):
        m = _LAYER_KEY.match(sub.name)
        if not sub.is_dir() or not m:
            continue
        shards = sorted(sub.glob("*.safetensors"))
        if not shards:
            continue
        parts = []
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    parts.append(f.get_tensor(key).to(torch.int64).reshape(-1, top_k))
        out[int(m.group(1))] = torch.cat(parts, dim=0)
    if not out:
        raise SystemExit(f"no expert_ids_layer_* subdirs with safetensors under {routing_dir}")
    return out


def _load_from_flat(path: Path, top_k: int) -> dict[int, torch.Tensor]:
    """Read a flat ``expert_ids_layer_{N}`` file -> {layer: [tokens, top_k]}."""
    out: dict[int, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            m = _LAYER_KEY.match(key)
            if not m:
                continue
            out[int(m.group(1))] = f.get_tensor(key).to(torch.int64).reshape(-1, top_k)
    if not out:
        raise SystemExit(f"no expert_ids_layer_* keys in {path}")
    return out


def _load_from_capture(path: Path, top_k: int) -> dict[int, dict[int, torch.Tensor]]:
    """Read the probe's per-invocation capture -> {layer: {chunk: [tokens, top_k]}}."""
    out: dict[int, dict[int, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            m = _LAYER_CHUNK_KEY.match(key)
            if not m:
                continue
            layer, chunk = int(m.group(1)), int(m.group(2))
            out.setdefault(layer, {})[chunk] = f.get_tensor(key).to(torch.int64).reshape(-1, top_k)
    if not out:
        raise SystemExit(
            f"no expert_ids_layer_*_chunk_* keys in {path} "
            "(use --source flat for a per-layer file, or --source capture for a probe capture)"
        )
    return out


def column_shares(picks: torch.Tensor, num_routed_experts: int) -> list[float]:
    """Per-column in-col share (percent) for one layer's [tokens, top_k] picks."""
    experts_per_col = num_routed_experts // NUM_DISPATCH_GROUPS
    flat = picks.reshape(-1)
    valid = flat[(flat >= 0) & (flat < num_routed_experts)]
    if valid.numel() == 0:
        return [0.0] * NUM_DISPATCH_GROUPS
    hist = torch.bincount(valid // experts_per_col, minlength=NUM_DISPATCH_GROUPS)
    return [100.0 * int(hist[c]) / valid.numel() for c in range(NUM_DISPATCH_GROUPS)]


def rank_cases(
    layers: dict[int, torch.Tensor], num_routed_experts: int, n_worst: int, n_nominal: int
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Rank every (layer, col) by in-col share; return (worst, nominal) picks.

    Worst = highest share. Nominal = closest to the uniform 25%. Both are de-duplicated by
    layer so the cases exercise distinct routing rather than four columns of one layer.
    """
    scored = [
        (layer, col, share)
        for layer, picks in sorted(layers.items())
        for col, share in enumerate(column_shares(picks, num_routed_experts))
    ]
    uniform = 100.0 / NUM_DISPATCH_GROUPS

    def take(candidates, n):
        seen_layers, out = set(), []
        for layer, col, share in candidates:
            if layer in seen_layers:
                continue
            seen_layers.add(layer)
            out.append((layer, col, share))
            if len(out) == n:
                break
        return out

    worst = take(sorted(scored, key=lambda t: -t[2]), n_worst)
    worst_layers = {layer for layer, _, _ in worst}
    nominal_pool = [t for t in scored if t[0] not in worst_layers]
    nominal = take(sorted(nominal_pool, key=lambda t: abs(t[2] - uniform)), n_nominal)
    return worst, nominal


def rank_invocations(
    per_invocation: dict[int, dict[int, torch.Tensor]],
    num_routed_experts: int,
    n_worst: int,
    n_nominal: int,
) -> tuple[list[tuple[int, int, int, float]], list[tuple[int, int, int, float]]]:
    """Rank individual dispatch invocations — one (layer, chunk, col) each.

    Every (layer, chunk, col) is scored on its own, so a layer that spikes in one chunk is not
    averaged away against its quiet chunks. Both sets are de-duplicated by layer: without that,
    the worst four are typically four chunks of the same layer, which replays near-identical
    routing four times instead of covering four distinct cases.

    Returns (worst, nominal) as (layer, chunk, col, share) tuples.
    """
    scored: list[tuple[int, int, int, float]] = []
    for layer, per_chunk in sorted(per_invocation.items()):
        for chunk, picks in sorted(per_chunk.items()):
            for col, share in enumerate(column_shares(picks, num_routed_experts)):
                scored.append((layer, chunk, col, share))
    uniform = 100.0 / NUM_DISPATCH_GROUPS

    def take(candidates, n):
        seen_layers, out = set(), []
        for layer, chunk, col, share in candidates:
            if layer in seen_layers:
                continue
            seen_layers.add(layer)
            out.append((layer, chunk, col, share))
            if len(out) == n:
                break
        return out

    worst = take(sorted(scored, key=lambda t: -t[3]), n_worst)
    worst_layers = {layer for layer, _, _, _ in worst}
    nominal_pool = [t for t in scored if t[0] not in worst_layers]
    nominal = take(sorted(nominal_pool, key=lambda t: abs(t[3] - uniform)), n_nominal)
    return worst, nominal


def _print_picks(title: str, picks: list[tuple[int, int, float]]) -> None:
    print(f"\n# {title}")
    for layer, col, share in picks:
        print(f"    ({layer}, {col}),  # {share:.1f}% in-col share")


def _print_invocation_picks(title: str, picks: list[tuple[int, int, int, float]]) -> None:
    print(f"\n# {title}")
    for layer, chunk, col, share in picks:
        print(f"    ({layer}, {col}),  # {share:.1f}% in-col share, from chunk {chunk}")


def report_per_invocation(
    per_invocation: dict[int, dict[int, torch.Tensor]], num_routed_experts: int, top_n: int
) -> None:
    """Rank dispatch invocations — one (layer, chunk) each — by peak column share.

    A chunked prefill calls dispatch once per (layer, chunk), and the column load is not
    stationary across chunks, so the ranking is reported per invocation rather than per layer.
    """
    scored: list[tuple[int, int, int, float]] = []  # (layer, chunk, hottest_col, share)
    for layer, per_chunk in per_invocation.items():
        for chunk, picks in per_chunk.items():
            shares = column_shares(picks, num_routed_experts)
            hottest = max(range(NUM_DISPATCH_GROUPS), key=lambda c: shares[c])
            scored.append((layer, chunk, hottest, shares[hottest]))

    print(f"\n# hottest {top_n} dispatch invocations (layer, chunk) by peak column share")
    for layer, chunk, col, share in sorted(scored, key=lambda t: -t[3])[:top_n]:
        print(f"    layer={layer:3d} chunk={chunk:2d} col={col}  share={share:.1f}%")

    # Per-chunk peak share, so drift across the ISL is visible at a glance.
    print("\n# per-chunk peak column share (max over layers)")
    for chunk in sorted({c for _, c, _, _ in scored}):
        rows = [t for t in scored if t[1] == chunk]
        layer, _, col, share = max(rows, key=lambda t: t[3])
        mean = sum(t[3] for t in rows) / len(rows)
        print(f"    chunk {chunk:2d}: peak {share:5.1f}% (layer {layer}, col {col})   mean {mean:5.1f}%")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source",
        choices=["trace-routing", "flat", "capture"],
        required=True,
        help="trace-routing: a golden trace's routing/ stream. flat: a per-layer "
        "expert_ids_layer_{N} file. capture: the probe's per-invocation "
        "expert_ids_layer_{L}_chunk_{C} file.",
    )
    p.add_argument("--in", dest="inp", type=Path, required=True, help="routing dir or safetensors file")
    p.add_argument(
        "--chunk",
        default="picks",
        help="which chunk's routing to emit. 'picks' (default) gives each selected layer the chunk "
        "its own pick came from, so every case replays one real dispatch invocation. An integer "
        "forces one chunk for all layers; 'all' concatenates the chunks (aggregated, which hides "
        "per-chunk spikes). Ranking always covers every chunk regardless.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="for --source flat/trace-routing, split each layer's tokens into chunks of this many "
        "rows to recover the per-dispatch-invocation view (row axis is token position, so chunk c "
        "is rows [c*size, (c+1)*size)). 5120 matches CHUNK in the chunked-prefill tests. 0 = off.",
    )
    p.add_argument("--out", type=Path, help="output expert_routing.safetensors (omit with --rank-only)")
    p.add_argument("--num-routed-experts", type=int, required=True, help="256 for DeepSeek V3, 384 for Kimi K2.6")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--tokens",
        type=int,
        default=25600,
        help="tokens to keep per layer; must be divisible by --dispatch-group-size. The worker's "
        "expected numel is dispatch_group_size * seq_len_per_chip * top_k, so 25600 tokens "
        "(8 x 3200) matches the current LB 8-device replay. 0 keeps everything.",
    )
    p.add_argument("--dispatch-group-size", type=int, default=8)
    p.add_argument("--n-worst", type=int, default=4)
    p.add_argument("--n-nominal", type=int, default=4)
    p.add_argument("--rank-only", action="store_true", help="report picks without writing a file")
    a = p.parse_args(argv)

    if not a.rank_only and a.out is None:
        p.error("--out is required unless --rank-only")

    per_invocation: dict[int, dict[int, torch.Tensor]] | None = None
    if a.source == "capture":
        per_invocation = _load_from_capture(a.inp, a.top_k)
        n_chunks = max(len(v) for v in per_invocation.values())
        print(
            f"loaded {len(per_invocation)} layers x {n_chunks} chunks "
            f"= {sum(len(v) for v in per_invocation.values())} dispatch invocations from {a.inp}"
        )
    else:
        loader = _load_from_trace_routing if a.source == "trace-routing" else _load_from_flat
        layers = loader(a.inp, a.top_k)
        if a.chunk_size:
            # Row axis is token position, so a chunk is a contiguous row block — the same set of
            # picks the corresponding dispatch invocation saw.
            per_invocation = {
                layer: {
                    c: picks[c * a.chunk_size : (c + 1) * a.chunk_size] for c in range(picks.shape[0] // a.chunk_size)
                }
                for layer, picks in layers.items()
            }
    experts_per_col = a.num_routed_experts // NUM_DISPATCH_GROUPS
    print(
        f"num_routed_experts={a.num_routed_experts} -> {experts_per_col}/column, "
        f"{experts_per_col // a.dispatch_group_size}/chip"
    )

    # Rank per invocation whenever the chunk axis is known; only fall back to the aggregate
    # ranking when it isn't (a flat file with no --chunk-size), since aggregating over the ISL
    # averages away exactly the per-chunk spikes worth replaying.
    if per_invocation is not None:
        report_per_invocation(per_invocation, a.num_routed_experts, max(a.n_worst, 8))
        worst, nominal = rank_invocations(per_invocation, a.num_routed_experts, a.n_worst, a.n_nominal)
        _print_invocation_picks(f"{a.n_worst} worst (layer, col) — highest in-col share", worst)
        _print_invocation_picks(f"{a.n_nominal} nominal (layer, col) — closest to uniform 25%", nominal)
        chunk_for_layer = {layer: chunk for layer, chunk, _, _ in worst + nominal}
    else:
        worst_agg, nominal_agg = rank_cases(layers, a.num_routed_experts, a.n_worst, a.n_nominal)
        _print_picks(f"{a.n_worst} worst (layer, col) — highest in-col share [AGGREGATED]", worst_agg)
        _print_picks(f"{a.n_nominal} nominal (layer, col) — closest to uniform 25% [AGGREGATED]", nominal_agg)
        chunk_for_layer = {}

    # Resolve what actually gets written.
    if per_invocation is not None:
        if a.chunk == "picks":
            # Each picked layer keeps the chunk its own pick came from; unpicked layers fall back
            # to chunk 0 so the file stays complete and every layer tensor is the same size.
            layers = {
                layer: pc[chunk_for_layer.get(layer, min(pc))]
                for layer, pc in per_invocation.items()
                if chunk_for_layer.get(layer, min(pc)) in pc
            }
        elif a.chunk == "all":
            layers = {layer: torch.cat([pc[c] for c in sorted(pc)], dim=0) for layer, pc in per_invocation.items()}
        else:
            want = int(a.chunk)
            missing = [layer for layer, pc in per_invocation.items() if want not in pc]
            if missing:
                raise SystemExit(f"chunk {want} missing for layers {missing[:5]}")
            layers = {layer: pc[want] for layer, pc in per_invocation.items()}

    n_tokens = next(iter(layers.values())).shape[0]
    print(f"\nemitting from {len(layers)} layers x {n_tokens} tokens x top-{a.top_k} (chunk={a.chunk})")
    if a.chunk == "picks" and chunk_for_layer:
        print(f"per-layer chunk selection: {dict(sorted(chunk_for_layer.items()))}")

    max_id = max(int(t.max()) for t in layers.values())
    if max_id >= a.num_routed_experts:
        raise SystemExit(f"expert id {max_id} >= --num-routed-experts {a.num_routed_experts}")

    if a.rank_only:
        return 0

    keep = a.tokens or n_tokens
    if keep > n_tokens:
        raise SystemExit(f"--tokens {keep} exceeds the available {n_tokens}")
    if keep % a.dispatch_group_size:
        raise SystemExit(f"--tokens {keep} not divisible by --dispatch-group-size {a.dispatch_group_size}")

    # Chip-major: the loader views the flat tensor as [dispatch_group_size, seq_len_per_chip, top_k],
    # so slice each chip's contiguous token block rather than the first `keep` rows outright.
    per_chip_src = n_tokens // a.dispatch_group_size
    per_chip_keep = keep // a.dispatch_group_size
    tensors: dict[str, torch.Tensor] = {}
    for layer, picks in sorted(layers.items()):
        chip_blocks = [picks[c * per_chip_src : c * per_chip_src + per_chip_keep] for c in range(a.dispatch_group_size)]
        tensors[f"expert_ids_layer_{layer}"] = torch.cat(chip_blocks, dim=0).reshape(-1).to(torch.int32).contiguous()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(a.out))
    numel = next(iter(tensors.values())).numel()
    print(
        f"\nwrote {len(tensors)} layers x {numel} ids "
        f"({a.dispatch_group_size} chips x {per_chip_keep} tokens x top-{a.top_k}) -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
