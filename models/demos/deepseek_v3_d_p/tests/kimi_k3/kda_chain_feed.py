# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 KDA layers 1-2 teacher-fed from the FP64 chain (``fp64_chain.py``): the CPU half. CPU only, no ttnn.

The chain stores neither the KDA input (post ``input_layernorm``) nor the KDA output of layers 1-2, but it stores
everything needed to recompute them for any segment without rerunning it: the three KDA states after every segment
(``state_<i>.pt``, FP64) and the MoE outputs of layers 1-2 (``segment_<i>.pt``, fp32). Three steps:

``rebuild``: per chain segment, from the KDA states after the previous segment, recompute in FP64 the chain's walk
through layers 0-2 with every expensive MoE replaced by its stored output (``chain_segment``'s math, no routed
experts loaded): layer 0 KDA and dense FFN, then for layers 1 and 2 the KDA input (rounded once to bf16, what the
device gets), the FP64 KDA output and the live AttnRes sum before the layer's attention (both fp32; the scorer routes
the device output from it). Segments are independent, so ``--shard w/W`` splits them over processes; each writes
``rebuilt_<i>.pt`` atomically and a rerun skips the ones present. Each segment checks itself against what the chain
stored, so a rebuild that does not reproduce the chain fails: the recomputed KDA states after the segment against
``state_<i>.pt`` (all three layers, relative L2 <= ``STATE_BAR`` of the layer), the recomputed MoE inputs and ``dsa_input_3``
against the stored ones (relative L2 <= ``PRODUCT_BAR``; fp32 storage, and the live sum carries the stored fp32 MoE
outputs instead of the chain's FP64 ones). It also records the GPU trace's layer-0 KDA output against FP64 per row
(``chain_vs_trace.key_sums``): the only KDA output the trace has, the GPU's calibration.

``input --layer L``: the device input of layer L as a ``teacher_feed`` input directory (``normed`` per segment).

``score --layer L``: a device run (``test_kda_chain_feed.py``) against FP64 per window of positions: the KDA output
(per-row NRMSE, PCC), the device recurrent state at the rows where a device chunk and a chain segment end together
(every 20480 rows) against the chain's state (relative L2 per head), and downstream the layer's MoE routing from the
device output (the FFN read of the FP64 live sum plus the device output, routed in FP64) against the chain's routing,
next to the GPU's routing recall of the same rows (the trace).

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_chain_feed rebuild --chain C --trace T \\
        --out R --shard 0/4 --threads 12
    python -m ...kda_chain_feed input --rebuilt R --layer 1 --out I1
    python -m ...kda_chain_feed score --rebuilt R --run D1 --trace T --window 65536
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, atomic_save, segment_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3.chain_vs_trace import key_sums, pcc_nrmse
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import (
    DSA_LAYER,
    F64,
    KDA_LAYERS,
    MOE_LAYERS,
    ChainModel,
    attn_res64,
    dense_mlp64,
    rms64,
    state_path,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import (
    Trace,
    build_input,
    chain_identity,
    chunk_path,
    complete_run,
    quantile,
    topk_recall,
)

FED_LAYERS = MOE_LAYERS  # (1, 2): the KDA layers whose input is an AttnRes read (layer 0's is the embedding)
REBUILD_VERSION = 1
# The rebuild must reproduce the chain. Layers 0-1 see only recomputed FP64, so their KDA states match to rounding
# (a different thread count changes FP64 summation order only). Layer 2's input carries the stored fp32 MoE output of
# layer 1 instead of the chain's FP64 one, a ~1e-8 relative perturbation (pilot: state 2.5e-9, convolution 5.4e-9),
# as do the stored fp32 products compared against.
STATE_BAR = {0: 1e-9, 1: 1e-9, 2: 1e-6}
PRODUCT_BAR = 1e-5


def rebuilt_path(out: Path, index: int) -> Path:
    return out / f"rebuilt_{index:06d}.pt"


def rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """||got - want|| / ||want|| in FP64."""
    return float((got.double() - want.double()).norm() / want.double().norm())


def rebuild_segment(model: ChainModel, chain: Path, index: int, token_ids: torch.Tensor, trace: Trace | None) -> dict:
    """Chain segment ``index`` recomputed through layers 0-2 from the stored states and MoE outputs (module docstring).
    Returns the payload ``rebuilt_<i>.pt`` holds, its ``checks`` included."""
    seg = torch.load(segment_path(chain, index), weights_only=True, mmap=True)
    start, end = seg["start"], seg["end"]
    states: dict[int, KDAReferenceState | None] = {l: None for l in KDA_LAYERS}
    if index:
        before = torch.load(state_path(chain, index - 1), weights_only=True, mmap=True)["kda"]
        states = {l: KDAReferenceState(**{k: v.clone() for k, v in before[l].items()}) for l in KDA_LAYERS}
    eps = model.eps
    x = model.embedding[token_ids[start:end].long()].to(F64)
    out: dict = {"start": start, "end": end}
    checks: dict[str, float] = {}
    sealed, live = x.new_zeros(x.shape[0], 0, x.shape[1]), x
    for l in KDA_LAYERS:
        L = model.layers[l]
        hidden = live if sealed.shape[1] == 0 else attn_res64(live, sealed, L["attn_res"], eps)
        hidden = rms64(hidden, L["input_layernorm"], eps)
        if l % model.block_size == 0:
            sealed, live = torch.cat([sealed, live.unsqueeze(1)], dim=1), None
        attn, states[l] = kda_forward_reference(hidden.unsqueeze(0), L["kda"], model.kda_config, states[l], dtype=F64)
        attn = attn[0]
        if l in FED_LAYERS:
            out[f"kda_input_{l}"] = hidden.to(torch.bfloat16)
            out[f"kda_output_{l}"] = attn.float()
            out[f"live_{l}"] = live.float()
        elif trace is not None:
            out["gpu_attn_output_0_sums"] = key_sums(trace.rows("attn_output_layer_0", start, end), attn)
        live = attn if live is None else live + attn
        normed = rms64(attn_res64(live, sealed, L["mlp_res"], eps), L["post_attention_layernorm"], eps)
        if l == 0:
            ffn = dense_mlp64(normed, L["mlp"])
        else:
            checks[f"moe_input_{l}_rel"] = rel(normed, seg[f"moe_input_{l}"])
            ffn = seg[f"moe_output_{l}"].to(F64)
        live = live + ffn
    read = attn_res64(live, sealed, model.layers[DSA_LAYER]["attn_res"], eps)
    checks["dsa_input_3_rel"] = rel(read, seg["dsa_input_3"])
    after = torch.load(state_path(chain, index), weights_only=True, mmap=True)["kda"]
    for l in KDA_LAYERS:
        checks[f"state_{l}_rel"] = rel(states[l].recurrent, after[l]["recurrent"])
        checks[f"conv_{l}_rel"] = max(
            rel(getattr(states[l], f"{c}_convolution"), after[l][f"{c}_convolution"]) for c in "qkv"
        )
    out["checks"] = checks
    return out


def check_failures(checks: dict[str, float]) -> dict[str, float]:
    """The checks of a rebuilt segment that miss their bar (``STATE_BAR`` for states, ``PRODUCT_BAR`` otherwise)."""
    bar = lambda k: STATE_BAR[int(k.split("_")[1])] if k.startswith(("state_", "conv_")) else PRODUCT_BAR
    return {k: v for k, v in checks.items() if not v <= bar(k)}


def rebuild(
    *, model: ChainModel, chain: Path, out: Path, token_ids: torch.Tensor, trace: Trace | None, shard=(0, 1), log=print
) -> list[int]:
    """Rebuild this shard's segments of ``chain`` into ``out`` (module docstring); returns the indices written."""
    manifest = json.loads((chain / MANIFEST).read_text())
    identity = {
        "producer": "kda_chain_feed.rebuild",
        "rebuild_version": REBUILD_VERSION,
        "chain": chain_identity(chain),
        "chain_model": manifest["model"],
        "fed_layers": list(FED_LAYERS),
        "segments": len(manifest["segments"]),
        "segment_rows": [[s["start"], s["end"]] for s in manifest["segments"]],
        "gpu_calibration": trace is not None,
    }
    digests = {k: v for k, v in model.identity()["weights_sha256"].items()}
    want = manifest["model"]["weights_sha256"]
    differ = [k for k, v in digests.items() if want.get(k) != v]
    if differ:
        raise ValueError(f"weights {differ} differ from the chain's {chain}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST
    if path.is_file():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"{out} holds a rebuild of something else: {json.loads(path.read_text())}")
    else:
        temporary = out / f"{MANIFEST}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(identity, indent=1))
        temporary.replace(path)
    w, n = shard
    written = []
    for index in range(w, identity["segments"], n):
        if rebuilt_path(out, index).is_file():
            continue
        began = time.perf_counter()
        payload = rebuild_segment(model, chain, index, token_ids, trace)
        failed = check_failures(payload["checks"])
        if failed:
            raise ValueError(f"segment {index}: the rebuild does not reproduce the chain: {failed}")
        atomic_save(payload, rebuilt_path(out, index))
        written.append(index)
        log(
            f"rebuilt segment {index + 1}/{identity['segments']} rows [{payload['start']}, {payload['end']}) "
            f"{time.perf_counter() - began:.1f} s " + json.dumps(payload["checks"])
        )
    return written


def rebuilt_manifest(rebuilt: Path) -> dict:
    """The rebuild's manifest; every segment must be present."""
    manifest = json.loads((rebuilt / MANIFEST).read_text())
    missing = [i for i in range(manifest["segments"]) if not rebuilt_path(rebuilt, i).is_file()]
    if missing:
        raise ValueError(f"{rebuilt} lacks {len(missing)} rebuilt segments (first {missing[:5]})")
    return manifest


def build_kda_input(*, rebuilt: Path, layer: int, out: Path, count: int | None = None) -> dict:
    """Layer ``layer``'s device input from a complete rebuild, as a ``teacher_feed`` input directory."""
    if layer not in FED_LAYERS:
        raise ValueError(f"layer {layer} is not a fed KDA layer {FED_LAYERS}")
    manifest = rebuilt_manifest(rebuilt)
    rows = manifest["segment_rows"]
    count = count or rows[-1][1]
    segment = rows[0][1] - rows[0][0]

    def make(start: int, end: int):
        p = torch.load(rebuilt_path(rebuilt, start // segment), weights_only=True, mmap=True)
        lo = start - p["start"]
        return p[f"kda_input_{layer}"][lo : lo + end - start].clone(), {
            k: v for k, v in p["checks"].items() if k.endswith(f"_{layer}_rel")
        }

    expected = {
        "kind": "kda_chain_feed",
        "layer": layer,
        "rebuilt": str(rebuilt),
        "rebuild": {k: manifest[k] for k in ("rebuild_version", "chain")},
        "rounding": "FP64 -> bf16 once",
        # the device test's manifest copies these (teacher-feed input contract)
        "trace": manifest["chain"]["trace"],
        "trace_commit": manifest["chain"]["trace_commit"],
        "input": f"kda_input_{layer}",
        "checkpoint": manifest["chain"]["checkpoint"],
    }
    return build_input(out=out, expected=expected, count=count, segment=segment, make=make, check_bars={})


# --- scoring a device run against the rebuilt FP64 ------------------------------------------------------------------


def route64(normed: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, top_k: int) -> torch.Tensor:
    """The K3 router's selection in FP64 (sigmoid scores + selection bias, top-k, best first), as ``MoE64.route``."""
    return ((normed @ gate.T).sigmoid() + bias).topk(top_k, dim=-1).indices


def score(*, rebuilt: Path, run: Path, trace: Trace | None, model: ChainModel, window: int, log=print) -> dict:
    """A device run of ``test_kda_chain_feed.py`` against the rebuilt FP64 (module docstring); writes
    ``metrics.json`` into the run directory."""
    run_manifest = complete_run(run)
    layer, count = run_manifest["layer"], run_manifest["count"]
    rb = rebuilt_manifest(rebuilt)
    seg_rows = rb["segment_rows"]
    segment = seg_rows[0][1] - seg_rows[0][0]
    chain = Path(rb["chain"]["chain"])
    L = model.layers[layer]
    eps = model.eps
    sums, nrmse_rows, recall_dev, recall_gpu, recall_ref = [], [], [], [], []
    token_ids = torch.tensor(trace.metadata["token_ids"][:count], dtype=torch.int64) if trace is not None else None
    chunk = run_manifest["chunk"]
    pending: dict | None = None
    for index in range(-(-count // segment)):
        p = torch.load(rebuilt_path(rebuilt, index), weights_only=True, mmap=True)
        lo, hi = p["start"], min(p["end"], count)
        got = torch.cat(
            [
                c["attn_out"][max(lo, c["start"]) - c["start"] : min(hi, c["end"]) - c["start"]]
                for c in (
                    torch.load(chunk_path(run, k), weights_only=True) for k in range(lo // chunk, -(-hi // chunk))
                )
            ]
        )
        want = p[f"kda_output_{layer}"][: hi - lo]
        sums.append(key_sums(got, want))
        nrmse_rows.append(((got.double() - want.double()).norm(dim=1) / want.double().norm(dim=1)).float())
        seg = torch.load(segment_path(chain, index), weights_only=True, mmap=True)
        emb = model.embedding[token_ids[lo:hi]].to(F64).unsqueeze(1)
        live = p[f"live_{layer}"][: hi - lo].double() + got.double()
        normed = rms64(attn_res64(live, emb, L["mlp_res"], eps), L["post_attention_layernorm"], eps)
        ids = route64(normed, L["gate"], L["gate_bias"], model.top_k)
        chain_ids = seg[f"routing_ids_{layer}"][: hi - lo].long()
        recall_dev.append(topk_recall(ids, chain_ids))
        live_ref = p[f"live_{layer}"][: hi - lo].double() + want.double()
        normed_ref = rms64(attn_res64(live_ref, emb, L["mlp_res"], eps), L["post_attention_layernorm"], eps)
        recall_ref.append(topk_recall(route64(normed_ref, L["gate"], L["gate_bias"], model.top_k), chain_ids))
        recall_gpu.append(topk_recall(trace.rows(f"expert_ids_layer_{layer}", lo, hi).long(), chain_ids))
        log(f"scored segment {index + 1} rows [{lo}, {hi})") if index % 32 == 0 else None
    sums, nrmse_rows = torch.cat(sums), torch.cat(nrmse_rows)
    recall_dev, recall_gpu, recall_ref = torch.cat(recall_dev), torch.cat(recall_gpu), torch.cat(recall_ref)
    states = []
    for entry in run_manifest.get("states", []):
        row = entry["row"]
        if row % segment:
            continue
        want = torch.load(state_path(chain, row // segment - 1), weights_only=True, mmap=True)["kda"][layer]
        got = torch.load(run / entry["file"], weights_only=True)
        per_head = (got["recurrent"].double() - want["recurrent"]).flatten(2).norm(dim=2) / want["recurrent"].flatten(
            2
        ).norm(dim=2)
        states.append(
            {
                "row": row,
                "recurrent_rel": rel(got["recurrent"], want["recurrent"]),
                "recurrent_head_rel_max": float(per_head.max()),
                "recurrent_head_rel_median": float(per_head.median()),
                "conv_rel": max(rel(got[f"{c}_convolution"], want[f"{c}_convolution"]) for c in "qkv"),
            }
        )

    def measures(lo: int, hi: int) -> dict:
        pcc, nrmse = pcc_nrmse(sums[lo:hi])
        return {
            "start": lo,
            "end": hi,
            "kda_pcc": pcc,
            "kda_nrmse": nrmse,
            "kda_row_nrmse_median": float(nrmse_rows[lo:hi].median()),
            "kda_row_nrmse_p99": quantile(nrmse_rows[lo:hi], 0.99),
            "routing_recall_device": float(recall_dev[lo:hi].mean()),
            "routing_set_equal_device": float((recall_dev[lo:hi] == 1).float().mean()),
            "routing_recall_gpu": float(recall_gpu[lo:hi].mean()),
            # the stored FP64 output (fp32) routed the same way: the routing's sensitivity to that storage alone
            "routing_recall_reference": float(recall_ref[lo:hi].mean()),
            "routing_set_equal_gpu": float((recall_gpu[lo:hi] == 1).float().mean()),
        }

    windows = [measures(lo, min(lo + window, count)) for lo in range(0, count, window)]
    result = {
        "layer": layer,
        "count": count,
        "window": window,
        "reference": rb["chain"],
        "run": str(run),
        "overall": measures(0, count),
        "windows": windows,
        "states": states,
    }
    (run / "metrics.json").write_text(json.dumps(result, indent=1))
    return result


def gpu_calibration(rebuilt: Path, window: int) -> list[dict]:
    """The GPU trace's layer-0 KDA output against FP64 per window (from the rebuild's per-row sums)."""
    manifest = rebuilt_manifest(rebuilt)
    sums = torch.cat(
        [
            torch.load(rebuilt_path(rebuilt, i), weights_only=True, mmap=True)["gpu_attn_output_0_sums"]
            for i in range(manifest["segments"])
        ]
    )
    out = []
    for lo in range(0, sums.shape[0], window):
        pcc, nrmse = pcc_nrmse(sums[lo : lo + window])
        out.append({"start": lo, "end": min(lo + window, sums.shape[0]), "gpu_kda0_pcc": pcc, "gpu_kda0_nrmse": nrmse})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["rebuild", "input", "score", "calibration"])
    parser.add_argument("--chain", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--rebuilt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--shard", default="0/1", help="w/W: this process rebuilds segments i with i %% W == w")
    parser.add_argument("--window", type=int, default=65536)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    def load_model(chain: Path) -> ChainModel:
        checkpoint = Path(json.loads((chain / MANIFEST).read_text())["identity"]["checkpoint"])
        log(f"loading weights (no routed experts) from {checkpoint}")
        return ChainModel.from_checkpoint(checkpoint, args.threads, log, routed_experts=False)

    if args.command == "rebuild":
        trace = Trace(args.trace)
        token_ids = torch.tensor(trace.metadata["token_ids"][: trace.prompt_len], dtype=torch.int64)
        model = load_model(args.chain)
        w, n = (int(v) for v in args.shard.split("/"))
        written = rebuild(
            model=model, chain=args.chain, out=args.out, token_ids=token_ids, trace=trace, shard=(w, n), log=log
        )
        log(f"shard {w}/{n} done: {len(written)} segments written")
    elif args.command == "input":
        m = build_kda_input(rebuilt=args.rebuilt, layer=args.layer, out=args.out, count=args.count)
        log(f"input layer {args.layer}: {m['segments'][-1]['end']} rows -> {args.out}")
    elif args.command == "score":
        chain = Path(rebuilt_manifest(args.rebuilt)["chain"]["chain"])
        result = score(
            rebuilt=args.rebuilt,
            run=args.run,
            trace=Trace(args.trace),
            model=load_model(chain),
            window=args.window,
            log=log,
        )
        print(json.dumps({k: result[k] for k in ("overall", "states")}, indent=1))
    else:
        print(json.dumps(gpu_calibration(args.rebuilt, args.window), indent=1))


if __name__ == "__main__":
    main()
