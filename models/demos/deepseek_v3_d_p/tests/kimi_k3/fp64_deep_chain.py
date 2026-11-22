# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-major FP64 CPU chain of Kimi-K3.1 layers 0..L-1 (L <= 12, one AttnRes block) from prompt token ids, and its
score against the GPU trace. CPU only; never imports ttnn.

``fp64_chain`` runs layers 0-3 segment-major, which needs every layer's weights resident at once (two MoE layers'
routed experts are ~118 GB). Deeper layers need more MoE layers and, from layer 4 on, the AttnRes live sum of every
earlier attention and FFN output for every row, including the DSA attention output of every row (``fp64_chain``
computes it on sampled rows only). This driver runs one *pass* at a time over all rows: each layer's attention site,
then its FFN site, each over the rows in segments, with the same per-layer math as ``fp64_chain`` (so layers 0-3
reproduce its products bit for bit at the same segment size and thread count). Only that layer's weights are loaded.
The carried state is the live sum ``[rows, hidden]`` in FP64; the sealed snapshot of one block is the embedding.

* KDA attention: the FP64 reference with its state carried across segments.
* DSA attention: keys, the FP64 indexer top-k over all causal keys, and the gated MLA output of **every** row over
  its top-k (``attend_rows``: a block of rows scores the union of their keys, each row masked to its own).
* FFN: layer 0's dense FFN, else the FP64 MoE (``MoE64``).

Products per pass (``<run>/L<l>_<site>/segment_<i>.pt``; fp32 except the keys and KDA states, FP64): DSA
``dsa_input_L`` (the read before ``input_layernorm``), ``latent`` / ``kpe`` / ``index_key``, ``topk_indices_L`` /
``topk_scores_L`` and ``attn_output_L``; KDA ``kda_input_L`` (the read), ``attn_output_L`` and the state after the
segment's last row, ``kda_state_L_<field>`` (``KDAReferenceState``'s fields, batch dropped); MoE ``moe_input_L`` (after
``post_attention_layernorm``), ``moe_output_L``, ``routing_ids_L`` (best first), ``routing_weights_L``,
``routing_margin_L``; dense ``dense_input_L`` / ``dense_output_L``. After each pass the live sum is checkpointed
(``live.pt``); an interrupted pass reruns from it.

``backfill`` brings a complete version-1 run (whose KDA and dense passes stored nothing) to version 2: it replays
the stored passes (the live sum adds their stored fp32 outputs, and each pass's reconstructed input is compared with
its stored one) and reruns the KDA and dense passes, so their products carry the fp32 rounding of the replayed
outputs (recorded per pass as ``input_rel``).

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_deep_chain run --trace T --count N --layers 8 --out G
    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_deep_chain score --chain G --trace T --out S
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    MANIFEST,
    atomic_save,
    resume_manifest,
    token_ids_sha256,
    write_manifest,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import dsa_fp64_golden as dsa
from models.demos.deepseek_v3_d_p.tests.kimi_k3.chain_vs_trace import CUT_GAP_RANKS, direction, key_sums, pcc_nrmse
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import (
    F64,
    attn_res64,
    dense_mlp64,
    dsa_topk,
    kda_layer,
    load_layer,
    rms64,
    tensor_digest,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import Trace, checkpoint_tensors, quantile, topk_recall

PRODUCER_VERSION = 2  # 2: KDA and dense passes store their products too (``backfill`` brings a version-1 run there)
ATTN_BLOCK = 16  # rows per attend_rows block (16 measured 11 ms/row at 1M context against 29 for per-row gathers)
KEYS = ("latent", "kpe", "index_key")
KDA_STATE = ("recurrent", "q_convolution", "k_convolution", "v_convolution")  # KDAReferenceState's fields
SCORER_VERSION = 1


def layer_kinds(config: dict, layers: int) -> dict[int, tuple[str, str]]:
    """``{layer: (attention, ffn)}`` for layers ``0..layers-1``: attention ``kda`` or ``dsa`` (config
    ``linear_attn_config``, 1-based layer lists), ffn ``dense`` for the first ``first_k_dense_replace`` layers, else
    ``moe``."""
    c = config.get("text_config", config)
    linear = c["linear_attn_config"]
    kda, full = {l - 1 for l in linear["kda_layers"]}, {l - 1 for l in linear["full_attn_layers"]}
    kinds = {}
    for l in range(layers):
        if (l in kda) == (l in full):
            raise ValueError(f"layer {l} is not exactly one of KDA / full attention in the config")
        kinds[l] = ("kda" if l in kda else "dsa", "dense" if l < int(c["first_k_dense_replace"]) else "moe")
    return kinds


def attend_rows(q: dict, sel: torch.Tensor, latent: torch.Tensor, kpe: torch.Tensor, w: dict, dims: dsa.Dims):
    """``dsa.attend`` of every row of ``q`` over its selected keys ``sel`` ``[n, topk]`` (-1: none), FP64. A block
    of ``ATTN_BLOCK`` rows scores the union of its rows' keys once; each row's softmax is masked to its own keys."""
    kvb = w["kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)
    out = []
    for b in range(0, sel.shape[0], ATTN_BLOCK):
        s = sel[b : b + ATTN_BLOCK].long()
        valid = s >= 0
        u = torch.unique(s[valid])
        row, col = valid.nonzero(as_tuple=True)
        mask = torch.full((s.shape[0], u.numel()), -float("inf"), dtype=F64)
        mask[row, torch.searchsorted(u, s[row, col])] = 0.0
        q_latent = torch.einsum("bhd,hdl->bhl", q["q_nope"][b : b + ATTN_BLOCK], kvb[:, : dims.nope])
        scores = (q_latent @ latent[u].T + q["q_pe"][b : b + ATTN_BLOCK] @ kpe[u].T) * (dims.nope + dims.rope) ** -0.5
        mixed = (scores + mask[:, None]).softmax(-1) @ latent[u]
        o = (
            torch.einsum("bhl,hvl->bhv", mixed, kvb[:, dims.nope :]).reshape(s.shape[0], -1)
            * q["gate"][b : b + ATTN_BLOCK]
        )
        out.append(o @ w["o_proj.weight"].T)
    return torch.cat(out)


# --- the run ---------------------------------------------------------------------------------------------------------


def pass_dir(run_dir: Path, layer: int, site: str) -> Path:
    return Path(run_dir) / f"L{layer}_{site}"


def passes(layers: int) -> list[tuple[int, str]]:
    return [(l, site) for l in range(layers) for site in ("attn", "ffn")]


def run_pass(
    *,
    layer: int,
    site: str,
    kind: str,
    L: dict,
    emb: torch.Tensor,
    live: torch.Tensor,
    segment: int,
    kda_config: KDAConfig,
    dims: dsa.Dims,
    eps: float,
    out: Path,
    log: Callable[[str], None],
) -> None:
    """One pass over all rows; updates ``live`` in place and writes the pass's products per segment."""
    n = emb.shape[0]
    segments = -(-n // segment)
    out.mkdir(parents=True, exist_ok=True)
    state, keys = None, {k: [] for k in KEYS}
    for index in range(segments):
        began = time.perf_counter()
        lo, hi = index * segment, min((index + 1) * segment, n)
        sealed = emb[lo:hi].to(F64).unsqueeze(1)
        products = {}
        if site == "attn":
            read = emb[lo:hi].to(F64) if layer == 0 else attn_res64(live[lo:hi], sealed, L["attn_res"], eps)
            hidden = rms64(read, L["input_layernorm"], eps)
            if kind == "kda":
                attn, state = kda_layer("fp64", hidden.unsqueeze(0), L["kda"], kda_config, state)
                attn = attn[0]
                products = {f"kda_input_{layer}": read.float(), f"attn_output_{layer}": attn.float()}
                products.update({f"kda_state_{layer}_{f}": getattr(state, f)[0].clone() for f in KDA_STATE})
            else:
                k = dsa.layer_keys(hidden, L["dsa"], dims)
                for name in KEYS:
                    keys[name].append(k[name])
                indices, scores = dsa_topk(hidden, L["dsa"], dims, eps, torch.cat(keys["index_key"]), lo)
                q = dsa.layer_queries(hidden, L["dsa"], dims)
                attn = attend_rows(q, indices, torch.cat(keys["latent"]), torch.cat(keys["kpe"]), L["dsa"], dims)
                products = {f"dsa_input_{layer}": read.float(), **k, f"topk_indices_{layer}": indices.int()}
                products.update({f"topk_scores_{layer}": scores.float(), f"attn_output_{layer}": attn.float()})
            if layer == 0:
                live[lo:hi] = attn  # layer 0 seals the embedding; the live sum restarts at its attention output
            else:
                live[lo:hi] += attn
        else:
            normed = rms64(attn_res64(live[lo:hi], sealed, L["mlp_res"], eps), L["post_attention_layernorm"], eps)
            if kind == "dense":
                ffn = dense_mlp64(normed, L["mlp"])
                products = {f"dense_input_{layer}": normed.float(), f"dense_output_{layer}": ffn.float()}
            else:
                m = L["moe"](normed)
                ffn = m["output"]
                products = {f"moe_input_{layer}": normed.float(), f"moe_output_{layer}": ffn.float()}
                products.update(
                    {f"routing_ids_{layer}": m["ids"].int(), f"routing_weights_{layer}": m["weights"].float()}
                )
                products[f"routing_margin_{layer}"] = m["margin"].float()
            live[lo:hi] += ffn
        atomic_save({"start": lo, "end": hi, **products}, out / f"segment_{index:06d}.pt")
        log(
            f"L{layer} {site} ({kind}) segment {index + 1}/{segments} rows [{lo}, {hi}) {time.perf_counter() - began:.1f} s"
        )


def run_deep_chain(
    *,
    run_dir: Path,
    load: Callable[[int, str], tuple[dict, dict]],
    embedding: torch.Tensor,
    kinds: dict[int, tuple[str, str]],
    kda_config: KDAConfig,
    dims: dsa.Dims,
    eps: float,
    token_ids: torch.Tensor,
    segment: int,
    identity: dict,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (or resume) every pass of layers ``kinds`` over ``token_ids``; returns the manifest. ``load(l, kind)``
    gives layer l's weights for one site's ``kind`` (``kda`` / ``dsa`` / ``dense`` / ``moe``; ``fp64_chain.load_layer``
    form, the layer's norms and AttnRes queries included) and their digests, so only that site's weights are resident. A resume refuses another identity, token
    sequence, segment size or thread count, and reruns the first pass not listed as complete from ``live.pt``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    layers = len(kinds)
    if layers > 12:
        raise ValueError("the chain keeps one AttnRes block (block size 12); deeper layers need a second seal")
    expected = {
        "producer": "fp64_deep_chain",
        "producer_version": PRODUCER_VERSION,
        "identity": identity,
        "kinds": {str(l): list(k) for l, k in kinds.items()},
        "kda_config": asdict(kda_config),
        "dsa_dims": asdict(dims),
        "eps": eps,
        "embedding_sha256": tensor_digest({"w": embedding}),
        "count": int(token_ids.numel()),
        "token_ids_sha256": token_ids_sha256(token_ids.tolist()),
        "segment": segment,
        "dtype": "float64",
        "threads": torch.get_num_threads(),
    }
    todo = passes(layers)
    manifest = resume_manifest(run_dir, expected, len(todo))  # its "segments": the completed passes, in order
    done = len(manifest["segments"])
    emb = embedding[token_ids.long()]
    if done:
        saved = torch.load(run_dir / "live.pt", weights_only=True)
        if saved["after"] != done:
            raise ValueError(f"live.pt holds the state after pass {saved['after']}, the manifest lists {done}")
        live = saved["live"]
        log(f"resuming after pass {done}/{len(todo)} ({todo[done - 1]})")
    else:
        live = torch.zeros(emb.shape, dtype=F64)
    for p in range(done, len(todo)):
        layer, site = todo[p]
        kind = kinds[layer][0 if site == "attn" else 1]
        began = time.perf_counter()
        L, digests = load(layer, kind)
        log(f"pass {p + 1}/{len(todo)}: layer {layer} {site} ({kind}); weights in {time.perf_counter() - began:.0f} s")
        run_pass(
            layer=layer,
            site=site,
            kind=kind,
            L=L,
            emb=emb,
            live=live,
            segment=segment,
            kda_config=kda_config,
            dims=dims,
            eps=eps,
            out=pass_dir(run_dir, layer, site),
            log=log,
        )
        del L
        atomic_save({"after": p + 1, "live": live}, run_dir / "live.pt")
        entry = {"layer": layer, "site": site, "kind": kind, "digests": digests}
        entry["seconds"] = round(time.perf_counter() - began, 1)
        entry["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        manifest["segments"].append(entry)
        write_manifest(run_dir, manifest)
        log(f"pass {p + 1}/{len(todo)} done in {entry['seconds']:.0f} s; rss {entry['max_rss_gb']} GB")
    return manifest


def replay_pass(
    *, layer: int, site: str, kind: str, L: dict, emb: torch.Tensor, live: torch.Tensor, eps: float, out: Path
) -> float:
    """Add a stored DSA or MoE pass's fp32 outputs to ``live`` (in place); returns the largest per-segment relative
    norm distance between the input reconstructed from ``live`` and the pass's stored input."""
    worst = 0.0
    for f in sorted(out.glob("segment_*.pt")):
        d = torch.load(f, weights_only=True, mmap=True)
        lo, hi = d["start"], d["end"]
        sealed = emb[lo:hi].to(F64).unsqueeze(1)
        if site == "attn":
            got, want = attn_res64(live[lo:hi], sealed, L["attn_res"], eps), d[f"dsa_input_{layer}"]
            output = d[f"attn_output_{layer}"]
        else:
            got = rms64(attn_res64(live[lo:hi], sealed, L["mlp_res"], eps), L["post_attention_layernorm"], eps)
            want, output = d[f"moe_input_{layer}"], d[f"moe_output_{layer}"]
        worst = max(worst, float((got - want.to(F64)).norm() / want.to(F64).norm()))
        live[lo:hi] += output.to(F64)
    return worst


def backfill(
    *,
    run_dir: Path,
    load: Callable[[int, str], tuple[dict, dict]],
    embedding: torch.Tensor,
    token_ids: torch.Tensor,
    kda_config: KDAConfig,
    log: Callable[[str], None] = print,
) -> dict:
    """Bring a complete version-1 run to version 2 (module docstring); resumable after each rerun pass from
    ``live_backfill.pt``. ``load(l, "norms")`` gives a layer's norms and AttnRes queries only. Returns the manifest,
    whose ``backfill`` entry lists every pass with its ``input_rel`` (replayed) or seconds (rerun)."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST).read_text())
    kinds = {int(l): tuple(k) for l, k in manifest["kinds"].items()}
    todo = passes(len(kinds))
    if manifest["producer_version"] != 1 or len(manifest["segments"]) != len(todo):
        raise ValueError(f"{run_dir} is not a complete version-1 run")
    if manifest["token_ids_sha256"] != token_ids_sha256(token_ids.tolist()) or manifest[
        "embedding_sha256"
    ] != tensor_digest({"w": embedding}):
        raise ValueError(f"{run_dir} was run on other tokens or another embedding")
    record = manifest.setdefault("backfill", {"to_version": PRODUCER_VERSION, "passes": [], "complete": False})
    if record["complete"]:
        return manifest
    dims, eps, segment = dsa.Dims(**manifest["dsa_dims"]), manifest["eps"], manifest["segment"]
    emb = embedding[token_ids.long()]
    checkpoint = run_dir / "live_backfill.pt"
    done = len(record["passes"])
    if done:
        saved = torch.load(checkpoint, weights_only=True)
        if saved["after"] != done:
            raise ValueError(f"live_backfill.pt holds the state after pass {saved['after']}, the manifest lists {done}")
        live = saved["live"]
        log(f"backfill resuming after pass {done}/{len(todo)}")
    else:
        live = torch.zeros(emb.shape, dtype=F64)
    for p in range(done, len(todo)):
        layer, site = todo[p]
        kind = kinds[layer][0 if site == "attn" else 1]
        began = time.perf_counter()
        entry = {"layer": layer, "site": site, "kind": kind}
        if kind in ("kda", "dense"):
            L, entry["digests"] = load(layer, kind)
            if entry["digests"] != manifest["segments"][p]["digests"]:
                raise ValueError(f"layer {layer} {kind} weights differ from the run's")
            run_pass(
                layer=layer,
                site=site,
                kind=kind,
                L=L,
                emb=emb,
                live=live,
                segment=segment,
                kda_config=kda_config,
                dims=dims,
                eps=eps,
                out=pass_dir(run_dir, layer, site),
                log=log,
            )
            entry["seconds"] = round(time.perf_counter() - began, 1)
        else:
            L, _ = load(layer, "norms")
            entry["input_rel"] = replay_pass(
                layer=layer, site=site, kind=kind, L=L, emb=emb, live=live, eps=eps, out=pass_dir(run_dir, layer, site)
            )
        del L
        record["passes"].append(entry)
        if kind in ("kda", "dense") or p == len(todo) - 1:
            atomic_save({"after": p + 1, "live": live}, checkpoint)
            write_manifest(run_dir, manifest)
        log(f"backfill pass {p + 1}/{len(todo)} {entry}")
    final = torch.load(run_dir / "live.pt", weights_only=True)["live"]
    record["live_rel"] = float((live - final).norm() / final.norm())
    record["complete"] = True
    write_manifest(run_dir, manifest)
    checkpoint.unlink()
    log(f"backfill complete; final live sum relative distance {record['live_rel']:.3e}")
    return manifest


def load_deep(run_dir: Path, layer: int, site: str, names: list[str], start: int = 0, end: int | None = None):
    """Products ``names`` of pass (layer, site) for rows ``[start, end)`` (default: all)."""
    files = sorted(pass_dir(run_dir, layer, site).glob("segment_*.pt"))
    parts: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    for f in files:
        d = torch.load(f, weights_only=True, mmap=True)
        lo, hi = d["start"], d["end"]
        if end is not None and lo >= end or hi <= start:
            continue
        a, b = max(start, lo) - lo, (hi if end is None else min(end, hi)) - lo
        for n in names:
            parts[n].append(d[n][a:b])
    return {n: torch.cat(v) for n, v in parts.items()}


# --- score against the GPU trace -------------------------------------------------------------------------------------


def score_segment(run_dir: Path, kinds: dict, trace: Trace, lo: int, hi: int) -> dict:
    """Per-row measures of the trace against the chain on rows ``[lo, hi)``: routing of every MoE layer, keys and
    top-k of every DSA layer (``chain_vs_trace``'s definitions)."""
    out: dict = {"start": lo, "end": hi}
    for l, (attn, ffn) in kinds.items():
        if ffn == "moe":
            p = load_deep(
                run_dir, l, "ffn", [f"routing_ids_{l}", f"routing_weights_{l}", f"routing_margin_{l}"], lo, hi
            )
            want, got = p[f"routing_ids_{l}"].long(), trace.rows(f"expert_ids_layer_{l}", lo, hi).long()
            rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
            missed = ~torch.isin(rows + want, rows + got)
            out[f"routing_{l}"] = {
                "recall": topk_recall(got, want).float(),
                "dropped": (p[f"routing_weights_{l}"].double() * missed).sum(1).float(),
                "margin": p[f"routing_margin_{l}"].float(),
            }
        if attn == "dsa":
            p = load_deep(run_dir, l, "attn", [*KEYS, f"topk_indices_{l}", f"topk_scores_{l}"], lo, hi)
            kv = trace.rows(f"kv_post_transform_layer_{l}", lo, hi)
            lora = p["latent"].shape[1]
            got = {"latent": kv[:, :lora], "kpe": kv[:, lora:], "index_key": trace.rows(f"indexer_k_layer_{l}", lo, hi)}
            out[f"keys_{l}"] = {k: key_sums(got[k], p[k]) for k in KEYS}
            want = p[f"topk_indices_{l}"].long()
            topk = want.shape[1]
            full = torch.arange(lo, hi) >= topk - 1
            ordered = p[f"topk_scores_{l}"][full].double().sort(dim=1, descending=True).values
            out[f"topk_{l}"] = {
                "recall": topk_recall(trace.rows(f"dsa_topk_indices_layer_{l}", lo, hi).long(), want)[full].float(),
                "cut_gap": (ordered[:, topk - 1 - min(CUT_GAP_RANKS, topk - 1)] - ordered[:, topk - 1]).float(),
            }
    return out


def summarize(results: list[dict], window: int) -> dict:
    """Per window of ``window`` rows (aligned to segments) and overall: routing recall / identical sets / dropped
    weight (mean, p99) / FP64 margin median; key PCC and NRMSE; top-k recall (mean, p01) and the FP64 cut gap
    median. Top-k recall falls with cut density even at constant key error (review r1 m4): read it with the gap."""
    groups: list[list[dict]] = []
    for r in results:
        if not groups or r["start"] // window != groups[-1][0]["start"] // window:
            groups.append([])
        groups[-1].append(r)

    def measures(rs: list[dict]) -> dict:
        m = {"start": rs[0]["start"], "end": rs[-1]["end"]}
        for key in rs[0]:
            cat = lambda k: torch.cat([r[key][k] for r in rs])  # noqa: E731
            if key.startswith("routing_"):
                recall, dropped = cat("recall"), cat("dropped")
                m[f"{key}_recall_mean"], m[f"{key}_set_equal"] = float(recall.mean()), float(
                    (recall == 1).float().mean()
                )
                m[f"{key}_dropped_mean"], m[f"{key}_dropped_p99"] = float(dropped.mean()), quantile(dropped, 0.99)
                m[f"{key}_margin_median_all"] = quantile(cat("margin"), 0.5)
            elif key.startswith("keys_"):
                for k in KEYS:
                    m[f"{k}_{key[5:]}_pcc"], m[f"{k}_{key[5:]}_nrmse"] = pcc_nrmse(cat(k))
            elif key.startswith("topk_"):
                recall = cat("recall")
                m[f"{key}_recall_mean"] = float(recall.mean()) if recall.numel() else float("nan")
                m[f"{key}_recall_p01"] = quantile(recall, 0.01)
                m[f"{key}_cut_gap_median"] = quantile(cat("cut_gap"), 0.5)
        return m

    windows = [measures(g) for g in groups]
    trend = {}
    for key in windows[0] if windows else []:
        values = [(w[key], w["start"]) for w in windows if key not in ("start", "end") and w[key] == w[key]]
        if len(values) < 2:
            continue
        first, last = values[0][0], values[-1][0]
        trend[key] = {"first": first, "last": last, "last_over_first": last / first if first else float("nan")}
        sign = direction(key)
        if sign is not None:
            worst = max(values, key=lambda v: sign * v[0])
            trend[key].update(worst=worst[0], worst_at=worst[1])
    return {"window": window, "overall": measures(results) if results else {}, "windows": windows, "trend": trend}


def score(run_dir: Path, trace: Trace, out: Path, window: int, log: Callable[[str], None] = print) -> dict:
    """Score every row the complete run holds (cached per segment in ``out``) and write metrics.json / windows.tsv."""
    manifest = json.loads((Path(run_dir) / MANIFEST).read_text())
    if len(manifest["segments"]) < 2 * len(manifest["kinds"]):
        raise ValueError(f"{run_dir}: {len(manifest['segments'])} of {2 * len(manifest['kinds'])} passes complete")
    kinds = {int(l): tuple(k) for l, k in manifest["kinds"].items()}
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "chain": str(run_dir),
        "producer_version": manifest["producer_version"],
        "token_ids_sha256": manifest["token_ids_sha256"],
        "passes_digests": [p["digests"] for p in manifest["segments"]],
        "trace": str(trace.path),
        "scorer_version": SCORER_VERSION,
    }
    id_path = out / "identity.json"
    if id_path.is_file() and json.loads(id_path.read_text()) != identity:
        raise ValueError(f"{out} holds scores of another chain or trace")
    id_path.write_text(json.dumps(identity, indent=1))
    results = []
    count, seg = manifest["count"], manifest["segment"]
    for index, lo in enumerate(range(0, count, seg)):
        cached = out / f"segment_{index:06d}.pt"
        if not cached.is_file():
            began = time.perf_counter()
            atomic_save(score_segment(run_dir, kinds, trace, lo, min(lo + seg, count)), cached)
            log(f"scored rows [{lo}, {min(lo + seg, count)}) {time.perf_counter() - began:.1f} s")
        results.append(torch.load(cached, weights_only=True))
    summary = summarize(results, window)
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    columns = list(summary["windows"][0])
    (out / "windows.tsv").write_text(
        "\t".join(columns)
        + "\n"
        + "".join("\t".join(f"{w[c]:.6g}" for c in columns) + "\n" for w in summary["windows"])
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "score", "backfill"])
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="run: the run directory; score: the score directory")
    parser.add_argument("--chain", type=Path, help="score / backfill: the run directory")
    parser.add_argument("--count", type=int, help="run: prompt rows (default: the trace's prompt)")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--segment", type=int, default=4096)
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads; a resume must use the same count")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get("KIMI_K31_HF_MODEL", "/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized")
        ),
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    trace = Trace(args.trace)
    if args.command == "score":
        summary = score(args.chain, trace, args.out, args.window, log)
        log(f"scored {summary['windows'][-1]['end']} rows in {len(summary['windows'])} windows -> {args.out}")
        return
    config = json.loads((args.checkpoint / "config.json").read_text())
    kinds = layer_kinds(config, args.layers)
    count = args.count or trace.prompt_len
    token_ids = torch.tensor(trace.metadata["token_ids"][:count], dtype=torch.int64)
    embedding = checkpoint_tensors(args.checkpoint, None, ["embed_tokens.weight"])["embed_tokens.weight"]
    eps = float(config.get("text_config", config)["rms_norm_eps"])

    def load(l: int, kind: str) -> tuple[dict, dict]:  # kind "norms": the norms and AttnRes queries only
        return load_layer(
            args.checkpoint,
            l,
            kda=kind == "kda",
            dense=kind == "dense",
            moe=kind == "moe",
            dsa=kind == "dsa",
            threads=args.threads,
            log=log,
        )

    if args.command == "backfill":
        manifest = json.loads((args.chain / MANIFEST).read_text())
        token_ids = token_ids[: manifest["count"]]
        backfill(
            run_dir=args.chain,
            load=load,
            embedding=embedding,
            token_ids=token_ids,
            kda_config=KDAConfig.from_model_config(config),
            log=log,
        )
        return
    run_deep_chain(
        run_dir=args.out,
        load=load,
        embedding=embedding,
        kinds=kinds,
        kda_config=KDAConfig.from_model_config(config),
        dims=dsa.Dims(eps=eps),
        eps=eps,
        token_ids=token_ids,
        segment=args.segment,
        identity={
            "model": "Kimi-K3.1",
            "checkpoint": str(args.checkpoint),
            "token_source": f"vllm trace {trace.path.name} metadata token_ids[:{count}]",
            "trace_commit": trace.metadata.get("trace_commit"),
        },
        log=log,
    )


if __name__ == "__main__":
    main()
