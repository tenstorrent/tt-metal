# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Requests, the calibration/held-out split, and synthetic prefill traffic built from captured routing.

Every request has its own attention context, so its routing is (approximately) independent of what it is
batched with. Synthetic 5120-token chunks are therefore built by splicing whole captured requests.

Split: inside every source, requests (not chunks) are assigned by a fixed per-source permutation until the
TEST half holds half the tokens; the rest is TRAIN. No test token is ever seen by a calibrated plan.

Mixing ("balanced"): each draw goes to the stratum (domain, then source) furthest below its equal token
share; random jitter < 1 token only breaks ties. Request lengths differ ~100x across domains, so drawing
strata uniformly per request would let long-document domains dominate.
"""

from __future__ import annotations

import json
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import routing
from .corpus import DOMAINS, tokens_sha256, write_sequence

LANG_GROUP = {"en": "english", "cpp": "code"}
LANG_GROUP.update({l: "european" for l in ["de", "es", "fr", "ca", "hr", "ru", "it", "pt", "id"]})
LANG_GROUP.update({l: "cjk" for l in ["zh", "ja", "ko"]})
LANG_GROUP.update({l: "indic_thai_arabic" for l in ["hi", "bn", "te", "th", "ar"]})
LANG_GROUP.update({l: "african" for l in ["sw", "yo"]})
SPECIAL_PREFIXES = ("control", "edge")  # scored whole, never split into requests


def is_special(name: str) -> bool:
    return name.startswith(SPECIAL_PREFIXES)


def usable_capture(name: str) -> bool:
    return not (name.startswith("mixed") or name.endswith("_repeat") or "failed" in name)


@dataclass
class Corpus:
    ids: dict  # sequence name -> expert_ids [layers, tokens, top_k]
    layers: list
    samples: list  # requests: seq, start, end, domain, lang, group, split

    @property
    def domains(self) -> list:
        return sorted({s["domain"] for s in self.samples})


def request_bounds(workdir: Path, name: str):
    meta = json.loads((workdir / "sequences" / name / "samples.json").read_text())
    so = np.load(workdir / "sequences" / name / "token_meta.npz")["sample_of"]
    bounds = np.flatnonzero(np.diff(so)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(so)]
    if len(starts) != len(meta) or not (so[starts] == np.arange(len(meta))).all():
        raise ValueError(f"{name}: token_meta does not match samples.json")
    return meta, starts, ends


def split_of(name: str, starts, ends) -> dict:
    """request index -> 'test' | 'train' (fixed per source)."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    total, test_tok, split = ends[-1], 0, {}
    for k in rng.permutation(len(starts)):
        if test_tok < total / 2:
            split[k] = "test"
            test_tok += ends[k] - starts[k]
        else:
            split[k] = "train"
    return split


def load_corpus(workdir: Path, layers=None, skip_prefix: int = 0, captures: str = "captures") -> Corpus:
    ids, samples, all_layers = {}, [], None
    for d in sorted((workdir / captures).iterdir()):
        n = d.name
        if not (d / "routing.npz").exists() or not usable_capture(n):
            continue
        x, ls = routing.load(d / "routing.npz")
        all_layers = all_layers or ls
        if ls != all_layers:
            raise ValueError(f"{n}: layers {ls} differ from {all_layers}")
        ids[n] = x
        if is_special(n):
            continue
        meta, starts, ends = request_bounds(workdir, n)
        split = split_of(n, starts, ends)
        for k, m in enumerate(meta):
            if starts[k] + skip_prefix >= ends[k]:
                continue  # nothing left once the prefix is dropped
            samples.append(
                dict(
                    seq=n,
                    start=int(starts[k]) + skip_prefix,
                    end=int(ends[k]),
                    domain=m["domain"],
                    lang=m["lang"] or "?",
                    group=LANG_GROUP.get(m["lang"], "other"),
                    split=split[k],
                )
            )
    if not ids:
        raise ValueError(f"no routing captures under {workdir / captures}")
    if layers:
        keep = [all_layers.index(int(l)) for l in layers]
        ids = {n: v[keep] for n, v in ids.items()}
        all_layers = [all_layers[i] for i in keep]
    return Corpus(ids, all_layers, samples)


def build_frames(
    corpus: Corpus,
    pool,
    n_chunks: int,
    seed: int,
    rows: int,
    num_experts: int,
    chunk_tokens: int,
    strata: str = "domain",
):
    """counts [layers, n_chunks, rows, num_experts] of synthetic chunks, and their token composition."""
    rng = np.random.default_rng(seed)
    tree = defaultdict(lambda: defaultdict(list))
    for s in pool:
        tree[s["domain"] if strata == "domain" else s["seq"]][s["seq"]].append(s)
    queues = {}

    def draw(top, src):
        q = queues.get(src)
        if not q:
            q = queues[src] = list(rng.permutation(len(tree[top][src])))
        return tree[top][src][q.pop()]

    tops = sorted(tree)
    tok_top = {t: 0 for t in tops}
    tok_src = {(t, s): 0 for t in tops for s in tree[t]}
    comp_dom, comp_grp = defaultdict(int), defaultdict(int)
    L = len(corpus.layers)
    out = np.zeros((L, n_chunks, rows, num_experts), np.int32)
    for c in range(n_chunks):
        parts, fill = [], 0
        while fill < chunk_tokens:
            top = min(tops, key=lambda t: tok_top[t] + rng.random())
            src = min(sorted(tree[top]), key=lambda s: tok_src[top, s] + rng.random())
            s = draw(top, src)
            take = min(s["end"] - s["start"], chunk_tokens - fill)
            tok_top[top] += take
            tok_src[top, src] += take
            parts.append(corpus.ids[s["seq"]][:, s["start"] : s["start"] + take])
            fill += take
            comp_dom[s["domain"]] += take
            comp_grp[s["group"]] += take
        out[:, c] = routing.chunk_counts(np.concatenate(parts, axis=1), rows, num_experts)
    tot = n_chunks * chunk_tokens
    return out, dict(
        chunks=n_chunks,
        domain={k: round(v / tot, 3) for k, v in sorted(comp_dom.items())},
        lang_group={k: round(v / tot, 3) for k, v in sorted(comp_grp.items())},
    )


def contiguous_frames(x: np.ndarray, rows: int, num_experts: int, chunk_tokens: int) -> np.ndarray:
    return np.stack(
        [
            routing.chunk_counts(x[:, c * chunk_tokens : (c + 1) * chunk_tokens], rows, num_experts)
            for c in range(x.shape[1] // chunk_tokens)
        ],
        1,
    )


def pack_heldout(workdir: Path, name: str = "heldout_mix_bal", seed: int = 777, tokens: int = 56320) -> dict:
    """A domain-balanced sequence from TEST requests only, for the on-device A/B; plus the spliced routing of
    the same requests (heldout/<name>/routing_spliced.npz) for a modeled estimate of exactly this sequence."""
    samples = []
    for d in sorted((workdir / "captures").iterdir()):
        n = d.name
        if not (d / "routing.npz").exists() or is_special(n) or not usable_capture(n):
            continue
        meta, starts, ends = request_bounds(workdir, n)
        split = split_of(n, starts, ends)
        for k, m in enumerate(meta):
            if split[k] == "test":
                samples.append(
                    dict(
                        seq=n,
                        start=int(starts[k]),
                        end=int(ends[k]),
                        domain=m["domain"],
                        lang=m["lang"],
                        source=m["source"],
                        kind=m["kind"],
                    )
                )
    rng = np.random.default_rng(seed)
    tree = defaultdict(lambda: defaultdict(list))
    for s in samples:
        tree[s["domain"]][s["seq"]].append(s)
    # canonical order first: the tie-breaking draws follow this order, so it fixes the sequence
    domains = [d for d in DOMAINS if d in tree] + sorted(d for d in tree if d not in DOMAINS)
    queues, tok_dom, tok_src, token_cache = {}, defaultdict(int), defaultdict(int), {}
    ids, sample_of, out_samples, parts = [], [], [], []
    while len(ids) < tokens:
        dom = min(domains, key=lambda t: tok_dom[t] + rng.random())
        src = min(sorted(tree[dom]), key=lambda s: tok_src[s] + rng.random())
        q = queues.get(src)
        if not q:
            q = queues[src] = list(rng.permutation(len(tree[dom][src])))
        s = tree[dom][src][q.pop()]
        if s["seq"] not in token_cache:
            token_cache[s["seq"]] = json.loads((workdir / "sequences" / s["seq"] / "metadata.json").read_text())[
                "token_ids"
            ]
        take = min(s["end"] - s["start"], tokens - len(ids))
        ids += token_cache[s["seq"]][s["start"] : s["start"] + take]
        sample_of += [len(out_samples)] * take
        out_samples.append(
            dict(
                source=s["source"],
                domain=dom,
                lang=s["lang"],
                kind=s["kind"],
                tokens=take,
                from_seq=s["seq"],
                from_start=s["start"],
            )
        )
        parts.append((s["seq"], s["start"], take))
        tok_dom[dom] += take
        tok_src[src] += take
    if (workdir / "sequences" / name).exists():
        raise FileExistsError(f"sequence {name} already exists")
    info = write_sequence(workdir, name, ids, sample_of, out_samples)
    caps = {}
    spliced = []
    for seq, start, take in parts:
        if seq not in caps:
            caps[seq] = routing.load(workdir / "captures" / seq / "routing.npz")
        spliced.append(caps[seq][0][:, start : start + take])
    layers = next(iter(caps.values()))[1]
    routing.save(workdir / "heldout" / name / "routing_spliced.npz", np.concatenate(spliced, axis=1), layers)
    langs = defaultdict(int)
    for s in out_samples:
        langs[s["lang"]] += s["tokens"]
    return dict(
        name=name,
        requests=len(out_samples),
        tokens_sha256=tokens_sha256(ids),
        domain_share={k: round(v / tokens, 3) for k, v in sorted(tok_dom.items())},
        top_langs=dict(sorted(langs.items(), key=lambda x: -x[1])[:10]),
        **{k: v for k, v in info.items() if k == "unique_tokens"},
    )
