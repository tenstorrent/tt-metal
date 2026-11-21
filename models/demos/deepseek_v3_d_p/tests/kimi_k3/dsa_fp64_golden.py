# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP64 CPU golden of a Kimi-K3.1 DSA layer (the one the teacher-fed input names: 92 for the device test) and the GPU /
device / fp32 errors against it.

Runs on CPU only and never imports ttnn. Reads the bf16 input that ``dsa_teacher_feed.py input`` wrote: the same rows
the device test ran on. Three steps, each resumable by segment with a ``manifest.json`` (the KDA long-context golden's
helpers):

1. ``keys``: every row's KV latent (``kv_a_layernorm``), its unrotated k_pe and its indexer key (LayerNorm with
   bias), in FP64. Linear in the context, about 1.3 TFLOP at 128K and 11 at 1M. Every segment is checked against the
   trace's ``kv_post_transform`` and ``indexer_k`` and fails closed below ``KEY_CHECK_PCC``.
2. ``queries``: a sampled set of query rows, evenly spread over the context, because scoring every row is quadratic.
   Per row, in FP64:
   * the indexer score of every causal key, ``sum_h relu(q_h . k) w_h`` with ``w = weights_proj(x) 32^-0.5 128^-0.5``;
   * its top-2048 (``order`` keeps the best 4096 by score, best first);
   * the gated attention over that selection: absorbed MLA, softmax scale 192^-0.5, NoPE, ``sigmoid(g_proj x)``
     before ``o_proj``.
   Arms, each over the FP64 keys and changing one thing:
   * ``kv_bf16``, ``kv_fp8``, ``kv_fp8_ue8m0``: the selected keys as a cache would store them. That is a bf16
     latent and rope; for the fp8 arms the latent is then fp8 e4m3 with one fp32 scale per 128 columns, amax/448,
     the device's ``ue8m0`` arm rounding that scale up to a power of two.
   * ``idx_fp8``: the selection of an indexer that reads fp8 keys and queries as vLLM's does (bf16 key, then fp8
     with one ue8m0 scale per token; query fp8 with one amax/448 scale per token and head), with exact attention.
3. ``compare``: per sampled row, the GPU trace, a device run of ``test_dsa_teacher_feed.py`` and the fp32 host
   recompute of ao3i.5.11.1 (``fp32_recompute``) against the golden. The output is a per-row attention NRMSE, floored
   at 1% of the median golden row norm as the KDA golden does. For the trace's and the device's own selections it
   recomputes FP64 attention over those keys with each cache format, and over the path's own cache contents (the
   device's saved cache; the trace's kv_post_transform as an ue8m0 FP8 cache). That splits each path's error into
   selection, cache contents and attention arithmetic, and shows which cache format its arithmetic matches. Selection agreement with the golden is
   rank-based: each missed key's rank in the golden's order, as the teacher feed's miss bar.

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_fp64_golden keys --input IN --out G
    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_fp64_golden queries --out G --rows 512
    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_fp64_golden compare --out G --trace T --device DEV
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import golden_floor, pcc
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    MANIFEST,
    atomic_save,
    resume_manifest,
    segment_path,
    write_manifest,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_teacher_feed import (
    EPS,
    Trace,
    checkpoint_tensors,
    chunk_path,
    load_input,
    missed_keys,
    rms_norm,
    topk_recall,
)

C = KimiK31Config
KEY_CHECK_PCC = 0.9999  # FP64 keys vs the trace's bf16 ones; measured >= 0.99999 at 128K (report)
ORDER = 2 * C.INDEX_TOPK  # keys of the golden's score order kept per row: the selection and as many runners-up
ARMS = ("fp64", "kv_bf16", "kv_fp8", "kv_fp8_ue8m0", "idx_fp8")
KV_FORMATS = ("exact", "bf16", "fp8", "fp8_ue8m0")
WEIGHTS = {  # name in the checkpoint's self_attn -> what it is
    "q_a_proj.weight": "q down",
    "q_a_layernorm.weight": "q norm",
    "q_b_proj.weight": "q up (96 x (128 nope + 64 rope))",
    "kv_a_proj_with_mqa.weight": "kv down (512 latent + 64 rope)",
    "kv_a_layernorm.weight": "latent norm",
    "kv_b_proj.weight": "kv up (96 x (128 k + 128 v))",
    "g_proj.weight": "output gate",
    "o_proj.weight": "output projection",
    "indexer.wq_b.weight": "index query (32 x 128)",
    "indexer.wk.weight": "index key",
    "indexer.k_norm.weight": "index key LayerNorm weight",
    "indexer.k_norm.bias": "index key LayerNorm bias",
    "indexer.weights_proj.weight": "index head weights",
}


@dataclass(frozen=True)
class Dims:
    """The layer's dimensions; K3.1's by default, smaller ones in the contract tests."""

    heads: int = C.NUM_ATTENTION_HEADS
    nope: int = C.QK_NOPE_HEAD_DIM
    rope: int = C.QK_ROPE_HEAD_DIM
    v: int = C.V_HEAD_DIM
    lora: int = C.KV_LORA_RANK
    index_heads: int = C.INDEX_N_HEADS
    index_dim: int = C.INDEX_HEAD_DIM
    topk: int = C.INDEX_TOPK
    eps: float = EPS


def layer_weights(checkpoint: Path, layer: int) -> dict[str, torch.Tensor]:
    """The layer's attention weights in FP64."""
    tensors = checkpoint_tensors(checkpoint, layer, [f"self_attn.{n}" for n in WEIGHTS])
    return {n: tensors[f"self_attn.{n}"].double() for n in WEIGHTS}


def rms64(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def layer_keys(x: torch.Tensor, w: dict, dims: Dims = Dims()) -> dict[str, torch.Tensor]:
    """Per row of ``x`` (any float dtype, computed in FP64): KV latent, k_pe, indexer key."""
    x = x.double()
    kv = x @ w["kv_a_proj_with_mqa.weight"].T
    latent = rms64(kv[:, : dims.lora], w["kv_a_layernorm.weight"], dims.eps)
    key = torch.nn.functional.layer_norm(
        x @ w["indexer.wk.weight"].T,
        (dims.index_dim,),
        w["indexer.k_norm.weight"],
        w["indexer.k_norm.bias"],
        dims.eps,
    )
    return {"latent": latent, "kpe": kv[:, dims.lora :], "index_key": key}


def layer_queries(x: torch.Tensor, w: dict, dims: Dims = Dims()) -> dict[str, torch.Tensor]:
    """Per row of ``x``, in FP64: attention query (nope, rope), output gate, index query and index head weights."""
    x = x.double()
    qr = rms64(x @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"], dims.eps)
    q = (qr @ w["q_b_proj.weight"].T).reshape(-1, dims.heads, dims.nope + dims.rope)
    return {
        "q_nope": q[..., : dims.nope],
        "q_pe": q[..., dims.nope :],
        "gate": torch.sigmoid(x @ w["g_proj.weight"].T),
        "index_q": (qr @ w["indexer.wq_b.weight"].T).reshape(-1, dims.index_heads, dims.index_dim),
        "index_w": (x @ w["indexer.weights_proj.weight"].T) * dims.index_heads**-0.5 * dims.index_dim**-0.5,
    }


def index_scores(index_q: torch.Tensor, index_w: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """One row's indexer score of each key: ``sum_h relu(q_h . k) w_h``; ``index_q`` [H, D], ``index_w`` [H]."""
    return torch.relu(index_q @ keys.T).T @ index_w


def fp8_e4m3(x: torch.Tensor, block: int, pow2_scale: bool) -> torch.Tensor:
    """``x`` through fp8 e4m3 with one scale per ``block`` consecutive last-dim values, amax / 448 (amax clamped at
    1e-4, as the device and DeepEP), rounded up to a power of two when ``pow2_scale`` (ue8m0); dequantized, in x's
    dtype."""
    g = x.reshape(*x.shape[:-1], -1, block)
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    if pow2_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return ((g / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale).reshape(x.shape)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(x.dtype)


def cache_format(latent: torch.Tensor, kpe: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected keys as a cache of format ``fmt`` returns them (``KV_FORMATS``)."""
    if fmt == "exact":
        return latent, kpe
    latent, kpe = bf16(latent), bf16(kpe)
    if fmt == "fp8":
        latent = fp8_e4m3(latent, 128, pow2_scale=False)
    elif fmt == "fp8_ue8m0":
        latent = fp8_e4m3(latent, 128, pow2_scale=True)
    elif fmt != "bf16":
        raise ValueError(f"unknown cache format {fmt}")
    return latent, kpe


def index_fp8(index_q: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Index query and keys as vLLM's fp8 indexer reads them (ao3i.5.7 emulation): bf16 key in fp8 with one ue8m0
    scale per token, query in fp8 with one amax/448 scale per token and head."""
    return fp8_e4m3(bf16(index_q), index_q.shape[-1], False), fp8_e4m3(bf16(keys), keys.shape[-1], True)


def attend(q: dict, row: int, latent: torch.Tensor, kpe: torch.Tensor, w: dict, dims: Dims = Dims()) -> torch.Tensor:
    """Gated MLA output ``[hidden]`` of query row ``row`` of ``q`` (``layer_queries``) over the given keys
    (``latent`` [n, lora], ``kpe`` [n, rope]), absorbed: q_nope W_k scores the latent and W_v maps the latent mix."""
    kvb = w["kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)
    q_latent = torch.einsum("hd,hdl->hl", q["q_nope"][row], kvb[:, : dims.nope])
    scores = (q_latent @ latent.T + q["q_pe"][row] @ kpe.T) * (dims.nope + dims.rope) ** -0.5
    mixed = scores.softmax(-1) @ latent  # [heads, lora]
    out = torch.einsum("hl,hvl->hv", mixed, kvb[:, dims.nope :]).reshape(-1) * q["gate"][row]
    return out @ w["o_proj.weight"].T


# --- step 1: keys -----------------------------------------------------------------------------------------------


def input_identity(input_dir: Path) -> dict:
    manifest = json.loads((input_dir / MANIFEST).read_text())
    return {k: manifest.get(k) for k in ("layer", "trace", "trace_commit", "input", "norm", "checkpoint", "count")}


def build_keys(
    *, input_dir: Path, checkpoint: Path, out: Path, segment: int, threads: int, trace: Trace | None
) -> dict:
    """Write (or resume) the FP64 keys of every input row into ``out/keys``."""
    run = out / "keys"
    run.mkdir(parents=True, exist_ok=True)
    source = input_identity(input_dir)
    count, layer = source["count"], source["layer"]
    expected = {
        "step": "keys",
        "input": str(input_dir),
        "input_identity": source,
        "checkpoint": str(checkpoint),
        "count": count,
        "segment": segment,
        "dtype": "float64",
        "threads": threads,
    }
    segments = -(-count // segment)
    manifest = resume_manifest(run, expected, segments)
    w = layer_weights(checkpoint, layer)
    for index in range(len(manifest["segments"]), segments):
        start, end = index * segment, min((index + 1) * segment, count)
        began = time.perf_counter()
        keys = layer_keys(load_input(input_dir, start, end), w)
        check = {}
        if trace is not None:
            kv = trace.rows(f"kv_post_transform_layer_{layer}", start, end)
            check = {
                "latent_pcc": pcc(kv[:, : C.KV_LORA_RANK], keys["latent"]),
                "kpe_pcc": pcc(kv[:, C.KV_LORA_RANK :], keys["kpe"]),
                "index_key_pcc": pcc(trace.rows(f"indexer_k_layer_{layer}", start, end), keys["index_key"]),
            }
            if min(check.values()) < KEY_CHECK_PCC:
                raise ValueError(f"rows [{start}, {end}): FP64 keys fail their check {check} (< {KEY_CHECK_PCC})")
        atomic_save({"start": start, "end": end, **keys}, segment_path(run, index))
        seconds = time.perf_counter() - began
        manifest["segments"].append({"start": start, "end": end, "seconds": round(seconds, 2), **check})
        write_manifest(run, manifest)
        print(
            f"{time.strftime('%H:%M:%S')} keys segment {index + 1}/{segments} rows [{start}, {end}) {seconds:.1f} s "
            + " ".join(f"{k}={v:.7f}" for k, v in check.items()),
            flush=True,
        )
    return manifest


def load_keys(out: Path, end: int | None = None) -> dict[str, torch.Tensor]:
    """The FP64 keys of rows ``[0, end)`` (default: all); the keys step must be complete up to ``end``."""
    run = out / "keys"
    manifest = json.loads((run / MANIFEST).read_text())
    end = manifest["count"] if end is None else end
    done = manifest["segments"][-1]["end"] if manifest["segments"] else 0
    if end > done:
        raise ValueError(f"{run} holds keys for {done} rows, {end} needed")
    parts = [torch.load(segment_path(run, i), weights_only=True) for i in range(-(-end // manifest["segment"]))]
    return {k: torch.cat([p[k] for p in parts])[:end] for k in ("latent", "kpe", "index_key")}


# --- step 2: sampled queries -------------------------------------------------------------------------------------


def input_rows(input_dir: Path, rows: list[int]) -> torch.Tensor:
    """The teacher-fed input of the given rows, ``[len(rows), hidden]`` bf16, reading each input segment once."""
    segment = json.loads((input_dir / MANIFEST).read_text())["segment"]
    out = [None] * len(rows)
    for index in sorted({r // segment for r in rows}):
        payload = torch.load(segment_path(input_dir, index), weights_only=True)
        for i, r in enumerate(rows):
            if r // segment == index:
                out[i] = payload["normed"][r - payload["start"]]
    return torch.stack(out)


def sample_rows(count: int, n: int, lo: int) -> list[int]:
    """``n`` query rows evenly spread over ``[lo, count)``, both ends included, distinct."""
    return sorted(set(torch.linspace(lo, count - 1, n).round().long().tolist()))


def golden_row(q: dict, i: int, row: int, keys: dict, keys_fp8: torch.Tensor, w: dict, dims: Dims = Dims()) -> dict:
    """Every arm of one query row (``row``, the ``i``-th row of ``q``) over the keys of rows ``[0, row]``."""
    causal = row + 1
    scores = index_scores(q["index_q"][i], q["index_w"][i], keys["index_key"][:causal])
    top = scores.topk(min(ORDER, causal))
    sel = top.indices[: min(dims.topk, causal)]
    outs = {}
    for arm, fmt in zip(ARMS[:4], KV_FORMATS):
        outs[arm] = attend(q, i, *cache_format(keys["latent"][sel], keys["kpe"][sel], fmt), w, dims)
    iq8, _ = index_fp8(q["index_q"][i], keys["index_key"][:1])
    sel8 = index_scores(iq8, q["index_w"][i], keys_fp8[:causal]).topk(min(dims.topk, causal)).indices
    outs["idx_fp8"] = attend(q, i, keys["latent"][sel8], keys["kpe"][sel8], w, dims)
    pad = ORDER - top.indices.numel()
    return {
        "out": outs,
        "order": torch.nn.functional.pad(top.indices, (0, pad), value=-1),
        "order_score": torch.nn.functional.pad(top.values, (0, pad), value=float("nan")),
        "sel_idx_fp8": torch.nn.functional.pad(sel8, (0, dims.topk - sel8.numel()), value=-1),
    }


def build_queries(*, out: Path, checkpoint: Path, name: str, rows: list[int], segment: int, threads: int) -> dict:
    """Write (or resume) the golden of the sampled ``rows`` into ``out/queries-<name>``."""
    keys_manifest = json.loads((out / "keys" / MANIFEST).read_text())
    run = out / f"queries-{name}"
    run.mkdir(parents=True, exist_ok=True)
    expected = {
        "step": "queries",
        "keys": {k: keys_manifest[k] for k in ("input", "input_identity", "checkpoint", "count", "segment")},
        "checkpoint": str(checkpoint),
        "rows": rows,
        "segment": segment,
        "order": ORDER,
        "arms": list(ARMS),
        "dtype": "float64",
        "threads": threads,
    }
    segments = -(-len(rows) // segment)
    manifest = resume_manifest(run, expected, segments)
    if len(manifest["segments"]) == segments:
        return manifest
    input_dir = Path(keys_manifest["input"])
    w = layer_weights(checkpoint, keys_manifest["input_identity"]["layer"])
    keys = load_keys(out, max(rows) + 1)
    inputs = dict(zip(rows, input_rows(input_dir, rows)))
    _, keys_fp8 = index_fp8(keys["index_key"][:1], keys["index_key"])
    for index in range(len(manifest["segments"]), segments):
        batch = rows[index * segment : (index + 1) * segment]
        began = time.perf_counter()
        q = layer_queries(torch.stack([inputs[r] for r in batch]), w)
        per_row = [golden_row(q, i, r, keys, keys_fp8, w) for i, r in enumerate(batch)]
        payload = {
            "rows": torch.tensor(batch),
            "out": {a: torch.stack([p["out"][a] for p in per_row]) for a in ARMS},
            **{k: torch.stack([p[k] for p in per_row]) for k in ("order", "order_score", "sel_idx_fp8")},
        }
        atomic_save(payload, segment_path(run, index))
        seconds = time.perf_counter() - began
        manifest["segments"].append({"rows": [batch[0], batch[-1]], "seconds": round(seconds, 2)})
        write_manifest(run, manifest)
        print(
            f"{time.strftime('%H:%M:%S')} queries segment {index + 1}/{segments} rows {batch[0]}..{batch[-1]} "
            f"{seconds:.1f} s",
            flush=True,
        )
    return manifest


def load_queries(out: Path, name: str) -> dict:
    run = out / f"queries-{name}"
    manifest = json.loads((run / MANIFEST).read_text())
    if len(manifest["segments"]) != -(-len(manifest["rows"]) // manifest["segment"]):
        raise ValueError(f"{run} is incomplete")
    parts = [torch.load(segment_path(run, i), weights_only=True) for i in range(len(manifest["segments"]))]
    golden = {k: torch.cat([p[k] for p in parts]) for k in ("rows", "order", "order_score", "sel_idx_fp8")}
    golden["out"] = {a: torch.cat([p["out"][a] for p in parts]) for a in ARMS}
    return golden


# --- step 3: compare -----------------------------------------------------------------------------------------------


def fp32_recompute(x: torch.Tensor, w32: dict, kv: torch.Tensor, dims: Dims = Dims()) -> torch.Tensor:
    """ao3i.5.11.1's fp32 host recompute of one row: q through the bf16 KimiRMSNorm, the trace's bf16
    ``kv_post_transform`` rows at the trace's own selection, fp32 arithmetic."""
    x = x.float()
    qr = rms_norm(x @ w32["q_a_proj.weight"].T, w32["q_a_layernorm.weight"]).float()
    qq = (qr @ w32["q_b_proj.weight"].T).reshape(-1, dims.heads, dims.nope + dims.rope)
    q = {
        "q_nope": qq[..., : dims.nope],
        "q_pe": qq[..., dims.nope :],
        "gate": torch.sigmoid(x @ w32["g_proj.weight"].T),
    }
    kv = kv.float()
    return attend(q, 0, kv[:, : dims.lora], kv[:, dims.lora :], w32, dims)


def rank_agreement(got: torch.Tensor, order: torch.Tensor, selected: int) -> dict:
    """Selection ``got`` [k] (-1 = none) against the golden's: recall of the golden's ``selected`` best keys, each
    missed key's rank in the golden's order, and how many of ``got``'s extra keys rank beyond ``ORDER``."""
    want = order[:selected].unsqueeze(0)
    recall = float(topk_recall(got.unsqueeze(0), want)[0])
    m = missed_keys(got.unsqueeze(0), want, -torch.arange(selected, dtype=torch.float64).unsqueeze(0))
    extra = got[(got >= 0) & ~torch.isin(got, order[:selected])]
    return {
        "recall": recall,
        "misses": int(m["rank"].numel()),
        "miss_rank_min": int(m["rank"].min()) if m["rank"].numel() else -1,
        "miss_top_half": int((2 * m["rank"] < selected).sum()),
        "extra_beyond_order": int((~torch.isin(extra, order[order >= 0])).sum()),
    }


def compare(*, out: Path, name: str, trace: Trace, device: Path | None, checkpoint: Path, report: Path) -> dict:
    """Score the trace, a device run and the fp32 recompute against the golden of the ``name`` query set."""
    golden = load_queries(out, name)
    keys_manifest = json.loads((out / "keys" / MANIFEST).read_text())
    input_dir = Path(keys_manifest["input"])
    rows = golden["rows"].tolist()
    keys = load_keys(out, max(rows) + 1)
    layer = keys_manifest["input_identity"]["layer"]
    w = layer_weights(checkpoint, layer)
    w32 = {k: v.float() for k, v in w.items()}
    dev_manifest = json.loads((device / MANIFEST).read_text()) if device else None
    if dev_manifest is not None and not dev_manifest.get("complete"):
        raise ValueError(f"{device} holds no complete device run")
    if dev_manifest is not None and dev_manifest["layer"] != layer:
        raise ValueError(f"{device} ran layer {dev_manifest['layer']}, the golden is layer {layer}")
    inputs = input_rows(input_dir, rows)
    kv_all = trace.rows(f"kv_post_transform_layer_{layer}", 0, max(rows) + 1)
    lora = C.KV_LORA_RANK
    # Each path's own cache contents: the trace's bf16 kv_post_transform as the ue8m0 FP8 cache the GPU's attention
    # matches best (report), and the device's saved cache (its own projection, FP8-quantized, read back).
    own_cache = {"trace": torch.cat(cache_format(kv_all[:, :lora].float(), kv_all[:, lora:].float(), "fp8_ue8m0"), -1)}
    if dev_manifest is not None:
        own_cache["device"] = torch.load(device / "caches.pt", weights_only=True)["kvpe"]
    ref = golden["out"]["fp64"]
    floor = golden_floor(ref.norm(dim=-1))

    def nr(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.double() - b.double()).norm() / max(float(b.double().norm()), floor))

    records = []
    for i, r in enumerate(rows):
        x = inputs[i : i + 1]
        q = layer_queries(x, w)
        selected = min(C.INDEX_TOPK, r + 1)
        order = golden["order"][i]
        g = ref[i]
        sel = {"trace": trace.rows(f"dsa_topk_indices_layer_{layer}", r, r + 1)[0].long()}
        prod = {"trace": trace.rows(f"attn_output_layer_{layer}", r, r + 1)[0]}
        if dev_manifest is not None:
            payload = torch.load(chunk_path(device, r // dev_manifest["chunk"]), weights_only=True)
            sel["device"] = payload["topk"][r - payload["start"]].long()
            prod["device"] = payload["attn_out"][r - payload["start"]]
        tsel = sel["trace"][sel["trace"] >= 0]
        prod["fp32"] = fp32_recompute(x, w32, kv_all[tsel])
        rec = {"row": r, "golden_norm": float(g.norm())}
        for path, out_row in prod.items():
            rec[f"{path}_vs_fp64"] = nr(out_row, g)
        rec["device_vs_trace"] = nr(prod["device"], prod["trace"]) if "device" in prod else float("nan")
        rec["fp32_vs_trace"] = nr(prod["fp32"], prod["trace"])
        for arm in ARMS[1:]:
            rec[f"{arm}_vs_fp64"] = nr(golden["out"][arm][i], g)
        for path, s in sel.items():
            s = s[(s >= 0) & (s <= r)].unique()
            for fmt in KV_FORMATS:
                at = attend(q, 0, *cache_format(keys["latent"][s], keys["kpe"][s], fmt), w)
                rec[f"fp64_{path}sel_{fmt}_vs_fp64"] = nr(at, g)  # selection (+ format) alone
                rec[f"{path}_vs_fp64_{path}sel_{fmt}"] = nr(prod[path], at)  # the path's arithmetic beyond that
            cached = own_cache[path][s].double()
            at = attend(q, 0, cached[:, :lora], cached[:, lora:], w)
            rec[f"fp64_{path}sel_owncache_vs_fp64"] = nr(at, g)  # selection and the path's cache contents
            rec[f"{path}_vs_fp64_{path}sel_owncache"] = nr(prod[path], at)  # the path's attention arithmetic alone
            for k, v in rank_agreement(sel[path], order, selected).items():
                rec[f"{path}_{k}"] = v
        for k, v in rank_agreement(golden["sel_idx_fp8"][i], order, selected).items():
            rec[f"idx_fp8_{k}"] = v
        # convention pin: the golden's score of the trace's keys against the trace's own scores
        tscore = trace.rows(f"dsa_topk_scores_layer_{layer}", r, r + 1)[0][sel["trace"] >= 0].double()
        gscore = index_scores(q["index_q"][0], q["index_w"][0], keys["index_key"][tsel])
        rec["trace_score_pcc"] = pcc(gscore, tscore) if tsel.numel() > 1 else float("nan")
        rec["trace_score_ratio"] = float(tscore.norm() / gscore.norm())
        records.append(rec)
        print(f"{time.strftime('%H:%M:%S')} compared row {r} ({i + 1}/{len(rows)})", flush=True)
    result = summarize(records, floor)
    result.update({"golden": str(out), "queries": name, "trace": str(trace.path), "device": str(device or "")})
    report.mkdir(parents=True, exist_ok=True)
    (report / "metrics.json").write_text(json.dumps(result, indent=1))
    columns = list(records[0])
    (report / "rows.tsv").write_text(
        "\t".join(columns) + "\n" + "".join("\t".join(f"{rec[c]:.6g}" for c in columns) + "\n" for rec in records)
    )
    return result


def summarize(records: list[dict], floor: float, windows: int = 8) -> dict:
    """Median (and p90 for NRMSE) of every per-row metric, over all rows and per position window; recall means."""
    count = max(r["row"] for r in records) + 1
    gated = [r for r in records if r["row"] >= C.INDEX_TOPK]  # rows < top-k select every causal key

    def stats(recs: list[dict]) -> dict:
        s = {"rows": len(recs)}
        for key in records[0]:
            if key in ("row",):
                continue
            v = torch.tensor([r[key] for r in recs], dtype=torch.float64)
            v = v[~v.isnan()]
            if not v.numel():
                continue
            if key.endswith(("recall",)):
                s[key + "_mean"] = float(v.mean())
            elif key.endswith(("misses", "miss_top_half", "extra_beyond_order")):
                s[key + "_sum"] = float(v.sum())
            elif key.endswith("miss_rank_min"):
                s[key] = float(v[v >= 0].min()) if (v >= 0).any() else -1.0
            else:
                s[key + "_median"] = float(v.median())
                s[key + "_p90"] = float(v.quantile(0.9))
        return s

    edges = torch.linspace(0, count, windows + 1).round().long().tolist()
    return {
        "floor": floor,
        "overall": stats(records),
        "overall_rows_ge_topk": stats(gated) if gated else {},
        "windows": [
            {"start": lo, "end": hi, **stats(recs)}
            for lo, hi in zip(edges[:-1], edges[1:])
            if (recs := [r for r in records if lo <= r["row"] < hi])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    checkpoint = Path(
        os.environ.get("KIMI_K31_HF_MODEL", "/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized")
    )
    for cmd in ("keys", "queries", "compare"):
        p = sub.add_parser(cmd)
        p.add_argument("--out", type=Path, required=True, help="golden directory")
        p.add_argument("--checkpoint", type=Path, default=checkpoint)
        p.add_argument("--threads", type=int, default=16)
    sub.choices["keys"].add_argument("--input", type=Path, required=True, help="dsa_teacher_feed input directory")
    sub.choices["keys"].add_argument("--trace", type=Path, help="check every segment against this trace")
    sub.choices["keys"].add_argument("--segment", type=int, default=4096)
    for cmd in ("queries", "compare"):
        sub.choices[cmd].add_argument("--name", default="main", help="query set")
    sub.choices["queries"].add_argument("--rows", type=int, default=512)
    sub.choices["queries"].add_argument("--lo", type=int, default=0, help="first sampled row")
    sub.choices["queries"].add_argument("--segment", type=int, default=16)
    sub.choices["compare"].add_argument("--trace", type=Path, required=True)
    sub.choices["compare"].add_argument("--device", type=Path, help="a complete test_dsa_teacher_feed run directory")
    sub.choices["compare"].add_argument("--report", type=Path, help="default: <out>/compare-<name>")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "keys":
        build_keys(
            input_dir=args.input,
            checkpoint=args.checkpoint,
            out=args.out,
            segment=args.segment,
            threads=args.threads,
            trace=Trace(args.trace) if args.trace else None,
        )
    elif args.command == "queries":
        count = json.loads((args.out / "keys" / MANIFEST).read_text())["count"]
        rows = sample_rows(count, args.rows, args.lo)
        build_queries(
            out=args.out,
            checkpoint=args.checkpoint,
            name=args.name,
            rows=rows,
            segment=args.segment,
            threads=args.threads,
        )
    else:
        result = compare(
            out=args.out,
            name=args.name,
            trace=Trace(args.trace),
            device=args.device,
            checkpoint=args.checkpoint,
            report=args.report or args.out / f"compare-{args.name}",
        )
        print(json.dumps(result["overall_rows_ge_topk"] or result["overall"], indent=1))


if __name__ == "__main__":
    main()
