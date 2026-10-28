# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E6 datatype sweep harness for the text tower: accuracy vs HF fp32 + traced device time per
precision candidate, plus the report (CSV, Pareto charts).

    python .../tests/perf/dtype_sweep.py run --candidates base_bfp8_hifi2,bf16w_hifi2 --out sweep.jsonl
    python .../tests/perf/dtype_sweep.py run --candidates all --out sweep.jsonl
    python .../tests/perf/dtype_sweep.py list
    python .../tests/perf/dtype_sweep.py report --in sweep.jsonl --doc-dir .../doc/datatype_sweep --selected <id>

A candidate is a delta over BASELINE (the E5a text policy) in the `selected_precision_config.json`
schema (tt/precision.py). Every candidate is built through `TextModel.from_state_dict(...,
precision=<candidate>)`, the same construction path the default model uses with the selected file.

Per candidate, in one device session:
  accuracy (eager; replay == eager bit for bit, tests/pcc/test_trace.py), vs the live HF fp32
  reference, on the test cases of tests/pcc (tests/common.CASES + the batch-8 padded case):
    layer_pcc_gated  teacher-forced layer PCC on the set the gates check (every layer on
                     test_every_layer's cases, layers 0 / 5 on every case): bar 0.995
    layer_pcc_all    every layer on every case (reported, not gated)
    module_pcc       attention / MLP output of layers 0 / 5 on every case: bar 0.995
    last_hidden_pcc  free-running last hidden state over valid tokens: floor 0.99
    embedding_cos    per-prompt pooled embedding cosine: bar 0.99
    retrieval        demo text retrieval: TT argmax == HF fp32 argmax (and == [0, 2, 1])
  performance: traced device time (execute_trace + one sync per window, inputs resident) at
    1x128, 1x512, 1x2048, 8x512, 1x8192: burst median of `--reps` windows after `--idle` s idle,
    then sustained median after a `--soak` s soak. Ranking aggregate: burst 1x512 + burst 8x512.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch

GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down", "ple_proj", "ple_block", "final_proj")
FULL_LAYERS = [5, 11, 17, 23]


def _mm(dtype="bfp8", fidelity="HiFi2", fp32_acc=True, out_dtype="bf16"):
    return {"dtype": dtype, "fidelity": fidelity, "fp32_acc": fp32_acc, "out_dtype": out_dtype}


BASELINE = {
    "schema_version": 1,
    "config_id": "base_bfp8_hifi2",
    "description": "",
    "text": {
        "weights": {g: _mm() for g in GROUPS},
        "layer_exceptions": [],
        "sdpa": {"fidelity": "HiFi2", "fp32_acc": True, "exp_approx": False},
        "norms": {"fidelity": "HiFi4", "fp32_acc": True},
        "residual_dtype": "bf16",
        "mask_dtype": "bfp4",
    },
    # text candidates keep the E5 tower policies (precision.E5_VISION_POLICY / E5_AUDIO_POLICY)
    "vision": None,
    "audio": None,
    "kv_cache_dtype": "n/a: encoder, single chip (no KV cache)",
    "ccl_dtype": "n/a: encoder, single chip (no CCL)",
    "logits_dtype": "n/a: encoder, single chip (pooled fp32 embedding, no logits)",
    "sampling": "n/a: encoder, single chip (no sampling)",
}


def _weights(groups, **kw):
    return {"text": {"weights": {g: _mm(**kw) for g in groups}}}


def _candidates() -> dict[str, tuple[str, dict]]:
    c = {
        "base_bfp8_hifi2": ("E5a policy: bfp8 weights, HiFi2 + fp32 acc", {}),
        "bf16w_hifi2": ("all weights bf16, HiFi2", _weights(GROUPS, dtype="bf16")),
        "bfp8_lofi": ("all weights bfp8, LoFi", _weights(GROUPS, fidelity="LoFi")),
        "bfp8_hifi2_nofp32acc": ("all matmuls without fp32 dest accumulation", _weights(GROUPS, fp32_acc=False)),
    }
    for g in GROUPS:
        c[f"bfp4_{g}_hifi2"] = (f"{g} bfp4 HiFi2", _weights([g], dtype="bfp4"))
        c[f"bfp4_{g}_lofi"] = (f"{g} bfp4 LoFi", _weights([g], dtype="bfp4", fidelity="LoFi"))
    sd = BASELINE["text"]["sdpa"]
    c["sdpa_nofp32acc"] = ("SDPA fp32_dest_acc off", {"text": {"sdpa": {**sd, "fp32_acc": False}}})
    c["sdpa_exp_approx"] = ("SDPA exp_approx_mode on", {"text": {"sdpa": {**sd, "exp_approx": True}}})
    c["sdpa_lofi"] = ("SDPA LoFi", {"text": {"sdpa": {**sd, "fidelity": "LoFi"}}})
    c["norms_hifi2"] = ("norms HiFi2", {"text": {"norms": {"fidelity": "HiFi2", "fp32_acc": True}}})
    c["norms_nofp32acc"] = (
        "norms without fp32 accumulation",
        {"text": {"norms": {"fidelity": "HiFi4", "fp32_acc": False}}},
    )
    # activation dtype = projection output dtype, per group (`out_dtype`)
    c["act_bfp8"] = ("every projection output bfp8", _weights(GROUPS, out_dtype="bfp8"))
    c["act_bfp8_nofinal"] = (
        "every projection output bfp8 except final_proj",
        _weights([g for g in GROUPS if g != "final_proj"], out_dtype="bfp8"),
    )
    for g in GROUPS[:-1]:
        c[f"act_bfp8_{g}"] = (f"{g} output bfp8", _weights([g], out_dtype="bfp8"))
    c["resid_bfp8"] = ("residual stream bfp8", {"text": {"residual_dtype": "bfp8"}})
    c["mask_bfp8"] = ("masks bfp8", {"text": {"mask_dtype": "bfp8"}})
    c["mask_bf16"] = ("masks bf16", {"text": {"mask_dtype": "bf16"}})
    return c


CANDIDATES = _candidates()


def register_combo(cid: str, description: str, delta: dict) -> None:
    CANDIDATES[cid] = (description, delta)


def candidate_dict(cid: str, delta: dict | None = None, description: str | None = None) -> dict:
    from models.demos.embedding_gemma2.tt.precision import E5_AUDIO_POLICY, E5_VISION_POLICY, deep_merge

    if delta is None:
        description, delta = CANDIDATES[cid]
    base = dict(BASELINE, vision=E5_VISION_POLICY, audio=E5_AUDIO_POLICY)
    d = deep_merge(base, delta)
    d["config_id"] = cid
    d["description"] = description or ""
    return d


# ---------------------------------------------------------------------- accuracy
PERF_SHAPES = ((1, 128), (1, 512), (1, 2048), (8, 512), (1, 8192))
RANK_SHAPES = ((1, 512), (8, 512))
BAR_LAYER = 0.995
BAR_LAST = 0.99
BAR_COS = 0.99


def accuracy_cases():
    from models.demos.embedding_gemma2.tests.common import CASES

    batch8 = (512, 480, 400, 300, 257, 129, 64, 7)
    return tuple(CASES) + (batch8,)


PER_LAYER_CASES = ((1025,), (8192,), (1025, 513, 31))  # test_text_model.test_every_layer


class References:
    """HF fp32 reference outputs, computed once per process."""

    def __init__(self, snapshot, state_dict):
        from models.demos.embedding_gemma2.reference.hf_reference import load_reference_text_model

        self.snapshot = snapshot
        self.model, _ = load_reference_text_model(snapshot, state_dict)
        self.cache = {}

    def get(self, lengths):
        from models.demos.embedding_gemma2.reference.hf_reference import run_reference
        from models.demos.embedding_gemma2.tests.common import make_batch

        key = tuple(lengths)
        if key not in self.cache:
            ids, mask = make_batch(self.snapshot, list(lengths))
            with torch.no_grad():
                ref = run_reference(self.model, ids, mask, capture_layers=None, capture_sublayers=(0, 5))
            self.cache[key] = (ids, mask, ref)
        return self.cache[key]

    def retrieval(self):
        if "retrieval" not in self.cache:
            from transformers import AutoTokenizer

            from models.demos.embedding_gemma2.demo.demo import DOC_PREFIX, DOCUMENTS, QUERIES, QUERY_PREFIX
            from models.demos.embedding_gemma2.reference.hf_reference import run_reference

            tok = AutoTokenizer.from_pretrained(str(self.snapshot))
            out = []
            for texts in ([QUERY_PREFIX + t for t in QUERIES], [DOC_PREFIX + t for t in DOCUMENTS]):
                b = tok(texts, padding=True, return_tensors="pt")
                with torch.no_grad():
                    ref = run_reference(self.model, b["input_ids"], b["attention_mask"], capture_layers=())
                out.append((b["input_ids"], b["attention_mask"], ref.embedding))
            self.cache["retrieval"] = out
        return self.cache["retrieval"]


def run_accuracy(device, model, refs: References) -> dict:
    import ttnn
    from models.demos.embedding_gemma2.tests.common import (
        cosine_rows,
        download,
        pcc,
        upload_activation,
        upload_inputs,
        valid_rows,
    )

    m = model
    layer_gated, layer_all, module, last, cos = [], [], [], [], []
    for lengths in accuracy_cases():
        ids, mask, ref = refs.get(lengths)
        B, L = ids.shape
        case = list(lengths)
        ids_d, keep_d, S = upload_inputs(device, ids, mask, m.config.max_seq_len)
        m.prepare(S)
        # free running
        last_t, emb = m(ids_d, keep_d)
        p_last = pcc(valid_rows(ref.last_hidden, mask), valid_rows(download(last_t, B, L), mask))
        last.append((p_last, case))
        emb_t = ttnn.to_torch(emb).float().reshape(B, -1)
        for b, v in enumerate(cosine_rows(ref.embedding, emb_t).tolist()):
            cos.append((v, case, b))
        ttnn.deallocate(last_t)
        ttnn.deallocate(emb)
        # teacher forced per layer + modules of layers 0 / 5
        masks = m.masks(keep_d)
        proj = m.per_layer_input(m.embedding(ids_d))
        for i, layer in enumerate(m.layers):
            kind = m.config.layer_types[i]
            pli = m.per_layer_input.layer_input(proj, i)
            x = upload_activation(device, ref.layer_in[i], S)
            y = layer(x, pli, masks[kind])
            v = pcc(valid_rows(ref.layer_out[i], mask), valid_rows(download(y, B, L), mask))
            layer_all.append((v, case, i))
            if tuple(lengths) in PER_LAYER_CASES or i in (0, 5):
                layer_gated.append((v, case, i))
            for t in (pli, x, y):
                ttnn.deallocate(t)
            if i in (0, 5):
                xa = upload_activation(device, ref.attn_in[i], S)
                ya = layer.self_attn(xa, masks[kind])
                module.append(
                    (pcc(valid_rows(ref.attn_out[i], mask), valid_rows(download(ya, B, L), mask)), case, f"attn{i}")
                )
                xm = upload_activation(device, ref.mlp_in[i], S)
                ym = layer.mlp(xm)
                module.append(
                    (pcc(valid_rows(ref.mlp_out[i], mask), valid_rows(download(ym, B, L), mask)), case, f"mlp{i}")
                )
                for t in (xa, ya, xm, ym):
                    ttnn.deallocate(t)
        ttnn.deallocate(proj)
        for t in {id(v): v for v in masks.values()}.values():
            ttnn.deallocate(t)
        ttnn.deallocate(ids_d)
        ttnn.deallocate(keep_d)

    # demo text retrieval
    tt_embs, hf_embs = [], []
    for ids, mask, hf_emb in refs.retrieval():
        ids_d, keep_d, S = upload_inputs(device, ids, mask, m.config.max_seq_len)
        m.prepare(S)
        last_t, emb = m(ids_d, keep_d)
        tt_embs.append(ttnn.to_torch(emb).float().reshape(ids.shape[0], -1))
        hf_embs.append(hf_emb.float())
        for b, v in enumerate(cosine_rows(hf_emb, tt_embs[-1]).tolist()):
            cos.append((v, "retrieval", b))
    tt_arg = (tt_embs[0] @ tt_embs[1].T).argmax(-1).tolist()
    hf_arg = (hf_embs[0] @ hf_embs[1].T).argmax(-1).tolist()

    def worst(rows):
        r = min(rows, key=lambda t: t[0])
        return {"value": r[0], "where": list(r[1:])}

    res = {
        "layer_pcc_gated": worst(layer_gated),
        "layer_pcc_all": worst(layer_all),
        "module_pcc": worst(module),
        "last_hidden_pcc": worst(last),
        "embedding_cos": worst(cos),
        "retrieval_tt_argmax": tt_arg,
        "retrieval_hf_argmax": hf_arg,
        "retrieval_agree": tt_arg == hf_arg and tt_arg == [0, 2, 1],
        "n_layer_gated": len(layer_gated),
        "n_cos": len(cos),
    }
    fails = []
    if res["layer_pcc_gated"]["value"] < BAR_LAYER:
        fails.append(
            f"layer_pcc {res['layer_pcc_gated']['value']:.5f} < {BAR_LAYER} at {res['layer_pcc_gated']['where']}"
        )
    if res["module_pcc"]["value"] < BAR_LAYER:
        fails.append(f"module_pcc {res['module_pcc']['value']:.5f} < {BAR_LAYER} at {res['module_pcc']['where']}")
    if res["last_hidden_pcc"]["value"] < BAR_LAST:
        fails.append(f"last_hidden_pcc {res['last_hidden_pcc']['value']:.5f} < {BAR_LAST}")
    if res["embedding_cos"]["value"] < BAR_COS:
        fails.append(
            f"embedding_cos {res['embedding_cos']['value']:.5f} < {BAR_COS} at {res['embedding_cos']['where']}"
        )
    if not res["retrieval_agree"]:
        fails.append(f"retrieval argmax TT {tt_arg} HF {hf_arg}")
    res["screen_pass"] = not fails
    res["failing"] = fails
    return res


# ---------------------------------------------------------------------- performance
def _window_ms(enc, b, s, n) -> float:
    import ttnn

    t0 = time.perf_counter()
    for _ in range(n):
        enc.execute_captured(b, s)
    ttnn.synchronize_device(enc.device)
    return (time.perf_counter() - t0) * 1e3 / n


def parse_shapes(text: str) -> tuple[tuple[int, int], ...]:
    """'1x2048,8x512' -> ((1, 2048), (8, 512)); empty -> PERF_SHAPES."""
    if not text:
        return PERF_SHAPES
    return tuple(tuple(int(v) for v in s.split("x")) for s in text.split(","))


def run_perf(device, model, snapshot, args) -> dict:
    from models.demos.embedding_gemma2.tests.common import make_batch
    from models.demos.embedding_gemma2.tt.text_encoder import TextEncoder

    shapes = parse_shapes(args.shapes)
    enc = TextEncoder(model, device)
    enc.capture_all(shapes)
    out = {}
    try:
        for b, s in shapes:
            ids, mask = make_batch(snapshot, [s] * b)
            emb = TextEncoder.read_embedding(enc.replay(enc.prepare_inputs(ids, mask))[1], b)
            assert torch.isfinite(emb).all()
            _window_ms(enc, b, s, 3)
            burst = []
            for _ in range(args.reps):
                time.sleep(args.idle)
                burst.append(_window_ms(enc, b, s, args.burst))
            t_end = time.perf_counter() + args.soak
            while time.perf_counter() < t_end:
                _window_ms(enc, b, s, 5)
            n_sus = max(5, int(args.sustained_ms / max(burst)))
            sus = [_window_ms(enc, b, s, n_sus) for _ in range(args.reps)]
            out[f"{b}x{s}"] = {
                "burst_ms_median": statistics.median(burst),
                "burst_windows": burst,
                "sustained_ms_median": statistics.median(sus),
                "sustained_windows": sus,
                "sustained_replays_per_window": n_sus,
            }
            print(
                f"  PERF {b}x{s}: burst {out[f'{b}x{s}']['burst_ms_median']:.3f} ms "
                f"sustained {out[f'{b}x{s}']['sustained_ms_median']:.3f} ms",
                flush=True,
            )
            time.sleep(args.idle)
    finally:
        enc.release_traces()
    out["regime"] = {
        shape_key(b, s): f"{len(shapes)} trace(s) captured in this process: {','.join(shape_key(*x) for x in shapes)}"
        for b, s in shapes
    }
    _rank(out)
    return out


def shape_key(b: int, s: int) -> str:
    return f"{b}x{s}"


def _rank(perf: dict) -> None:
    """Ranking aggregate (burst 1x512 + 8x512), when both shapes were measured."""
    if all(shape_key(b, s) in perf for b, s in RANK_SHAPES):
        perf["rank_ms"] = sum(perf[shape_key(b, s)]["burst_ms_median"] for b, s in RANK_SHAPES)


def construction_evidence(model) -> dict:
    """What the constructed model actually holds: weight dtypes, compute configs, mask dtype."""

    def fid(ckc):
        return str(ckc.math_fidelity).split(".")[-1]

    def mm(p):
        return p.to_dict()

    ev = {"layers": {}}
    for i, layer in enumerate(model.layers):
        a, mlp, ple = layer.self_attn, layer.mlp, layer.ple_block
        ev["layers"][i] = {
            "wqkv": str(a.wqkv.dtype).split(".")[-1],
            "wo": str(a.wo.dtype).split(".")[-1],
            "w_gate": str(mlp.w_gate.dtype).split(".")[-1],
            "w_up": str(mlp.w_up.dtype).split(".")[-1],
            "w_down": str(mlp.w_down.dtype).split(".")[-1],
            "ple_gate": str(ple.w_gate.dtype).split(".")[-1],
            "ple_proj": str(ple.w_proj.dtype).split(".")[-1],
            "qkv_policy": mm(a.qkv_policy),
            "o_policy": mm(a.o_policy),
            "gate_up_policy": mm(mlp.gate_up_policy),
            "down_policy": mm(mlp.down_policy),
            "ple_block_policy": mm(ple.policy),
            "sdpa": {
                "fidelity": fid(a.sdpa_compute_kernel_config),
                "fp32_acc": bool(a.sdpa_compute_kernel_config.fp32_dest_acc_en),
                "exp_approx": a.sdpa_policy.exp_approx,
            },
            "norm_fidelity": fid(layer.input_layernorm.compute_kernel_config),
            "norm_fp32_acc": bool(layer.input_layernorm.compute_kernel_config.fp32_dest_acc_en),
            "residual_dtype": str(layer.residual_dtype).split(".")[-1],
        }
    ev["ple_proj"] = str(model.per_layer_input.proj_weight.dtype).split(".")[-1]
    ev["final_proj"] = str(model.w_out.dtype).split(".")[-1]
    ev["mask_dtype"] = str(model.masks.dtype).split(".")[-1]
    ev["final_proj_policy"] = model.out_policy.to_dict()
    ev["ple_proj_policy"] = model.per_layer_input.policy.to_dict()
    return ev


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def cmd_run(args) -> None:
    import ttnn
    from models.demos.embedding_gemma2.conftest import MODEL_ID, open_eg2_device
    from models.demos.embedding_gemma2.tt.config import TextModelConfig, default_weight_cache_dir
    from models.demos.embedding_gemma2.tt.precision import PrecisionConfig
    from models.demos.embedding_gemma2.tt.text_model import TextModel
    from models.demos.embedding_gemma2.tt.weights import load_text_config_dict, load_text_state_dict, resolve_snapshot

    if args.combos:
        for cid, spec in json.loads(Path(args.combos).read_text()).items():
            register_combo(cid, spec["description"], spec["delta"])
    ids = list(CANDIDATES) if args.candidates == "all" else args.candidates.split(",")
    snapshot = resolve_snapshot(MODEL_ID)
    state_dict = load_text_state_dict(snapshot)
    config = TextModelConfig.from_hf(load_text_config_dict(snapshot), max_seq_len=8192)
    cache_dir = default_weight_cache_dir() / snapshot.name
    refs = References(snapshot, state_dict) if not args.no_acc else None
    head = _git_head()
    for cid in ids:
        prec_dict = candidate_dict(cid)
        rec = {
            "id": cid,
            "description": prec_dict["description"],
            "delta": CANDIDATES[cid][1],
            "text_precision": prec_dict["text"],
            "commit": head,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": "sjc-snva-tp100 p150a",
        }
        print(f"== {cid}: {prec_dict['description']}", flush=True)
        device = open_eg2_device(0)  # EMBEDDING_GEMMA2_GRID; default: the device's own grid
        grid = device.compute_with_storage_grid_size()
        rec["compute_grid"] = [grid.x, grid.y]
        try:
            t0 = time.time()
            model = TextModel.from_state_dict(
                state_dict, config, device, cache_dir=cache_dir, precision=PrecisionConfig.from_dict(prec_dict, cid)
            )
            rec["construction"] = construction_evidence(model)
            if refs is not None:
                rec["accuracy"] = run_accuracy(device, model, refs)
                a = rec["accuracy"]
                print(
                    f"  ACC layer {a['layer_pcc_gated']['value']:.5f} module {a['module_pcc']['value']:.5f} "
                    f"last {a['last_hidden_pcc']['value']:.5f} cos {a['embedding_cos']['value']:.5f} "
                    f"retrieval {a['retrieval_agree']} -> {'PASS' if a['screen_pass'] else 'FAIL ' + '; '.join(a['failing'])}",
                    flush=True,
                )
            if not args.no_perf:
                rec["perf"] = run_perf(device, model, snapshot, args)
                if "rank_ms" in rec["perf"]:
                    print(f"  RANK {cid}: {rec['perf']['rank_ms']:.3f} ms (1x512 + 8x512 burst)", flush=True)
            if args.merge:
                rec["merge"] = True
            rec["seconds"] = time.time() - t0
        except Exception as e:  # noqa: BLE001 - a candidate the runtime cannot build/run is recorded as blocked
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-4000:]
            print(f"  ERROR {cid}: {rec['error'][:400]}", flush=True)
        finally:
            ttnn.close_device(device)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def cmd_list(_args) -> None:
    for cid, (desc, delta) in CANDIDATES.items():
        print(f"{cid:28s} {desc}   {json.dumps(delta)}")


# ---------------------------------------------------------------------- report
def _load(paths) -> list[dict]:
    """A later run of the same id replaces an earlier one, except `merge` records (`run --merge`,
    e.g. one perf shape per process), which add their accuracy / perf shapes to the earlier one."""
    recs = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            old = recs.get(r["id"])
            if r.pop("merge", False) and old is not None:
                if "accuracy" in r:
                    old["accuracy"] = r["accuracy"]
                if "perf" in r:
                    perf = old.setdefault("perf", {})
                    regime = {**perf.get("regime", {}), **r["perf"].get("regime", {})}
                    perf.update({k: v for k, v in r["perf"].items() if k not in ("regime", "rank_ms")})
                    perf["regime"] = regime
                    _rank(perf)
                if "error" in r:
                    old["error"] = r["error"]
                old.setdefault("merged_commits", []).append(r["commit"])
            else:
                recs[r["id"]] = r
    return list(recs.values())


def _pareto(points):
    """Non-dominated (lower ms, higher metric) points, sorted by ms."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1.0
    for p in pts:
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


def _chart(recs, metric_key, ylabel, bar, path, selected, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pts = [
        (
            r["perf"]["rank_ms"],
            r["accuracy"][metric_key]["value"],
            r["id"],
            r["accuracy"]["screen_pass"],
            _one_trace_per_process(r["perf"]),
        )
        for r in recs
        if "rank_ms" in r.get("perf", {}) and "accuracy" in r
    ]
    front = _pareto([(x, y, cid) for x, y, cid, _, _ in pts])
    front_ids = {p[2] for p in front}
    ys_ok = [y for _, y, _, ok, _ in pts if ok] or [bar]
    probe_key = {"last_hidden_pcc": "min_last_hidden_pcc", "embedding_cos": "min_embedding_cos"}.get(metric_key)
    probe_min = {r["id"]: r["probe"]["all"][probe_key] for r in recs if probe_key and r.get("probe")}
    ys_ok += [v for v in probe_min.values()]
    xs = [x for x, *_ in pts]
    # left: every evaluated config; right: zoom on the region around the gate minimum
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.0), dpi=150, gridspec_kw={"width_ratios": [1, 1.25]})
    span = max(max(ys_ok) - min(ys_ok), 1e-4)
    zoom_lo = min(min(ys_ok) - 0.25 * span, bar - 0.15 * span)
    zoom_hi = max(ys_ok) + 0.10 * (max(ys_ok) - zoom_lo)
    for k, ax in enumerate(axes):
        ax.set_facecolor("#fbfaf7")
        ax.set_xlim(min(xs) - 0.8, max(xs) + 1.6)
        if k == 1:
            ax.set_ylim(zoom_lo, zoom_hi)
            ax.set_title("zoom: the passing configs and the gate line", fontsize=9)
        else:
            lo = min([y for _, y, *_ in pts] + [bar])  # the gate line stays in view
            ax.set_ylim(lo - 0.04 * (1 - lo), 1 + 0.02 * (1 - lo))
            ax.set_ylabel(ylabel)
            ax.set_title("every evaluated config", fontsize=9)
        ax.plot([p[0] for p in front], [p[1] for p in front], color="#3b6ea8", linewidth=1.4, alpha=0.75, zorder=2)
        ax.axhline(bar, linestyle=":", color="#444444", linewidth=1.3, zorder=1)
        for x, y, cid, ok, one in pts:
            if probe_key and cid in probe_min:  # robustness probe: min over gate + probe inputs
                ax.plot([x, x], [y, probe_min[cid]], color="#7a5195", linewidth=0.9, alpha=0.8, zorder=2)
                ax.scatter(x, probe_min[cid], s=46, facecolor="none", edgecolor="#7a5195", linewidth=1.3, zorder=4)
            if cid == selected:
                ax.scatter(x, y, s=170, color="#d62728", edgecolor="black", linewidth=0.8, zorder=5)
            else:
                ax.scatter(
                    x,
                    y,
                    s=46,
                    marker="D" if one else "o",
                    color="#3b6ea8" if ok else "#b0b0b0",
                    edgecolor="#22323f",
                    linewidth=0.6,
                    zorder=3,
                )
        # labels: selected first, then the frontier / baseline / precision candidates, then the
        # rest; a label that would overlap one already placed tries other offsets, else is dropped
        fig.canvas.draw()
        order = sorted(
            pts,
            key=lambda p: (
                p[2] != selected,
                not (p[2] in front_ids or p[2].startswith(("base", "p3_", "bf16w"))),
                p[0],
            ),
        )
        placed = [ax.get_window_extent()]  # index 0 is the axes box (labels must fit inside)
        marks = [ax.transData.transform((x, y)) for x, y, *_ in pts]
        marks += [ax.transData.transform((x, probe_min[cid])) for x, _, cid, *_ in pts if cid in probe_min]
        for x, y, cid, ok, _ in order:
            if k == 1 and not (zoom_lo <= y <= zoom_hi):
                continue
            if k == 0 and not (cid == selected or cid in front_ids or cid.startswith(("base", "bf16w"))):
                continue
            color = "#d62728" if cid == selected else ("#22323f" if ok else "#8a8a8a")
            # left panel (dense): only the offsets next to the point, so a label never drifts
            # onto a neighbour; a label that fits nowhere is dropped (the right panel / CSV have it)
            offsets = ((6, 3), (6, -10), (-6, 3), (-6, -10), (6, 12), (6, -19), (-6, 12), (-6, -19))
            for dx, dy in offsets[:4] if k == 0 else offsets:
                t = ax.annotate(
                    cid,
                    (x, y),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=6.6 if cid != selected else 7.6,
                    fontweight="bold" if cid == selected else "normal",
                    color=color,
                    ha="left" if dx > 0 else "right",
                    zorder=6,
                )
                bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.04, 1.15)
                inside = placed[0].contains(bb.x0, bb.y0) and placed[0].contains(bb.x1, bb.y1)
                on_marker = any(bb.expanded(1.0, 1.6).contains(mx, my) for mx, my in marks)
                if inside and not on_marker and not any(bb.overlaps(o) for o in placed[1:]):
                    placed.append(bb)
                    break
                t.remove()
        ax.set_xlabel("traced device ms, burst median, 1x512 + 8x512 (lower is faster)")
        ax.grid(True, color="#e3e0d8", linewidth=0.6)
    handles = [
        Line2D([], [], color="#3b6ea8", linewidth=1.4, alpha=0.75, label="Pareto frontier (every evaluated config)"),
        Line2D([], [], color="#444444", linestyle=":", linewidth=1.3, label=f"gate minimum {bar}"),
        Line2D(
            [], [], marker="o", linestyle="", markersize=12, color="#d62728", mec="black", label=f"selected: {selected}"
        ),
        Line2D([], [], marker="o", linestyle="", markersize=7, color="#3b6ea8", mec="#22323f", label="passes screen"),
        Line2D([], [], marker="o", linestyle="", markersize=7, color="#b0b0b0", mec="#22323f", label="fails screen"),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markersize=6,
            color="#3b6ea8",
            mec="#22323f",
            label="perf: one trace per process",
        ),
    ]
    if probe_min:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                mfc="none",
                mec="#7a5195",
                label="min over gate + robustness probe inputs",
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _one_trace_per_process(perf: dict) -> bool:
    """True when the ranking shapes were measured with a single trace captured in the process."""
    regime = perf.get("regime", {})
    return any(regime.get(shape_key(b, s), "").startswith("1 trace(s)") for b, s in RANK_SHAPES)


def _compact(rec: dict) -> dict:
    """Report copy of a record: per-layer construction evidence grouped by identical content
    ({"layers": [i, ...], **evidence}), traceback trimmed."""
    r = copy.deepcopy(rec)
    con = r.get("construction")
    if con and "layers" in con:
        groups = {}
        for i, ev in con["layers"].items():
            groups.setdefault(json.dumps(ev, sort_keys=True), []).append(int(i))
        con["layers"] = [{"layers": ls, **json.loads(k)} for k, ls in groups.items()]
    if "traceback" in r:
        r["traceback"] = r["traceback"][-600:]
    return r


def cmd_report(args) -> None:
    import csv

    recs = _load(args.inputs.split(","))
    doc = Path(args.doc_dir)
    doc.mkdir(parents=True, exist_ok=True)
    recs.sort(key=lambda r: r.get("perf", {}).get("rank_ms", 1e9))
    # robustness probe (tests/perf/dtype_probe.py): per-candidate summary on the gate + probe inputs
    probe = {}
    for p in filter(None, args.probe.split(",")):
        for line in Path(p).read_text().splitlines():
            if line.strip():
                pr = json.loads(line)
                key = "probe" if pr.get("inputs", "short") == "short" else f"probe_{pr['inputs']}"
                probe.setdefault(pr["id"], {})[key] = pr["summary"]
    for r in recs:
        r.update(probe.get(r["id"], {}))
    (doc / "sweep_results.json").write_text(json.dumps([_compact(r) for r in recs], indent=1) + "\n")
    cols = ["id", "description", "screen_pass", "failing", "rank_ms"]
    cols += [f"burst_{b}x{s}" for b, s in PERF_SHAPES] + [f"sustained_{b}x{s}" for b, s in PERF_SHAPES]
    cols += ["perf_regime"]
    cols += ["layer_pcc_gated", "layer_pcc_gated_at", "layer_pcc_all", "module_pcc", "module_pcc_at"]
    cols += ["last_hidden_pcc", "embedding_cos", "embedding_cos_at", "retrieval_agree"]
    cols += ["probe_min_last_hidden_pcc", "probe_min_last_hidden_at", "probe_min_embedding_cos", "probe_pass"]
    cols += ["error", "commit"]
    with open(doc / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            a, p = r.get("accuracy", {}), r.get("perf", {})
            row = {"id": r["id"], "description": r["description"], "error": r.get("error", ""), "commit": r["commit"]}
            if a:
                row.update(
                    screen_pass=a["screen_pass"],
                    failing="; ".join(a["failing"]),
                    layer_pcc_gated=f"{a['layer_pcc_gated']['value']:.6f}",
                    layer_pcc_gated_at=a["layer_pcc_gated"]["where"],
                    layer_pcc_all=f"{a['layer_pcc_all']['value']:.6f}",
                    module_pcc=f"{a['module_pcc']['value']:.6f}",
                    module_pcc_at=a["module_pcc"]["where"],
                    last_hidden_pcc=f"{a['last_hidden_pcc']['value']:.6f}",
                    embedding_cos=f"{a['embedding_cos']['value']:.6f}",
                    embedding_cos_at=a["embedding_cos"]["where"],
                    retrieval_agree=a["retrieval_agree"],
                )
            if p:
                if "rank_ms" in p:
                    row["rank_ms"] = f"{p['rank_ms']:.3f}"
                for b, s in PERF_SHAPES:
                    k = shape_key(b, s)
                    if k in p:
                        row[f"burst_{k}"] = f"{p[k]['burst_ms_median']:.3f}"
                        row[f"sustained_{k}"] = f"{p[k]['sustained_ms_median']:.3f}"
                regimes = set(p.get("regime", {}).values())
                row["perf_regime"] = (
                    "one trace per process"
                    if regimes and all(v.startswith("1 trace(s)") for v in regimes)
                    else "all five traces captured in one process"
                )
            if r.get("probe"):
                both = r["probe"]["all"]
                row.update(
                    probe_min_last_hidden_pcc=f"{both['min_last_hidden_pcc']:.6f}",
                    probe_min_last_hidden_at=both["min_last_hidden_at"],
                    probe_min_embedding_cos=f"{both['min_embedding_cos']:.6f}",
                    probe_pass=both["pass"],
                )
            w.writerow(row)
    _chart(
        recs,
        "embedding_cos",
        "min end-to-end embedding cosine vs HF fp32",
        BAR_COS,
        doc / "text_cosine_perf_pareto.png",
        args.selected,
        "EmbeddingGemma 2 text tower: embedding cosine vs traced device time (P150a)",
    )
    _chart(
        recs,
        "layer_pcc_gated",
        "min per-layer PCC vs HF fp32 (teacher forced)",
        BAR_LAYER,
        doc / "text_pcc_perf_pareto.png",
        args.selected,
        "EmbeddingGemma 2 text tower: per-layer PCC vs traced device time (P150a)",
    )
    _chart(
        recs,
        "last_hidden_pcc",
        "min free-running last-hidden PCC vs HF fp32",
        BAR_LAST,
        doc / "text_lasthidden_perf_pareto.png",
        args.selected,
        "EmbeddingGemma 2 text tower: last-hidden PCC (the binding metric) vs traced device time (P150a)",
    )
    print(f"wrote {len(recs)} records to {doc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--candidates", default="all")
    r.add_argument("--combos", default="", help="json {id: {description, delta}} of extra candidates")
    r.add_argument("--out", required=True)
    r.add_argument("--no-acc", action="store_true")
    r.add_argument("--no-perf", action="store_true")
    r.add_argument("--burst", type=int, default=20)
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--idle", type=float, default=3.0)
    r.add_argument("--soak", type=float, default=5.0)
    r.add_argument("--sustained-ms", type=float, default=1500.0, help="target length of one sustained window")
    r.add_argument("--shapes", default="", help="perf shapes BxS,... captured in this process (default: all five)")
    r.add_argument("--merge", action="store_true", help="report merges this record into the earlier one")
    sub.add_parser("list")
    rp = sub.add_parser("report")
    rp.add_argument("--inputs", required=True, help="comma-separated sweep jsonl files")
    rp.add_argument("--doc-dir", required=True)
    rp.add_argument("--selected", default="")
    rp.add_argument("--probe", default="", help="comma-separated dtype_probe.py jsonl files")
    args = ap.parse_args()
    {"run": cmd_run, "list": cmd_list, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
