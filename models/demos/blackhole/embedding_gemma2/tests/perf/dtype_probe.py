# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E6a robustness probe: free-running last-hidden PCC and per-prompt embedding cosine of text
precision candidates on the fixed gate cases plus a broader set of short / padded real-text inputs.

    python .../tests/perf/dtype_probe.py --candidates base_bfp8_hifi2,p3_qkv_gu_exp --out probe.jsonl
    python .../tests/perf/dtype_probe.py --candidates p3_x --combos combos.json --out probe.jsonl
    python .../tests/perf/dtype_probe.py list

Inputs (`probe_inputs()`):
  gate   the end-to-end cases of tests/pcc/test_text_model.py except b1_s32768 (tests/common.CASES
         + the batch-8 padded case), same corpus offsets as the tests
  probe  batch-1 lengths 1-64 at corpus offsets the tests do not use; padded batches that mix a
         long row with short rows (512 + 33 and similar) at new offsets; the demo queries and
         documents (with the demo prefixes) one at a time and as the demo's padded batches
  mid    (`--inputs mid`) batch-1 lengths 100-2048 at new offsets, 505-520 three times each: the
         gate's worst cases sit at 511 / 512 / 513
  e9     (`--inputs e9`, E9.1-r1) batch-1 lengths 160-224 at corpus offset 0: the b1_s256 contract
         gate failure family. The 10 E9 contract inputs are gated by tests/perf/prefill.py.
Grid: EMBEDDING_GEMMA2_GRID (default: the device's own); recorded as `compute_grid`.
Metrics: the code of test_text_model (tests/common: pcc over the valid token rows of the whole
batch, cosine per prompt) against the live HF fp32 reference; bars 0.99 / 0.99. Eager forward
(trace replay == eager bit for bit, tests/pcc/test_trace.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

BAR_LAST = 0.99
BAR_COS = 0.99
BATCH8 = (512, 480, 400, 300, 257, 129, 64, 7)

# batch-1 lengths in 1..64 (every tile / bucket boundary in that range plus odd lengths)
PROBE_SHORT = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 17, 20, 24, 28, 31, 32, 33, 36, 40, 45, 48, 52, 57, 63, 64)
# the most sensitive gate lengths again, at a second set of offsets
PROBE_SHORT_REPEAT = (7, 31, 33, 5, 12, 48)
PROBE_PADDED = (
    (512, 33),
    (512, 33),
    (512, 33),
    (512, 7),
    (512, 31),
    (512, 1),
    (512, 64, 17),
    (1025, 33),
    (2048, 33),
    (256, 33, 9),
    (128, 31, 7, 3),
    (512, 480, 33, 7),
    (64, 33, 31, 7, 1),
)


# `--inputs mid`: batch-1 lengths 100-2048 (the gate's worst cases sit at 511 / 512 / 513 and the
# padded 512 + 33), each at new offsets; 505-520 three times
PROBE_MID = (100, 128, 200, 256, 300, 384, 450, 500, 640, 768, 1000, 1024, 1025, 1500, 2048)
PROBE_MID_REPEAT = (505, 510, 511, 512, 513, 514, 520)

# `--inputs e9` (E9.1-r1): the b1_s256 contract-gate failure family, batch 1 at corpus offset 0
# (the model-card start; E9 stage 1 gate256_probe). The 10 contract inputs (3/4 of each bucket,
# offset 0) are gated on the traced replay output by tests/perf/prefill.py (HF_CHECK).
PROBE_E9_256 = (160, 176, 192, 200, 208, 224)


def _offset(k: int, row: int, n_corpus: int) -> int:
    """Deterministic corpus offsets away from the tests' (row * 997)."""
    return (1234 + 7919 * k + 3001 * row) % n_corpus


def probe_inputs(snapshot, which: str = "short") -> list[dict]:
    """[{name, set, ids [B, L], mask [B, L]}] (see module docstring)."""
    from transformers import AutoTokenizer

    from models.demos.blackhole.embedding_gemma2.demo.demo import DOC_PREFIX, DOCUMENTS, QUERIES, QUERY_PREFIX
    from models.demos.blackhole.embedding_gemma2.tests.common import CASES, _corpus_ids, case_id, make_batch

    n = len(_corpus_ids(str(snapshot)))
    out = []
    if which == "e9":
        for L in PROBE_E9_256:
            ids, mask = make_batch(snapshot, [L], offsets=[0])
            out.append({"name": f"b1_s{L}_o0", "set": "e9_256", "ids": ids, "mask": mask})
        return out
    if which == "mid":
        lengths_list = PROBE_MID + PROBE_MID_REPEAT * 3
        for k, L in enumerate(lengths_list):
            off = _offset(1000 + k, 0, n)
            ids, mask = make_batch(snapshot, [L], offsets=[off])
            out.append({"name": f"b1_s{L}_o{off}", "set": "mid", "ids": ids, "mask": mask})
        return out
    for lengths in tuple(CASES) + (BATCH8,):
        ids, mask = make_batch(snapshot, list(lengths))
        out.append({"name": case_id(lengths), "set": "gate", "ids": ids, "mask": mask})
    k = 0
    for lengths in [(L,) for L in PROBE_SHORT + PROBE_SHORT_REPEAT] + list(PROBE_PADDED):
        k += 1
        offs = [_offset(k, r, n) for r in range(len(lengths))]
        ids, mask = make_batch(snapshot, list(lengths), offsets=offs)
        out.append({"name": f"{case_id(lengths)}_o{offs[0]}", "set": "probe", "ids": ids, "mask": mask})
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    for tag, prefix, texts in (("q", QUERY_PREFIX, QUERIES), ("d", DOC_PREFIX, DOCUMENTS)):
        for i, t in enumerate(texts):
            b = tok([prefix + t], return_tensors="pt")
            out.append({"name": f"demo_{tag}{i}", "set": "probe", "ids": b["input_ids"], "mask": b["attention_mask"]})
        b = tok([prefix + t for t in texts], padding=True, return_tensors="pt")
        out.append({"name": f"demo_{tag}_batch", "set": "probe", "ids": b["input_ids"], "mask": b["attention_mask"]})
    return out


def references(model, inputs) -> list[dict]:
    from models.demos.blackhole.embedding_gemma2.reference.hf_reference import run_reference

    refs = []
    for inp in inputs:
        with torch.no_grad():
            r = run_reference(model, inp["ids"], inp["mask"], capture_layers=())
        refs.append({"last_hidden": r.last_hidden.float(), "embedding": r.embedding.float()})
    return refs


def run_candidate(device, model, inputs, refs) -> list[dict]:
    import ttnn
    from models.demos.blackhole.embedding_gemma2.tests.common import (
        cosine_rows,
        download,
        pcc,
        upload_inputs,
        valid_rows,
    )

    rows = []
    for inp, ref in zip(inputs, refs):
        ids, mask = inp["ids"], inp["mask"]
        B, L = ids.shape
        ids_d, keep_d, S = upload_inputs(device, ids, mask, model.config.max_seq_len)
        model.prepare(S)
        last, emb = model(ids_d, keep_d)
        last_t = download(last, B, L)
        p_last = pcc(valid_rows(ref["last_hidden"], mask), valid_rows(last_t, mask))
        per_row = [pcc(ref["last_hidden"][b][mask[b].bool()], last_t[b][mask[b].bool()]) for b in range(B)]
        cos = cosine_rows(ref["embedding"], ttnn.to_torch(emb).float().reshape(B, -1)).tolist()
        for t in (last, emb, ids_d, keep_d):
            ttnn.deallocate(t)
        rows.append(
            {
                "name": inp["name"],
                "set": inp["set"],
                "lengths": mask.sum(-1).tolist(),
                "bucket": S,
                "last_hidden_pcc": p_last,
                "last_hidden_pcc_per_row": per_row,
                "embedding_cos": cos,
            }
        )
    return rows


def summarize(rows) -> dict:
    s = {}
    for which in sorted({r["set"] for r in rows}) + ["all"]:
        sel = [r for r in rows if which == "all" or r["set"] == which]
        last = min(sel, key=lambda r: r["last_hidden_pcc"])
        cos = min(((c, r["name"], b) for r in sel for b, c in enumerate(r["embedding_cos"])), key=lambda t: t[0])
        s[which] = {
            "n_inputs": len(sel),
            "n_prompts": sum(len(r["embedding_cos"]) for r in sel),
            "min_last_hidden_pcc": last["last_hidden_pcc"],
            "min_last_hidden_at": last["name"],
            "min_embedding_cos": cos[0],
            "min_embedding_cos_at": [cos[1], cos[2]],
            "n_last_below_bar": sum(r["last_hidden_pcc"] < BAR_LAST for r in sel),
            "n_cos_below_bar": sum(c < BAR_COS for r in sel for c in r["embedding_cos"]),
        }
        s[which]["pass"] = s[which]["min_last_hidden_pcc"] >= BAR_LAST and s[which]["min_embedding_cos"] >= BAR_COS
    return s


def cmd_run(args) -> None:
    import ttnn
    from models.demos.blackhole.embedding_gemma2.conftest import MODEL_ID, open_eg2_device
    from models.demos.blackhole.embedding_gemma2.reference.hf_reference import load_reference_text_model
    from models.demos.blackhole.embedding_gemma2.tests.perf.dtype_sweep import (
        CANDIDATES,
        _git_head,
        candidate_dict,
        register_combo,
    )
    from models.demos.blackhole.embedding_gemma2.tt.config import TextModelConfig, default_weight_cache_dir
    from models.demos.blackhole.embedding_gemma2.tt.precision import PrecisionConfig
    from models.demos.blackhole.embedding_gemma2.tt.text_model import TextModel
    from models.demos.blackhole.embedding_gemma2.tt.weights import (
        load_text_config_dict,
        load_text_state_dict,
        resolve_snapshot,
    )

    for path in filter(None, args.combos.split(",")):
        for cid, spec in json.loads(Path(path).read_text()).items():
            register_combo(cid, spec["description"], spec["delta"])
    snapshot = resolve_snapshot(MODEL_ID)
    state_dict = load_text_state_dict(snapshot)
    config = TextModelConfig.from_hf(load_text_config_dict(snapshot), max_seq_len=8192)
    cache_dir = default_weight_cache_dir() / snapshot.name
    inputs = probe_inputs(snapshot, args.inputs)
    hf, _ = load_reference_text_model(snapshot, state_dict)
    t0 = time.time()
    refs = references(hf, inputs)
    del hf
    print(f"{len(inputs)} inputs, HF fp32 references in {time.time() - t0:.0f} s", flush=True)
    for cid in args.candidates.split(","):
        prec = candidate_dict(cid)
        device = open_eg2_device(0)  # EMBEDDING_GEMMA2_GRID; default: the device's own grid
        grid = device.compute_with_storage_grid_size()
        try:
            t0 = time.time()
            model = TextModel.from_state_dict(
                state_dict, config, device, cache_dir=cache_dir, precision=PrecisionConfig.from_dict(prec, cid)
            )
            rows = run_candidate(device, model, inputs, refs)
        finally:
            ttnn.close_device(device)
        rec = {
            "id": cid,
            "description": prec["description"],
            "delta": CANDIDATES[cid][1],
            "commit": _git_head(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": time.time() - t0,
            "compute_grid": [grid.x, grid.y],
            "inputs": args.inputs,
            "summary": summarize(rows),
            "rows": rows,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        for which, s in rec["summary"].items():
            print(
                f"PROBE {cid} {which}: last {s['min_last_hidden_pcc']:.5f} ({s['min_last_hidden_at']}, "
                f"{s['n_last_below_bar']} below) cos {s['min_embedding_cos']:.5f} ({s['min_embedding_cos_at']}, "
                f"{s['n_cos_below_bar']} below) -> {'PASS' if s['pass'] else 'FAIL'}",
                flush=True,
            )


def cmd_list(args) -> None:
    from models.demos.blackhole.embedding_gemma2.conftest import MODEL_ID
    from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot

    for inp in probe_inputs(resolve_snapshot(MODEL_ID), args.inputs):
        print(f"{inp['set']:6s} {inp['name']:32s} lengths {inp['mask'].sum(-1).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--candidates", required=True)
    r.add_argument("--combos", default="", help="comma-separated json files {id: {description, delta}}")
    r.add_argument("--out", required=True)
    r.add_argument("--inputs", default="short", choices=("short", "mid", "e9"), help="input set (module docstring)")
    ls = sub.add_parser("list")
    ls.add_argument("--inputs", default="short", choices=("short", "mid", "e9"))
    args = ap.parse_args()
    {"run": cmd_run, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    main()
