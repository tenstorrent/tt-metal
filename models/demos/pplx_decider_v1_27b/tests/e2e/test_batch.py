# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched requests: B prompts in one forward give the batch-1 answers.

Batch contract (``tt/model.py``, ``demo/decider.py: TTDecider.plan_batches``): the rows of a batch are
right-padded to one bucket. Both mixers are causal, so the padding cannot change a row's real-token
hidden states; the head reads each row's own last real token and applies each row's own option count
and mask (the temperature is the checkpoint's single value). ``TTDecider`` groups requests by bucket
and splits each group into batches of ``batch_cap(bucket, max_batch_size)``.

Gates:
- equivalence: for B in {2, 4, 8} the 25 text golden rows (and the 8 image rows) batched by bucket give
  the batch-1 decision on every row and logit PCC >= 0.9999 against batch 1 (bit identity recorded);
  against the HF golden, >= 24/25 text (>= 7/8 image) decisions with the near-tie rule and logit PCC >= 0.99;
- a mixed text + image batch, a batch holding a max-bucket row next to short rows, a single-row
  batch, shuffled input order through ``TTDecider.predict_batch`` and the split above ``max_batch_size``;
- rejections (a row over ``max_seq_len``, ``max_batch_size`` outside 1..32), 0 host calls inside a batched
  forward, determinism, and that every ``BATCH_CAPS`` entry runs (the DRAM limit).

Results: ``$PPLX_DECIDER_STAGE6_DIR/batch_*.json`` (default ``$PPLX_DECIDER_ARTIFACTS/e2e``).

    pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_batch.py -q -s
"""

from __future__ import annotations

import json
import random
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.pplx_decider_v1_27b.tests.e2e.test_image_model import Golden as ImageGolden
from models.demos.pplx_decider_v1_27b.tests.e2e.test_image_model import encode_row
from models.demos.pplx_decider_v1_27b.tests.e2e.test_model import Golden as TextGolden
from models.demos.pplx_decider_v1_27b.tests.runtime_audit import count_host_calls
from models.demos.pplx_decider_v1_27b.tt import paths
from models.demos.pplx_decider_v1_27b.tt.model import (
    BATCH_CAPS,
    PplxDeciderModel,
    batch_cap,
    bucket_for,
    dram_view,
    validate_max_batch_size,
)

OUT_DIR = paths.E2E_DIR
BATCH_PCC = 0.9999  # per-row logit PCC against batch 1
PADDING_PCC = 0.999  # a row padded beyond its own bucket (test_model.py::test_bucket_padding bar)
HF_LOGIT_PCC = 0.99
NEAR_TIE_GAP = 0.05
TEXT_MIN_AGREE, IMAGE_MIN_AGREE = 24, 7
TEST_MAX_BATCH = 8

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def _write(name: str, payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture(scope="module")
def model(_device_module_impl):
    return PplxDeciderModel.from_snapshot(_device_module_impl, vision=True, max_batch_size=TEST_MAX_BATCH)


@pytest.fixture(scope="module")
def text_golden():
    return TextGolden()


@pytest.fixture(scope="module")
def image_golden():
    return ImageGolden()


def text_encoded(golden: TextGolden, rid: str) -> dict:
    return {"input_ids": golden.tensor(rid, "input_ids").reshape(1, -1)}


def run_batch(model, encoded: list[dict], counts: list[int], bucket=None) -> list[dict]:
    """One forward over the rows -> per row probs [count] and logits [255] (host)."""
    tokens, last_indices, images = model.prepare_batch(encoded, bucket)
    probs, logits, _ = model(tokens, last_indices, counts, images=images)
    p = ttnn.to_torch(probs).float().reshape(len(counts), -1)
    l = ttnn.to_torch(logits).float().reshape(len(counts), -1)
    for t in (tokens, probs, logits):
        ttnn.deallocate(t)
    if images is not None:
        images.deallocate()
    return [{"probs": p[b, :n].clone(), "logits": l[b, :255].clone()} for b, n in enumerate(counts)]


def run_single(model, encoded: dict, count: int, bucket=None) -> dict:
    """The batch-1 path (``upload_tokens`` / ``prepare_images``, int last index and count)."""
    tokens, last_index, images = model.prepare_images(encoded, bucket)
    probs, logits, _ = model(tokens, last_index, count, images=images)
    out = {
        "probs": ttnn.to_torch(probs).float().reshape(-1)[:count],
        "logits": ttnn.to_torch(logits).float().reshape(-1)[:255],
    }
    for t in (tokens, probs, logits):
        ttnn.deallocate(t)
    if images is not None:
        images.deallocate()
    return out


@pytest.fixture(scope="module")
def text_batch1(model, text_golden):
    return {g["id"]: run_single(model, text_encoded(text_golden, g["id"]), g["count"]) for g in text_golden.rows}


@pytest.fixture(scope="module")
def image_batch1(model, image_golden):
    return {g["id"]: run_single(model, encode_row(image_golden, g["id"]), g["count"]) for g in image_golden.rows}


def plan(lengths: dict[str, int], max_batch: int, max_seq_len: int) -> list[list[str]]:
    """``TTDecider.plan_batches`` on ids: group by bucket (input order kept), split by ``batch_cap``."""
    by_bucket = {}
    for rid, n in lengths.items():
        by_bucket.setdefault(bucket_for(n, max_seq_len), []).append(rid)
    out = []
    for bucket in sorted(by_bucket):
        cap = batch_cap(bucket, max_batch)
        ids = by_bucket[bucket]
        out += [ids[s : s + cap] for s in range(0, len(ids), cap)]
    return out


def compare_rows(golden, ref: dict, got: dict, batches: list[list[str]]) -> tuple[list[dict], dict]:
    rows = []
    batch_of = {rid: (i, len(b)) for i, b in enumerate(batches) for rid in b}
    for g in golden.rows:
        rid, count = g["id"], g["count"]
        tt, one = got[rid], ref[rid]
        hf_logits = golden.tensor(rid, "logits").float()
        rows.append(
            {
                "id": rid,
                "seq_len": g["seq_len"],
                "count": count,
                "batch": batch_of[rid][0],
                "batch_rows": batch_of[rid][1],
                "argmax": int(torch.argmax(tt["probs"])),
                "batch1_argmax": int(torch.argmax(one["probs"])),
                "hf_argmax": g["argmax"],
                "hf_top2_gap": g["top2_gap"],
                "same_as_batch1": int(torch.argmax(tt["probs"])) == int(torch.argmax(one["probs"])),
                "agree_hf": int(torch.argmax(tt["probs"])) == g["argmax"],
                "logit_pcc_vs_batch1": pcc(tt["logits"], one["logits"]),
                "max_abs_prob_diff_vs_batch1": float((tt["probs"] - one["probs"]).abs().max()),
                "bit_identical_vs_batch1": bool(
                    torch.equal(tt["probs"], one["probs"]) and torch.equal(tt["logits"], one["logits"])
                ),
                "logit_pcc_valid_vs_hf": pcc(tt["logits"][:count], hf_logits[:count]),
            }
        )
    summary = {
        "rows": len(rows),
        "batches": [len(b) for b in batches],
        "same_as_batch1": sum(r["same_as_batch1"] for r in rows),
        "bit_identical_vs_batch1": sum(r["bit_identical_vs_batch1"] for r in rows),
        "logit_pcc_vs_batch1_min": min(r["logit_pcc_vs_batch1"] for r in rows),
        "max_abs_prob_diff_vs_batch1": max(r["max_abs_prob_diff_vs_batch1"] for r in rows),
        "agree_hf": sum(r["agree_hf"] for r in rows),
        "hf_misses": [(r["id"], r["hf_top2_gap"]) for r in rows if not r["agree_hf"]],
        "logit_pcc_valid_vs_hf_min": min(r["logit_pcc_valid_vs_hf"] for r in rows),
    }
    return rows, summary


def check(summary: dict, min_agree: int) -> None:
    assert summary["same_as_batch1"] == summary["rows"], summary
    assert summary["logit_pcc_vs_batch1_min"] >= BATCH_PCC, summary
    assert summary["agree_hf"] >= min_agree, summary
    assert all(gap < NEAR_TIE_GAP for _, gap in summary["hf_misses"]), summary
    assert summary["logit_pcc_valid_vs_hf_min"] >= HF_LOGIT_PCC, summary


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_text_batch_equivalence(model, text_golden, text_batch1, batch_size):
    lengths = {g["id"]: g["seq_len"] for g in text_golden.rows}
    batches = plan(lengths, batch_size, model.config.max_seq_len)
    count_of = {g["id"]: g["count"] for g in text_golden.rows}
    got = {}
    start = time.perf_counter()
    for batch in batches:
        outs = run_batch(model, [text_encoded(text_golden, rid) for rid in batch], [count_of[rid] for rid in batch])
        got.update(zip(batch, outs))
    rows, summary = compare_rows(text_golden, text_batch1, got, batches)
    summary.update(batch_size=batch_size, seconds=round(time.perf_counter() - start, 2))
    _write(f"batch_text_B{batch_size}.json", {"summary": summary, "rows": rows})
    logger.info(f"text batch B={batch_size}: {json.dumps(summary)}")
    check(summary, TEXT_MIN_AGREE)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_image_batch_equivalence(model, image_golden, image_batch1, batch_size):
    lengths = {g["id"]: g["seq_len"] for g in image_golden.rows}
    batches = plan(lengths, batch_size, model.config.max_seq_len)
    count_of = {g["id"]: g["count"] for g in image_golden.rows}
    got = {}
    for batch in batches:
        outs = run_batch(model, [encode_row(image_golden, rid) for rid in batch], [count_of[rid] for rid in batch])
        got.update(zip(batch, outs))
    rows, summary = compare_rows(image_golden, image_batch1, got, batches)
    summary["batch_size"] = batch_size
    _write(f"batch_image_B{batch_size}.json", {"summary": summary, "rows": rows})
    logger.info(f"image batch B={batch_size}: {json.dumps(summary)}")
    check(summary, IMAGE_MIN_AGREE)


@pytest.mark.timeout(1800)
def test_bucket128_batch(model):
    """The 128 bucket (no golden rows there): 8 seeded prompts of 20..128 tokens, B=8 vs batch 1."""
    g = torch.Generator().manual_seed(128)
    lengths = [128, 20, 77, 101, 64, 33, 127, 90]
    ids = [torch.randint(1000, 100000, (n,), generator=g).reshape(1, -1) for n in lengths]
    counts = [2, 5, 3, 255, 2, 12, 7, 4]
    one = [run_single(model, {"input_ids": i}, n) for i, n in zip(ids, counts)]
    outs = run_batch(model, [{"input_ids": i} for i in ids], counts)
    result = [
        {
            "seq_len": n,
            "same_decision": int(torch.argmax(o["probs"])) == int(torch.argmax(r["probs"])),
            "logit_pcc_vs_batch1": pcc(o["logits"], r["logits"]),
            "max_abs_prob_diff_vs_batch1": float((o["probs"] - r["probs"]).abs().max()),
            "bit_identical_vs_batch1": bool(
                torch.equal(o["probs"], r["probs"]) and torch.equal(o["logits"], r["logits"])
            ),
        }
        for n, o, r in zip(lengths, outs, one)
    ]
    _write("batch_bucket128.json", result)
    logger.info(f"bucket 128 B=8: {json.dumps(result)}")
    assert all(r["same_decision"] and r["logit_pcc_vs_batch1"] >= BATCH_PCC for r in result), result


@pytest.mark.timeout(1800)
def test_mixed_text_image_batch(model, text_golden, image_golden, text_batch1, image_batch1):
    """Text and image rows of one bucket in one forward: each row equals its batch-1 output."""
    picks = [("text", "s01_ticket_routing"), ("image", "v01_dominant_color"), ("text", "s08_itinerary_airport")]
    picks += [("image", "v02_count_circles")]
    encoded, counts, refs = [], [], []
    for kind, rid in picks:
        golden, ref = (text_golden, text_batch1) if kind == "text" else (image_golden, image_batch1)
        encoded.append(text_encoded(golden, rid) if kind == "text" else encode_row(golden, rid))
        counts.append(golden.row(rid)["count"])
        refs.append(ref[rid])
    outs = run_batch(model, encoded, counts)
    result = [
        {
            "id": rid,
            "kind": kind,
            "same_decision": int(torch.argmax(o["probs"])) == int(torch.argmax(r["probs"])),
            "logit_pcc_vs_batch1": pcc(o["logits"], r["logits"]),
            "bit_identical_vs_batch1": bool(
                torch.equal(o["probs"], r["probs"]) and torch.equal(o["logits"], r["logits"])
            ),
        }
        for (kind, rid), o, r in zip(picks, outs, refs)
    ]
    _write("batch_mixed_text_image.json", result)
    logger.info(f"mixed text+image batch: {json.dumps(result)}")
    assert all(r["same_decision"] and r["logit_pcc_vs_batch1"] >= BATCH_PCC for r in result), result


@pytest.mark.timeout(1800)
def test_max_length_row_with_short_rows(model, text_golden, text_batch1):
    """A batch whose longest row fills the 8192 bucket: the short rows are padded to 8192 too."""
    ids = ["x04_customer_lookup", "s01_ticket_routing", "s05_unit_conversion"][: BATCH_CAPS[8192]]
    outs = run_batch(model, [text_encoded(text_golden, rid) for rid in ids], [text_golden.row(r)["count"] for r in ids])
    result = []
    for rid, o in zip(ids, outs):
        r = text_batch1[rid]
        result.append(
            {
                "id": rid,
                "seq_len": text_golden.row(rid)["seq_len"],
                "own_bucket": bucket_for(text_golden.row(rid)["seq_len"]),
                "batch_bucket": 8192,
                "same_decision": int(torch.argmax(o["probs"])) == int(torch.argmax(r["probs"])),
                "logit_pcc_vs_batch1": pcc(o["logits"], r["logits"]),
                "max_abs_prob_diff_vs_batch1": float((o["probs"] - r["probs"]).abs().max()),
                "bit_identical_vs_batch1": bool(torch.equal(o["logits"], r["logits"])),
            }
        )
    _write("batch_max_length_with_short.json", result)
    logger.info(f"max-length row + short rows: {json.dumps(result)}")
    for r in result:
        bar = BATCH_PCC if r["own_bucket"] == r["batch_bucket"] else PADDING_PCC
        assert r["same_decision"] and r["logit_pcc_vs_batch1"] >= bar, r


@pytest.mark.timeout(1800)
def test_single_row_batch_equals_batch1(model, text_golden, image_golden, text_batch1, image_batch1):
    """The batch path with one row (list last index / count) is bit-identical to the batch-1 path."""
    text = run_batch(model, [text_encoded(text_golden, "m06_server_500")], [text_golden.row("m06_server_500")["count"]])
    image = run_batch(
        model, [encode_row(image_golden, "v03_receipt_total")], [image_golden.row("v03_receipt_total")["count"]]
    )
    for got, ref in ((text[0], text_batch1["m06_server_500"]), (image[0], image_batch1["v03_receipt_total"])):
        assert torch.equal(got["probs"], ref["probs"]) and torch.equal(got["logits"], ref["logits"])


@pytest.mark.timeout(3600)
def test_predict_batch_order_and_split(model, text_golden, image_golden, text_batch1, image_batch1):
    """``TTDecider.predict_batch``: shuffled input order comes back in input order, and more rows than
    ``max_batch_size`` per bucket are split into batches of at most ``max_batch_size``."""
    from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider
    from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer

    prompts = {
        json.loads(l)["id"]: json.loads(l)["row"]
        for l in (paths.DECISION_GOLDEN_DIR / "prompts.jsonl").read_text().splitlines()
    }
    text_ids = [g["id"] for g in text_golden.rows if g["seq_len"] <= 2048]  # 10 rows at 1024, 6 at 2048
    image_ids = ["v01_dominant_color", "v04_tallest_bar", "v05_red_circle_yes"]
    items = [(rid, prompts[rid], text_batch1[rid]) for rid in text_ids]
    items += [(rid, image_golden.prompts[rid]["row"], image_batch1[rid]) for rid in image_ids]
    random.Random(1234).shuffle(items)
    rows = [row for _, row, _ in items]

    decider = TTDecider(model, AppTokenizer())
    saved = model.config.max_batch_size
    model.config.max_batch_size = 4
    try:
        _, _, batches = decider.plan_batches(rows)
        probs = decider.predict_probabilities_batch(rows)
        answers = decider.predict_batch(rows[:3])
    finally:
        model.config.max_batch_size = saved
    result = {
        "order": [rid for rid, _, _ in items],
        "batches": batches,
        "rows": [
            {
                "id": rid,
                "same_decision": int(torch.tensor(p).argmax()) == int(ref["probs"].argmax()),
                "max_abs_prob_diff_vs_batch1": float((torch.tensor(p) - ref["probs"]).abs().max()),
            }
            for (rid, _, ref), p in zip(items, probs)
        ],
    }
    _write("batch_predict_order_split.json", result)
    logger.info(f"predict_batch: batches {batches}")
    assert sorted(i for b in batches for i in b) == list(range(len(rows)))
    assert all(1 <= len(b) <= 4 for b in batches) and len(batches) >= 5  # 13 rows at 1024 -> 4+4+4+1, 6 at 2048 -> 4+2
    assert all(r["same_decision"] and r["max_abs_prob_diff_vs_batch1"] < 1e-3 for r in result["rows"]), result
    for (rid, row, ref), answer in zip(items[:3], answers):
        assert answer == decider.tokenizer.autojev.answer(row["question"], ref["probs"].tolist()), rid


@pytest.mark.timeout(600)
def test_rejections(model, expect_error):
    from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider
    from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer

    with expect_error(ValueError, "limit"):
        model.upload_batch([[1] * 100, [1] * (model.config.max_seq_len + 1)])
    with expect_error(ValueError, "empty"):
        model.upload_batch([])
    for bad in (0, max(BATCH_CAPS.values()) + 1, 2.0):
        with expect_error(ValueError, "max_batch_size"):
            validate_max_batch_size(bad)
    decider = TTDecider(model, AppTokenizer())
    long_row = {"state": "word " * 9000, "question": {"type": "noul", "instructions": "Is it long?"}}
    short_row = {"state": "hi", "question": {"type": "noul", "instructions": "Is it short?"}}
    with expect_error(ValueError, "limit"):
        decider.plan_batches([short_row, long_row])


@pytest.mark.timeout(1800)
def test_batch_forward_stays_on_device_and_deterministic(model, text_golden):
    """B=4 at bucket 1024: 0 host calls between the upload and the readback; two runs bit-identical."""
    ids = [g["id"] for g in text_golden.rows if g["seq_len"] <= 1024][:4]
    encoded = [text_encoded(text_golden, rid) for rid in ids]
    counts = [text_golden.row(rid)["count"] for rid in ids]
    tokens, last_indices, _ = model.prepare_batch(encoded)
    warm = model(tokens, last_indices, counts)
    ttnn.synchronize_device(model.mesh_device)
    first = ttnn.to_torch(warm[0]).float()
    for t in warm[:2]:
        ttnn.deallocate(t)
    with count_host_calls() as counts_host:
        probs, logits, _ = model(tokens, last_indices, counts)
        ttnn.synchronize_device(model.mesh_device)
    second = ttnn.to_torch(probs).float()
    result = {
        "ids": ids,
        "bucket": tokens.shape[-1],
        "batch": tokens.shape[0],
        "host_calls": dict(counts_host),
        "bit_identical_runs": bool(torch.equal(first, second)),
    }
    for t in (tokens, probs, logits):
        ttnn.deallocate(t)
    _write("batch_runtime_audit.json", result)
    assert not counts_host, f"Host calls inside the batched forward: {dict(counts_host)}"
    assert result["bit_identical_runs"]


@pytest.mark.timeout(3600)
def test_batch_caps_fit(model):
    """Every ``BATCH_CAPS`` entry runs with the full model and the vision tower resident (full-bucket rows)."""
    g = torch.Generator().manual_seed(0)
    result = {"dram_after_load": dram_view(model.mesh_device)}
    for bucket, cap in BATCH_CAPS.items():
        if bucket > model.config.max_seq_len:
            continue
        rows = [torch.randint(1000, 100000, (bucket,), generator=g).tolist() for _ in range(cap)]
        tokens, last_indices = model.upload_batch(rows)
        start = time.perf_counter()
        probs, logits, _ = model(tokens, last_indices, [2] * cap)
        host = ttnn.to_torch(probs).float()
        result[str(bucket)] = {
            "batch": cap,
            "seconds_incl_compile": round(time.perf_counter() - start, 2),
            "finite": bool(torch.isfinite(host).all()),
            "dram_after": dram_view(model.mesh_device),
        }
        for t in (tokens, probs, logits):
            ttnn.deallocate(t)
        logger.info(f"batch cap bucket {bucket} x {cap}: {result[str(bucket)]}")
        assert result[str(bucket)]["finite"]
    _write("batch_caps.json", result)
