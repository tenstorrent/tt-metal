# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end image decisions (stage 12B): vision tower -> splice -> 64 layers (3D mRoPE) -> head.

Golden (``reference/hf_image_decision_golden.py``; 8 image rows, bf16 HF, layer-streamed on CPU):
``$PPLX_DECIDER_IMAGE_GOLDEN`` (default artifacts/pplx_decider/goldens/vision/e2e/), files
``golden_bf16.safetensors``, ``summary_bf16.json`` and ``prompts.jsonl`` (rows with the PNG paths).
Every row is re-encoded with the app's processor here (input ids checked against the golden), so
the test covers the real input prep.

Gates (person-approved, decisions.tsv 2026-10-09):
- decision agreement: TT top-1 == HF top-1 on >= 7/8 rows; a miss only on a near-tie
  (HF top-2 gap < 0.05);
- readout-logit PCC >= 0.99 per row (``logits[:count]``);
- splice (TT vs TT, same tensors): image-token rows of the spliced ``inputs_embeds`` bit-exact
  equal to the device vision-tower output, text rows bit-exact equal to the text embedding; the
  spliced-embedding PCC vs the HF golden is reported per row and held to the 12A tower bar (0.99),
  because any gap to the golden is the tower's feature error, not the splice's;
- 3D position ids from input prep == golden ``position_ids``.
Recorded: final-hidden PCC, max |prob diff|, last-token residual PCC after each of the 64 layers
(rows in ``TRACE_ROWS``). Also: ``TTDecider.predict(..., images=[path])`` on a golden row,
determinism, and the runtime fallback audit of an image request. Results go to
``$PPLX_DECIDER_STAGE12B_DIR``.

Run::

    pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_image_model.py -q -s
"""

from __future__ import annotations

import json
import os
import statistics
import time
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

import ttnn
from models.demos.pplx_decider_v1_27b.tests.runtime_audit import count_host_calls
from models.demos.pplx_decider_v1_27b.tt.model import IMAGE_TOKEN_ID, PplxDeciderModel

GOLDEN_DIR = Path(
    os.environ.get("PPLX_DECIDER_IMAGE_GOLDEN", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision/e2e")
)
OUT_DIR = Path(
    os.environ.get("PPLX_DECIDER_STAGE12B_DIR", "/local/ttuser/gtobar/artifacts/pplx_decider/stage12b/image_e2e")
)
MIN_AGREE = 7
NEAR_TIE_GAP = 0.05
LOGIT_PCC = 0.99
TOWER_PCC = 0.99  # stage-12A vision-tower feature bar; the splice itself is checked bit-exact
TRACE_ROWS = ("v02_count_circles", "v07_brightness_dark")  # the near-tie row and the lowest-confidence other row

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


class Golden:
    def __init__(self, directory: Path = GOLDEN_DIR):
        path = directory / "golden_bf16.safetensors"
        if not path.exists():
            pytest.fail(
                f"Missing {path}; run python -m models.demos.pplx_decider_v1_27b.reference.hf_image_decision_golden"
            )
        self.summary = json.loads((directory / "summary_bf16.json").read_text())
        self.rows = self.summary["rows"]
        self.prompts = {r["id"]: r for r in map(json.loads, (directory / "prompts.jsonl").read_text().splitlines())}
        self._file = safe_open(str(path), framework="pt")

    def tensor(self, row_id: str, name: str) -> torch.Tensor:
        return self._file.get_tensor(f"{row_id}.{name}")

    def row(self, row_id: str) -> dict:
        return next(r for r in self.rows if r["id"] == row_id)


@lru_cache(maxsize=1)
def tokenizer():
    from models.demos.pplx_decider_v1_27b.reference.image_decision_prompts import image_tokenizer

    return image_tokenizer()


def encode_row(golden: Golden, row_id: str) -> dict:
    """The app's processor output for a golden row (from its PNG); input ids must equal the golden."""
    from models.demos.pplx_decider_v1_27b.reference.image_decision_prompts import encode

    enc = encode(tokenizer(), golden.prompts[row_id]["row"])
    assert enc["input_ids"][0].tolist() == golden.tensor(row_id, "input_ids").tolist(), f"{row_id}: re-encoding drifted"
    return enc


@pytest.fixture(scope="module")
def golden():
    return Golden()


@pytest.fixture(scope="module")
def model(_device_module_impl):
    return PplxDeciderModel.from_snapshot(_device_module_impl, vision=True)


def _write(name: str, payload) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_image_row(model, enc: dict, count: int, *, layer_trace=False) -> dict:
    tokens, last_index, images = model.prepare_images(enc)
    probs, logits, extras = model(
        tokens, last_index, count, images=images, return_hidden=True, collect_layer_hidden=layer_trace
    )
    out = {
        "probs": ttnn.to_torch(probs).float().reshape(-1)[:count],
        "logits": ttnn.to_torch(logits).float().reshape(-1)[:255],
        "final_hidden": ttnn.to_torch(extras["final_hidden"]).float().reshape(-1),
        "position_ids": images.position_ids,
        "bucket": tokens.shape[-1],
    }
    if layer_trace:
        out["layer_hidden"] = torch.stack([ttnn.to_torch(t).float().reshape(-1) for t in extras["layer_hidden"]])
        for t in extras["layer_hidden"]:
            ttnn.deallocate(t)
    for t in (tokens, probs, logits, extras["final_hidden"]):
        ttnn.deallocate(t)
    images.deallocate()
    return out


@pytest.mark.timeout(1800)
def test_splice(model, golden):
    """The splice copies rows exactly; its PCC to the golden is the 12A tower's (bar 0.99).

    Per row, TT vs TT on the same tensors: image-token rows of the spliced ``inputs_embeds`` ==
    the device vision-tower output (``torch.equal``), text rows == the text embedding output.
    The spliced-embedding PCC vs the HF golden ``inputs_embeds`` is reported per row and must be
    >= ``TOWER_PCC``; it measures the tower's feature error, which the splice passes through.
    """
    results = []
    for g in golden.rows:
        rid = g["id"]
        enc = encode_row(golden, rid)
        seq = enc["input_ids"].shape[1]
        tokens, _, images = model.prepare_images(enc)
        features = model.image_features(images)
        tt_features = torch.cat([ttnn.to_torch(f).reshape(-1, f.shape[-1]) for f in features])
        x = model.splice(tokens, images, features)
        tt = ttnn.to_torch(x).reshape(-1, x.shape[-1])[:seq]
        ttnn.deallocate(x)
        text_only = model.embedding(tokens)
        tt_text = ttnn.to_torch(text_only).reshape(-1, x.shape[-1])[:seq]
        ttnn.deallocate(text_only)
        ttnn.deallocate(tokens)
        images.deallocate()
        want = golden.tensor(rid, "inputs_embeds")
        is_image = enc["input_ids"][0] == IMAGE_TOKEN_ID
        row = {
            "id": rid,
            "seq_len": seq,
            "image_tokens": int(is_image.sum()),
            "pcc_all": pcc(tt, want),
            "pcc_image_rows": pcc(tt[is_image], want[is_image]),
            "text_rows_bit_exact_vs_golden": bool(torch.equal(tt[~is_image], want[~is_image].to(tt.dtype))),
            "text_rows_bit_exact_vs_text_path": bool(torch.equal(tt[~is_image], tt_text[~is_image])),
            # The splice copies rows: image rows must equal the TT tower output exactly; any gap to
            # the golden is the vision tower's (stage 12A) error, not the splice's.
            "image_rows_bit_exact_vs_tt_tower": bool(torch.equal(tt[is_image], tt_features)),
            "tower_features_pcc_vs_golden": pcc(tt_features, want[is_image]),
            "max_abs_diff_image_rows": float((tt[is_image].float() - want[is_image].float()).abs().max()),
        }
        results.append(row)
        logger.info(f"splice {json.dumps(row)}")
    _write("splice.json", results)
    for r in results:
        assert r["image_rows_bit_exact_vs_tt_tower"], f"{r['id']}: spliced image rows != TT tower output: {r}"
        assert r["text_rows_bit_exact_vs_text_path"], f"{r['id']}: spliced text rows != TT text embedding: {r}"
        assert r["text_rows_bit_exact_vs_golden"], f"{r['id']}: spliced text rows != golden: {r}"
    low = [(r["id"], r["pcc_all"]) for r in results if not r["pcc_all"] >= TOWER_PCC]
    assert not low, f"Spliced inputs_embeds PCC vs golden < {TOWER_PCC} (12A tower bar, see splice.json): {low}"


@pytest.mark.timeout(3600)
def test_image_decision_agreement(model, golden):
    rows, layer_pcc, raw = [], {}, {}
    for g in golden.rows:
        rid, count = g["id"], g["count"]
        enc = encode_row(golden, rid)
        start = time.perf_counter()
        tt = run_image_row(model, enc, count, layer_trace=rid in TRACE_ROWS)
        seconds = time.perf_counter() - start
        hf_probs, hf_logits = golden.tensor(rid, "probs").float(), golden.tensor(rid, "logits").float()
        tt_top = int(torch.argmax(tt["probs"]))
        top2 = torch.topk(tt["probs"], min(2, count)).values
        row = {
            "id": rid,
            "type": g["type"],
            "seq_len": g["seq_len"],
            "bucket": tt["bucket"],
            "patches": g["patches"],
            "image_tokens": g["image_tokens"],
            "count": count,
            "hf_argmax": g["argmax"],
            "hf_choice": g["choice"],
            "hf_prob": g["max_prob"],
            "hf_top2_gap": g["top2_gap"],
            "tt_argmax": tt_top,
            "tt_prob": float(tt["probs"][tt_top]),
            "tt_top2_gap": float(top2[0] - top2[1]) if count > 1 else 1.0,
            "agree": tt_top == g["argmax"],
            "max_abs_prob_diff": float((tt["probs"] - hf_probs).abs().max()),
            "logit_pcc_valid": pcc(tt["logits"][:count], hf_logits[:count]),
            "logit_pcc_all255": pcc(tt["logits"], hf_logits),
            "max_abs_logit_diff_valid": float((tt["logits"][:count] - hf_logits[:count]).abs().max()),
            "final_hidden_pcc": pcc(tt["final_hidden"], golden.tensor(rid, "final_hidden")),
            "position_ids_equal_golden": bool(torch.equal(tt["position_ids"], golden.tensor(rid, "position_ids"))),
            "seconds_incl_prep_trace_readback": round(seconds, 3),
        }
        for name in ("probs", "logits", "final_hidden"):
            raw[f"{rid}.{name}"] = tt[name].contiguous()
        if "layer_hidden" in tt:
            hf_layers = golden.tensor(rid, "layer_last_hidden").float()
            layer_pcc[rid] = [pcc(tt["layer_hidden"][i], hf_layers[i]) for i in range(hf_layers.shape[0])]
            row["layer_pcc_min"] = min(layer_pcc[rid])
            row["layer_pcc_last"] = layer_pcc[rid][-1]
            raw[f"{rid}.layer_hidden"] = tt["layer_hidden"].contiguous()
        rows.append(row)
        logger.info(
            f"{rid:<22} S={row['seq_len']:<4} HF {g['choice']!r} {g['max_prob']:.4f} gap {g['top2_gap']:.3f} | "
            f"TT argmax {tt_top} {row['tt_prob']:.4f} agree={row['agree']} dprob={row['max_abs_prob_diff']:.4f} "
            f"logitPCC={row['logit_pcc_valid']:.5f} hidPCC={row['final_hidden_pcc']:.5f}"
        )

    agree = sum(r["agree"] for r in rows)
    misses = [r for r in rows if not r["agree"]]
    summary = {
        "agree": agree,
        "rows": len(rows),
        "misses": [{k: r[k] for k in ("id", "hf_top2_gap", "hf_prob", "tt_prob")} for r in misses],
        "logit_pcc_valid_min": min(r["logit_pcc_valid"] for r in rows),
        "logit_pcc_valid_median": statistics.median(r["logit_pcc_valid"] for r in rows),
        "final_hidden_pcc_min": min(r["final_hidden_pcc"] for r in rows),
        "max_abs_prob_diff_max": max(r["max_abs_prob_diff"] for r in rows),
        "policy": model.config.optimizations.policy.name,
        "vision_policy": model.vision.config.optimizations.policy.name,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = _write("image_decisions.json", {"summary": summary, "rows": rows, "layer_pcc": layer_pcc})
    save_file(raw, str(OUT_DIR / "image_outputs.safetensors"), metadata={"policy": summary["policy"]})
    logger.info(f"image decision agreement {agree}/{len(rows)}; {json.dumps(summary)} -> {path}")

    assert all(r["position_ids_equal_golden"] for r in rows)
    assert agree >= MIN_AGREE, f"Only {agree}/{len(rows)} image rows agree with HF top-1 (need {MIN_AGREE})"
    assert len(misses) <= len(rows) - MIN_AGREE
    for r in misses:
        assert r["hf_top2_gap"] < NEAR_TIE_GAP, f"{r['id']}: disagreement on a non-tie (HF gap {r['hf_top2_gap']:.4f})"
    low = [(r["id"], r["logit_pcc_valid"]) for r in rows if not r["logit_pcc_valid"] >= LOGIT_PCC]
    assert not low, f"Readout-logit PCC < {LOGIT_PCC}: {low}"


@pytest.mark.timeout(1800)
def test_decider_predict_image(model, golden):
    """``TTDecider.predict(state, question, images=[path])`` (the app API) on a golden row."""
    from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider
    from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer

    rid = "v01_dominant_color"
    g, row = golden.row(rid), golden.prompts[rid]["row"]
    decider = TTDecider(model, AppTokenizer())
    answer = decider.predict(row["state"], row["question"], images=row["images"])
    probs = decider.predict_probabilities(row["state"], row["question"], images=row["images"])
    result = {"id": rid, "answer": answer, "hf_answer": g["answer"], "probs": probs}
    _write("decider_predict_image.json", result)
    logger.info(f"TTDecider.predict image: {json.dumps(result)}")
    assert answer["choice"] == g["answer"]["choice"]
    hf = golden.tensor(rid, "probs").float()
    assert max(abs(p - float(h)) for p, h in zip(probs, hf)) < 0.05


@pytest.mark.timeout(1800)
def test_image_bucket_padding(model, golden):
    """An image request in its own bucket (1024) and padded to 8192: same decision, logit PCC >= 0.999.

    Also the DRAM view (text C0 + vision resident) right after the 8192-bucket image forward.
    """
    from models.demos.pplx_decider_v1_27b.tt.model import dram_view

    rid = "v04_tallest_bar"  # largest image (1024 patches, 256 image tokens); v02 is a TT tie, see README
    g = golden.row(rid)
    enc = encode_row(golden, rid)
    out = {}
    for bucket in (1024, 8192):
        tokens, last_index, images = model.prepare_images(enc, bucket=bucket)
        probs, logits, _ = model(tokens, last_index, g["count"], images=images)
        ttnn.synchronize_device(model.mesh_device)
        dram = dram_view(model.mesh_device)
        out[bucket] = {
            "probs": ttnn.to_torch(probs).float().reshape(-1)[: g["count"]],
            "logits": ttnn.to_torch(logits).float().reshape(-1)[:255],
            "dram_after_forward": dram,
        }
        for t in (tokens, probs, logits):
            ttnn.deallocate(t)
        images.deallocate()
    a, b = out[1024], out[8192]
    result = {
        "id": rid,
        "seq_len": g["seq_len"],
        "buckets": [1024, 8192],
        "argmax": [int(torch.argmax(a["probs"])), int(torch.argmax(b["probs"]))],
        "logit_pcc_valid": pcc(a["logits"][: g["count"]], b["logits"][: g["count"]]),
        "logit_pcc_all255": pcc(a["logits"], b["logits"]),
        "max_abs_prob_diff": float((a["probs"] - b["probs"]).abs().max()),
        "bit_identical_probs": bool(torch.equal(a["probs"], b["probs"])),
        "dram_after_forward": {k: v["dram_after_forward"] for k, v in out.items()},
    }
    _write("image_bucket_padding.json", result)
    logger.info(f"image bucket padding {json.dumps(result)}")
    assert result["argmax"][0] == result["argmax"][1]
    assert result["logit_pcc_valid"] >= 0.999 and result["logit_pcc_all255"] >= 0.999


@pytest.mark.timeout(1800)
def test_image_determinism(model, golden):
    """The same image request twice gives bit-identical probabilities and logits."""
    rid = "v02_count_circles"
    g = golden.row(rid)
    enc = encode_row(golden, rid)
    first, second = run_image_row(model, enc, g["count"]), run_image_row(model, enc, g["count"])
    result = {
        "id": rid,
        "probs_bit_identical": bool(torch.equal(first["probs"], second["probs"])),
        "logits_bit_identical": bool(torch.equal(first["logits"], second["logits"])),
        "final_hidden_bit_identical": bool(torch.equal(first["final_hidden"], second["final_hidden"])),
        "probs": first["probs"].tolist(),
    }
    _write("image_determinism.json", result)
    assert result["probs_bit_identical"] and result["logits_bit_identical"], result


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("row_id", ["v02_count_circles", "v04_tallest_bar"])
def test_image_forward_stays_on_device(model, golden, row_id):
    """0 host conversions / torch ops between the input upload and the probability readback.

    Uploaded inputs: token ids, splice index, cos/sin tables (``prepare_images``), and per image the
    pixel rows, pos-embed rows, vision cos/sin and the key mask (``PplxVisionTower.prepare_inputs``).
    """
    g = golden.row(row_id)
    enc = encode_row(golden, row_id)
    tokens, last_index, images = model.prepare_images(enc)
    warm = model(tokens, last_index, g["count"], images=images)  # program cache warm for this shape
    ttnn.synchronize_device(model.mesh_device)
    for t in warm[:2]:
        ttnn.deallocate(t)
    with count_host_calls() as counts:
        probs, logits, _ = model(tokens, last_index, g["count"], images=images)
        ttnn.synchronize_device(model.mesh_device)
    host = ttnn.to_torch(probs).float().reshape(-1)[: g["count"]]
    _write(
        f"runtime_audit_{row_id}.json",
        {"id": row_id, "bucket": tokens.shape[-1], "patches": g["patches"], "host_calls": dict(counts)},
    )
    for t in (tokens, probs, logits):
        ttnn.deallocate(t)
    images.deallocate()
    assert not counts, f"Host calls inside the image forward: {dict(counts)}"
    assert int(torch.argmax(host)) == g["argmax"]
