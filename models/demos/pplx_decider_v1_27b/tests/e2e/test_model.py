# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: the full 64-layer TT model against the stage-6 bf16 HF decision golden.

Golden (``reference/hf_decision_golden.py``; 25 rows, bf16 HF, layer-streamed on CPU):
``$PPLX_DECIDER_DECISION_GOLDEN`` (default artifacts/pplx_decider/goldens/decisions/), files
``golden_bf16.safetensors`` and ``summary_bf16.json``.

Gates (person-approved, decisions.tsv 2026-10-09):
- decision agreement: TT top-1 == HF top-1 on >= 24/25 rows, and a disagreement is allowed only
  on a near-tie (HF top-2 probability gap < 0.05);
- readout-logit PCC >= 0.99 per row over the valid options (``logits[:count]``).
Recorded, not gated: last-token final-hidden PCC, max |prob diff|, PCC over all 255 logits, and
the last-token residual PCC after every layer (64 values per row).

Also here (one model load per module): the bucket padding check, determinism, and the runtime
fallback audit of the full forward. Results go to ``$PPLX_DECIDER_STAGE6_DIR`` (JSON).

Run::

    pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_model.py -q -s
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

import ttnn
from models.demos.pplx_decider_v1_27b.tests.runtime_audit import count_host_calls
from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel, bucket_for

GOLDEN_DIR = Path(
    os.environ.get("PPLX_DECIDER_DECISION_GOLDEN", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens/decisions")
)
OUT_DIR = Path(os.environ.get("PPLX_DECIDER_STAGE6_DIR", "/local/ttuser/gtobar/artifacts/pplx_decider/stage6"))
MIN_AGREE = 24
NEAR_TIE_GAP = 0.05
LOGIT_PCC = 0.99
PADDING_PCC = 0.999

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
            pytest.fail(f"Missing {path}; run python -m models.demos.pplx_decider_v1_27b.reference.hf_decision_golden")
        self.summary = json.loads((directory / "summary_bf16.json").read_text())
        self.rows = self.summary["rows"]
        self._file = safe_open(str(path), framework="pt")

    def tensor(self, row_id: str, name: str) -> torch.Tensor:
        return self._file.get_tensor(f"{row_id}.{name}")

    def row(self, row_id: str) -> dict:
        return next(r for r in self.rows if r["id"] == row_id)


@pytest.fixture(scope="module")
def golden():
    return Golden()


@pytest.fixture(scope="module")
def model(_device_module_impl):
    return PplxDeciderModel.from_snapshot(_device_module_impl)


def run_row(model, ids: list[int], count: int, *, bucket=None, layer_trace=False) -> dict:
    tokens, last_index = model.upload_tokens(ids, bucket)
    probs, logits, extras = model(tokens, last_index, count, return_hidden=True, collect_layer_hidden=layer_trace)
    out = {
        "probs": ttnn.to_torch(probs).float().reshape(-1)[:count],
        "logits": ttnn.to_torch(logits).float().reshape(-1)[:255],
        "final_hidden": ttnn.to_torch(extras["final_hidden"]).float().reshape(-1),
    }
    if layer_trace:
        out["layer_hidden"] = torch.stack([ttnn.to_torch(t).float().reshape(-1) for t in extras["layer_hidden"]])
        for t in extras["layer_hidden"]:
            ttnn.deallocate(t)
    for t in (tokens, probs, logits, extras["final_hidden"]):
        ttnn.deallocate(t)
    return out


def _write(name: str, payload) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


@pytest.mark.timeout(3600)
def test_decision_agreement(model, golden):
    rows, layer_pcc, raw = [], {}, {}
    for g in golden.rows:
        rid, count = g["id"], g["count"]
        ids = golden.tensor(rid, "input_ids").tolist()
        assert len(ids) == g["seq_len"]
        start = time.perf_counter()
        tt = run_row(model, ids, count, layer_trace=True)
        seconds = time.perf_counter() - start
        hf_probs, hf_logits = golden.tensor(rid, "probs").float(), golden.tensor(rid, "logits").float()
        hf_layers = golden.tensor(rid, "layer_last_hidden").float()
        tt_top = int(torch.argmax(tt["probs"]))
        top2 = torch.topk(tt["probs"], min(2, count)).values
        row = {
            "id": rid,
            "type": g["type"],
            "seq_len": g["seq_len"],
            "bucket": bucket_for(g["seq_len"]),
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
            "seconds_incl_trace_readback": round(seconds, 3),
        }
        layer_pcc[rid] = [pcc(tt["layer_hidden"][i], hf_layers[i]) for i in range(hf_layers.shape[0])]
        for name in ("probs", "logits", "final_hidden", "layer_hidden"):
            raw[f"{rid}.{name}"] = tt[name].contiguous()
        row["layer_pcc_min"] = min(layer_pcc[rid])
        row["layer_pcc_last"] = layer_pcc[rid][-1]
        rows.append(row)
        logger.info(
            f"{rid:<26} S={row['seq_len']:<5} HF {g['choice']!r} {g['max_prob']:.4f} | TT argmax {tt_top} "
            f"{row['tt_prob']:.4f} agree={row['agree']} dprob={row['max_abs_prob_diff']:.4f} "
            f"logitPCC={row['logit_pcc_valid']:.5f} hidPCC={row['final_hidden_pcc']:.5f}"
        )

    agree = sum(r["agree"] for r in rows)
    misses = [r for r in rows if not r["agree"]]
    logit_pccs = [r["logit_pcc_valid"] for r in rows]
    summary = {
        "agree": agree,
        "rows": len(rows),
        "misses": [{k: r[k] for k in ("id", "hf_top2_gap", "hf_prob", "tt_prob")} for r in misses],
        "logit_pcc_valid_min": min(logit_pccs),
        "logit_pcc_valid_median": statistics.median(logit_pccs),
        "final_hidden_pcc_min": min(r["final_hidden_pcc"] for r in rows),
        "final_hidden_pcc_median": statistics.median(r["final_hidden_pcc"] for r in rows),
        "max_abs_prob_diff_max": max(r["max_abs_prob_diff"] for r in rows),
        "policy": model.config.optimizations.policy.name,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = _write("e2e_decisions.json", {"summary": summary, "rows": rows, "layer_pcc": layer_pcc})
    # Raw TT outputs, for bit-identity checks across code changes (tests/e2e/compare_outputs.py).
    save_file(raw, str(OUT_DIR / "e2e_outputs.safetensors"), metadata={"policy": summary["policy"]})
    logger.info(f"decision agreement {agree}/{len(rows)}; {json.dumps(summary)} -> {path}")

    assert agree >= MIN_AGREE, f"Only {agree}/{len(rows)} rows agree with HF top-1 (need {MIN_AGREE})"
    for r in misses:
        assert r["hf_top2_gap"] < NEAR_TIE_GAP, f"{r['id']}: disagreement on a non-tie (HF gap {r['hf_top2_gap']:.4f})"
    low = [(r["id"], r["logit_pcc_valid"]) for r in rows if not r["logit_pcc_valid"] >= LOGIT_PCC]
    assert not low, f"Readout-logit PCC < {LOGIT_PCC}: {low}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("row_id", ["s01_ticket_routing", "l05_league_leader"])
def test_bucket_padding(model, golden, row_id):
    """The same prompt padded to its own bucket and to the next one: same decision, logit PCC >= 0.999."""
    g = golden.row(row_id)
    ids = golden.tensor(row_id, "input_ids").tolist()
    natural = bucket_for(len(ids))
    larger = {128: 1024, 1024: 2048, 2048: 4096, 4096: 8192}[natural]
    a = run_row(model, ids, g["count"], bucket=natural)
    b = run_row(model, ids, g["count"], bucket=larger)
    result = {
        "id": row_id,
        "seq_len": len(ids),
        "buckets": [natural, larger],
        "argmax": [int(torch.argmax(a["probs"])), int(torch.argmax(b["probs"]))],
        "logit_pcc_valid": pcc(a["logits"][: g["count"]], b["logits"][: g["count"]]),
        "logit_pcc_all255": pcc(a["logits"], b["logits"]),
        "max_abs_prob_diff": float((a["probs"] - b["probs"]).abs().max()),
        "bit_identical_probs": bool(torch.equal(a["probs"], b["probs"])),
    }
    _write(f"bucket_padding_{row_id}.json", result)
    logger.info(f"bucket padding {json.dumps(result)}")
    assert result["argmax"][0] == result["argmax"][1]
    assert result["logit_pcc_all255"] >= PADDING_PCC and result["logit_pcc_valid"] >= PADDING_PCC


@pytest.mark.timeout(1800)
def test_determinism(model, golden):
    """The same prompt twice gives bit-identical probabilities and logits."""
    row_id = "m06_server_500"
    g = golden.row(row_id)
    ids = golden.tensor(row_id, "input_ids").tolist()
    first, second = run_row(model, ids, g["count"]), run_row(model, ids, g["count"])
    result = {
        "id": row_id,
        "probs_bit_identical": bool(torch.equal(first["probs"], second["probs"])),
        "logits_bit_identical": bool(torch.equal(first["logits"], second["logits"])),
        "probs": first["probs"].tolist(),
    }
    _write("determinism.json", result)
    assert result["probs_bit_identical"] and result["logits_bit_identical"], result


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("row_id", ["s01_ticket_routing", "x01_log_most_errors"])
def test_full_forward_stays_on_device(model, golden, row_id):
    """0 host conversions / torch ops between the token upload and the probability readback."""
    g = golden.row(row_id)
    tokens, last_index = model.upload_tokens(golden.tensor(row_id, "input_ids").tolist())
    warm = model(tokens, last_index, g["count"])  # program cache warm for this bucket
    ttnn.synchronize_device(model.mesh_device)
    for t in warm[:2]:
        ttnn.deallocate(t)
    with count_host_calls() as counts:
        probs, logits, _ = model(tokens, last_index, g["count"])
        ttnn.synchronize_device(model.mesh_device)
    host = ttnn.to_torch(probs).float().reshape(-1)[: g["count"]]
    _write(f"runtime_audit_{row_id}.json", {"id": row_id, "bucket": tokens.shape[-1], "host_calls": dict(counts)})
    assert not counts, f"Host calls inside the full forward: {dict(counts)}"
    assert int(torch.argmax(host)) == g["argmax"]
