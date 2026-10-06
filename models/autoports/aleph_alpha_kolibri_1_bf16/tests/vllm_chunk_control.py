# SPDX-License-Identifier: Apache-2.0
"""Exact canonical chunk-schedule control, without adapter or scheduler calls."""

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import torch


def run(g):
    root = Path(__file__).resolve().parents[1]
    path = root / "readiness_vllm/full-tracked-final-requests.json"
    traffic = json.loads(
        path.read_text() if path.exists() else gzip.decompress(path.with_name(path.name + ".gz").read_bytes())
    )
    req = next(x for x in traffic["results"] if x["label"] == "concurrent-b")
    ids = req["prompt_ids"]
    saved = g.initial_pages
    tables = {i: torch.zeros_like(table) for i, table in g.state.host_page_tables.items()}
    for i, table in tables.items():
        for slot in (0, 3):
            first = i * 256 + slot * 64
            table[slot, :64] = torch.arange(first, first + 64, dtype=torch.int32)
        assert first + 64 <= g.state.layers[i][0].shape[0]
    records = []
    logit_controls = {}
    try:
        g.initial_pages = tables
        for slot, ends in (
            (0, [1025]),
            (3, [1025]),
            (0, [251, 763, 1025]),
            (3, [251, 763, 1025]),
            (0, [512, 1024, 1025]),
        ):
            g.reset()
            g.set_sampling(temperature=0.0)
            start = 0
            began = time.monotonic()
            for end in ends:
                token = g.prefill_forward(
                    torch.tensor([ids[start:end]]),
                    page_table=tables,
                    kv_cache=g.state,
                    prompt_lens=[end - start],
                    start_pos=[start],
                    slots=[slot],
                    output_mask=[end == len(ids)],
                )
                start = end
            logits = g.read_logits()[0, 0, slot].float()
            assert torch.isfinite(logits).all(), "Nonfinite canonical logits"
            key = f"slot{slot}-ends{ends}"
            comparisons = {name: float((logits - ref).abs().max()) for name, ref in logit_controls.items()}
            logit_controls[key] = logits.clone()
            vals, idx = logits.topk(20)
            seq = [int(token[slot])]
            first_token = time.monotonic()
            for _ in range(len(req["output_token_ids"]) - 1):
                token = g.decode_forward(None, None, page_table=None, kv_cache=g.state)
                seq.append(int(token[slot]))
            finished = time.monotonic()
            records.append(
                dict(
                    slot=slot,
                    ends=ends,
                    starts=[0] + ends[:-1],
                    generated_ids=seq,
                    top20_ids=idx.tolist(),
                    top20_logits=vals.tolist(),
                    disputed_logits={str(i): float(logits[i]) for i in (46, 21499)},
                    logit_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                    max_abs_logit_differences=comparisons,
                    matches_served=seq == req["output_token_ids"],
                    ttft_ms=1000 * (first_token - began),
                    decode_ms_per_token=1000 * (finished - first_token) / (len(seq) - 1),
                )
            )
        matched = all(x["matches_served"] for x in records if x["ends"] == [251, 763, 1025])
        result = dict(
            status="pass" if matched else "chunk-control-mismatch",
            prompt_ids=ids,
            served_ids=req["output_token_ids"],
            records=records,
            path="canonical prefill_forward/decode_forward; external cache; no adapter, scheduler, or overlap",
        )
        Path(os.environ["KOLIBRI_VLLM_STARTUP_CONTROL"]).write_text(json.dumps(result, indent=2) + "\n")
        print("CANONICAL_CHUNK_CONTROL_STATUS", result["status"], flush=True)
    finally:
        g.initial_pages = saved
        g.reset()
