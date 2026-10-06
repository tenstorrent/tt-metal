# SPDX-License-Identifier: Apache-2.0
"""Opt-in test-only canonical-generator baseline before any HTTP requests exist."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import torch


def run(generator):
    if os.environ.get("KOLIBRI_VLLM_CHUNK_CONTROL") == "1":
        from .vllm_chunk_control import run as chunk_control

        return chunk_control(generator)
    g = generator
    root = Path(__file__).resolve().parents[1]
    quality_prompts = json.loads((root / "doc/datatype_sweep/selected_default/qualitative_prompts.json").read_text())
    prompts = quality_prompts[:3]
    prior = json.loads((root / "doc/datatype_sweep/selected_default/qualitative_tt.json").read_text())
    prompts += [
        dict(id=f"{x['id']}-continuation16", token_ids=x["prompt_ids"] + x["generated_ids"][:16]) for x in prior[:3]
    ]
    saved = g.initial_pages
    # The vLLM-owned pool may alias layers. Give every layer distinct physical
    # pages for this short setup control, using only the supplied allocation.
    tables = {i: torch.zeros_like(table) for i, table in g.state.host_page_tables.items()}
    for i, table in tables.items():
        table[0, :128] = torch.arange(i * 128, (i + 1) * 128, dtype=torch.int32)
        assert (i + 1) * 128 <= g.state.layers[i][0].shape[0]
    records = []
    try:
        g.initial_pages = tables
        for repeat in range(2):
            for prompt in prompts:
                ids = prompt["token_ids"]
                sampled = g.generate(ids, 1, temperature=0.0, stop_on_eos=False)
                logits = g.read_logits()[0, 0, 0].float()
                scores = torch.log_softmax(logits, dim=-1)
                values, indices = scores.topk(20)
                records.append(
                    dict(
                        id=prompt["id"],
                        repeat=repeat,
                        prompt_ids=ids,
                        token=sampled[0],
                        logit_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                        top20_token_ids=indices.tolist(),
                        top20_logprobs=values.tolist(),
                    )
                )
        assert [x["logit_sha256"] for x in records[: len(prompts)]] == [
            x["logit_sha256"] for x in records[len(prompts) :]
        ]
        traffic_path = root / "readiness_vllm/full-tracked-final-requests.json"
        traffic_controls = []
        traffic_archive = traffic_path.with_name(traffic_path.name + ".gz")
        if traffic_path.exists() or traffic_archive.exists():
            seen = set()
            traffic = json.loads(
                traffic_path.read_text() if traffic_path.exists() else gzip.decompress(traffic_archive.read_bytes())
            )
            for request in traffic["results"]:
                key = tuple(request["prompt_ids"])
                if request["sampled"] or key in seen:
                    continue
                seen.add(key)
                generated = g.generate(
                    request["prompt_ids"], len(request["output_token_ids"]), temperature=0.0, stop_on_eos=False
                )
                traffic_controls.append(
                    dict(
                        label=request["label"],
                        prompt_ids=request["prompt_ids"],
                        generated_ids=generated,
                        served_ids=request["output_token_ids"],
                        exact_match=generated == request["output_token_ids"],
                    )
                )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["KOLIBRI_CHECKPOINT_DIR"])
        quality_controls = []
        for prompt in quality_prompts:
            generated = g.generate(prompt["token_ids"], 256, temperature=0.0, stop_on_eos=False)
            quality_controls.append(
                dict(
                    id=prompt["id"],
                    prompt=prompt["prompt"],
                    prompt_ids=prompt["token_ids"],
                    generated_ids=generated,
                    text=tokenizer.decode(generated, skip_special_tokens=False),
                )
            )
        passed = all(x["exact_match"] for x in traffic_controls)
        Path(os.environ["KOLIBRI_VLLM_STARTUP_CONTROL"]).write_text(
            json.dumps(
                dict(
                    status="pass" if passed else "traffic-control-mismatch",
                    path="canonical KolibriGenerator.generate; no adapter forward or scheduler",
                    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    batch=g.batch_size,
                    capacity=g.capacity,
                    records=records,
                    traffic_controls=traffic_controls,
                    quality_controls=quality_controls,
                ),
                indent=2,
            )
            + "\n"
        )
        # Keep the diagnostic server available if a comparison needs follow-up.
        # The saved non-pass status remains a required gate, never a waiver.
        print("CANONICAL_CONTROL_STATUS", "pass" if passed else "traffic-control-mismatch", flush=True)

    finally:
        g.initial_pages = saved
        g.reset()
