# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline native rendering of retained completed questions in both content formats."""

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

root = Path("/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/benchmark/run")
t = AutoTokenizer.from_pretrained(
    "/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    local_files_only=True,
)
inputs = {(r["task"], r["doc_id"]): r for r in map(json.loads, (root / "benchmark-inputs.jsonl").open())}
responses = {r["id"]: r for r in map(json.loads, (root / "mmlu_pro/responses.jsonl").open())}
rows = []
for link in map(json.loads, (root / "mmlu_pro/request_links.jsonl").open()):
    x = inputs[(link["task"], link["doc_id"])]
    messages = json.loads(x["arguments"][0][0])
    raw = responses[link["response_id"]]
    variants = {
        "string": messages,
        "openai_parts": [{**m, "content": [{"type": "text", "text": m["content"]}]} for m in messages],
    }
    result = {}
    for name, msg in variants.items():
        ids = t.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True
        )["input_ids"]
        text = t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        result[name] = {
            "length": len(ids),
            "bos_count": ids.count(t.bos_token_id),
            "tokens_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "rendered_prefix": text[:300],
        }
    rows.append(
        {
            "task": link["task"],
            "doc_id": link["doc_id"],
            "response_id": link["response_id"],
            "api_prompt_tokens": raw["usage"]["prompt_tokens"],
            "variants": result,
        }
    )
assert len(rows) == 8
assert all(r["variants"]["openai_parts"]["length"] == r["api_prompt_tokens"] for r in rows)
assert all(v["bos_count"] == 1 for r in rows for v in r["variants"].values())
(root / "template-audit.json").write_text(
    json.dumps({"scope": "Offline pinned native tokenizer only; no server or inference", "rows": rows}, indent=2) + "\n"
)
print(
    [(r["api_prompt_tokens"], r["variants"]["string"]["length"], r["variants"]["openai_parts"]["length"]) for r in rows]
)
