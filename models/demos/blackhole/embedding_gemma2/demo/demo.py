# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text-retrieval demo: embed queries and documents on device (traced) and rank documents.

    pytest models/demos/blackhole/embedding_gemma2/demo/demo.py -s
    python models/demos/blackhole/embedding_gemma2/demo/demo.py        (same, as a script)

Prompts follow the model card (`config_sentence_transformers.json`): queries get
"task: search result | query: ", documents "title: none | text: ".
"""

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder
from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "

QUERIES = [
    "What causes the northern lights?",
    "How do I bake sourdough bread?",
    "Which planet is the largest in the solar system?",
]
DOCUMENTS = [
    "The northern lights are caused by charged particles from the sun hitting the upper atmosphere.",
    "Jupiter is the largest planet in our solar system, more than twice as massive as all others combined.",
    "Sourdough bread uses a fermented starter of flour and water instead of commercial yeast.",
    "The stock market closed higher on Tuesday after a strong earnings season.",
]


def run_demo(device, model_id="google/embeddinggemma-2"):
    from transformers import AutoTokenizer

    snapshot = resolve_snapshot(model_id)
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    enc = TextEncoder.from_pretrained(device, str(snapshot), max_seq_len=2048)

    batches = [
        tok([QUERY_PREFIX + t for t in QUERIES], padding=True, return_tensors="pt"),
        tok([DOC_PREFIX + t for t in DOCUMENTS], padding=True, return_tensors="pt"),
    ]
    # one trace per (batch, bucket), all captured before the first replay
    shapes = [(b["input_ids"].shape[0], enc.prepare_inputs(b["input_ids"]).seq_len) for b in batches]
    enc.capture_all(shapes)
    q, d = (enc.encode(b["input_ids"], b["attention_mask"], traced=True) for b in batches)
    scores = q @ d.T
    for i, query in enumerate(QUERIES):
        best = int(scores[i].argmax())
        logger.info(f"query: {query!r}\n    best doc ({scores[i, best]:.3f}): {DOCUMENTS[best]!r}")
    enc.release_traces()
    return scores


def test_demo(eg2_device):
    scores = run_demo(eg2_device)
    # query i -> its matching document
    assert scores.argmax(dim=-1).tolist() == [0, 2, 1]
    assert torch.isfinite(scores).all()


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, trace_region_size=128 * 1024 * 1024)
    try:
        run_demo(dev)
    finally:
        ttnn.close_device(dev)
