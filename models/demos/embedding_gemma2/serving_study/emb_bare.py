"""Bare tt-metal path: tokenize (as the media server does) + EmbeddingGemma2ForEmbedding.forward, timed per call.

Usage: emb_bare.py out.json [reps]   (chip from TT_VISIBLE_DEVICES; EMB_TIMING_LOG adds the per-phase breakdown)
"""

import json
import os
import sys
import time

import torch

import ttnn
from emb_common import MODEL_ID, POINTS, load_tokenizer, make_texts
from models.demos.embedding_gemma2.demo.generator_vllm import EmbeddingGemma2ForEmbedding

out_path = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
t0 = time.perf_counter()
model = EmbeddingGemma2ForEmbedding(device=mesh, max_batch_size=32, max_seq_len=8192, model_name=MODEL_ID)
print(f"model + warmup {time.perf_counter() - t0:.1f}s", flush=True)
tok = load_tokenizer()
rows, golden = [], {}
for batch, n_tok in POINTS:
    texts = make_texts(tok, n_tok, batch)
    tok_ms, fwd_ms = [], []
    for r in range(reps + 3):
        a = time.perf_counter()
        enc = tok(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        b = time.perf_counter()
        emb = model.forward(enc["input_ids"], attention_mask=enc["attention_mask"])
        ttnn.synchronize_device(mesh)
        c = time.perf_counter()
        if r >= 3:
            tok_ms.append((b - a) * 1e3)
            fwd_ms.append((c - b) * 1e3)
    golden[f"{batch}x{n_tok}"] = emb.tolist()
    med = lambda v: sorted(v)[len(v) // 2]
    row = dict(batch=batch, tokens=n_tok, actual_tokens=int(enc["attention_mask"].sum(-1).max()), reps=reps,
               tokenize_ms_p50=med(tok_ms), forward_ms_p50=med(fwd_ms), e2e_ms_p50=med([x + y for x, y in zip(tok_ms, fwd_ms)]))
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(dict(rows=rows, golden=golden), open(out_path, "w"))
model.release()
ttnn.close_mesh_device(mesh)
