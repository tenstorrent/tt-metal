"""Model-only soak of one serving configuration (no HTTP): N forwards of random (rows, tokens) drawn from given ranges,
as the server worker builds the model (EmbeddingGemma2ForEmbedding, MeshShape(1,1), trace region 512 MiB).

Used to clear a batch ladder for the DP server (EMBEDDING_GEMMA2_SERVE_BATCHES picks the rungs). faulthandler dumps the
stack if a forward takes over 60 s.

Usage: emb_shape_soak.py --rows 1 --tokens 2-2048 --n 6000 [--max-batch 1] [--seed 0]
  --rows / --tokens: "K" or "A-B" (uniform). --max-batch: the model's max_batch_size (its trace ladder).
"""

import argparse
import faulthandler
import random
import sys
import time

import torch
from emb_common import MODEL_ID, load_tokenizer, make_texts

import ttnn
from models.demos.embedding_gemma2.demo.generator_vllm import EmbeddingGemma2ForEmbedding


def rng_range(spec):
    a, _, b = spec.partition("-")
    return int(a), int(b or a)


ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="1")
ap.add_argument("--tokens", default="128")
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--max-batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
r_lo, r_hi = rng_range(a.rows)
t_lo, t_hi = rng_range(a.tokens)

device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
model = EmbeddingGemma2ForEmbedding(device=device, max_batch_size=a.max_batch, max_seq_len=8192, model_name=MODEL_ID)
tok = load_tokenizer()
pool = torch.tensor(tok(make_texts(tok, 2048, 1)[0]).input_ids, dtype=torch.int64)  # 2048 real ids
rng = random.Random(a.seed)
print(f"model ready, {len(model.shapes)} traces, rows {a.rows} tokens {a.tokens}", flush=True)
t0 = time.time()
for i in range(a.n):
    rows, n = rng.randint(r_lo, r_hi), rng.randint(t_lo, t_hi)
    ids = pool[:n].repeat(rows, 1)
    faulthandler.dump_traceback_later(60, exit=False)
    model.forward(ids, attention_mask=torch.ones_like(ids))
    ttnn.synchronize_device(device)
    faulthandler.cancel_dump_traceback_later()
    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{a.n} forwards, {time.time() - t0:.0f} s", flush=True)
print(f"PASS: {a.n} forwards rows {a.rows} tokens {a.tokens} in {time.time() - t0:.0f} s", flush=True)
model.release()
ttnn.close_mesh_device(device)
sys.exit(0)
