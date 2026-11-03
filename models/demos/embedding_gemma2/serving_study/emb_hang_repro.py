"""Model-only repro of the FastAPI --max-batch 32 hang (2026-10-29): no HTTP, no worker process.

Calls EmbeddingGemma2ForEmbedding.forward in the pattern that hung: N1 batch-1 forwards, then N4 forwards of 4 rows
(batch-8 trace), then N_mix forwards of random 1..32 rows, all at 128 tokens. Prints progress every 500 forwards;
faulthandler dumps the Python stack if one forward takes over 60 s (the caller's timeout kills the process).

Usage: emb_hang_repro.py --open fastapi|media [--n1 3000 --n4 3000 --nmix 2000]
  fastapi: ttnn.open_mesh_device(MeshShape(1,1), trace_region_size=512 MiB), as scripts/emb_serve_fastapi.py
  media:   + DispatchCoreConfig() and torch threads 1, as tt-media-server's EmbeddingGemma2Runner; set
           TT_MESH_GRAPH_DESC_PATH to the P150 descriptor in the shell (the media server does it before import)
"""

import argparse
import faulthandler
import random
import sys
import time

import torch

import ttnn
from emb_common import MODEL_ID, load_tokenizer, make_texts
from models.demos.embedding_gemma2.demo.generator_vllm import EmbeddingGemma2ForEmbedding

ap = argparse.ArgumentParser()
ap.add_argument("--open", choices=["fastapi", "media"], required=True)
ap.add_argument("--n1", type=int, default=3000)
ap.add_argument("--n4", type=int, default=3000)
ap.add_argument("--nmix", type=int, default=2000)
ap.add_argument("--nalt", type=int, default=0, help="forwards of random 1..7 rows (b1/b8 traces alternate) with 0-5 ms gaps")
ap.add_argument("--phases", default="b1,rows4,mix,alt", help="phase order")
args = ap.parse_args()

params = {"trace_region_size": 512 * 1024 * 1024}
if args.open == "media":
    torch.set_num_threads(1)
    params["dispatch_core_config"] = ttnn.DispatchCoreConfig(None, None, None)
device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
model = EmbeddingGemma2ForEmbedding(device=device, max_batch_size=32, max_seq_len=8192, model_name=MODEL_ID)

tok = load_tokenizer()
ids = torch.tensor([tok(t).input_ids for t in make_texts(tok, 128, 32)], dtype=torch.int64)  # [32, 128]
mask = torch.ones_like(ids)

rng = random.Random(0)
print(f"model ready ({args.open} open), {len(model.shapes)} traces", flush=True)
all_phases = {"b1": (args.n1, lambda: 1), "rows4": (args.n4, lambda: 4), "mix": (args.nmix, lambda: rng.randint(1, 32)),
              "alt": (args.nalt, lambda: rng.randint(1, 7))}
phases = [(p, *all_phases[p]) for p in args.phases.split(",") if all_phases[p][0] > 0]
done, t0 = 0, time.time()
for name, n, rows_fn in phases:
    for i in range(n):
        r = rows_fn()
        faulthandler.dump_traceback_later(60, exit=False)
        model.forward(ids[:r], attention_mask=mask[:r])
        ttnn.synchronize_device(device)
        faulthandler.cancel_dump_traceback_later()
        if name == "alt":
            time.sleep(rng.random() * 0.005)
        done += 1
        if done % 250 == 0:
            print(f"{name} {i + 1}/{n} total {done} forwards, {time.time() - t0:.0f} s", flush=True)
    print(f"phase {name} done ({n} forwards)", flush=True)
print(f"PASS: {done} forwards without a hang ({args.open} open) in {time.time() - t0:.0f} s", flush=True)
model.release()
ttnn.close_mesh_device(device)
sys.exit(0)
