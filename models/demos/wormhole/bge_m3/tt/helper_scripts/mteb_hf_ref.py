"""HF/CPU MTEB reference for BGE-M3 at a chosen max sequence length.

mteb's BAAI/bge-m3 loader (sentence-transformers, CLS + normalize, cosine) uses
max_tokens 8194. The single-chip TT path runs S512, so its fair reference is HF with
max_seq_length 512.

Usage: python mteb_hf_ref.py <max_len> <out_dir> [task ...]   (default task: ArguAna)
"""

import json
import sys
from pathlib import Path

import mteb
import torch

max_len = int(sys.argv[1])
out = Path(sys.argv[2])
tasks = sys.argv[3:] or ["ArguAna"]
torch.set_num_threads(int(__import__("os").environ.get("HF_THREADS", "32")))

model = mteb.get_model("BAAI/bge-m3", device="cpu")
model.model.max_seq_length = max_len
print("HF_REF max_seq_length", model.model.max_seq_length, flush=True)
results = mteb.MTEB(tasks=mteb.get_tasks(tasks=tasks)).run(
    model, output_folder=str(out), eval_splits=["test"], overwrite_results=True, encode_kwargs={"batch_size": 16}
)
scores = {}
for r in results:
    for split in r.scores.values():
        scores[r.task_name] = float(split[0]["main_score"])
print("HF_REF max_len=%d %s" % (max_len, json.dumps(scores)), flush=True)
out.mkdir(parents=True, exist_ok=True)
(out / f"hf_ref_len{max_len}.json").write_text(json.dumps(scores, indent=2))
