"""HF fp32 reference embeddings (CPU) for the DP check: writes the check texts and their sentence-transformers
embeddings (EmbeddingGemma2Model, masked mean over every token, L2 normalize).

Needs transformers >= 5.20 first on PYTHONPATH (the overlay in models/demos/embedding_gemma2/README.md, Setup).
Usage: emb_hf_ref.py OUT_DIR   -> OUT_DIR/check_texts.json, OUT_DIR/hf_ref.npy
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from emb_common import REVISION, load_tokenizer, make_texts

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

tok = load_tokenizer()
texts = [
    "hello world",
    "task: search result | query: how do the northern lights form?",
    "title: none | text: Aurora borealis appears when solar wind particles collide with oxygen and nitrogen.",
    "task: question answering | query: What is the capital of France?",
    "Paris is the capital and largest city of France.",
    "def add(a, b):\n    return a + b",
    "Tenstorrent Blackhole Galaxy: 32 chips in an 8x4 mesh.",
    "¿Dónde está la biblioteca? 图书馆在哪里？",
]
for n in (32, 100, 128, 300, 512, 1000, 2048):
    texts += make_texts(tok, n, 2, seed=n)

from huggingface_hub import snapshot_download

from models.demos.embedding_gemma2.reference.hf_full_reference import load_reference_full_model
from models.demos.embedding_gemma2.reference.hf_reference import sentence_embedding

snap = snapshot_download("google/embeddinggemma-2", revision=REVISION, local_files_only=True)
model = load_reference_full_model(snap, torch.float32)
embs = []
with torch.no_grad():
    for t in texts:  # one at a time: no padding in the reference
        f = tok([t], return_tensors="pt")
        h = model(input_ids=f["input_ids"], attention_mask=f["attention_mask"]).last_hidden_state.float()
        embs.append(sentence_embedding(h, f["attention_mask"])[0].numpy())
np.save(out / "hf_ref.npy", np.stack(embs).astype(np.float32))
json.dump(
    {"texts": texts, "n_tokens": [len(tok(t).input_ids) for t in texts]}, open(out / "check_texts.json", "w"), indent=1
)
print(f"wrote {len(texts)} reference embeddings to {out}")
