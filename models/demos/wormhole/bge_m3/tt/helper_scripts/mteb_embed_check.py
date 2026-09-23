"""Embedding-level check of the single-chip MTEB adapter against HF on ArguAna texts.

Encodes the same ArguAna texts (mteb's title + text for documents, and queries) with
  - TT: demo/mteb_eval_minimal.py TTSingleChipEmbedder (S512, masked path, CLS, L2 norm)
  - HF: sentence-transformers BAAI/bge-m3 on CPU at max_seq_length 512 and 8192
and prints per-text cosine between them, split by token length (<= 512 / > 512),
plus the agreement of the query x document similarity matrices.

Usage: TT_VISIBLE_DEVICES=0 python mteb_embed_check.py [batch] [n_docs] [n_queries]
"""

import sys

import mteb
import numpy as np
from sentence_transformers import SentenceTransformer

import ttnn
from models.demos.wormhole.bge_m3.demo.mteb_eval_minimal import TTSingleChipEmbedder

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 8
n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 192
n_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 64

task = mteb.get_tasks(tasks=["ArguAna"])[0]
task.load_data()
split = task.dataset["default"]["test"]
corpus, queries = split["corpus"], split["queries"]


def doc_text(r):
    return (r["title"] + " " + r["text"]).strip() if r.get("title") else r["text"].strip()


hf = SentenceTransformer("BAAI/bge-m3", device="cpu")
tok = hf.tokenizer
all_docs = [doc_text(r) for r in corpus]
lens = np.array([len(tok(t)["input_ids"]) for t in all_docs])
# Half the documents are the longest ones (truncated at 512), half are the first ones.
long_idx = list(np.argsort(-lens)[: n_docs // 2])
short_idx = [i for i in range(len(all_docs)) if lens[i] <= 512][: n_docs - len(long_idx)]
doc_idx = long_idx + short_idx
docs = [all_docs[i] for i in doc_idx]
qs = [q["text"] for q in queries.select(range(n_queries))]
texts = docs + qs
tlen = np.array([len(tok(t)["input_ids"]) for t in texts])
print(
    "CHECK texts %d (docs %d, queries %d); > 512 tokens: %d" % (len(texts), len(docs), len(qs), int((tlen > 512).sum()))
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000, num_command_queues=1)
try:
    tt = TTSingleChipEmbedder(mesh, batch)
    tt_emb = tt.encode([{"text": texts}])
    tt.release()
finally:
    ttnn.close_mesh_device(mesh)

hf.max_seq_length = 512
hf512 = hf.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
hf.max_seq_length = 8192
hf8k = hf.encode(texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True)

tt_n = np.linalg.norm(tt_emb, axis=1)
print("CHECK TT norms min %.6f max %.6f; finite %s" % (tt_n.min(), tt_n.max(), np.isfinite(tt_emb).all()))


def cos_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


for name, a, b in [("TT vs HF512", tt_emb, hf512), ("TT vs HF8192", tt_emb, hf8k), ("HF512 vs HF8192", hf512, hf8k)]:
    c = cos_rows(a, b)
    for sel_name, sel in [("<=512 tok", tlen <= 512), (">512 tok", tlen > 512)]:
        if sel.any():
            print(
                "CHECK %-16s %-9s n=%3d cos mean %.5f min %.5f"
                % (name, sel_name, sel.sum(), c[sel].mean(), c[sel].min())
            )

# Query x document similarity agreement (what the retrieval score uses).
nd = len(docs)
for name, e in [("HF512", hf512), ("HF8192", hf8k)]:
    s_tt = tt_emb[nd:] @ tt_emb[:nd].T
    s_hf = e[nd:] @ e[:nd].T
    top_tt = s_tt.argmax(1)
    top_hf = s_hf.argmax(1)
    corr = np.corrcoef(s_tt.ravel(), s_hf.ravel())[0, 1]
    print(
        "CHECK sim-matrix TT vs %-6s pearson %.5f top1 agree %d/%d maxabs %.4f"
        % (name, corr, int((top_tt == top_hf).sum()), len(qs), np.abs(s_tt - s_hf).max())
    )
