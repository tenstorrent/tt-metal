# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gates P1.4-P1.6: run the CPU reference over book text in chunks and dump goldens.

Output: generated/ernie45_d_p/golden/s{seq}_c{chunk}/
    manifest.json                    run description + index of what is stored
    metadata.json                    prefill-server golden-trace style (token_ids, num_layers, ...)
    kv_cache/layer_{i}.safetensors   key_cache_layer_{i} / value_cache_layer_{i}: [1, n_kv, seq, head_dim] bf16,
                                     K post-RoPE in native interleaved order (= Meta order, no permutation)
    chunk_{c:02d}/layer_{i:02d}.safetensors  per-layer intermediates of chunk c (see ernie_ref Recorder names)
    chunk_{c:02d}/model.safetensors  embed, final_norm, top32 logits (values+ids) for every position,
                                     full logits of the last 32 positions of the chunk

Per-layer intermediates are stored for every chunk with --full-dumps, otherwise only for the last chunk.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from models.demos.ernie45_d_p.bringup import metrics  # noqa: E402
from models.demos.ernie45_d_p.reference.ernie_ref import HF_MODEL_ID, ErnieReference, load_book_tokens  # noqa: E402

TASK = os.environ.get("ERNIE_BRINGUP_TASK", "golden")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
FULL_LOGITS_TAIL = 32
TOPK = 32


def golden_dir(seq: int, chunk: int) -> str:
    root = os.environ.get("ERNIE_GOLDEN_ROOT", os.path.join(REPO, "generated/ernie45_d_p/golden"))
    return os.path.join(root, f"s{seq}_c{chunk}")


def _store_dtype(name: str, t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.int64, torch.int32):
        return t.contiguous()
    if name.endswith(("router_logits", "topk_w")):
        return t.float().contiguous()
    return t.to(torch.bfloat16).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--full-dumps", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="first N layers only (debug)")
    a = ap.parse_args()
    assert a.seq % a.chunk == 0, "seq must be a multiple of chunk"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoTokenizer

    out = golden_dir(a.seq, a.chunk)
    os.makedirs(os.path.join(out, "kv_cache"), exist_ok=True)
    layers = list(range(a.layers)) if a.layers else None
    t0 = time.time()
    ref = ErnieReference(dtype=torch.float32, layers=layers)
    tok = AutoTokenizer.from_pretrained(ref.model_path)
    tokens = load_book_tokens(tok, a.seq)
    print(f"loaded reference in {time.time() - t0:.0f}s; {len(ref.layer_ids)} layers; seq={a.seq} chunk={a.chunk}")

    n_chunks = a.seq // a.chunk
    cache = ref.new_cache(a.seq)
    chunk_times, acc_top1, acc_top5 = [], [], []
    for c in range(n_chunks):
        dump = a.full_dumps or c == n_chunks - 1
        per_layer = defaultdict(dict)
        model_t = {}

        def rec(name, t, per_layer=per_layer, model_t=model_t, dump=dump):
            if name.startswith("L"):
                if dump:
                    li, key = name[1:].split(".", 1)
                    per_layer[int(li)][key] = _store_dtype(name, t.detach().clone())
            elif name != "logits":
                model_t[name] = _store_dtype(name, t.detach().clone())

        s = c * a.chunk
        tc = time.time()
        normed, logits = ref.forward_chunk(tokens[s : s + a.chunk], s, cache, rec, logits_last_n=a.chunk)
        chunk_times.append(time.time() - tc)

        vals, ids = torch.topk(logits.float(), TOPK, dim=-1)
        model_t["top32_values"] = vals.contiguous()
        model_t["top32_ids"] = ids.to(torch.int32).contiguous()
        model_t["logits_tail"] = logits[-FULL_LOGITS_TAIL:].float().contiguous()
        model_t["tokens"] = tokens[s : s + a.chunk].to(torch.int32).contiguous()
        # Self-consistency vs the book text (teacher-forced next-token prediction).
        nxt = tokens[s + 1 : s + a.chunk + 1]
        n = nxt.shape[0]
        acc_top1.append((ids[:n, 0] == nxt).float().mean().item())
        acc_top5.append((ids[:n, :5] == nxt[:, None]).any(-1).float().mean().item())

        cdir = os.path.join(out, f"chunk_{c:02d}")
        os.makedirs(cdir, exist_ok=True)
        save_file(model_t, os.path.join(cdir, "model.safetensors"))
        for li, tensors in per_layer.items():
            save_file(tensors, os.path.join(cdir, f"layer_{li:02d}.safetensors"))
        print(
            f"chunk {c + 1}/{n_chunks} [{s},{s + a.chunk}) {chunk_times[-1]:.0f}s "
            f"top1={acc_top1[-1]:.3f} top5={acc_top5[-1]:.3f} dumped_layers={len(per_layer)}",
            flush=True,
        )

    for i in ref.layer_ids:
        save_file(
            {
                f"key_cache_layer_{i}": cache.k[i][None].to(torch.bfloat16).contiguous(),
                f"value_cache_layer_{i}": cache.v[i][None].to(torch.bfloat16).contiguous(),
            },
            os.path.join(out, "kv_cache", f"layer_{i}.safetensors"),
        )

    cfg = ref.cfg
    meta = {
        "model": HF_MODEL_ID,
        "token_ids": tokens.tolist(),
        "num_layers": len(ref.layer_ids),
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "kv_cache_format": "separate_k_v",
        "k_rope_layout": "interleaved",
        "seq_len": a.seq,
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f)
    manifest = {
        "model": HF_MODEL_ID,
        "seq": a.seq,
        "chunk": a.chunk,
        "n_chunks": n_chunks,
        "layers": ref.layer_ids,
        "moe_layers": [i for i in ref.layer_ids if cfg.is_moe_layer(i)],
        "full_dumps": a.full_dumps,
        "dumped_chunks": list(range(n_chunks)) if a.full_dumps else [n_chunks - 1],
        "compute_dtype": "float32",
        "store_dtype": "bfloat16 (router_logits/topk_w fp32, topk_idx int64)",
        "text": "A Tale of Two Cities (Project Gutenberg), BOS-prefixed",
        "chunk_times_s": [round(t, 1) for t in chunk_times],
        "book_top1_acc": acc_top1,
        "book_top5_acc": acc_top5,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    metrics.record(TASK, "golden_layers", len(ref.layer_ids))
    metrics.record(TASK, "golden_chunks", n_chunks)
    metrics.record(TASK, "cpu_seconds", round(time.time() - t0, 1))
    metrics.record(TASK, "book_top5_acc_last_chunk", acc_top5[-1])
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
