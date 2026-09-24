# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""STS-B accuracy through the *batched* prefill path (the demo's bs8/16/32 configuration).

``eval_accuracy_tt.py`` encodes one text per forward (batch_size=1), so it never exercises the
batched kernels and program configs. This script builds the model exactly as the perf demo does
for ``--batch B`` (``apply_workload_env(B, 512)``), runs the STS-B test texts B at a time through
``ttnn_prefill_forward`` (eager), applies the final RMSNorm on host over the full sequence and
pools two ways: mean over the real tokens ("masked") and mean over the padded ISL ("fast", what
the bs1 script's default does on device). It prints the Spearman correlation for both, and saves
the embeddings so runs at different batch sizes can be compared text by text
(``--compare /tmp/embs_B1.pt``: per-text cosine of this run vs a reference run).

    TT_VISIBLE_DEVICES=10 python eval_accuracy_batched.py --batch 8
    TT_VISIBLE_DEVICES=9  python eval_accuracy_batched.py --batch 1 --save /tmp/embs_B1.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="only the first N pairs (0 = all 1379)")
    p.add_argument("--save", default="", help="save per-text embeddings (masked pool, pre-normalisation)")
    p.add_argument("--compare", default="", help="reference embeddings file from another --save run")
    args = p.parse_args()
    B, S = args.batch, args.max_length

    from scipy.stats import spearmanr

    from models.demos.blackhole.pplx_embed_4b.demo.eval_accuracy import load_stsb

    s1, s2, gold = load_stsb()
    if args.limit:
        s1, s2, gold = s1[: args.limit], s2[: args.limit], gold[: args.limit]
    texts = s1 + s2
    print(f"{len(s1)} pairs, {len(texts)} texts, batch {B}, ISL {S}", flush=True)

    import ttnn
    from models.demos.blackhole.pplx_embed_4b.demo._common import apply_workload_env, build_single_device_model
    from models.demos.blackhole.pplx_embed_4b.demo.live_demo import _extract_final_norm
    from models.tt_transformers.tt.common import copy_host_to_device

    apply_workload_env(B, S)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000, num_command_queues=1)
    try:
        generator, model_args, kv_caches, page_table = build_single_device_model(device, batch_size=B, seq_len=S)
        model = generator.model[0]
        tok = model_args.tokenizer
        norm_w, eps = _extract_final_norm(model)
        pad_id = getattr(tok, "pad_token_id", None) or 0
        padded_batch = model_args.max_batch_size
        dim = model_args.dim

        # tokenize everything up front
        ids_all = torch.full((len(texts), S), pad_id, dtype=torch.long)
        lens = torch.zeros(len(texts), dtype=torch.long)
        for i, t in enumerate(texts):
            enc = tok(t, truncation=True, max_length=S, return_tensors="pt")["input_ids"][0]
            ids_all[i, : len(enc)] = enc
            lens[i] = len(enc)

        embs_masked = torch.zeros(len(texts), dim)
        embs_fast = torch.zeros(len(texts), dim)
        t0 = time.perf_counter()
        for start in range(0, len(texts), B):
            idx = list(range(start, min(start + B, len(texts))))
            ids = torch.full((padded_batch, S), pad_id, dtype=torch.long)
            ids[: len(idx)] = ids_all[idx]
            if B > 1:
                host = model.prepare_prefill_inputs_trace(ids, page_table=page_table, batch_size=B, user_id=0)
                fwd = dict(batch_size=B, user_id=0)
            else:
                host = model.prepare_prefill_inputs_trace(ids, page_table=page_table[0:1])
                fwd = {}
            dev = copy_host_to_device((host[0], host[3], host[4]), mesh_device=device)
            tr = model.transform_and_embed_prefill_inputs_device(*dev, tt_chunk_start_idx=None)
            out = model.ttnn_prefill_forward(
                x=tr[0],
                page_table=tr[1],
                chunk_page_table=tr[2],
                rot_mats_global=host[1],
                rot_mats_local=host[2],
                kv_cache=kv_caches[0],
                **fwd,
            )
            h = ttnn.to_torch(out).float().reshape(-1, S, dim)[: len(idx)]  # [b, S, H]
            ttnn.deallocate(out)
            h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + eps) * norm_w  # final RMSNorm on host
            for j, i in enumerate(idx):
                n = int(lens[i])
                embs_masked[i] = h[j, :n].mean(0)
                embs_fast[i] = h[j].mean(0)
            if start == 0 or (start // B) % 50 == 0:
                print(f"  {start + len(idx)}/{len(texts)} texts, {time.perf_counter() - t0:.0f}s", flush=True)
        wall = time.perf_counter() - t0

        def spear(e):
            a = torch.nn.functional.normalize(e[: len(s1)], dim=-1)
            b = torch.nn.functional.normalize(e[len(s1) :], dim=-1)
            return spearmanr((a * b).sum(-1).numpy(), np.array(gold))[0]

        print("=" * 60)
        print(f"  STS-B through the batch-{B} prefill path  ({len(texts)} texts, {wall:.0f}s eager)")
        print(f"  Spearman, masked mean over real tokens: {spear(embs_masked):.4f}")
        print(f"  Spearman, mean over the padded ISL:     {spear(embs_fast):.4f}")
        finite = bool(torch.isfinite(embs_masked).all())
        print(f"  all embeddings finite: {finite}")
        if args.save:
            torch.save({"masked": embs_masked, "fast": embs_fast, "lens": lens}, args.save)
            print(f"  saved {args.save}")
        if args.compare and os.path.exists(args.compare):
            ref = torch.load(args.compare)["masked"][: len(texts)]
            cos = (torch.nn.functional.normalize(ref, dim=-1) * torch.nn.functional.normalize(embs_masked, dim=-1)).sum(
                -1
            )
            print(
                f"  per-text cosine vs {args.compare}: mean {cos.mean():.5f}  min {cos.min():.5f}  p1 {cos.quantile(0.01):.5f}"
            )
        print("=" * 60)
    finally:
        sys.stdout.flush()
        os._exit(0)  # same teardown caveat as eval_accuracy_tt.py


if __name__ == "__main__":
    main()
