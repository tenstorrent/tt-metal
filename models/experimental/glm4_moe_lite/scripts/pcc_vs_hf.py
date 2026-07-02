# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth PCC gate: TT prefill last-token logits vs the HF reference.

The HF reference (full-precision torch forward of GLM-4.7-Flash on CPU) is expensive
to compute (~59 GB model, 47 MoE layers). This script computes it ONCE for a given
(prompt-source, seq_len) and caches the last-token logits to disk, so every later run
just loads the cached tensor and compares — turning a multi-minute HF forward into an
instant load.

Flow:
  1. Tokenize `seq_len` real tokens (Tale of Two Cities corpus, same as the official
     pipeline test) or a custom --prompt.
  2. HF reference: load cache if it matches (input_ids, vocab); else run the HF model
     on CPU, take last-token logits [1,1,vocab], save cache, free the model.
  3. TT: open the 2x4 mesh, run prefill, take last-token logits.
  4. comp_pcc(HF, TT). Exit 0 if PCC >= --min-pcc, else 1.

The TT model runs under whatever GLM4_MOE_LITE_* flags are set in the environment, so
run it with the SAME flags as the config you are optimizing to measure that config's
real accuracy vs ground truth.

Usage (from tt-metal root, python_env active) — see agent_logs/pcc_vs_hf.sh wrapper.
"""
from __future__ import annotations

import argparse
import bz2
import gc
import os
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.glm4_moe_lite.scripts.debug_run_full_tt_greedy import (
    _dispatch_core_config,
    _parse_tt_dtype,
    _set_default_fabric_config,
)
from models.experimental.glm4_moe_lite.tests.pipeline_tests.test_utils import PROMPT_FILE
from models.experimental.glm4_moe_lite.tt.layer0_tt import (
    _alloc_contiguous_page_table,
    _alloc_paged_kvpe_cache,
    _round_up,
)
from models.experimental.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.experimental.glm4_moe_lite.tt.weights import resolve_best_effort_snapshot_dir

DEFAULT_HF_CACHE_DIR = "models/experimental/glm4_moe_lite/experiments/hf_ref"


def _tokenize(snap, prompt, seq_len):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
    if prompt:
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        if len(ids) < seq_len:  # repeat to reach seq_len
            reps = (seq_len + len(ids) - 1) // len(ids)
            ids = (ids * reps)[:seq_len]
        else:
            ids = ids[:seq_len]
    else:
        with bz2.open(PROMPT_FILE, "rt", encoding="utf-8") as f:
            corpus = f.read()
        ids = tok(corpus, add_special_tokens=True)["input_ids"][:seq_len]
    if len(ids) < seq_len:
        raise SystemExit(f"Could not get {seq_len} tokens (got {len(ids)}).")
    return torch.tensor([ids], dtype=torch.int64)


def _compute_or_load_hf(snap, input_ids_i64, vocab, cache_path: Path):
    if cache_path.exists():
        blob = torch.load(cache_path)
        if (
            int(blob.get("vocab", -1)) == int(vocab)
            and blob["input_ids"].shape == input_ids_i64.shape
            and torch.equal(blob["input_ids"], input_ids_i64)
        ):
            print(f"[HF] loaded cached reference logits from {cache_path}", flush=True)
            return blob["hf_logits"].to(torch.float32)
        print(f"[HF] cache {cache_path} stale (input mismatch) -> recomputing", flush=True)

    print("[HF] loading HF reference model on CPU (one-time; ~59 GB, may take minutes)...", flush=True)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        snap,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    with torch.no_grad():
        out = model(input_ids=input_ids_i64, use_cache=False, return_dict=True)
    hf_logits = out.logits[:, -1:, : int(vocab)].to(torch.float32).clone()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": input_ids_i64,
            "vocab": int(vocab),
            "seq_len": int(input_ids_i64.shape[1]),
            "hf_logits": hf_logits,
        },
        cache_path,
    )
    print(f"[HF] saved reference logits to {cache_path}", flush=True)
    del model, out
    gc.collect()
    return hf_logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--prompt", default="", help="Custom prompt; empty = Tale of Two Cities corpus (official ref).")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--mesh-rows", type=int, default=2)
    ap.add_argument("--mesh-cols", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--kv-cache-dtype", default="bf16")
    ap.add_argument("--cache-dir", default="~/.cache/ttnn/models/glm4_moe_lite/vllm")
    ap.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    ap.add_argument("--min-pcc", type=float, default=0.97)
    ap.add_argument(
        "--shard-probe", type=int, default=4, help="Report PCC split into this many vocab shards (TP size)."
    )
    args = ap.parse_args()

    os.environ["GLM4_MOE_LITE_ENABLE_MOE"] = "1"
    snap = Path(resolve_best_effort_snapshot_dir(str(args.model_id)))
    seq_len = int(args.seq_len)

    input_ids_i64 = _tokenize(snap, args.prompt, seq_len)  # [1, seq_len] int64
    tag = "corpus" if not args.prompt else "prompt"
    hf_cache = Path(args.hf_cache_dir) / f"hf_prefill_lasttok_{tag}_isl{seq_len}.pt"

    # ---- HF ground-truth (cached) — do this before opening the mesh ----
    # vocab from config to slice logits consistently.
    import json

    with open(snap / "config.json") as f:
        vocab = int(json.load(f)["vocab_size"])
    hf_logits = _compute_or_load_hf(snap, input_ids_i64, vocab, hf_cache)  # [1,1,vocab] f32

    # ---- TT prefill ----
    prompt_ids = input_ids_i64.to(torch.int32).repeat(int(args.batch_size), 1)
    prompt_lens = [seq_len] * int(args.batch_size)
    mesh_rows, mesh_cols = int(args.mesh_rows), int(args.mesh_cols)
    n_devices = mesh_rows * mesh_cols
    block_size = int(args.block_size)
    total_len = max(seq_len + 32, 256)
    blocks_per_seq = max(1, _round_up(total_len, block_size) // block_size)
    kv_cache_dtype = _parse_tt_dtype(str(args.kv_cache_dtype))

    _set_default_fabric_config(n_devices)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(mesh_rows, mesh_cols),
        dispatch_core_config=_dispatch_core_config(),
    )
    try:
        runner = Glm4MoeLiteDenseOnlyTT.create(
            device=mesh_device,
            snapshot_dir=snap,
            cache_dir=Path(os.path.expanduser(str(args.cache_dir))),
            max_seq_len=int(blocks_per_seq * block_size),
        )
        for li in range(runner.num_layers_to_run):
            runner._ensure_layer_weights(li)
        kvpe_dim = int(runner.hparams.kv_lora_rank + runner.hparams.qk_rope_head_dim)
        kv_cache = [
            _alloc_paged_kvpe_cache(
                device=mesh_device,
                max_num_blocks=int(args.batch_size * blocks_per_seq),
                block_size=block_size,
                kvpe_dim=kvpe_dim,
                dtype=kv_cache_dtype,
            )
            for _ in range(int(runner.num_layers_to_run))
        ]
        page_table = _alloc_contiguous_page_table(batch=int(args.batch_size), blocks_per_seq=blocks_per_seq)

        tt_logits = runner.prefill(
            tokens=prompt_ids,
            prompt_lens=prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            seq_pad_multiple=block_size,
        )
        tt_cmp = tt_logits.reshape(int(args.batch_size), -1)[0:1, :vocab].to(torch.float32).unsqueeze(1)  # [1,1,vocab]

        passing, pcc_msg = comp_pcc(hf_logits, tt_cmp, float(args.min_pcc))
        argmax_hf = int(torch.argmax(hf_logits.flatten()).item())
        argmax_tt = int(torch.argmax(tt_cmp.flatten()).item())
        print("=" * 64, flush=True)
        print(f"[PCC vs HF] seq_len={seq_len} vocab={vocab} kv={args.kv_cache_dtype}", flush=True)
        print(f"[PCC vs HF] comp_pcc     : {pcc_msg}", flush=True)
        print(
            f"[PCC vs HF] argmax HF/TT : {argmax_hf} / {argmax_tt} "
            f"({'MATCH' if argmax_hf == argmax_tt else 'DIFFER'})",
            flush=True,
        )
        print(f"[PCC vs HF] threshold    : {args.min_pcc}", flush=True)
        print(f"[PCC vs HF] RESULT       : {'PASS' if passing else 'FAIL'}", flush=True)

        # --- Per-vocab-shard PCC breakdown (diagnose lm_head TP vocab-sharding assembly). ---
        # lm_head is split across the TP columns; a mis-assembled shard shows as one bad chunk.
        def _pcc(a, b):
            a = a.flatten().double() - a.flatten().double().mean()
            b = b.flatten().double() - b.flatten().double().mean()
            d = (a.norm() * b.norm()).item()
            return (torch.dot(a, b).item() / d) if d > 0 else 1.0

        hf_v = hf_logits.flatten()
        tt_v = tt_cmp.flatten()
        for nshards in (int(args.shard_probe), 8):
            if nshards <= 1 or vocab % nshards != 0:
                continue
            w = vocab // nshards
            parts = [f"{i}:{_pcc(hf_v[i*w:(i+1)*w], tt_v[i*w:(i+1)*w]):.4f}" for i in range(nshards)]
            print(f"[PCC vs HF] per-{nshards}-shard PCC (width {w}): {' '.join(parts)}", flush=True)
        # also report max-abs-diff location to see if error is localized
        diff = (hf_v - tt_v).abs()
        print(
            f"[PCC vs HF] global max|diff|={diff.max().item():.3f} at idx {int(torch.argmax(diff))}; "
            f"mean|diff|={diff.mean().item():.4f}",
            flush=True,
        )
        print("=" * 64, flush=True)
        return 0 if passing else 1
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    raise SystemExit(main())
