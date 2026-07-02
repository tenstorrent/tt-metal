# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A/B accuracy guard for the adaptive `prefill_pcm` MoE chunking optimization.

Runs the *same* TT model on the *same* 2x4 mesh with the *same* production flags,
computing prefill last-token logits twice:

  A) OLD behavior: GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM=1  -> sparse MoE runs in
     many small chunks (e.g. 4 chunks at ISL-128).  This is the already-shipping,
     validated code path.
  B) NEW behavior: unset -> adaptive prefill_pcm runs the prefill in one chunk.

The chunking change is intended to be numerically identical (MoE is token-wise),
so PCC(A, B) must be ~1.0.  This directly measures any accuracy loss introduced by
the optimization without needing an HF reference (which is Blackhole-gated + heavy).

Usage (from tt-metal root, python_env active):
    TT_METAL_GTEST_ETH_DISPATCH=1 GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
    GLM4_MOE_LITE_CCL_NUM_LINKS=1 GLM4_MOE_LITE_CCL_TOPOLOGY=linear \
    GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
    GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_BATCHED_PREFILL=1 \
    GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 \
    GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 \
    python models/experimental/glm4_moe_lite/scripts/ab_prefill_pcm_pcc.py \
      --simulate-context-len 128 --mesh-rows 2 --mesh-cols 4 --min-pcc 0.999

Exit code 0 = PASS (PCC >= --min-pcc), 1 = FAIL.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.glm4_moe_lite.scripts.debug_run_full_tt_greedy import (
    _dispatch_core_config,
    _parse_tt_dtype,
    _set_default_fabric_config,
)
from models.experimental.glm4_moe_lite.tt.layer0_tt import (
    _alloc_contiguous_page_table,
    _alloc_paged_kvpe_cache,
    _round_up,
)
from models.experimental.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.experimental.glm4_moe_lite.tt.weights import resolve_best_effort_snapshot_dir


def _prefill_last_token_logits(runner, prompt_ids, prompt_lens, page_table, kv_cache, block_size):
    logits = runner.prefill(
        tokens=prompt_ids,
        prompt_lens=prompt_lens,
        page_table=page_table,
        kv_cache=kv_cache,
        seq_pad_multiple=block_size,
    )
    # logits: [B, 1, vocab] (logits trace mode / eager). Take batch 0, last token.
    lg = logits.reshape(
        int(prompt_ids.shape[0]),
        -1,
    )[
        0
    ]  # [vocab] for last token
    return lg.to(dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--prompt", default="Summarize")
    ap.add_argument("--simulate-context-len", type=int, default=128)
    ap.add_argument("--min-cache-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--mesh-rows", type=int, default=2)
    ap.add_argument("--mesh-cols", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--kv-cache-dtype", default="bf16")
    ap.add_argument("--cache-dir", default="~/.cache/ttnn/models/glm4_moe_lite/vllm")
    ap.add_argument("--min-pcc", type=float, default=0.999)
    ap.add_argument("--time-iters", type=int, default=5, help="Timed prefill iters per config (after 1 warmup).")
    args = ap.parse_args()

    os.environ["GLM4_MOE_LITE_ENABLE_MOE"] = "1"

    snap = Path(resolve_best_effort_snapshot_dir(str(args.model_id)))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids_single = enc["input_ids"].to(dtype=torch.int32)
    prompt_len = int(prompt_ids_single.shape[1])
    sim_ctx = int(args.simulate_context_len)
    if sim_ctx > prompt_len:
        repeats = (sim_ctx + prompt_len - 1) // prompt_len
        prompt_ids_single = prompt_ids_single.repeat(1, repeats)[:, :sim_ctx]
        prompt_len = sim_ctx
    batch_size = int(args.batch_size)
    prompt_ids = prompt_ids_single.repeat(batch_size, 1)
    prompt_lens = [prompt_len] * batch_size

    mesh_rows, mesh_cols = int(args.mesh_rows), int(args.mesh_cols)
    n_devices = mesh_rows * mesh_cols
    block_size = int(args.block_size)
    total_len = max(prompt_len + 32, int(args.min_cache_tokens))
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

        def _fresh_cache():
            kv_cache = [
                _alloc_paged_kvpe_cache(
                    device=mesh_device,
                    max_num_blocks=int(batch_size * blocks_per_seq),
                    block_size=block_size,
                    kvpe_dim=kvpe_dim,
                    dtype=kv_cache_dtype,
                )
                for _ in range(int(runner.num_layers_to_run))
            ]
            page_table = _alloc_contiguous_page_table(batch=batch_size, blocks_per_seq=blocks_per_seq)
            return kv_cache, page_table

        # ---- A) OLD: force many-chunk sparse MoE (shipping, validated path) ----
        os.environ["GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM"] = "1"
        kv_a, pt_a = _fresh_cache()
        logits_old = _prefill_last_token_logits(runner, prompt_ids, prompt_lens, pt_a, kv_a, block_size)
        print(f"[A/B] OLD (PREFILL_PCM=1, chunked) logits shape={tuple(logits_old.shape)}", flush=True)

        # ---- B) NEW: adaptive prefill_pcm (single-chunk at this ISL) ----
        os.environ.pop("GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM", None)
        kv_b, pt_b = _fresh_cache()
        logits_new = _prefill_last_token_logits(runner, prompt_ids, prompt_lens, pt_b, kv_b, block_size)
        print(f"[A/B] NEW (adaptive, 1-chunk) logits shape={tuple(logits_new.shape)}", flush=True)

        # ---- Timing: median prefill wall-clock per config (device-inclusive; prefill
        # returns host logits so the call blocks until device work completes). The
        # correctness prefill above already warmed/compiled each config. ----
        def _time_config(pcm_env_val):
            if pcm_env_val is None:
                os.environ.pop("GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM", None)
            else:
                os.environ["GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM"] = pcm_env_val
            samples = []
            for _ in range(max(1, int(args.time_iters))):
                kv, pt = _fresh_cache()
                t0 = time.perf_counter()
                _ = _prefill_last_token_logits(runner, prompt_ids, prompt_lens, pt, kv, block_size)
                samples.append(time.perf_counter() - t0)
            return samples

        t_old = _time_config("1")
        t_new = _time_config(None)
        med_old, med_new = statistics.median(t_old), statistics.median(t_new)
        speedup = (med_old / med_new) if med_new > 0 else float("nan")
        print("-" * 60, flush=True)
        print(
            f"[A/B TIME] OLD (4-chunk) prefill median = {med_old*1e3:.2f} ms  "
            f"(min {min(t_old)*1e3:.2f}, n={len(t_old)})",
            flush=True,
        )
        print(
            f"[A/B TIME] NEW (1-chunk) prefill median = {med_new*1e3:.2f} ms  "
            f"(min {min(t_new)*1e3:.2f}, n={len(t_new)})",
            flush=True,
        )
        print(
            f"[A/B TIME] delta = {(med_old-med_new)*1e3:+.2f} ms  "
            f"speedup = {speedup:.3f}x  ({(speedup-1)*100:+.1f}%)",
            flush=True,
        )

        passing, pcc_val = comp_pcc(logits_old, logits_new, float(args.min_pcc))
        # comp_pcc returns (bool, message-with-number); also compute a raw float for clarity.
        with torch.no_grad():
            a = logits_old.flatten().double()
            b = logits_new.flatten().double()
            a = a - a.mean()
            b = b - b.mean()
            denom = (a.norm() * b.norm()).item()
            raw_pcc = (torch.dot(a, b).item() / denom) if denom > 0 else 1.0
        max_abs = (logits_old - logits_new).abs().max().item()
        argmax_old = int(torch.argmax(logits_old).item())
        argmax_new = int(torch.argmax(logits_new).item())

        print("=" * 60, flush=True)
        print(f"[A/B PCC] comp_pcc msg : {pcc_val}", flush=True)
        print(f"[A/B PCC] raw pcc      : {raw_pcc:.8f}", flush=True)
        print(f"[A/B PCC] max_abs_diff : {max_abs:.6e}", flush=True)
        print(
            f"[A/B PCC] argmax old/new: {argmax_old} / {argmax_new} "
            f"({'MATCH' if argmax_old == argmax_new else 'DIFFER'})",
            flush=True,
        )
        print(f"[A/B PCC] threshold    : {args.min_pcc}", flush=True)
        ok = bool(passing) and raw_pcc >= float(args.min_pcc)
        print(f"[A/B PCC] RESULT       : {'PASS' if ok else 'FAIL'}", flush=True)
        print("=" * 60, flush=True)
        return 0 if ok else 1
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    raise SystemExit(main())
