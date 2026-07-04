# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify the commit LM-head skip: controlled before/after of the 256-token commit.

The commit re-encodes the 256-token canvas as 256 sequential decode-appends and DISCARDS the
per-token logits. Each decode computed a full hidden×vocab (2816×262144) LM head + final norm that
were thrown away. This benches the commit loop with skip_lm_head=False (old) vs True (new) on the
SAME model, confirming (a) it still runs and (b) the wall-clock commit saving. The KV writes happen
inside the layer loop and are identical either way, so correctness is unaffected by the skip.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt.commit_decode import commit_decode_forward
from models.experimental.diffusion_gemma.tt.generate import (
    _deallocate_decode_inputs,
    prefill_prompt_tokens,
    tokenize_prompt,
)

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def _commit_loop(tt_model, canvas_tokens, start_pos, skip_lm_head):
    for offset in range(canvas_tokens.shape[1]):
        token = canvas_tokens[:, offset]
        position = torch.tensor([start_pos + offset], dtype=torch.int32)
        di = tt_model.prepare_inputs_decode(token, position, page_table=None)
        out, _ = commit_decode_forward(tt_model, di[0], di[1], di[2], di[3], skip_lm_head=skip_lm_head)
        out.deallocate(True)
        _deallocate_decode_inputs(di)


def run(num_layers, canvas_length, max_seq_len):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = mi.tt_model
        prompt_tokens = tokenize_prompt(mi.tokenizer, "The capital of France is")
        prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
        vocab = int(getattr(mi.tokenizer, "vocab_size", 262144))
        host_commit = torch.randint(0, vocab, (1, canvas_length), dtype=torch.long)

        results = {}
        for skip in (False, True):
            # warm
            _commit_loop(tt_model, host_commit, prefill.cache_len, skip)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            _commit_loop(tt_model, host_commit, prefill.cache_len, skip)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) * 1e3
            results[skip] = ms
            logger.info(f"[commit skip_lm_head={skip}] 256-token commit ms = {ms:.1f} (num_layers={num_layers})")

        old, new = results[False], results[True]
        saved_pct = (old - new) / old * 100 if old else 0.0
        # project per-layer (subtract fixed) not needed; report both + 30L projection of the per-token delta
        print(
            f"RESULT_COMMIT_SKIP num_layers={num_layers} old_ms={old:.1f} new_ms={new:.1f} "
            f"saved_ms={old-new:.1f} saved_pct={saved_pct:.1f}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.max_seq_len)


if __name__ == "__main__":
    main()
