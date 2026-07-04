# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-step component breakdown of the DiffusionGemma traced denoise step.

The traced serving loop's per-step slope (~0.25 s/step @30L, session 9) is the sum of:
  embed + self_cond.forward(gated MLP) + [30 x (attn+norm+MoE)] + final_norm
        + LM head + soft_embedding(prev logits over 262k vocab) + terminal(denoise_step).

This probe times each component in isolation on the REAL 26B model (mesh (1,4), TP=4),
so we know which one is the binding per-step cost after the MoE was already OPT-004-tuned.
Fixed (L-independent) components — LM head, soft-embedding, terminal, embed, final norm —
are reported at their true value; the layer-loop (attn+MoE) is reported per-layer from the
reduced L used here. Async-pipelined timing (warm + iters + one sync) = device compute,
which is what a trace replay pays.

    DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 DG_CKPT=... \
      python -u models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py --num-layers 2 --iters 15

Markers: RESULT_COMPONENT name=.. ms=..   RESULT_BREAKDOWN <json>
*** DEVICE-OWNERSHIP: run only when QB2 is free. ***
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import denoise_loop as DL
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    embed_canvas_tokens,
    make_generation_logits_fn_builder_from_checkpoint_state,
)
from models.experimental.diffusion_gemma.tt.generate import (
    host_canvas_to_device,
    prefill_prompt_tokens,
    tokenize_prompt,
)
from models.experimental.diffusion_gemma.tt.sparse_moe import sparse_experts_forward

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
TEMP = 0.6
BUDGET = 0.1


def _time(fn, iters, mesh, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e3 / iters


def run(num_layers, canvas_length, iters, prompt, max_seq_len):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = mi.tt_model
        H = tt_model.hf_config.hidden_size
        n_moe = sum(1 for l in tt_model.layers if getattr(l, "enable_moe_block", False))
        logger.info(f"built L={num_layers} (moe layers={n_moe}) H={H}")

        prompt_tokens = tokenize_prompt(mi.tokenizer, prompt)
        prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
        ttnn.synchronize_device(mesh)

        adapter_kwargs = {}
        cfg = getattr(tt_model, "hf_config", None)
        if cfg is not None:
            adapter_kwargs["config"] = cfg
        logits_builder = make_generation_logits_fn_builder_from_checkpoint_state(mi.state_dict, **adapter_kwargs)
        adapter = logits_builder(tt_model, prompt_tokens=prompt_tokens, prompt_len=prefill.cache_len)
        # trace-safe self-cond so soft_embedding is active exactly like the serving path.
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_length)
        adapter.reset_signal_buffer()

        vocab = int(getattr(mi.tokenizer, "vocab_size", 262144))
        gen = torch.Generator(device="cpu").manual_seed(0)
        host_canvas = torch.randint(0, vocab, (1, canvas_length), dtype=torch.long, generator=gen)
        canvas = host_canvas_to_device(mesh, host_canvas)
        noise_tokens = host_canvas_to_device(mesh, torch.randint(0, vocab, (1, canvas_length), dtype=torch.long))
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=BUDGET)

        res = {"num_layers": num_layers, "n_moe": n_moe}

        # ---- FULL trace-safe step (embed + selfcond.fwd + fwd + LMhead + soft_emb) + terminal ----
        def full_step():
            logits = adapter._trace_safe_call(canvas, 1)
            r = DL.denoise_step(
                logits,
                temperature=TEMP,
                entropy_budget=BUDGET,
                gumbel_noise=None,
                noise_tokens=noise_tokens,
                constants=consts,
            )
            for t in (r.canvas, r.accept_mask, r.entropy, r.sampled, r.argmax):
                t.deallocate(True)
            logits.deallocate(True)

        res["full_step_ms"] = _time(full_step, iters, mesh)

        # ---- hidden forward (L layers + final norm), no LM head ----
        emb = embed_canvas_tokens(tt_model, canvas)
        cond0 = adapter.self_conditioning.forward(emb, adapter.signal_buf)
        emb.deallocate(True)

        def hidden_fwd():
            h = DF.denoise_hidden_forward(
                tt_model,
                prompt_hidden_by_layer=adapter.prompt_hidden_by_layer,
                canvas_hidden=cond0,
                q_rope_offset=adapter.q_rope_offset,
                prompt_len=adapter.prompt_len,
            )
            h.deallocate(True)

        res["hidden_fwd_ms"] = _time(hidden_fwd, iters, mesh)

        # a persistent hidden for LM head + a persistent logits for soft-emb/terminal
        hidden = DF.denoise_hidden_forward(
            tt_model,
            prompt_hidden_by_layer=adapter.prompt_hidden_by_layer,
            canvas_hidden=cond0,
            q_rope_offset=adapter.q_rope_offset,
            prompt_len=adapter.prompt_len,
        )

        def lm_head():
            lg = tt_model._apply_lm_head(hidden, is_decode=False)
            lg.deallocate(True)

        res["lm_head_ms"] = _time(lm_head, iters, mesh)
        logits = tt_model._apply_lm_head(hidden, is_decode=False)

        # ---- soft_embedding over the full 262k vocab ----
        def soft_emb():
            sig = adapter.self_conditioning.soft_embedding(
                logits,
                adapter.self_conditioning_embedding_weight,
                compute_kernel_config=adapter.self_conditioning_compute_kernel_config,
            )
            sig.deallocate(True)

        res["soft_emb_ms"] = _time(soft_emb, iters, mesh)

        # ---- terminal (denoise_step: sampling/entropy/accept) ----
        def terminal():
            r = DL.denoise_step(
                logits,
                temperature=TEMP,
                entropy_budget=BUDGET,
                gumbel_noise=None,
                noise_tokens=noise_tokens,
                constants=consts,
            )
            for t in (r.canvas, r.accept_mask, r.entropy, r.sampled, r.argmax):
                t.deallocate(True)

        res["terminal_ms"] = _time(terminal, iters, mesh)

        # ---- self_cond.forward (gated MLP over signal) ----
        emb2 = embed_canvas_tokens(tt_model, canvas)

        def selfcond_fwd():
            c = adapter.self_conditioning.forward(emb2, adapter.signal_buf)
            c.deallocate(True)

        res["selfcond_fwd_ms"] = _time(selfcond_fwd, iters, mesh)
        emb2.deallocate(True)

        # ---- one MoE layer (sparse_experts_forward, tuned per env) ----
        moe = None
        for l in tt_model.layers:
            if getattr(l, "enable_moe_block", False):
                moe = l.moe
                break
        if moe is not None:
            ri = ttnn.from_torch(
                torch.randn(1, 1, canvas_length, H) * 0.1,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            dense_routing = DF._denoise_router_forward(moe.router, ri)
            xin = ttnn.from_torch(
                torch.randn(1, 1, canvas_length, H) * 0.1,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

            def moe_layer():
                out = sparse_experts_forward(moe.experts, xin, dense_routing, capacity=32)
                out.deallocate(True)

            res["moe_layer_ms"] = _time(moe_layer, iters, mesh)
            ri.deallocate(True)
            dense_routing.deallocate(True)
            xin.deallocate(True)

        # per-layer estimate: (hidden_fwd - final_norm) / L ; approximate final_norm as small
        per_layer = res["hidden_fwd_ms"] / num_layers
        res["per_layer_est_ms"] = per_layer
        # project a 30L step: 30*per_layer + LMhead + soft_emb + terminal + selfcond_fwd + embed(small)
        res["proj_30L_step_ms"] = (
            30 * per_layer + res["lm_head_ms"] + res["soft_emb_ms"] + res["terminal_ms"] + res["selfcond_fwd_ms"]
        )

        for k, v in res.items():
            if k.endswith("_ms"):
                print(f"RESULT_COMPONENT name={k[:-3]} ms={v:.3f}", flush=True)
        print("RESULT_BREAKDOWN " + json.dumps(res), flush=True)

        logits.deallocate(True)
        hidden.deallocate(True)
        cond0.deallocate(True)
        canvas.deallocate(True)
        noise_tokens.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.iters, args.prompt, args.max_seq_len)


if __name__ == "__main__":
    main()
