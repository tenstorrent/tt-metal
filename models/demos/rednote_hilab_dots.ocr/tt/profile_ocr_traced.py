# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Warm, FULLY-TRACED OCR inference perf for rednote-hilab/dots.ocr (p150/blackhole).

Extends ``profile_ocr.py``: in addition to the already-traced AR decode step, this
captures and replays metal traces for the VISION tower forward (42 layers) and the
LM PREFILL forward (28 layers). It reports, per stage, untraced-vs-traced wall time
and the speedup, plus the new fully-traced end-to-end generation time vs the
all-untraced-except-decode baseline.

Trace lifetime / active-trace allocation conflict
--------------------------------------------------
vision -> scatter -> prefill -> decode each allocate device buffers. An armed trace
makes any subsequent FRESH allocation unsafe ("Allocating device buffers is unsafe
due to the existence of an active trace"). The robust pattern used here: pre-allocate
EVERY persistent host-facing buffer (vision input, prefill input, decode embed, the
KV cache, the per-layer RoPE decode buffers) BEFORE capturing any trace, then capture
all three traces (vision, prefill, decode) back-to-back during warmup. The transient
intermediates allocated *inside* each captured region belong to that trace and are
reused on replay; no fresh persistent allocation happens once a trace is armed.

Per-image / per-generation replay then only streams new data into the persistent
buffers (copy_host_to_device_tensor into the vision-input / prefill-input / decode
embed buffers, and write_pos / write_decode_rope for decode) and calls execute_trace.

Usage::

    python tt/profile_ocr_traced.py --lm-layers 28 --vision-layers 42

The model dir name contains a dot, so siblings are imported by file path.
"""
import argparse
import importlib.util
import os
import statistics
import time

import torch

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.normpath(os.path.join(_HERE, "..", "demo"))

CHECKPOINT_PATH = os.environ.get(
    "DOTS_OCR_CHECKPOINT",
    "/local/ttuser/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/"
    "c0111ce6bc07803dbc267932ffef0ae3a51dc951",
)
EOS = {151643, 151673}


def _load_by_path(name, filename, directory):
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="dots.ocr warm OCR perf with vision+prefill+decode all traced")
    ap.add_argument("--image", default=os.path.join(_DEMO_DIR, "demo_image1.jpg"))
    ap.add_argument("--prompt", default="Read the text in the image.")
    ap.add_argument("--lm-layers", type=int, default=28)
    ap.add_argument("--vision-layers", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--trace-region-size", type=int, default=300_000_000)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5, help="timed replay iterations for vision/prefill p50")
    args = ap.parse_args()

    demo = _load_by_path("dots_proft_demo", "demo_ocr.py", _DEMO_DIR)
    loader = _load_by_path("dots_proft_loader", "weight_loader.py", _HERE)
    ocrm = _load_by_path("dots_proft_ocr_model", "ocr_model.py", _HERE)
    kvc = _load_by_path("dots_proft_kv_cache", "kv_cache.py", _HERE)

    input_ids, pixel_values, grid_thw, tok = demo.host_preprocess(args.image, args.prompt, CHECKPOINT_PATH)
    pixel_values = pixel_values.to(torch.float32)
    lm_sd = loader.load_language_model_weights(CHECKPOINT_PATH, num_layers=args.lm_layers)
    vis_sd = loader.load_vision_tower_weights(CHECKPOINT_PATH, num_layers=args.vision_layers)

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768, trace_region_size=args.trace_region_size)
    try:
        model = ocrm.TtOcrModel(
            device=device,
            lm_state_dict=lm_sd,
            vision_state_dict=vis_sd,
            grid_thw=grid_thw,
            lm_num_layers=args.lm_layers,
            vision_num_layers=args.vision_layers,
        )

        ids = input_ids.reshape(-1).to(torch.int64).tolist()
        prompt_len = len(ids)
        max_seq_len = prompt_len + args.max_new_tokens + 1
        lm = model._lm_for_seq(prompt_len, max_seq_len=max_seq_len)
        cache = kvc.SelfAttentionKVCache(
            device=device,
            num_layers=args.lm_layers,
            batch=1,
            num_kv_heads=model.num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=model.head_dim,
            dtype=model.dtype,
        )

        # ============================================================== #
        # Pre-allocate EVERY persistent host-facing buffer BEFORE any     #
        # trace is captured (the active-trace allocation-safety rule).    #
        # ============================================================== #
        # Vision input: [num_patches, embed_dim] post host patch_embed.
        patch_tokens = model.vision_tower.patch_embed(pixel_values)
        num_patches = patch_tokens.shape[0]
        vision_in = ttnn.from_torch(
            patch_tokens.to(torch.float32),
            device=device,
            dtype=model.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Prefill input: [prompt_len, hidden] (post vision->text scatter).
        prefill_in = ttnn.from_torch(
            torch.zeros(prompt_len, model.hidden_size, dtype=torch.float32),
            device=device,
            dtype=model.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Decode input: single token embed [1, hidden].
        decode_embed = ttnn.from_torch(
            torch.zeros(1, model.hidden_size, dtype=torch.float32),
            device=device,
            dtype=model.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def write_vision_in(patch_tok):
            host = ttnn.from_torch(patch_tok.to(torch.float32), dtype=model.dtype, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, vision_in)

        def write_prefill_in(embeds):
            host = ttnn.from_torch(embeds.to(torch.float32), dtype=model.dtype, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, prefill_in)

        def write_embed(token_id):
            vec = model._embed_table[int(token_id)].reshape(1, model.hidden_size).to(torch.float32)
            host = ttnn.from_torch(vec, dtype=model.dtype, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, decode_embed)

        # ============================================================== #
        # WARMUP (compile) — run each path once untraced so kernels build #
        # and the persistent buffers settle, BEFORE capture.              #
        # ============================================================== #
        # vision compile
        _ = model.vision_tower(vision_in)
        ttnn.synchronize_device(device)

        # build a real prefill input + decode first-token for warmup.
        vis_out_tt = model.vision_tower(vision_in)
        vision_embeds = ttnn.to_torch(vis_out_tt).to(torch.float32).reshape(-1, model.hidden_size)
        inputs_embeds = model.build_inputs_embeds(torch.tensor(ids, dtype=torch.int64), vision_embeds)
        write_prefill_in(inputs_embeds)

        # prefill compile (populate cache)
        cache.reset()
        pf = lm.prefill_from_embeds(prefill_in, cache)
        ttnn.synchronize_device(device)
        first_id = int(torch.argmax(ttnn.to_torch(pf).to(torch.float32).reshape(-1)).item())

        # decode compile
        warm_pos = prompt_len
        write_embed(first_id)
        lm.write_decode_pos(warm_pos, cache)
        _ = lm.decode_step_traced(decode_embed, cache)
        ttnn.synchronize_device(device)

        # Also warm the UNTRACED vision/prefill/decode for the comparison numbers.
        _ = model.vision_tower(vision_in)
        ttnn.synchronize_device(device)
        cache.reset()
        _ = lm.prefill_from_embeds(prefill_in, cache)
        ttnn.synchronize_device(device)
        _ = lm.decode_step(decode_embed, warm_pos, cache)
        ttnn.synchronize_device(device)

        # ============================================================== #
        # CAPTURE the three traces back-to-back (no fresh persistent      #
        # allocation between/after — all buffers exist already).          #
        # ============================================================== #
        # vision trace
        vis_tid = ttnn.begin_trace_capture(device, cq_id=0)
        vision_out = model.vision_tower(vision_in)
        ttnn.end_trace_capture(device, vis_tid, cq_id=0)
        ttnn.synchronize_device(device)

        # prefill trace (writes the KV cache via fill_cache into persistent buffers)
        cache.reset()
        prefill_tid = ttnn.begin_trace_capture(device, cq_id=0)
        prefill_logits = lm.prefill_from_embeds(prefill_in, cache)
        ttnn.end_trace_capture(device, prefill_tid, cq_id=0)
        ttnn.synchronize_device(device)

        # decode trace
        write_embed(first_id)
        lm.write_decode_pos(warm_pos, cache)
        decode_tid = ttnn.begin_trace_capture(device, cq_id=0)
        decode_logits = lm.decode_step_traced(decode_embed, cache)
        ttnn.end_trace_capture(device, decode_tid, cq_id=0)
        ttnn.synchronize_device(device)

        # ============================================================== #
        # MEASURE untraced vs traced for vision and prefill (p50).        #
        # ============================================================== #
        def time_n(fn, n):
            xs = []
            for _ in range(n):
                t0 = time.perf_counter()
                fn()
                ttnn.synchronize_device(device)
                xs.append((time.perf_counter() - t0) * 1000.0)
            return statistics.median(xs)

        # VISION untraced
        vis_untraced = time_n(lambda: model.vision_tower(vision_in), args.iters)
        # VISION traced (replay)
        vis_traced = time_n(lambda: ttnn.execute_trace(device, vis_tid, cq_id=0, blocking=False), args.iters)

        # PREFILL untraced (reset cache each time so fill_cache writes are clean)
        def prefill_untraced_once():
            cache.reset()
            lm.prefill_from_embeds(prefill_in, cache)

        pf_untraced = time_n(prefill_untraced_once, args.iters)

        def prefill_traced_once():
            cache.reset()
            ttnn.execute_trace(device, prefill_tid, cq_id=0, blocking=False)

        pf_traced = time_n(prefill_traced_once, args.iters)

        # ============================================================== #
        # FULLY-TRACED end-to-end generation (vision + prefill + decode). #
        # ============================================================== #
        cache.reset()
        # vision replay on persistent vision_in (already loaded with this image).
        t0 = time.perf_counter()
        ttnn.execute_trace(device, vis_tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        e2e_vision_ms = (time.perf_counter() - t0) * 1000.0
        vision_embeds = ttnn.to_torch(vision_out).to(torch.float32).reshape(-1, model.hidden_size)

        inputs_embeds = model.build_inputs_embeds(torch.tensor(ids, dtype=torch.int64), vision_embeds)
        write_prefill_in(inputs_embeds)

        t0 = time.perf_counter()
        ttnn.execute_trace(device, prefill_tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        e2e_prefill_ms = (time.perf_counter() - t0) * 1000.0
        next_id = int(torch.argmax(ttnn.to_torch(prefill_logits).to(torch.float32).reshape(-1)).item())

        gen_tokens = [next_id]
        decode_ms = []
        cur_id = next_id
        for step in range(1, args.max_new_tokens):
            pos = prompt_len + step - 1
            write_embed(cur_id)
            lm.write_decode_pos(pos, cache)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, decode_tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            decode_ms.append((time.perf_counter() - t0) * 1000.0)
            cur_id = int(torch.argmax(ttnn.to_torch(decode_logits).to(torch.float32).reshape(-1)).item())
            gen_tokens.append(cur_id)
            if cur_id in EOS:
                break

        decode_p50 = statistics.median(decode_ms) if decode_ms else float("nan")
        text = tok.decode(gen_tokens, skip_special_tokens=True)

        # all-untraced-except-decode baseline e2e (vision+prefill untraced, decode traced).
        baseline_e2e = vis_untraced + pf_untraced + decode_p50 * max(len(gen_tokens) - 1, 0)
        traced_e2e = e2e_vision_ms + e2e_prefill_ms + decode_p50 * max(len(gen_tokens) - 1, 0)

        print("=" * 78)
        print(
            f"depth: lm_layers={args.lm_layers} vision_layers={args.vision_layers} "
            f"prompt_len={prompt_len} num_patches={num_patches}"
        )
        print(f"generated_tokens={len(gen_tokens)}  text={text!r}")
        print("-" * 78)
        print(f"{'stage':<14}{'untraced ms':>14}{'traced ms':>14}{'speedup':>12}")
        print(f"{'vision (42L)':<14}{vis_untraced:>14.2f}{vis_traced:>14.2f}{vis_untraced/vis_traced:>11.2f}x")
        print(f"{'prefill (28L)':<14}{pf_untraced:>14.2f}{pf_traced:>14.2f}{pf_untraced/pf_traced:>11.2f}x")
        print(f"{'decode/tok':<14}{'(see profile_ocr)':>14}{decode_p50:>14.2f}{'~2.4':>11}x")
        print("-" * 78)
        print(f"e2e (vision untraced + prefill untraced + decode traced): {baseline_e2e:8.2f} ms")
        print(f"e2e (vision traced   + prefill traced   + decode traced): {traced_e2e:8.2f} ms")
        print(f"e2e speedup: {baseline_e2e/traced_e2e:.2f}x")
        print("=" * 78)
        print(
            f"SUMMARY_TRACED lm={args.lm_layers} vis={args.vision_layers} prompt_len={prompt_len} "
            f"num_patches={num_patches} "
            f"vision_untraced_ms={vis_untraced:.2f} vision_traced_ms={vis_traced:.2f} "
            f"vision_speedup={vis_untraced/vis_traced:.2f} "
            f"prefill_untraced_ms={pf_untraced:.2f} prefill_traced_ms={pf_traced:.2f} "
            f"prefill_speedup={pf_untraced/pf_traced:.2f} "
            f"decode_traced_ms={decode_p50:.2f} "
            f"e2e_baseline_ms={baseline_e2e:.2f} e2e_traced_ms={traced_e2e:.2f} "
            f"e2e_speedup={baseline_e2e/traced_e2e:.2f} ntok={len(gen_tokens)} text={text!r}"
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
