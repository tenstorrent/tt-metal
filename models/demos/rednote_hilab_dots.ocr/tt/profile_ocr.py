# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Warm, post-trace OCR inference perf for rednote-hilab/dots.ocr (p150/blackhole).

Measures the real production OCR pipeline at FULL depth (28 LM / 42 vision layers)
on ``demo/demo_image1.jpg``:

  1. host patch_embed         (Conv2d + RMSNorm on host)
  2. vision encoder (42L)     (TtVisionTower forward, once per image)
  3. LM prefill (28L)         (full-causal forward over the prompt, fills KV cache)
  4. decode/token (traced)    (steady-state O(1) cached step, METAL TRACE REPLAY)
  5. total                    (end-to-end "HELLO 2026" generation)

Method: one WARMUP full generation compiles kernels and captures the decode-step
metal trace; a second TIMED generation measures each stage with
``ttnn.synchronize_device`` boundaries. The traced decode replays a single
captured trace at every position -- the position and per-step RoPE flow through
persistent device buffers (KV-cache ``pos_tt`` + per-layer cos/sin buffers), so
no Python int is baked into the captured kernel args (see skills/perf/SKILL.md
"The trace pitfall").

Also reports untraced decode-step time (same maths, host dispatch per op) for the
traced-vs-untraced speedup.

Usage::

    python tt/profile_ocr.py --lm-layers 28 --vision-layers 42

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
    ap = argparse.ArgumentParser(description="dots.ocr warm OCR inference perf (traced decode)")
    ap.add_argument("--image", default=os.path.join(_DEMO_DIR, "demo_image1.jpg"))
    ap.add_argument("--prompt", default="Read the text in the image.")
    ap.add_argument("--lm-layers", type=int, default=28)
    ap.add_argument("--vision-layers", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--trace-region-size", type=int, default=256_000_000)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    demo = _load_by_path("dots_prof_demo", "demo_ocr.py", _DEMO_DIR)
    loader = _load_by_path("dots_prof_loader", "weight_loader.py", _HERE)
    ocrm = _load_by_path("dots_prof_ocr_model", "ocr_model.py", _HERE)
    kvc = _load_by_path("dots_prof_kv_cache", "kv_cache.py", _HERE)

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

        # Persistent decode-input embed buffer (stable address for trace replay).
        persistent_embed = ttnn.from_torch(
            torch.zeros(1, model.hidden_size, dtype=torch.float32),
            device=device,
            dtype=model.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def write_embed(token_id):
            vec = model._embed_table[int(token_id)].reshape(1, model.hidden_size).to(torch.float32)
            host = ttnn.from_torch(vec, dtype=model.dtype, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, persistent_embed)

        # ---- helper: one full generation, with optional trace replay ----
        def encode_and_prefill(timed=False):
            t = {}
            # vision (host patch_embed split out)
            t0 = time.perf_counter()
            patch_tokens = model.vision_tower.patch_embed(pixel_values)
            t["host_patch_embed_ms"] = (time.perf_counter() - t0) * 1000.0
            tt_in = ttnn.from_torch(
                patch_tokens.to(torch.float32),
                device=device,
                dtype=model.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            t0 = time.perf_counter()
            tt_out = model.vision_tower(tt_in)
            ttnn.synchronize_device(device)
            t["vision_ms"] = (time.perf_counter() - t0) * 1000.0
            vision_embeds = ttnn.to_torch(tt_out).to(torch.float32).reshape(-1, model.hidden_size)

            inputs_embeds = model.build_inputs_embeds(torch.tensor(ids, dtype=torch.int64), vision_embeds)
            hin = ttnn.from_torch(
                inputs_embeds.to(torch.float32),
                device=device,
                dtype=model.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            t0 = time.perf_counter()
            pf = lm.prefill_from_embeds(hin, cache)
            ttnn.synchronize_device(device)
            t["prefill_ms"] = (time.perf_counter() - t0) * 1000.0
            nid = int(torch.argmax(ttnn.to_torch(pf).to(torch.float32).reshape(prompt_len, -1)[-1]).item())
            return t, nid

        # ================= WARMUP generation (compiles + captures trace) =======
        cache.reset()
        _, first_id = encode_and_prefill()

        # warmup one traced-shape decode step (compile), then capture the trace.
        warm_pos = prompt_len  # position of the token being fed (first generated)
        write_embed(first_id)
        lm.write_decode_pos(warm_pos, cache)
        _ = lm.decode_step_traced(persistent_embed, cache)
        ttnn.synchronize_device(device)

        # Capture the decode-step trace ONCE.
        write_embed(first_id)
        lm.write_decode_pos(warm_pos, cache)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        traced_logits = lm.decode_step_traced(persistent_embed, cache)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        # Also warm up the untraced decode_step path (for comparison).
        _ = lm.decode_step(persistent_embed, warm_pos, cache)
        ttnn.synchronize_device(device)

        # ================= TIMED generation (warm) =============================
        cache.reset()
        stage_t, next_id = encode_and_prefill(timed=True)
        gen_tokens = [next_id]

        total_t0 = time.perf_counter()
        # (re-run vision+prefill timing already captured in stage_t for this gen)
        decode_ms = []
        cur_id = next_id
        for step in range(1, args.max_new_tokens):
            pos = prompt_len + step - 1
            write_embed(cur_id)
            lm.write_decode_pos(pos, cache)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            decode_ms.append((time.perf_counter() - t0) * 1000.0)
            cur_id = int(torch.argmax(ttnn.to_torch(traced_logits).to(torch.float32).reshape(-1)).item())
            gen_tokens.append(cur_id)
            if cur_id in EOS:
                break
        total_ms = stage_t["vision_ms"] + stage_t["prefill_ms"] + sum(decode_ms) + stage_t["host_patch_embed_ms"]

        # ---- untraced decode timing (fresh cache, same path, host-dispatched) ----
        cache.reset()
        _, u_first = encode_and_prefill()
        untr_ms = []
        u_id = u_first
        for step in range(1, min(args.max_new_tokens, 9)):
            pos = prompt_len + step - 1
            write_embed(u_id)
            t0 = time.perf_counter()
            sl = lm.decode_step(persistent_embed, pos, cache)
            ttnn.synchronize_device(device)
            untr_ms.append((time.perf_counter() - t0) * 1000.0)
            u_id = int(torch.argmax(ttnn.to_torch(sl).to(torch.float32).reshape(-1)).item())

        # decode/token p50 -- all replays are steady-state (warmup excluded above).
        decode_p50 = statistics.median(decode_ms) if decode_ms else float("nan")
        untr_p50 = statistics.median(untr_ms) if untr_ms else float("nan")

        text = tok.decode(gen_tokens, skip_special_tokens=True)
        print("=" * 72)
        print(f"depth: lm_layers={args.lm_layers} vision_layers={args.vision_layers}  prompt_len={prompt_len}")
        print(f"generated_tokens={len(gen_tokens)}  text={text!r}")
        print("-" * 72)
        print(f"host patch_embed     : {stage_t['host_patch_embed_ms']:8.2f} ms")
        print(f"vision encoder (42L) : {stage_t['vision_ms']:8.2f} ms")
        print(f"LM prefill (28L)     : {stage_t['prefill_ms']:8.2f} ms")
        print(f"decode/token traced  : {decode_p50:8.2f} ms (p50)  = {1000.0/decode_p50:6.2f} tok/s")
        print(f"decode/token untraced: {untr_p50:8.2f} ms (p50)  = {1000.0/untr_p50:6.2f} tok/s")
        print(f"trace speedup        : {untr_p50/decode_p50:8.2f}x")
        print(f"total (HELLO 2026)   : {total_ms:8.2f} ms ({len(gen_tokens)} tok)")
        print("=" * 72)
        print(
            f"SUMMARY lm={args.lm_layers} vis={args.vision_layers} prompt_len={prompt_len} "
            f"host_patch_embed_ms={stage_t['host_patch_embed_ms']:.2f} vision_ms={stage_t['vision_ms']:.2f} "
            f"prefill_ms={stage_t['prefill_ms']:.2f} decode_traced_ms={decode_p50:.2f} "
            f"decode_untraced_ms={untr_p50:.2f} speedup={untr_p50/decode_p50:.2f} "
            f"total_ms={total_ms:.2f} ntok={len(gen_tokens)} text={text!r}"
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
