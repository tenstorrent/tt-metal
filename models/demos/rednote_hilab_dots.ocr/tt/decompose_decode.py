# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decompose the WARM traced decode step for rednote-hilab/dots.ocr (p150/blackhole).

The traced decode step measures ~16.4 ms/token at full 28-LM-layer depth. This
harness answers: is that time mostly PURE DEVICE TRACE REPLAY, or is it the
per-step HOST TAIL (logits D2H + argmax + token H2D) that runs OUTSIDE the trace
and serializes every step?

It reuses the exact same model assembly + trace-capture flow as ``profile_ocr.py``
(one warmup gen to compile + capture the decode trace, fresh cache, then timed
loop), but instead of one ``decode_ms`` it times each sub-component separately,
warm, p50 over many steps:

  1. pure_replay_ms : ttnn.execute_trace(blocking) + synchronize_device, NO host
                      work between replays (tight loop on the captured trace).
  2. logits_d2h_ms  : ttnn.to_torch(traced_logits)  -- the [1, vocab=151936] D2H.
  3. argmax_ms      : host torch.argmax over the [vocab] logits (+ EOS check).
  4. pos_rope_h2d_ms: write_decode_pos = pos H2D (kv_cache.write_pos) + per-layer
                      RoPE cos/sin H2D (write_decode_rope x num_layers).
  5. embed_h2d_ms   : write_embed = single token embed row H2D.
  6. full_step_ms   : the real per-step loop body (write_embed + write_decode_pos
                      + execute_trace + sync + to_torch + argmax) -- should ≈ 16.4.

Run::

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
    python models/demos/rednote_hilab_dots.ocr/tt/decompose_decode.py \
        --lm-layers 28 --vision-layers 42 --replay-iters 200

Set ``--tracy`` notes are handled by the caller running this under tracy; this
script just runs N pure replays so a tracy capture can sum DEVICE KERNEL DURATION.
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


def p50(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="dots.ocr traced-decode step decomposition")
    ap.add_argument("--image", default=os.path.join(_DEMO_DIR, "demo_image1.jpg"))
    ap.add_argument("--prompt", default="Read the text in the image.")
    ap.add_argument("--lm-layers", type=int, default=28)
    ap.add_argument("--vision-layers", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--replay-iters", type=int, default=200, help="pure-replay tight-loop count")
    ap.add_argument("--tail-iters", type=int, default=200, help="host-tail isolated count")
    ap.add_argument("--trace-region-size", type=int, default=256_000_000)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument(
        "--tracy-replay-only", action="store_true", help="only do pure replay loop (for clean tracy device-kernel sum)"
    )
    args = ap.parse_args()

    demo = _load_by_path("dots_dec_demo", "demo_ocr.py", _DEMO_DIR)
    loader = _load_by_path("dots_dec_loader", "weight_loader.py", _HERE)
    ocrm = _load_by_path("dots_dec_ocr_model", "ocr_model.py", _HERE)
    kvc = _load_by_path("dots_dec_kv_cache", "kv_cache.py", _HERE)

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

        def encode_and_prefill():
            patch_tokens = model.vision_tower.patch_embed(pixel_values)
            tt_in = ttnn.from_torch(
                patch_tokens.to(torch.float32),
                device=device,
                dtype=model.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_out = model.vision_tower(tt_in)
            ttnn.synchronize_device(device)
            vision_embeds = ttnn.to_torch(tt_out).to(torch.float32).reshape(-1, model.hidden_size)
            inputs_embeds = model.build_inputs_embeds(torch.tensor(ids, dtype=torch.int64), vision_embeds)
            hin = ttnn.from_torch(
                inputs_embeds.to(torch.float32),
                device=device,
                dtype=model.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pf = lm.prefill_from_embeds(hin, cache)
            ttnn.synchronize_device(device)
            nid = int(torch.argmax(ttnn.to_torch(pf).to(torch.float32).reshape(prompt_len, -1)[-1]).item())
            return nid

        # ============ WARMUP: compile + capture decode trace =================
        cache.reset()
        first_id = encode_and_prefill()
        warm_pos = prompt_len
        write_embed(first_id)
        lm.write_decode_pos(warm_pos, cache)
        _ = lm.decode_step_traced(persistent_embed, cache)
        ttnn.synchronize_device(device)

        write_embed(first_id)
        lm.write_decode_pos(warm_pos, cache)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        traced_logits = lm.decode_step_traced(persistent_embed, cache)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)

        vocab = int(traced_logits.shape[-1])

        # ---------------- (1) PURE TRACE REPLAY (device floor) --------------
        # Tight loop, NO host work between replays. pos/embed are already in the
        # persistent buffers from the capture; we never touch them. This is the
        # device-kernel time the trace covers.
        # warm a few
        for _ in range(5):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)
        pure_replay_ms = []
        for _ in range(args.replay_iters):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            pure_replay_ms.append((time.perf_counter() - t0) * 1000.0)

        if args.tracy_replay_only:
            print(f"TRACY_REPLAY_ONLY iters={args.replay_iters} pure_replay_p50_ms={p50(pure_replay_ms):.3f}")
            return

        # Also measure a batched-replay variant: N replays then one sync, to
        # separate per-replay host launch overhead from device time.
        ttnn.synchronize_device(device)
        N = args.replay_iters
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        batched_replay_ms = (time.perf_counter() - t0) * 1000.0 / N

        # ---------------- (3) LOGITS D2H (to_torch of [1, vocab]) -----------
        logits_d2h_ms = []
        for _ in range(args.tail_iters):
            t0 = time.perf_counter()
            lt = ttnn.to_torch(traced_logits)
            logits_d2h_ms.append((time.perf_counter() - t0) * 1000.0)
        # keep one host copy for argmax timing
        host_logits = ttnn.to_torch(traced_logits).to(torch.float32).reshape(-1)

        # ---------------- (2b) HOST ARGMAX (+ EOS check) -------------------
        argmax_ms = []
        for _ in range(args.tail_iters):
            t0 = time.perf_counter()
            nid = int(torch.argmax(host_logits).item())
            _ = nid in EOS
            argmax_ms.append((time.perf_counter() - t0) * 1000.0)

        # combined logits-D2H + cast + argmax (the actual decode tail compute)
        d2h_argmax_ms = []
        for _ in range(args.tail_iters):
            t0 = time.perf_counter()
            nid = int(torch.argmax(ttnn.to_torch(traced_logits).to(torch.float32).reshape(-1)).item())
            _ = nid in EOS
            d2h_argmax_ms.append((time.perf_counter() - t0) * 1000.0)

        # ---------------- (4) write_decode_pos: pos + per-layer RoPE H2D ----
        pos_rope_h2d_ms = []
        for i in range(args.tail_iters):
            pos = warm_pos + (i % 8)
            t0 = time.perf_counter()
            lm.write_decode_pos(pos, cache)
            pos_rope_h2d_ms.append((time.perf_counter() - t0) * 1000.0)

        # pos only (kv_cache.write_pos), to split it from the per-layer RoPE H2D
        pos_only_ms = []
        for i in range(args.tail_iters):
            pos = warm_pos + (i % 8)
            t0 = time.perf_counter()
            cache.write_pos(pos)
            pos_only_ms.append((time.perf_counter() - t0) * 1000.0)

        # ---------------- (5) write_embed: token embed row H2D --------------
        embed_h2d_ms = []
        for i in range(args.tail_iters):
            t0 = time.perf_counter()
            write_embed(first_id)
            embed_h2d_ms.append((time.perf_counter() - t0) * 1000.0)

        # ---------------- (6) FULL STEP: real per-step loop body ------------
        # Reset cache + prefill so cur_id flow is realistic, then run the exact
        # production loop body and time the whole thing per step.
        cache.reset()
        next_id = encode_and_prefill()
        full_step_ms = []
        cur_id = next_id
        for step in range(1, args.max_new_tokens + 64):  # extra to get many p50 samples
            pos = prompt_len + step - 1
            if pos >= max_seq_len - 1:
                pos = max_seq_len - 2  # clamp to valid cache range; timing not correctness
            t0 = time.perf_counter()
            write_embed(cur_id)
            lm.write_decode_pos(pos, cache)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            cur_id = int(torch.argmax(ttnn.to_torch(traced_logits).to(torch.float32).reshape(-1)).item())
            full_step_ms.append((time.perf_counter() - t0) * 1000.0)
            _ = cur_id in EOS  # don't break -- want many timing samples

        # ============================ REPORT =================================
        pr = p50(pure_replay_ms)
        ld = p50(logits_d2h_ms)
        am = p50(argmax_ms)
        da = p50(d2h_argmax_ms)
        ph = p50(pos_rope_h2d_ms)
        po = p50(pos_only_ms)
        eh = p50(embed_h2d_ms)
        fs = p50(full_step_ms)
        rope_only = ph - po

        print("=" * 78)
        print(
            f"DECOMPOSE traced decode  lm_layers={args.lm_layers} vocab={vocab} "
            f"prompt_len={prompt_len}  (p50 over many warm steps)"
        )
        print("-" * 78)
        print(f"(1) pure trace replay (device floor) : {pr:8.3f} ms   [{1000.0/pr:6.1f} tok/s ideal]")
        print(f"    batched replay (N/sync amortized): {batched_replay_ms:8.3f} ms")
        print(f"(3) logits D2H  to_torch[1,{vocab}]  : {ld:8.3f} ms")
        print(f"(2) host argmax over [{vocab}]       : {am:8.3f} ms")
        print(f"    D2H+cast+argmax (real tail compute): {da:8.3f} ms")
        print(f"(4) write_decode_pos (pos+RoPE H2D)  : {ph:8.3f} ms")
        print(f"      - pos-only (write_pos H2D)     : {po:8.3f} ms")
        print(f"      - per-layer RoPE H2D ({args.lm_layers}x cos/sin): {rope_only:8.3f} ms")
        print(f"(5) write_embed (token embed H2D)    : {eh:8.3f} ms")
        print("-" * 78)
        host_tail = da + ph + eh
        print(f"SUM host tail (D2H+argmax + posRoPE + embed): {host_tail:8.3f} ms")
        print(f"reconstructed step (1)+host_tail     : {pr + host_tail:8.3f} ms")
        print(f"(6) MEASURED full step               : {fs:8.3f} ms")
        print("=" * 78)
        host_frac = host_tail / fs * 100.0 if fs == fs else float("nan")
        print(f"host-tail fraction of full step      : {host_frac:5.1f}%")
        print(f"device-replay fraction of full step  : {pr/fs*100.0:5.1f}%")
        print("=" * 78)
        print(
            f"SUMMARY lm={args.lm_layers} vocab={vocab} pure_replay_ms={pr:.3f} "
            f"batched_replay_ms={batched_replay_ms:.3f} logits_d2h_ms={ld:.3f} argmax_ms={am:.3f} "
            f"d2h_argmax_ms={da:.3f} pos_rope_h2d_ms={ph:.3f} pos_only_ms={po:.3f} "
            f"rope_h2d_ms={rope_only:.3f} embed_h2d_ms={eh:.3f} host_tail_ms={host_tail:.3f} "
            f"full_step_ms={fs:.3f} host_frac_pct={host_frac:.1f}"
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
