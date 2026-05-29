# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr OCR decode path (KV-cache).

Exercises the cached autoregressive decode step (the per-token hot loop unlocked
by ``tt/kv_cache.py`` + the flash-decode SDPA path in ``tt/attention.py``) so a
tracy capture can attribute device-kernel time across the LM trunk. Reports
``prefill_ms`` and the median steady-state ``decode_step_ms`` (the O(1) cached
step), split from the warmup step.

The sub-pass-1 structural win in this model is the KV cache itself: it converts
the per-step decode from an O(N) full-trunk re-run into an O(1) cached step,
which is what makes full-depth (28 LM / 42 vision) generation tractable. This
harness measures the resulting cached step and (under tracy) exposes the
device-kernel hotspots for a future op-level pass.

Usage (raw timing)::

    python tt/profile_ocr.py --lm-layers 28 --vision-layers 42 --num-timed 8

Usage (tracy, device-kernel data)::

    python -m tracy -p -v -r --op-support-count 3000 --dump-device-data-mid-run \
        -n ocr models/demos/rednote_hilab_dots.ocr/tt/profile_ocr.py \
        --lm-layers 28 --vision-layers 42 --num-timed 8

Then read ``generated/profiler/.logs/tracy_ops_data.csv`` and bucket by op-code.

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


def _load_by_path(name, filename, directory):
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="dots.ocr OCR decode-path tracy harness")
    ap.add_argument("--image", default=os.path.join(_DEMO_DIR, "sample_ocr.png"))
    ap.add_argument("--prompt", default="Read the text in the image.")
    ap.add_argument("--lm-layers", type=int, default=28)
    ap.add_argument("--vision-layers", type=int, default=42)
    ap.add_argument("--num-timed", type=int, default=8)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    demo = _load_by_path("dots_prof_demo", "demo_ocr.py", _DEMO_DIR)
    loader = _load_by_path("dots_prof_loader", "weight_loader.py", _HERE)
    ocrm = _load_by_path("dots_prof_ocr_model", "ocr_model.py", _HERE)
    kvc = _load_by_path("dots_prof_kv_cache", "kv_cache.py", _HERE)

    input_ids, pixel_values, grid_thw, _tok = demo.host_preprocess(args.image, args.prompt, CHECKPOINT_PATH)
    pixel_values = pixel_values.to(torch.float32)
    lm_sd = loader.load_language_model_weights(CHECKPOINT_PATH, num_layers=args.lm_layers)
    vis_sd = loader.load_vision_tower_weights(CHECKPOINT_PATH, num_layers=args.vision_layers)

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768)
    try:
        model = ocrm.TtOcrModel(
            device=device,
            lm_state_dict=lm_sd,
            vision_state_dict=vis_sd,
            grid_thw=grid_thw,
            lm_num_layers=args.lm_layers,
            vision_num_layers=args.vision_layers,
        )

        vision_embeds = model.encode_image(pixel_values)
        ids = input_ids.reshape(-1).to(torch.int64).tolist()
        prompt_len = len(ids)
        max_seq = prompt_len + args.num_timed + 2
        lm = model._lm_for_seq(prompt_len, max_seq_len=max_seq)
        cache = kvc.SelfAttentionKVCache(
            device=device,
            num_layers=args.lm_layers,
            batch=1,
            num_kv_heads=model.num_kv_heads,
            max_seq_len=max_seq,
            head_dim=model.head_dim,
        )

        # Prefill (populates the cache for the whole prompt).
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
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        nid = int(torch.argmax(ttnn.to_torch(pf).to(torch.float32).reshape(prompt_len, -1)[-1]).item())

        # Warmup decode step (compiles kernels; excluded from steady-state).
        te = model._embed_token(nid)
        _ = lm.decode_step(te, prompt_len, cache)
        ttnn.synchronize_device(device)

        # Timed steady-state cached decode steps.
        step_ms = []
        for s in range(1, args.num_timed + 1):
            pos = prompt_len + s
            te = model._embed_token(nid)
            t0 = time.perf_counter()
            sl = lm.decode_step(te, pos, cache)
            ttnn.synchronize_device(device)
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            nid = int(torch.argmax(ttnn.to_torch(sl).to(torch.float32).reshape(-1)).item())

        print(
            f"SUMMARY lm_layers={args.lm_layers} vision_layers={args.vision_layers} "
            f"prompt_len={prompt_len} prefill_ms={prefill_ms:.2f} "
            f"decode_step_ms={statistics.median(step_ms):.2f} "
            f"decode_step_min_ms={min(step_ms):.2f} num_timed={args.num_timed}"
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
