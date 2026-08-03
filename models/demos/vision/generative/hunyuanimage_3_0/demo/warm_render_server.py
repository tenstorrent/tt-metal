#!/usr/bin/env python3
"""Stage-1 resident warm-render server: build the 32-layer on-device stack ONCE,
render a list of prompts back-to-back via generate_image_ondevice, report per-render
perf + a WARM AVERAGE (renders 2..N, excluding the first warmup). Measures whether the
133.7s step-1 compile amortizes across renders in one process (ttnn program cache)."""
import time

import torch  # noqa

import ttnn  # noqa
from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi
from models.demos.vision.generative.hunyuanimage_3_0.tt import host_glue_stage3 as hg3
from models.demos.vision.generative.hunyuanimage_3_0.tt.pipeline import _close_device, _open_selftest_device

STEPS = 50

# tag, prompt  -- cyberpunk base + 3 small tweaks + 1 completely different
PROMPTS = [
    (
        "cyber_base",
        "A cyberpunk style city with bright neon lights in the rain reflected in the puddles, the buildings are densely packed with markets and people. The style should be hyperrealistic. there should be cars on the ground and flying cars in the air as if there was a traffic system in the sky and on ground. People should look modified and enhanced with technology.",
    ),
    (
        "cyber_v1_gold",
        "A cyberpunk style city with warm golden neon lights in the rain reflected in the puddles, the buildings are densely packed with markets and people. The style should be hyperrealistic. there should be cars on the ground and flying cars in the air as if there was a traffic system in the sky and on ground. People should look modified and enhanced with technology.",
    ),
    (
        "cyber_v2_fog",
        "A cyberpunk style city with bright neon lights in heavy fog reflected in the puddles, the buildings are densely packed with vertical gardens and people. The style should be hyperrealistic. there should be cars on the ground and flying cars in the air as if there was a traffic system in the sky and on ground. People should look modified and enhanced with technology.",
    ),
    (
        "cyber_v3_dawn",
        "A cyberpunk style city with fading neon lights at foggy dawn reflected in the puddles, the buildings are densely packed with markets and drones. The style should be hyperrealistic. there should be cars on the ground and flying cars in the air as if there was a traffic system in the sky and on ground. People should look modified and enhanced with technology.",
    ),
    (
        "diff_zen",
        "A serene traditional Japanese zen garden at sunrise with a calm koi pond, blooming cherry blossom trees, a small wooden arched bridge, raked white gravel, and soft morning mist. The style should be hyperrealistic and peaceful with warm natural light.",
    ),
]


def step1_from_log(marker):
    # generate_image_ondevice prints "  step 1/50 NNN ms"; we scrape our own captured stdout per render.
    return None


def main():
    dev = _open_selftest_device()
    try:
        t_build = time.time()
        model, tt_pipe, _uninstall = gi.build_tt_backed_model(dev, num_layers=32, use_trace=False)
        print(f"BUILD_DONE {time.time()-t_build:.1f}s", flush=True)
        t_heads = time.time()
        heads = hg3.setup_ondevice_headglue_static(model, tt_pipe)  # ~114s conv-head compile, ONCE
        print(f"HEADS_DONE {time.time()-t_heads:.1f}s (built once, reused across renders)", flush=True)
        results = []
        for tag, prompt in PROMPTS:
            print(f"\n===== RENDER [{tag}] =====", flush=True)
            t0 = time.time()
            try:
                img, timing = hg3.generate_image_ondevice(
                    model,
                    tt_pipe,
                    prompt,
                    num_inference_steps=STEPS,
                    out_path=f"/tmp/render_{tag}.png",
                    heads=heads,
                )
                wall = time.time() - t0
                timing["tag"] = tag
                timing["wall_s"] = wall
                results.append(timing)
                print(
                    f"RENDER_RESULT[{tag}] total={timing['total_s']:.1f}s "
                    f"loop={timing['loop_s']:.1f}s vae={timing['vae_s']:.1f}s "
                    f"mean_ms_step={timing['ms_per_step']:.0f} wall={wall:.1f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"RENDER_FAIL[{tag}] {type(e).__name__}: {str(e)[:200]}", flush=True)
        print("\n======== SUMMARY ========", flush=True)
        for r in results:
            print(
                f"  {r['tag']:16s} total={r['total_s']:7.1f}s  loop={r['loop_s']:7.1f}s  vae={r['vae_s']:5.1f}s",
                flush=True,
            )
        warm = results[1:]
        if warm:
            at = sum(r["total_s"] for r in warm) / len(warm)
            al = sum(r["loop_s"] for r in warm) / len(warm)
            av = sum(r["vae_s"] for r in warm) / len(warm)
            print(
                f"WARM_AVG (renders 2..{len(results)}, n={len(warm)}): "
                f"total={at:.1f}s  loop={al:.1f}s  vae={av:.1f}s",
                flush=True,
            )
    finally:
        _close_device(dev)


if __name__ == "__main__":
    main()
