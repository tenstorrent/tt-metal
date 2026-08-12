#!/usr/bin/env python3
"""Live queue-driven server (HunyuanImage-3.0 on TT Galaxy): builds the 32-layer on-device stack
once and warms it, then tails a JSON-lines queue and renders each submitted prompt warm (~29.8 s
@ 50 steps, 1024²) with no per-render rebuild. Submit a request by appending one JSON line to
$HUNYUAN_DEMO_DIR/queue.jsonl: {"id": "...", "prompt": "...", "steps": 50}. Each render writes
$HUNYUAN_DEMO_DIR/out/<id>.png plus a <id>.done marker (holding render seconds, or "FAIL")."""
import json
import os
import time

import torch  # noqa: F401

import ttnn  # noqa: F401
from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi
from models.demos.vision.generative.hunyuanimage_3_0.tt import host_glue_stage3 as hg3
from models.demos.vision.generative.hunyuanimage_3_0.tt.pipeline import _close_device, _open_selftest_device

DEMO = os.environ.get("HUNYUAN_DEMO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_run"))
QUEUE = os.path.join(DEMO, "queue.jsonl")
OUT = os.path.join(DEMO, "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    open(QUEUE, "a").close()
    dev = _open_selftest_device()
    try:
        t = time.time()
        model, tt_pipe, _uninstall = gi.build_tt_backed_model(dev, num_layers=32, use_trace=False)
        heads = hg3.setup_ondevice_headglue_static(model, tt_pipe)
        print(f"BUILD_DONE {time.time()-t:.0f}s; warming up (one render compiles the kernels)...", flush=True)
        tw = time.time()
        hg3.generate_image_ondevice(
            model,
            tt_pipe,
            "a photorealistic red panda astronaut floating in space, warmup",
            num_inference_steps=50,
            out_path=os.path.join(OUT, "_warmup.png"),
            heads=heads,
        )
        print(f"WARMUP_DONE {time.time()-tw:.0f}s", flush=True)
        processed = len(open(QUEUE).read().splitlines())  # skip pre-existing entries
        print(f"READY — warm; tailing {QUEUE} (skipped {processed} old). Submit prompts now.", flush=True)
        while True:
            lines = open(QUEUE).read().splitlines()
            while len(lines) > processed:
                line = lines[processed]
                processed += 1
                if not line.strip():
                    continue
                rid = None
                try:
                    req = json.loads(line)
                    rid = str(req["id"])
                    prompt = req["prompt"]
                    steps = int(req.get("steps", 50))
                    print(f"RENDER {rid} ({steps} steps): {prompt[:70]}", flush=True)
                    t0 = time.time()
                    hg3.generate_image_ondevice(
                        model,
                        tt_pipe,
                        prompt,
                        num_inference_steps=steps,
                        out_path=os.path.join(OUT, f"{rid}.png"),
                        heads=heads,
                    )
                    dt = time.time() - t0
                    open(os.path.join(OUT, f"{rid}.done"), "w").write(f"{dt:.1f}")
                    print(f"DONE {rid} in {dt:.1f}s", flush=True)
                except Exception as e:
                    print(f"RENDER_FAIL {rid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    if rid:
                        open(os.path.join(OUT, f"{rid}.done"), "w").write("FAIL")
            time.sleep(1.5)
    finally:
        _close_device(dev)


if __name__ == "__main__":
    main()
