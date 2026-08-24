"""Measure the tool's port the way the hand-port is measured: steady-state ms per audio frame.

Runs the SHIPPED path (`run_tts`) at the demo's own settings — real prompt, 24 frames,
`early_stop` left at its default — so the number describes what a user actually gets.
`run_tts` already returns {prefill_s, decode_s, codec_s}, so no new instrumentation is needed.

One warm-up run first (kernel compilation and first-touch), then the measured run.
"""
import time

import ttnn

from models.demos.voxtral_tts_full.tt import pipeline as P
from models.demos.voxtral_tts_full.tt import reference as ref

FRAMES = 24  # demo_tts.py --max-frames default

dev = ttnn.open_device(device_id=0, trace_region_size=P.DEFAULT_TRACE_REGION_SIZE)
try:
    t0 = time.time()
    model = ref.load_hf_model()
    inputs = ref.encode_inputs()
    pipe = P.build_pipeline(dev, model=model)
    print(f"[measure] build (model load + stage) {time.time() - t0:.1f}s", flush=True)

    print("[measure] warm-up run…", flush=True)
    pipe.run_tts(inputs, max_frames=FRAMES)

    print("[measure] timed run…", flush=True)
    out = pipe.run_tts(inputs, max_frames=FRAMES)
    t = out["timings"]
    n = out["n_frames"]

    per_frame_ms = t["decode_s"] * 1000.0 / max(n, 1)
    total_ms = (t["prefill_s"] + t["decode_s"] + t["codec_s"]) * 1000.0
    audio_s = n / 12.5  # 12.5 frames per second of audio

    print("\n================ RESULT ================")
    print(f"frames generated      : {n}")
    print(f"prefill               : {t['prefill_s'] * 1000:9.1f} ms   (once)")
    print(f"decode                : {t['decode_s'] * 1000:9.1f} ms   total")
    print(f"  -> PER FRAME        : {per_frame_ms:9.2f} ms   <-- compare with the hand-port")
    print(f"codec                 : {t['codec_s'] * 1000:9.1f} ms   (once)")
    print(f"total utterance       : {total_ms:9.1f} ms  for {audio_s:.2f}s of audio")
    print(f"RTF (compute/audio)   : {total_ms / 1000.0 / audio_s:9.3f}")
    print("========================================")
finally:
    ttnn.close_device(dev)
