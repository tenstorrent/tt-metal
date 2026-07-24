# FIBO CFG on/off perf — `test_fibo_pipeline_perf_breakdown_json`

**Config:** 1024×1024, 30 steps, **traced** (steady-state replay), median of 8 runs (after 1 warmup + 1 settle)
**DiT prompt-branch padding:** `keep_padding=True` → fixed buckets `(256, 1024)` — the ~833-token JSON positive prompt is padded to the **1024** bucket.
**Prompt:** committed `fibo_vlm_prompt.json` (real VLM text→JSON caption); negative = `"blurry, low quality, distorted, watermark, text"`.

CFG toggle added via a `cfg` parametrize (`ids=["cfg", "nocfg"]`): `cfg=True` → gs=5.0 (2 fwd/step), `cfg=False` → gs=1.0 (the `guidance_scale > 1` gate skips the negative encode + uncond denoise branch → 1 fwd/step).

## Galaxy `(4, 8)` — `mesh_device1`, blackhole

**Date:** 2026-07-24

| Stage    | CFG on (gs=5.0, 2 fwd/step) | CFG off (gs=1.0, 1 fwd/step) |
|----------|-----------------------------|------------------------------|
| encode   | 0.34 s  (5.3%)              | 0.26 s  (6.7%)               |
| prepare  | 0.35 s  (5.5%)              | 0.28 s  (7.0%)               |
| denoise  | **5.47 s (85.8%) → 5.48 it/s** | **3.16 s (80.6%) → 9.49 it/s** |
| decode   | 0.22 s  (3.4%)              | 0.22 s  (5.7%)               |
| **total**| **6.39 s → 0.1566 img/s** (best 6.33 s → 0.1579) | **3.92 s → 0.2548 img/s** (best 3.85 s → 0.2600) |

**Takeaway:** CFG off is ~**1.63×** faster end-to-end (6.39 s → 3.92 s). The win is concentrated in denoise (5.47 s → 3.16 s, ~1.73×; 5.48 → 9.49 it/s) — roughly the expected halving of forward passes minus per-step overhead that doesn't scale with forward count. encode/prepare drop slightly (no negative encode / no uncond branch build); decode unchanged. Both runs produced valid, non-degenerate images.

**Output images:**

| Variant | File |
|---------|------|
| CFG on (gs=5.0) | `fibo_perf_json_1024x1024_30steps_gs5.0_20260724-102204.png` |
| CFG off (gs=1.0) | `fibo_perf_json_1024x1024_30steps_gs1.0_20260724-102608.png` |

## GB2 `(2, 2)` — `mesh_device0`, blackhole

**Date:** _TBD_

| Stage    | CFG on (gs=5.0, 2 fwd/step) | CFG off (gs=1.0, 1 fwd/step) |
|----------|-----------------------------|------------------------------|
| encode   | _TBD_                       | _TBD_                        |
| prepare  | _TBD_                       | _TBD_                        |
| denoise  | _TBD_                       | _TBD_                        |
| decode   | _TBD_                       | _TBD_                        |
| **total**| _TBD_                       | _TBD_                        |

**Takeaway:** _TBD_

**Output images:**

| Variant | File |
|---------|------|
| CFG on (gs=5.0) | _TBD_ |
| CFG off (gs=1.0) | _TBD_ |

## "No known best blocking" warnings

12 unique shapes, each warned **once at build/warmup only** (no recurrence in the measured window — program-cache hits suppress repeats), all falling back to a default blocking:

- **DiT spatial stream (M=4096):** `(4096,1024,3072)`, `(4096,1024,1024)` → default `8x8x8`
- **SmolLM3 encoder:** `(32|128,2048,768)`, `(…,2048,2752)`, `(…,2752,2048)`, `(128,2048,512)`, `(128,512,256)` → default `2x8x8`
- **Misc/small:** `(64,1024,512)`, `(32,64,64)`

The DiT **prompt-stream** matmuls did not warn (registered blockings from the M-bucket sweep hit). The untuned ones are the DiT spatial stream and the encoder shapes.

## Repro

```bash
# mesh_device1 = Galaxy (4,8); mesh_device0 = GB2 (2,2)
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  "models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown_json" \
  -k "mesh_device1 and traced and cfg" -v -s --timeout=1800     # CFG on
  # -k "mesh_device1 and traced and nocfg"                       # CFG off
```

_Galaxy (4,8) result: `2 passed in 518.83s`._
