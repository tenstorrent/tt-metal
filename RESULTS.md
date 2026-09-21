# MiniMax-H3 HyperFlow results

Measured on Blackhole Galaxy (4x8, TP=4 axis 0 / SP=8 axis 1, ring fabric, 2 links) with the
two-time adapter `minimax_h3_hyperflow_8step_v1.0.safetensors` (sha256 `9297f450...df447`,
published at `videorebirth/hyperflow`).

**Every row is 8 forwards.** The adapter publishes a fixed 9-point sigma grid; the pipeline reads
the count off the file and refuses a caller-supplied `num_inference_steps`, so the schedule cannot
silently fall back to the base model's 49.

**Attention is dense.** `vsa_config` is opt-in and unset here, so these are dense-attention numbers
-- as is the base-model reference below, which keeps the comparison like-for-like. The sparse (VSA)
path is a separate axis and is not measured in this table.

## Base model reference

| | |
|---|---|
| t2va 5 s, 49 forwards, dense | **58.0 s** compute (measured, `bh-glx-110-d07u08`) |
| t2va 5 s, 8 forwards, dense | **14.5 s** compute |
| Speedup | **4.0x** |

Independently corroborated on g03blx04, which measured 15.2 s for the same 8-forward point
(denoise 9.2 s, 1149.6 ms/forward) against this table's 14.5 s -- two boxes within 5%.

## Results

| Mode | Clip | Frames | Canvas | Fwd | Padded | Encoder | Keyframe encode | Denoise | VAE decode | Audio decode | Total | s / video s | CLIP | Node |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2va | 5 s | 124 | 1344x768 | 8 | 37888 | 0.4 | -- | 9.4 | 3.6 | 1.2 | 14.5 | 2.8 | 37.53 | bh-glx-120-b09u02 |
| t2va | 10 s | 243 | 1344x768 | 8 | 73472 | 0.4 | -- | 23.1 | 7.3 | 1.3 | 32.1 | 3.2 | 36.94 | bh-glx-120-b09u02 |
| fl2va | 10 s | 243 | 1344x768 | 8 | 77568 | 2.2 | 1.3 | 30.2 | 7.3 | 1.2 | 42.1 | 4.2 | 37.19 | bh-glx-120-c06u08 |
| fl2va | 15 s | 362 | 1344x768 | 8 | 113152 | 2.2 | 1.3 | 58.5 | 10.8 | 1.5 | 74.2 | 4.9 | 37.44 | bh-glx-120-c06u08 |

**Not measured:** t2va 15 s, fl2va 5 s, ref2va 5 s, ref2va 10 s, ref2va 15 s.

Timings are seconds of compute in the warm window: each point runs one full warmup generation at
its shape first, and prepares plus artifact export are excluded. CLIP is prompt alignment,
**recorded not gated** -- the calibrated bars elsewhere are set against the 49-forward base model.

## Reading the scaling

Denoise goes as roughly `O(N^1.65)` in padded sequence length (2.05x sequence -> 3.21x denoise;
2.99x -> 6.22x). That is the expected shape for dense attention: quadratic in the attention term,
linear in the projections and feed-forward. It is *not* a defect, and it is why the realtime factor
degrades with clip length (2.8x at 5 s, 4.9x at 15 s) rather than holding flat.

Consequences for where the remaining time goes:

* **Denoise dominates and grows** -- 65 % of compute at 5 s, 79 % at 15 s.
* **VAE decode scales about linearly with frames** (3.6 / 7.3 / 10.8 s for 124 / 243 / 362) and so
  *shrinks* as a share of total, 25 % -> 15 %. Optimising it pays less the longer the clip.
* **Keyframe encode is ~1.3 s**, so `fl2va` conditioning costs almost nothing beyond the longer
  sequence it implies.

## Provenance and caveats

The `Node` column is not decoration: two hosts swept into one artifact directory, so each row names
the box that produced it. `bh-glx-120-c06u08` threw a SIGBUS and lost its PCIe devices earlier in
the session; the rows it contributed passed every gate in the test and are real measurements, but a
reproduction on `bh-glx-120-b09u02` would settle them.

g03blx04 is excluded entirely: it drops tray 1 (chips 8-15) under MiniMax-H3 load, reproduced six
times across two different source trees, while non-H3 matmul at 29.9 TFLOP/s per chip runs 180 s
clean. A controlled A/B on the pre-change tree ruled out this branch's commits as the cause.

### Grid configuration changed mid-sweep

Rows measured before 22:50 ran with SDPA and the VSA pooled matmuls clamped to an 11x10 core grid;
rows after run unclamped at the device's full 12x10. The clamp was reverted because it broke
ref2va -- at `padded_len` 89856 the SDPA circular buffers on 11x10 reach 1647616 B against L1's
1572864 B, since a narrower grid gives each core more to hold. Matmuls were on 11x10 in both cases
(`get_matmul_core_grid` has always clamped them), so the difference is confined to SDPA and the
VSA stages, worth roughly 8 % more cores. Treat cross-row comparisons spanning that boundary as
approximate until the early points are re-measured.

### ref2va does not run on this build

All three `ref2va` cells are unmeasured because the mode fails at program compile, before any
device work:

```
Statically allocated circular buffers on core range [0-0 - 10-9]
grow to 1647616 B which is beyond max L1 size of 1572864 B
  references.py:257  encode_references
  vae_minimax_h3.py:847  encode
  conv_minimax_h3.py:359  forward
```

It is the video VAE's taps=3 encoder conv3d, which only `ref2va` reaches, 4.75 % over L1.

This is **not specific to the HyperFlow sweep**. The pre-existing, calibrated
`test_pipeline_ref2va_minimax_h3.py::test_ref2va_end_to_end` -- no adapter, no two-time schedule --
fails with the byte-identical size on the same build. That gate carries measured `padded_len` and
per-stage timings in its own comments, so `ref2va` ran on an earlier build: this is a regression,
not a mode that never worked.

Ruled out by experiment, not inference:

* **Core grid.** The blocking is grid-independent -- forcing 12x10 instead of 11x10 produced the
  same 1647616 B to the byte.
* **The rail-cap commit.** The same failure occurs with it reverted.
* **A missing blocking entry.** The exact `_BLOCKINGS` lookup hits; neither fallback warning fires.

The remaining suspects are the conv3d blocking chosen for this shape and the op's own L1
accounting, which needs a bisect against the build that produced the gate's recorded numbers.
