# Wan2.2 TI2V-5B — I2V enablement and optimization (nkira)

Everything done on `nkira/wan2.2-5B-i2v` (sprint-5 tip on `nkira/wan2.2-5B-i2v-step5`, sprint 4
on `-step3`) since branching from Teja's bring-up at `8d596eb242d`. Companion to `Wan2_2_TI2V_5B.md`, which stays the bring-up doc.

Hardware throughout: one Blackhole Galaxy, 4x8, SP=8 axis1 / TP=4 axis0, Ring, FSDP off.
All timings are 40 steps, warm-traced, and every figure below is a **mean of 3 invocations** —
the per-run `Std` the perf test prints is a single sample, not a spread.

---

## 1. Results

**Current tip (sprint 6, 2026-09-28, host u13-43, mean of 3, 81 f / 40 steps, warm-traced,
bf16 / HiFi2, 2cq).** Sprint 6 fix 1 removed the AdaLN modulation's tile/row-major round trip
(section 7.9, bit-exact); the sprint-5 numbers below are the "before" column, and a same-hour
control run of the sprint-5 `transformer_wan.py` at 720p (10.201 s denoise / 11.262 s total)
confirms the recorded sprint-5 mean had not drifted.

| Mode | Resolution | Text enc | Image enc | Denoise | VAE dec | Total | vs sprint 5 (11.23 / 12.90 / 5.49) |
|------|------------|----------|-----------|---------|---------|-------|---|
| T2V  | 1280x704   | 0.090s | — | **9.787s** | 0.938s | **10.83s** | -3.5% (denoise -4.0%; -4.1% / -3.8% vs the control) |
| I2V  | 1280x704   | 0.089s | 1.235s | **10.260s** | 0.920s | **12.52s** | -2.9% (denoise -3.0%) |
| T2V  | 832x480    | 0.088s | — | **4.438s** | 0.603s | **5.14s** | -6.4% (denoise -8.3%) |
| T2V  | 1280x704, opt-in `all_bf8_lofi` (2026-09-29, final preset definition, section 7.7) | 0.090s | — | **8.250s** | 0.95-1.33s | **9.32-9.46s** clean (mean of 3 incl. a VAE outlier: 9.48s) | -3.9% denoise vs its sprint-5 9.61s; **-15.7% denoise vs the bf16 default** |

Runs: 720p 9.807 / 9.780 / 9.773 s denoise (spread 0.34 %), totals spread 0.61 %; 480p
4.438 / 4.443 / 4.434 (0.20 %), totals 2.7 % (all VAE: 0.56-0.60 s); I2V 10.274 / 10.245 /
10.262 (0.29 %), totals 0.20 %. Per step: 10.3 ms at 720p, 10.1 ms at 480p, 8.0 ms I2V --
a fixed per-block cost, so the relative gain is largest where the step is shortest. Against the
sprint start (16.78 / 18.86 / 8.87 s) the totals are now **-35.4 % / -33.6 % / -42.1 %**. The
`all_bf8_lofi` preset has not been re-measured on this tip yet (it stacked with the sprint-5
hoist; section 7.7).

**Sprint-5 tip (2026-09-26, host u13-43, mean of 3, 81 f / 40 steps, warm-traced,
bf16 / HiFi2, 2cq), kept for the record.** Sprint 5 hoisted the per-block AdaLN modulation out of the two CFG passes
(section 7.2, bit-exact); it is the only model change since sprint 4, so the delta column is
that change alone. The `all_bf8_lofi` preset (section 7.7) has not been re-measured on top of it.

| Mode | Resolution | Text enc | Image enc | Denoise | VAE dec | Total | vs sprint 4 (11.50 / 13.49 / 6.03) |
|------|------------|----------|-----------|---------|---------|-------|---|
| T2V  | 1280x704   | 0.089s | — | **10.198s** | 0.922s | **11.23s** | -2.4% (denoise -2.3%) |
| I2V  | 1280x704   | 0.089s | 1.293s | **10.580s** | 0.923s | **12.90s** | -4.4% (denoise -4.5%) |
| T2V  | 832x480    | 0.088s | — | **4.842s** | 0.556s | **5.49s** | -8.9% (denoise -9.5%) |
| T2V  | 1280x704, opt-in `all_bf8_lofi` (2026-09-28) | 0.089s | — | **8.583s** | 0.919s | **9.61s** | -3.4% vs its sprint-4 9.95s; **-16.5% vs the bf16 default** |

Spreads across the three runs: denoise 0.6 % (720p T2V), 0.4 % (I2V), 0.07 % (480p), 0.3 %
(`all_bf8_lofi`); totals 0.7 / 0.7 / 0.4 / 0.2 %. The preset row (runs 9.598 / 9.620 /
9.608 s) shows the hoist and the quant preset stack: the preset stays opt-in pending the
visual sign-off in section 7.7. The gain grows as the step gets shorter (720p T2V 261 -> 255 ms/step, 480p
134 -> 121 ms/step) because what was removed is a fixed ~330 program launches per step, not
math. Against the sprint start (16.78 / 18.86 / 8.87 s) the totals are now **-33.1 % / -31.6 % /
-38.1 %**.

**Sprint-4 tip (2026-09-25, same host and method), kept for the record.** The
`all_bf8_lofi` row is opt-in (`WAN5B_QUANT_CONFIG=all_bf8_lofi`, section 7.7) and was the fastest
configuration that passes every gate.

| Mode | Resolution | Config | Text enc | Image enc | Denoise | VAE dec | Total | vs sprint 3 (11.77 / 13.96 / 6.34) |
|------|------------|--------|----------|-----------|---------|---------|-------|---|
| T2V  | 1280x704   | default (bf16, 2cq) | 0.091s | — | **10.434s** | 0.951s | **11.50s** | -2.3% |
| T2V  | 1280x704   | `all_bf8_lofi` (2cq) | 0.090s | — | **8.914s** | 0.930s | **9.95s** | **-15.5%** |
| I2V  | 1280x704   | default (bf16, 2cq) | 0.088s | 1.327s | **11.073s** | 0.985s | **13.49s** | -3.3% |
| T2V  | 832x480    | default (bf16, 2cq) | 0.088s | — | **5.350s** | 0.586s | **6.03s** | -4.9% |

Denoise and total spreads across the three runs are 0.1-0.7% for T2V; I2V total spreads 2.2%
because its host image encode spreads 12% (VAE alone spreads 1-11%, see section 8). Against
the sprint-3 tip measured the same way on the same host with blocking execution (720p T2V
10.747s denoise / 11.825s total, 480p 5.649 / 6.325, single-queue runs of 2026-09-24), the 2cq
default is -2.9% / -2.8% at 720p and -5.3% / -4.7% at 480p, i.e. 7.8 ms per step at 720p — the
outside bound estimated in section 7.5 before it was built.
Against the sprint start (16.78s) the 720p T2V total is now **-31.5% default, -40.7% opt-in**.

Sprint 3 results (2026-09-22), kept for the record:

| | before | after | delta |
|---|---|---|---|
| 720p T2V total (1280x704, 81f) | 16.78s | **11.77s** | **-29.8%** |
| 720p I2V total | 18.86s | **13.96s** | **-26.0%** |
| 480p T2V total (832x480, 81f) | 8.87s | **6.34s** | **-28.5%** |
| 720p VAE decode | 4.632s | **0.959s** | **-79.3%** |
| 720p denoise | 12.045s | **10.701s** | **-11.2%** |

121 frames at 720p T2V: **19.76s** traced (494 ms/step). No prior baseline — newly measured;
not re-run with the swept matmul table.

Section detail at the current tip (swept matmul table, 2026-09-22, host u13-43, mean of 3;
denoise and total spread <= 0.8%):

| Mode | Resolution | Text enc | Image enc | Denoise | VAE dec | Total |
|------|------------|----------|-----------|---------|---------|-------|
| T2V  | 1280x704   | 0.092s   | —         | 10.701s | 0.959s  | 11.77s |
| I2V  | 1280x704   | 0.088s   | 1.379s    | 11.513s | 0.964s  | 13.96s |
| T2V  | 832x480    | 0.089s   | —         | 5.667s  | 0.576s  | 6.34s |

Before the swept table (2026-09-17, previous host): T2V 720p 11.436s denoise / 12.51s total,
I2V 720p 12.072s / 14.68s, T2V 480p 6.301s / 6.98s. The "before" column above is from that host
too; the one same-host reference is the single 720p T2V run on u13-43 the day before the sweep
(11.238s denoise, 12.29s total), against which the swept table is -4.8% / -4.2%.

---

## 2. What was built: I2V

TI2V-5B is **architecturally different from the 14B I2V** and the difference is the whole job.
The 14B concatenates the conditioning frame onto the input channels and uses a CLIP image
encoder. The 5B does neither. It conditions by **pinning latent frame 0**:

- the seed image is encoded by the torch VAE on host, written into latent frame 0, and re-pinned
  after every solver step
- a **per-token timestep** gives frame-0 tokens t=0 and everything else the current t
- no CLIP

`models/tt_dit/pipelines/wan/pipeline_wan_ti2v_5b_i2v.py`, plus
`test_pipeline_wan_ti2v_5b_i2v.py`, `test_ti2v_5b_i2v_math.py`, `test_transformer_wan_ti2v_5b.py`.

**The per-token timestep is carried as a 2-row tensor**, not a full per-token one. The fully
per-token layout runs the timestep MLP at M=N/SP, which needs matmul blocking entries whose keys
collided with T2V projection shapes in the process-global table. The 2-row form is bit-exact
against the scalar path and leaves `combined_step`'s traced signature untouched.

Validation: two-row vs scalar **PCC 100.0000%**, transformer vs torch **99.9893%** (scalar) /
**99.9894%** (per-token), conditioning math **20/20** exact vs diffusers, E2E frame-0-vs-seed
**PCC 0.9984**.

**Prompting matters more than it does for T2V.** Only frame 0 is pinned, so frames 1..N are
unconstrained. A controlled A/B on one seed image, prompt the only variable:

| prompt | mean_frame_delta | outcome |
|---|---|---|
| "...add sharks in the water" | 19.62 | surfer **gone**, replaced by sharks |
| descriptive of the seed + motion | 16.29 | surfer preserved, coherent |

Ask for something absent from the seed and the model changes the scene, as instructed. For a
**static** subject the motion has to be put in the camera and the environment: a beaded-figurine
seed described with "slow gentle" motion gave `mean_frame_delta` 0.87 (effectively a still);
the same scene with "pushes in steadily / brisk wind / clouds roll visibly" gave **9.42**, a
10.8x increase at identical PCC 0.9982 and no subject drift.

---

## 3. What was optimized

### 3.1 VAE decode 4.63s -> 0.97s (-79%)

**`WanDupUp3D` was 78% of the decode** — 3.85s of the 4.92s device total, across eleven ttnn
calls in one module. It is a single channel-to-space index permutation, but the shipped chain
drove the innermost dimension down to 1, 4 or 8 elements, and **ROW_MAJOR pads every row to
32B**, so `concat([x_nc1] * repeats, dim=2)` on a size-1 axis was a 16x padded physical copy
(83.6ms per call on its own). ROW_MAJOR itself is necessary here — TILE would pad the size-2/4
factor dims to 32 — but the padding hurts regardless.

The rewrite keeps `oc` (256-1024 elements, already aligned) innermost, so the reshapes only
split/merge outer dimensions and are metadata-only. The indexing collapses further than
expected: offset `f = dt*fs*fs + dh*fs + dw` reads channel-slice `f // repeats` of the
*original* tensor with stride `factor // repeats`, so the `repeat_interleave` never needs
materialising, and when `repeats == factor` every offset reads the same channel — **exact
nearest-neighbour, one `ttnn.upsample`**. Two of the 5B's three instances are that case:

| instance | in -> out | factor | repeats | consequence |
|---|---|---|---|---|
| up0, up1 | 1024 -> 1024 | 8 | **8 = factor** | exact nearest-neighbour |
| up2 | 1024 -> 512 | 4 | 2 = `factor_s` | two channel slices, one per H offset |

`_forward_generic` is kept as fallback and as the equivalence reference; `__init__` gates the
fast path, so uncovered shapes are bit-for-bit unchanged. `WanDupUp3D` only exists inside
`WanResidualUpBlock` (`is_residual=True`), so the Wan2.1/14B decoder — which uses `WanUpBlock` —
has none and is untouched structurally.

Measured per instance (micro-benchmark, `test_dup_up3d_ti2v_5b.py`): up0 50.4ms -> 0.19ms
(270x), up1 186.0ms -> 0.56ms (332x), up2 206.8ms -> 5.74ms (36x). Section: **4.6915s ->
0.9236s**, with conv3d unchanged at 0.52s — the internal check that the change did one thing.

### 3.2 Ring SDPA q=128 -> 160 (denoise -6.1%)

Work items are `B*NH*ceil(M/q)` spread flat over the SDPA worker grid. At M=2336 with 6 local
heads (24/TP4), the q=128 inherited from the 14B gives 19x6 = **114 items against a ~110-core
grid** — a whole extra scheduling round to place 4 items. Swept under Tracy,
`DEVICE KERNEL DURATION`, mean over the 32 device instances of each config (480 instances):

| q\k | 128 | 256 | 512 |
|---|---|---|---|
| **128** (inherited) | 1954.4 | 1619.2 | **1480.0** |
| **160** | 1579.0 | 1174.8 | **1068.4  (-27.8%)** |
| 192 | 1480.7 | 1248.9 | 1183.6 |
| 224 | 2028.6 | 1457.4 | 1260.9 |
| 256 | 1837.0 | 1537.4 | 1433.0 |

Applied through `sdpa_chunk_size_overrides` on the 5B pipeline config — **not** by editing
`WanAttention.sdpa_chunk_size_map`, which is keyed only on `(is_blackhole, sp, tp)` and is
shared with the 14B. The 14B does not have this problem (740 items -> 7 rounds, 96% efficient).
The hook existed on `WanAttention` and `WanTransformerBlock` but was dead from
`WanTransformer3DModel` down; plumbing it through the model, `WanCheckpoint.build` and
`WanPipelineConfig` is what makes a per-variant retune possible at all.

E2E: 720p denoise **12.184 -> 11.436s (-6.14%, spread 0.31%)**, I2V denoise **12.726 ->
12.072s (-5.14%)**. 480p denoise **+0.54%** against a 1.11% spread — flat, because at M=1024
both chunk sizes already fit one round. The override key is not resolution-aware, so 480p is
along for the ride; both resolutions were gated for this reason. Transformer PCC unchanged to
four decimals.

### 3.3 I2V image encode 3.37s -> 1.38s

bf16 + `torch.compile` on the host encoder, falling back to eager if compilation raises, and
opt-out via `WAN5B_I2V_ENCODE_COMPILE=0`. Compilation cost lands in the constructor warmup, not
on a measured run, and a different image at the same resolution does not recompile.

---

## 4. Tooling built

Tracy op-level profiling was recorded as **blocked** on this pipeline. It is not — see §5. Until
that was found, two host-side tools did the work, and they remain the fastest way to get a
ranking without a profiled run:

- **`_ttnn_host_profiler.py`** — monkeypatches every module-level ttnn callable and times it in
  two modes. `dispatch` (no sync) measures how long the host spends *issuing*; `sync` (sync
  after every op) measures per-op device cost. Comparing the two is what identifies whether a
  section is host-bound or device-bound.
- **`test_vae_bench_ti2v_5b.py`** — decode only, ~1 min per repeat instead of a full 17s
  generation. `WAN5B_BENCH_WARMUP=0` skips the warmup generation.
- **`test_denoise_bench_ti2v_5b.py`** — the same treatment for the denoise loop.
- **`test_dup_up3d_ti2v_5b.py`** — bit-exactness gate for the VAE rewrite.
- **`test_adaln_chunk_ti2v_5b.py`** — micro-benchmark for the AdaLN six-way split.
- A `wan2_2_ti2v_5b_1xGLX` entry (`nhq=6`, `seq_len=2336`) in
  `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`, which had a 14B entry but no 5B one.

**Reading sync mode correctly.** It adds ~0.35-0.4ms per op, so high-count/low-cost ops are
inflated and must be corrected before ranking. Two ops that looked significant and were not:
`get_arch_name` showed 6.2% in sync mode but is 18.6us/call in dispatch — almost all of it was
the synchronisation itself; and `ttnn.chunk` measured 1239us sync vs 952us dispatch on a
4608-element tensor, i.e. it is host-dominated, and a captured trace pays host cost once rather
than per step. Ranking on sync alone would have overstated that one as a ~1.6s production win.
It is not one.

---

## 5. Tracy is not blocked

The 12000-marker limit is `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` — default **1000 programs**
(`profiler_state_manager.cpp:21`), budgeted at 48B/program -> 48000B/RISC -> 12000 uint32 words
(`kernel_profiler.hpp:62`). It is parsed at **runtime** (`rtoptions.cpp:143`), so raising it
needs **no host rebuild**, only an automatic JIT kernel recompile.

```bash
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000   # before the device opens
python_env/bin/python -m tracy -p -r -v -m pytest <node-ids> -sv --timeout=0
```

Result here: **0 dropped markers, 0 join assertions**, full `ops_perf_results_<ts>.csv` with
per-instance `DEVICE KERNEL DURATION`. The tell that it was always a sizing problem: the earlier
**19200 drop warnings = 120 cores x 5 RISCs x 32 devices**, i.e. every RISC on every chip —
uniform, not a hot spot.

`ttnn.ReadDeviceProfiler(mesh)` drains the buffer mid-run (one call covers all 32 chips) and is
the documented pattern for tests exceeding 1000 ops; both output CSVs append, so nothing is lost
across drains. Several repo models already do this.

Caveats: cost is ~4800B x count per DRAM bank (96MB/bank at 20000), reserved only in Tracy
builds, so start at 20000-30000 rather than 100000. **Wall-clock under Tracy is meaningless**;
`DEVICE KERNEL DURATION` is trustworthy, `OP TO OP LATENCY` is not. The
`process_ops_logs.py:683` join has **no tolerant mode**, so avoiding drops is the only route.

---

## 6. Where the time goes now

**VAE decode (1.25s device total):** conv3d **42%** — unchanged in absolute terms at 0.52s, and
independently matching an estimate derived from Teja's own swept `best_us` times invocation
counts, which is good evidence that axis is exhausted. `WanDupUp3D` is now ~0.07s.

**Denoise (~91% of the run)** is genuinely compute-bound, unlike the VAE. Host-profiler
estimate (pre-sweep, kept for the record): ring SDPA ~28 %, MMRS (ff2) ~27 %, AGMM ~17 %,
AdaLN chunk ~13 % (host-dominated), norms ~11 %.

**Tracy device ranking (2026-09-28, sprint 5; section 7.8 has the capture).** 6-block model,
720p, bf16 default, `DEVICE KERNEL DURATION` mean over 32 devices; "per bp" = per block per
CFG pass; full-model column = x60 (x30 for once-per-block ops) + fixed:

| op (input shapes) | call site | per bp | kernel us | share | full model ms/step |
|---|---|---|---|---|---|
| `RingJointSDPA` 6 x 2336 x 128 | self-attention (`attention_wan.py`) | 1 | 1,168 | 29.1 % | 70.1 |
| `AllGatherMinimalMatmulAsync` N=768 | `attn1.to_out` (+addcmul), `attn2.to_q`, `attn2.to_out` | 3 | 265 | 19.8 % | 47.6 |
| `AllGatherMinimalMatmulAsync` N=3584 | `ffn.ff1` (+gelu) | 1 | 433 | 10.8 % | 26.0 |
| `MinimalMatmulStridedReduceScatter` | `ffn.ff2` (+RS +addcmul) | 1 | 390 | 9.7 % | 23.4 |
| `AllGatherMinimalMatmulAsync` N=2304 | `attn1.to_qkv` | 1 | 300 | 7.5 % | 18.0 |
| `DitFusedDistributedRmsnorm`, bf16 weight | `norm_q`/`norm_k` (self, RoPE), `norm_q` (cross), `norm2` | 4 | 56 | 5.6 % | 13.5 |
| `DitFusedDistributedRmsnorm`, fp32 weight | `norm1`, `norm3` (+ `norm_out` x2/step) | 2 | 84 | 4.5 % | 10.2 |
| `TilizeWithValPadding` `[1,1,1,768]` fp32 | re-tiling 4 of 6 AdaLN chunks (+7/step) | 2 per block | 58 | 3.8 % | 7.4 |
| `MinimalMatmul` 512x3072x1536 | `attn2.to_kv` | 1 | 54 | 1.4 % | 3.3 |
| `SDPAOperation` 2336 x 512 | cross-attention | 1 | 50 | 1.2 % | 3.0 |
| heads split/merge (`NlpCreateHeads`, `NLPConcatHeads`) | both attentions | 4 | 11-22 | 2.4 % | 4.4 |
| `AllGatherAsync` fp32 2336x768 | `norm_out` gather (fixed) | 2/step | 278 | 1.2 % | 0.6 |
| `BinaryNg` 16x512x512 | cross-attention residual add | 1 | 25 | 0.6 % | 1.5 |
| AdaLN rest (`Untilize`, 6 `Slice`, 2 `Typecast`, 2 `BinaryNg` `1+x`) | `prepare_modulation` | ~11 per block | 1-13 | 1.0 % | 1.3 |
| timestep MLP, `proj_out`, lerp, RS | fixed per step | — | 30-260 | 1.1 % | 0.6 |

Sum **~235 ms kernel per full step vs 259.5 ms `execute_trace`**: ~25 ms/step (~10 %) of
inter-program gaps over ~1840 launches, ~13 us each. Corrections to the estimate above: ff2
is 10 %, not 27 %; the three **N=768 AGMM projections are 20 %** and run at ~41 TFLOP/s per
chip against 110-130 for qkv/ff1/ff2 (the all-gather of the 2336x3072 activation is the same
for every AGMM and 768 output columns cannot hide it); the AdaLN chunk's device cost is a
**tile/row-major round trip** (`ttnn.chunk` untilizes the `[1,1,6,768]` table+temb, and every
chunk that feeds a tile-layout consumer is re-tiled at 58 us per 3 KB tensor), not host time.
Ring SDPA moves 134 GFLOP in 1.17 ms (~115 TFLOP/s per chip, 35 % of HiFi2 peak) and is the
best-utilised op in the step.

A full read of the denoise path found **no ROW_MAJOR<->TILE padding pathology anywhere in it**,
so there is no second `WanDupUp3D` to find. Remaining wins are incremental.

**Sprint 6, after fix 1 (2026-09-28, same capture recipe, `s6_f1_blocks6`):** 45.86 ms kernel
and 328 programs per 6-block replay per device (was 48.15 ms / 404); the `TilizeWithValPadding`
and `UntilizeWithUnpadding` rows are gone and every matmul / SDPA / norm row is unchanged
(section 7.9 has the per-row comparison). Scaled to 30 blocks: ~223 ms kernel and ~1460
programs per step against the ~248 ms production `execute_trace` (255 -> 245 ms/step wall).

**Program count matters more than this table suggests (sprint 5).** Removing ~330 tiny
modulation programs per step (section 7.2) saved 6.9 ms of a 261 ms step, i.e. ~21 us per
program of pure launch cost inside the trace. At 480p (121 ms/step) the same removal was worth
9.5 % of the denoise. So the ranking above, which is by kernel time, undercounts the ~1300
programs a step still launches (~25 % of the step if all cost ~20 us): op fusion and op-count
reduction are a lever of the same order as the matmul blockings were. The Tracy capture (7.8)
would put a number on it directly via `ops per step per device` in `tracy_summarize_ops.py`.

---

## 7. Next, in order

All verified by reading the code; none implemented.

1. **Re-key the 5B matmul tables from `"11x10"` to `"12x9"`.** They are unreachable by *grid
   key*, not just by M: `get_agmm_config` looks up `agmm_worker_grid(...)` = `(12,9)` whenever
   `force_transpose=True`, which is every Wan call site. Also use M=**2336**, not the registered
   2368. This is the root cause behind `attn1.to_out`, `ffn.ff1`, `attn2.to_kv` and `proj_out`
   all landing on default blockings.

   *Keys fixed (2026-09-21):* `_register_5b_matmul_tables` now registers the eleven shapes the
   model requests — AGMM under `"12x9"`, proj_out and cross-attn `to_kv` under `"11x10"`, ff2 as
   fused MMRS — at M=2336 and M=1024. The pre-change resolution, for reference: qkv@2336 and
   to_out at both M came from the AGMM v3 rules, ff2 from the MMRS v2.3 rules, and ff1 at both
   M, qkv@1024, proj_out and to_kv were on the warned 8x8x8 default. 720p/121f (M=3424) is not
   tabled.

   *Swept (2026-09-22), all 11 shapes* — device kernel time, best of ~310-380 L1-feasible combos
   per shape, against the blocking that ran before:

   | shape | M | before | after | gain |
   |---|---|---|---|---|
   | qkv | 1024 | 160.2 us (default) | 149.5 | 6.7% |
   | qkv | 2336 | 315.3 us (v3) | 275.8 | 12.5% |
   | to_out | 1024 | 108.5 us (v3) | 108.5 | 0 — v3 was optimal |
   | to_out | 2336 | 236.0 us (v3) | 229.2 | 2.9% |
   | ff1 | 1024 | 239.3 us (default) | 203.8 | 14.8% |
   | ff1 | 2336 | 400.7 us (default) | 352.1 | 12.1% |
   | ff2 (MMRS) | 1024 | 270.5 us (v2.3) | 210.4 | 22.2% |
   | ff2 (MMRS) | 2336 | 410.6 us (v2.3) | 357.7 | 12.9% |
   | proj_out | 1024 | 35.4 us | 33.7 | 5.0% |
   | proj_out | 2336 | 74.7 us | 70.2 | 6.0% |
   | cross-attn to_kv | 512 | 58.7 us (default) | 54.7 | 6.8% |

   At 720p the pattern is one full M block per core (73 M tiles over 12 columns -> M_block 7) with
   K_block 6, and the top-5 combos per shape sit within ~1% of each other, so the winners are not
   noise picks.

   *E2E with the full table (2026-09-22, u13-43, mean of 3, all gates pass):* 720p T2V denoise
   11.436 -> **10.701s** (-6.4%; -4.8% against the 11.238s single run on this host the day
   before), total 12.51 -> **11.77s**. 480p T2V denoise 6.301 -> **5.667s** (-10.1%), total
   6.98 -> **6.34s**. 720p I2V denoise 12.072 -> **11.513s** (-4.6%), total 14.68 ->
   **13.96s**. Spreads 0.4-0.8% on denoise and total. 480p gains most because at M=1024 the
   defaults were worst (ff1 and ff2 were 15-22% off in kernel time). Text encoder, image
   encode and VAE are unchanged, as expected: none of them use these tables.

   **The sweep fills the disk.** It compiles one program per combo and the kernel JIT cache
   (`~/.cache/tt-metal-cache`) keeps every one: ~118 GB across 587k files for ten shapes, plus
   ~3 GB of profiler capture per shape. Two sweep runs died with `ENOSPC` on this box before
   that was understood. Budget ~15 GB per shape, or prune the sweep-window cache entries
   afterwards (they are unique blockings and never reused).
2. **Hoist the per-step modulation.** `combined_step` calls `inner_step` twice with the **same
   timestep**, so the timestep MLP, patch embed and every block's modulation are recomputed
   identically — ~26,400 op launches per generation, half of them exact duplicates.

   *Done, first half (2026-09-22):* `combined_step` now computes `prepare_timestep_conditioning`
   and `patch_embedding` once under CFG and hands both to the two `inner_step` passes through
   two keyword-only arguments that default to the old behaviour for every other caller. Nothing
   downstream writes into either tensor (the fused addcmul kernels return fresh outputs).
   Gate `test_cfg_hoist_ti2v_5b.py`: `max_abs_diff == 0.0` for the cond pass, the uncond pass
   and the combined output at 720p geometry; the transformer PCC suite is unchanged to four
   decimals (100.0000 / 99.9893 / 99.9894 %). Perf effect of this first half: see section 1
   (expected <= 1 %).

   *Done, second half (2026-09-26, sprint 5):* the per-block AdaLN modulation is hoisted too.
   `WanTransformerBlock.prepare_modulation(temb)` returns the six tensors the block consumes
   (`shift`, `1 + scale`, bf16 `gate`, and the FFN triple) with exactly the ops `forward` ran
   inline before, in the same order; `forward(modulation=)` takes them, and `combined_step`
   computes the 30 tuples plus the `norm_out` pair (`prepare_norm_out_modulation`) once per step
   under CFG and hands them to both `inner_step` passes. Every other caller (untraced `forward`,
   `inner_step` without CFG, the 14B) computes them inline as before. Gate
   `test_cfg_hoist_ti2v_5b.py` now passes the hoisted modulations explicitly and still asserts
   `max_abs_diff == 0.0` for the cond pass, the uncond pass and `combined_step`; the PCC suite is
   unchanged (100.0000 / 99.9893 / 99.9894 %) and `test_trace_modes_ti2v_5b` stays bit-identical.

   Measured, 720p T2V, 81 f / 40 steps, 2cq, host u13-43, same hour:

   | | denoise | total |
   |---|---|---|
   | sprint-4 file, control run (1 run) | 10.473 s | 11.535 s |
   | sprint-4 recorded mean of 3 (2026-09-24/25) | 10.434 s | 11.50 s |
   | **hoisted, mean of 3** (10.178 / 10.236 / 10.179; spread 0.6 %) | **10.198 s** | **11.228 s** |

   **-2.6 % denoise / -2.7 % total against the control, 6.9 ms per step** -- far above the
   "<= 1 %" this item was booked at. The reason is instructive: the hoist removes ~330 tiny
   device programs per step (30 blocks x (1 add + 6-way chunk + 2 typecasts + 2 scalar adds), one
   CFG pass' worth), so each removed program was worth ~21 us of device time inside the trace,
   which is launch/dispatch cost, not math. The remaining once-per-step copy of the same ~330
   programs is therefore worth another ~6-7 ms/step (2.5 %) if it can be removed -- see item 9.

   The other two geometries, mean of 3 against the sprint-4 means (section 1): 720p I2V denoise
   11.073 -> **10.580 s** (-4.5 %), total 13.49 -> **12.90 s**; 480p T2V denoise 5.350 ->
   **4.842 s** (-9.5 %, 12.7 ms/step), total 6.03 -> **5.49 s**. I2V gains twice as much per step
   as T2V because its modulation is per-token (7 MB fp32 tensors per slice instead of 3 KB), and
   480p gains most in relative terms because its step is half as long. Teja's 121 f generate
   passes with CLIP mean 40.69 (min 39.84 / max 41.21; bf16 before the hoist 40.38 -- the output
   is bit-identical, the CLIP spread is the gate's own); mid-frame preview
   `/home/ttuser/wan5b_s5_hoist_t2v_720p_121f_mid.png` is sharp and coherent. The generate
   test's mp4 export fails with `No module named 'imageio'` in this venv (also in the sprint-4
   bf8 generate log), so only the PNGs land; the demo CLI writes mp4s through a different path.
   The `+1` fold (item 3) is now only 60 ops/step (~1.2 ms) and costs bit-exactness, so it
   stays undone.
3. **Fold `+1.0` into `scale_shift_table`.** `1 + (table+temb) == (table+1) + temb` exactly;
   removes 4,800 ops per generation at zero numerical cost.
4. **Change the AdaLN split layout.** Measured bit-exact on the production shape: current
   `chunk` dim2 TILE ~495us, dim3 TILE flat ~194us (2.5x), dim2 ROW_MAJOR ~130us (3.8x).
   Host-dominated, so this helps untraced runs and trace capture, not the traced steady state.
5. **Trace the UniPC solver step, or stop blocking on `execute_trace`.** The solver runs outside
   the traced region while `execute_trace` blocks, so ~760 host-dispatched launches per
   generation happen with the device idle.

   *Measured and dropped (2026-09-22, `test_step_gap_ti2v_5b.py`, 720p T2V, 40 traced steps,
   stats over steps 2-39):*

   | per step | ms |
   |---|---|
   | wall (production timeline) | 266.35 |
   | `execute_trace` blocking | 258.52 |
   | solver host dispatch | 5.89 |
   | solver device tail after dispatch (sync) | 0.69 |
   | host-only gap = wall - trace - solver device | **1.37** (0.5 %) |

   The idle window a non-blocking trace could close is 1.4 ms/step strictly, 7.8 ms/step at
   the outside (if the solver's dispatch-bound 6.5 ms overlapped completely). Both are under
   the 10 ms/step bar set before measuring, so the ceiling is < 3 % and the realistic gain
   ~1 %. Not built; the bench stays so the number can be re-taken if the trace gets shorter.

   *Revisited and built anyway (2026-09-24), as an opt-in first.* `Tracer` gained
   `tracer_blocking_execution` and `tracer_input_cq_id` (`models/tt_dit/utils/tracing.py`): the
   per-call input copies go on a second command queue fenced with events (the input queue waits
   for the previous execution, the compute queue waits for the copies), the same scheme as the
   tt_cnn pipelines. `WanPipeline.configure_trace_execution(blocking, input_cq_id)` selects the
   mode per pipeline and `WAN5B_TRACE_MODE=blocking|nonblocking|2cq` drives the perf tests and
   the demo (`2cq` adds `num_command_queues=2` to the device params). Gate
   `test_trace_modes_ti2v_5b.py`: the same captured trace in all three modes is **bit-identical**
   (`torch.equal` on the latents) and two queues leave the compute grid at **12x10**, the same as
   one queue — the first version of the gate expected 11x10 and failed; that key is a derived cap
   for the matmul tables, not the device grid. The gate's own 8-step wall deltas are noise-level
   (-3.8 / -6.5 ms/step nonblocking / 2cq on 2026-09-24, -6.0 / -3.0 on 2026-09-25): use the
   full runs, not the gate, for the perf number. Full runs, mean of 3 (2026-09-24):

   | 81 f / 40 steps | blocking denoise / total | 2cq denoise / total | delta |
   |---|---|---|---|
   | 720p T2V | 10.747 / 11.825s | **10.434 / 11.498s** | -2.9% / -2.8% (7.8 ms/step) |
   | 480p T2V | 5.649 / 6.325s (1 run) | **5.350 / 6.030s** | -5.3% / -4.7% |
   | 720p I2V | not re-run blocking | **11.073 / 13.493s** | -3.8% / -3.3% vs sprint 3 |

   **2cq is the default since `b5dd469bd09`.** The measured gain sits at the *outside* bound of
   the estimate above (7.8 ms/step), not the strict one (1.4 ms): the solver's dispatch-bound
   6.5 ms does overlap once the compute queue no longer waits for the host. The strict figure
   was the wrong one to gate on; the bar of 10 ms/step still would not have been met, so the
   decision to build it was a judgement call that paid ~3%. `WAN5B_TRACE_MODE=blocking` restores
   the single-queue path.
6. **bf8 for the ring-SDPA K/V gather.** ~120GB/device/generation crosses the SP fabric in bf16
   and the `bfloat8_b` path already exists but is only enabled by a `QuantConfig` no 5B pipeline
   applies. Overlapped with compute, so precision-gate it and expect only what the fabric is
   actually binding.

   *Preset exists, isolated effect not measured (2026-09-25).* `bf8_weights_sdpa_bf8` is
   `all_weights_bf8` plus bf8 SDPA inputs at HiFi2. It has not been run on its own; the bf8 K/V
   path is exercised and gated as part of `all_bf8_lofi` (item 7), whose SDPA is bf8 HiFi2.

7. **`QuantConfig` presets on the 5B (opt-in, 2026-09-22).** `WAN5B_QUANT_CONFIG=<preset>` now
   applies a `QuantConfig` preset in the two 5B perf tests, the generate test and the
   transformer-vs-torch PCC tests (`quant_config.py: set_quant_config_from_env`). The perf tests
   re-run the eager warmup after applying it, because trace capture cannot compile the programs
   a new dtype or fidelity needs. The default is unchanged.

   All rows 2026-09-25 unless noted, 720p T2V, 2cq default, perf = mean of 3 (bf8 is mean of 2:
   the third run was interrupted). CLIP is Teja's 121 f generate, seed 42, gate 36.00, bf16 40.38.

   | preset | transformer PCC scalar / per-token (bf16: 99.9893 / 99.9894) | CLIP mean (min / max) | 720p T2V denoise / total (default 10.434 / 11.50) |
   |---|---|---|---|
   | `all_weights_bf8` | 99.9885 / 99.9885 % | 40.20 (38.65 / 41.67) | 10.048 / **11.14s** (-3.7% / -3.1%) |
   | `bf8_weights_sdpa_bf8` | not measured on its own | — | — |
   | `all_bf8_lofi` | **99.9651 / 99.9651 %** (-0.024 pp) | **41.34** (39.73 / 42.51) | **8.914 / 9.95s** (-14.6% / -13.5%) |
   | `all_lofi` (2026-09-22) | **hangs the device** in the first LoFi matmul (self-attn QKV; host blocked in `synchronize_device` inside `get_fused_norm_stats_buffer`, `wait_for_outstanding_reads`); needed `tt-smi -glx_reset_auto` | — | — |
   | `all_bf8_lofi_sdpa_lofi` | **hangs the device** in the scalar-timestep PCC test (4 min without output, `py-spy dump` captured no model frame within 60 s, the process needed SIGKILL and chip 0 then failed FW init until a `-glx_reset_auto`) | — | — |

   Read the two hangs together: LoFi matmuls with **bf8 operands** run (that is every matmul in
   `all_bf8_lofi`), LoFi matmuls with **bf16 operands** hang (`all_lofi`), and the ring SDPA at
   LoFi hangs even with bf8 inputs. Both hangs reproduce on the first affected op, so a preset
   that is going to hang does so inside the 20 s PCC test, not in a 4-minute perf run: gate new
   presets there first, with a watchdog, and expect to reset the box. The SDPA at bf8 HiFi2 is the
   floor for that op until someone debugs the LoFi ring-SDPA kernel.

   *Re-measured on the sprint-5 tip (2026-09-28, mean of 3):* denoise **8.583 s**, total
   **9.61 s** (9.598 / 9.620 / 9.608), i.e. the modulation hoist and the preset stack
   (-0.34 s on top of the preset's own -1.55 s); -16.5 % against the sprint-5 bf16 default.

   *Sprint-6 tip (2026-09-28, fixes 1 and 3, mean of 3, 720p T2V, operator request).* Two
   things changed for the preset (section 7.13): `cross_attn_out` must keep bf16 weights (the
   fused residual's ternary kernel asserts the residual and weight tile formats match), and with
   the residual add inside a LoFi epilogue the transformer PCC fell to 99.8863 / 99.8848 %.
   Measured in that state first: denoise **8.328 s**, total **9.383 s** (9.372 / 9.381 /
   9.396, spreads 0.09 / 0.26 %), -3.0 % / -2.4 % against the 9.61 s of the sprint-5 tip (the
   AdaLN fix stacks with the preset as the hoist did). The preset then pins `cross_attn_out`
   to HiFi2 with fp32 accumulate, and the fused cross-attention residual was dropped (7.13).
   With that final definition on the tip: transformer PCC **100.0000 / 99.9600 / 99.9600 %**
   (two-row-vs-scalar / scalar / per-token) -- 0.005 pp under the 99.9651 % recorded for the
   sprint-5 preset even though the only remaining difference is a *more* precise
   `cross_attn_out` (bf16 weights, HiFi2, fp32 acc instead of bf8 LoFi). A same-hour PCC run
   with the sprint-5 preset file swapped back in reproduces **99.9651 %** exactly, so the
   0.005 pp is the `cross_attn_out` change itself (a more precise layer lowering the PCC
   against the fp32 reference: error cancellation between bf8 layers, not a bug). The change is
   kept at the operator's request; it is no longer *required* now that the fused residual is
   gone, it buys no measurable speed (the projection is bandwidth-bound either way), and
   reverting it is the one-line `cross_attn_out=lc` in `all_bf8_lofi`. 720p x3 under
   the final definition (2026-09-29 00:00, 2cq): denoise **8.250 s** (8.217 / 8.258 / 8.274,
   spread 0.7 %), totals 9.666 / 9.316 / 9.458 s -- mean 9.48 s, median 9.46 s. Run 1's total
   carries a VAE decode of 1.33 s (the other two: 0.95 / 1.07 s), a 40 % outlier of the kind
   section 8 warns about, and it tripped the 1.2 s VAE gate while its denoise was the fastest of
   the three; read **9.32-9.46 s** as the clean total. Against the sprint-5 preset (8.583 /
   9.61 s): **-3.9 % denoise**, 8.3 ms/step (the AdaLN fix is a fixed per-block cost, so it is
   worth less per step where the step is shorter). Against the first pass with the LoFi
   epilogue (8.328 s): -0.9 % denoise, i.e. HiFi2 on `cross_attn_out` is indeed free. Against
   the sprint-6 bf16 default (9.787 / 10.835 s): -15.7 % denoise. The demo CLI on the same tip
   (81 f, seed 42, the perf-test prompt): warm traced 10.85 s bf16, **9.32 s** preset.

   *Visual verdict (operator, 2026-09-29): bf16 stays the default; the preset stays opt-in.*
   Side-by-side `/home/ttuser/wan5b_s6_ab_bf16_left_bf8lofi_right.mp4` (bf16 left). What the
   numbers say about the two clips: they are **different samples, not a degraded copy** --
   PSNR 17.9 dB / SSIM 0.78 between them (a trajectory divergence over 80 forwards), while
   per-clip sharpness is identical (Laplacian variance 310.6 vs 309.4, high-frequency energy
   share 41.12 vs 41.11 %). The operator saw a slight quality loss on this sample (glove logo,
   face detail), which is consistent with "different sample" rather than systematic softening.
   Judge the preset like a seed change with 2.4x the per-step error (relative RMSE 4.1 % vs
   1.7 % against fp32). `all_weights_bf8` (bf8 weights, HiFi2 everywhere, PCC 99.9885 %, ~-3 %)
   is the untried middle rung if a default change is wanted later.

   `all_bf8_lofi` is the sprint-4 result: -15.5% end to end at 720p against the sprint-3 tip, a
   0.024 pp PCC cost and a CLIP mean *above* bf16 (which says nothing about quality, section 8).
   The 121 f previews `/home/ttuser/wan5b_t2v_720p_bf8lofi_{first,mid,last}.png` and the mp4
   next to them are sharp and identity-stable to the eye; the decision to make it the default
   is a visual one against `/home/ttuser/wan5b_demo_t2v.mp4` (bf16) and has not been taken —
   it stays opt-in. If it is adopted, re-run 480p and I2V under it (neither has been measured
   with any preset) and recalibrate the three gate functions.

   `all_weights_bf8` as shipped asserted `ternary_a_tile_size == in1_tile_size` in the fused
   AGMM+addcmul kernel: the residual is a bf16 activation and must match the weight tile
   format, which `all_bf8_lofi` already accounted for by keeping `self_attn_out` at bf16. The
   preset now does the same (qkv, cross-attn q/kv/out, ff1, ff2 go to bf8). The bf8 preview
   frames (`/home/ttuser/wan5b_t2v_720p_bf8_{first,mid,last}.png`) are visually clean: sharp,
   coherent, stable identities across 121 frames.

8. **Tracy validation capture of the traced step (checklist item).** Attempted three times on
   2026-09-25 with the block in `Wan2_2_TI2V_5B_checklist.md`; every attempt died with
   `OSError: [Errno 28] No space left on device`. A Tracy run JIT-recompiles every kernel with
   profiler markers (thousands of `riscv-tt-elf-g++` invocations into `~/.cache/tt-metal-cache`)
   and streams the device log into `generated/profiler/.logs/`, and the root disk had 11-12 GB
   free. Not a code blocker; the per-step device time is already known from the blocking
   `execute_trace` measurement (258.5 ms of 266.4 ms, item 5).

   *Sprint 5 (2026-09-25/26): disk solved, host RAM is the real wall.* Three more attempts on
   `test_step_gap_ti2v_5b` (`WAN5B_GAP_PASSES=A`), with `-o /mnt/tt-data/nkira/profiler/<tag>`
   (sets `TT_METAL_PROFILER_DIR`) and `build/profiler/build_wasm/traces` symlinked onto NFS:

   | attempt | setup | outcome |
   |---|---|---|
   | 1 | 20 steps, `ReadDeviceProfiler` drain every 4 steps, count 30000 | `No available port found`: `tools/tracy/__init__.py:get_available_port` binds to `gethostbyname(hostname)`, which resolves to an IP this host does not own (10.81.14.43 vs 10.82.97.43). Fixed with `-t 8086`. |
   | 2 | same | drains took **321 s and 230 s** each (32 chips, most behind ethernet); the host pushes one Tracy zone per device marker, 1.2 G zones, pytest RSS 504 GB -> **OOM-killed** (host has 566 GB). |
   | 3 | 8 steps, no drains, `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1` | 0 marker drops, all 8 traced steps ran (264.3 ms wall, 259.5 ms `execute_trace`: production numbers, so the traced path is not distorted under Tracy); then **32 min in the end-of-run device read + C++ post-process** (`TT_METAL_PROFILER_CPP_POST_PROCESS`, on by default in `-r`) until the job was stopped for host memory pressure before any CSV was written. |

   Budget **programs, not steps**: a 720p step is ~1840 programs (Tracy count, 30 blocks), and
   the eager compile run plus the trace capture (2 steps each) already cost ~7400, so attempt
   3 held ~20k programs x 120 cores x 5 RISCs x 32 chips of markers on the host. The tooling
   for a capture under ~10k programs: `WAN5B_GAP_BLOCKS=n` truncates the transformer to n
   blocks (every block is the same op stream at the same shapes, so per-op numbers scale to 30
   exactly), `WAN5B_GAP_PROFILER_DRAIN` is there but is not a fix (see the drain times), and
   `tracy_summarize_ops.py --traced-only` ranks by `OP CODE` and by input shapes/dtypes, which
   is what separates the three AGMM call sites without a Python stack.

   *Attempt 4 succeeded (2026-09-28)* with the command below: 6 blocks, 4 traced steps, 0
   dropped markers, 404 programs per replay per device, **48.15 ms kernel per replay (min
   47.7, max 48.5 over 32 chips)** against 50.6 ms `execute_trace` / 56.6 ms wall, peak host
   RSS **246 GB** for this reduced capture (the full model would need ~1.2 TB; do not try).
   Two wrapper quirks: it prints `No profiling data could be captured` when the capture tool
   takes more than 15 s to save the `.tracy` (it did, 1.2 GB), while the file is in fact
   written -- run `python -m tracy --process-logs-only -r -o <dir>` afterwards (23 min: 37 GB
   host-side CSV export, the join, then it copies the 44 GB device log into `reports/`); and
   the report's shape columns are `padded[logical]` strings, which the summariser now keeps
   verbatim. The ranking is in section 6; artefacts in `/mnt/tt-data/nkira/profiler/blocks6/`.
   Command:

   ```bash
   free -g   # need a few hundred GB free
   WAN5B_GAP_BLOCKS=6 WAN5B_GAP_STEPS=4 WAN5B_GAP_PASSES=A TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 \
     python -m tracy -p -r -v -t 8086 -o /mnt/tt-data/nkira/profiler/blocks6 --op-support-count 30000 \
     -m pytest "models/tt_dit/tests/models/wan2_2/test_step_gap_ti2v_5b.py::test_step_gap_ti2v_5b[blackhole-bh_4x8]" -sv --timeout=0
   python models/tt_dit/tests/models/wan2_2/tracy_summarize_ops.py <reports>/ops_perf_results_<ts>.csv --steps 4 --traced-only
   ```

9. **Remove the once-per-step modulation cost (sprint-5 lead, Tracy-sized, not built).** After
   item 2 the traced step still runs the 30 blocks' modulation once, and Tracy (section 6) says
   what it costs and why: `ttnn.chunk` on the tile-layout `[1,1,6,768]` fp32 `table + temb`
   **untilizes** it (13 us), slices six `[1,768]` rows in row-major (1 us each), and the four
   chunks that feed tile-layout consumers (`1 + scale`, `1 + c_scale`, the two bf16 gate
   typecasts) are **re-tiled by `TilizeWithValPadding` at 58 us per 3 KB tensor**; only the
   two `shift` chunks stay row-major into the norm's bias. Per full step: 120 tilizes = 7.0 ms,
   plus ~1.3 ms for the rest, plus ~11 launches x 30 blocks of gaps -- **~8-12 ms/step, 3-5 %**,
   and more at 480p. Fix without new kernels: chunk the timestep projection *before* the
   `unflatten`, i.e. slice the `[1,1,1,6*768]` tile tensor at 768-column boundaries (tile
   aligned, no layout change), once per step; pre-split each block's table into six
   `[1,1,1,768]` tile constants at load time; per block do six tile-aligned adds, the two gate
   adds with `dtype=bfloat16` so the typecast disappears. Bit-exact for shift/scale/gates if
   `1 + x` stays a separate op (6 + 2 adds per block, all ~4 us); folding the `+1` into the
   table rows saves the last two but changes fp32 rounding. The per-token (I2V) arm already
   chunks on the feature axis and needs only the pre-split tables.

   The longer-range version -- compute all 40 steps' modulation once per generation and feed it
   to the trace -- still needs a zero-copy view or a row-offset argument on the norm kernel
   (180 tensors per step otherwise), and is now worth only the ~2 ms of adds the fix above
   leaves.

   *Built (2026-09-28, sprint 6, commit "AdaLN modulation without the tile/row-major round
   trip").* Exactly the design above, with two details worth knowing. The six table rows are
   derived **on device from the unchanged `[1,1,6,dim]` Parameter** (`WanTransformerBlock.
   _split_table`, a dim-2 `ttnn.slice` per row) rather than stored as new Parameters, so the
   weight cache's `.tensorbin` files and the 14B's cache are untouched; they are created eagerly
   after every load (`WanTransformer3DModel.prepare_modulation_constants`, called from `load` and
   `load_torch_state_dict`) and dropped in `deallocate_weights`, because their first creation
   goes through the untilize / row-major slice / re-tilize fallback and must never happen inside
   a trace capture. The flat `[1,1,1,6*D/tp]` projection is taken straight from the embedder
   (`prepare_timestep_conditioning(..., flat_proj=True)`, which also drops the per-step
   `unflatten` reshape) and cut once per step by `split_timestep_proj`: `begins[-1] % 32 == 0`
   and `begins[-2] == 0`, so `ttnn.slice` stays on its tile program factory (a NOC tile copy).
   The gates come out as bf16 straight from `ttnn.add(..., dtype=bfloat16)`: binary_ng keeps
   the fp32 sum in the fp32 destination and appends the same `typecast_tile<Float32,Float16_b>`
   LLK the standalone `ttnn.typecast` runs, which is why this is bit-identical and not merely
   close. `1.0 + scale` stays a separate op. The legacy `prepare_modulation` remains the inline
   path (untraced `forward`, `inner_step` without CFG, the 14B block unit test) and the reference.
   The per-token (I2V two-row) arm needed nothing beyond the same code: the flat `[1,1,N,6*D/tp]`
   projection is sliced the same way and each block does six `[1,1,1,D/tp] + [1,1,N,D/tp]`
   broadcast adds, which also removes the per-block table reshape and the six 1.2 MB slice
   copies per block the old arm paid.

   Gates: `test_cfg_hoist_ti2v_5b.py` now compares legacy vs split **per tensor** (six per block
   plus the two `norm_out` ones) as well as per pass and for `combined_step`, for the scalar and
   the two-row timestep: all 34 comparisons `max_abs_diff == 0.0`. Transformer PCC 100.0000 /
   99.9893 / 99.9894 % (unchanged), trace modes bit-identical, the 14B 4x8 ring model and
   inner_step tests pass (PCC 99.9886 %; their D/tp is 1280, also tile aligned).

   Measured, 81 f / 40 steps, 2cq, host u13-43, mean of 3 (section 1 has the run lists):

   | | denoise | total | per step |
   |---|---|---|---|
   | 720p T2V, sprint-5 file, same-hour control (1 run) | 10.201 s | 11.262 s | |
   | **720p T2V, fix 1** | **9.787 s** (-4.1 %) | **10.835 s** (-3.8 %) | -10.3 ms |
   | 480p T2V (vs sprint-5 mean 4.842 / 5.49) | **4.438 s** (-8.3 %) | **5.137 s** (-6.4 %) | -10.1 ms |
   | 720p I2V (vs 10.580 / 12.90) | **10.260 s** (-3.0 %) | **12.523 s** (-2.9 %) | -8.0 ms |

   Above the 8-12 ms/step booked: the removed kernels (120 tilizes x 58 us = 7 ms, untilizes,
   slices, typecasts ~1.3 ms) plus ~11 launches x 30 blocks x ~13 us ~ 4 ms account for it.
   Tracy (6-block capture `/mnt/tt-data/nkira/profiler/s6_f1_blocks6/reports/2026_09_28_21_15_41/`,
   4 traced steps, 0 drops, `--steps 6 --traced-only`): **45.86 ms kernel and 328 programs per
   replay per device** (sprint 5: 48.15 ms, 404), i.e. -2.3 ms kernel and -76 programs per
   6-block replay, x5 for the 30 blocks = ~11.5 ms kernel per step plus the launch gaps of 380
   programs -- consistent with the 10.3 ms/step measured end to end (the capture's
   `execute_trace` went 50.6 -> 48.4 ms). `TilizeWithValPadding` and `UntilizeWithUnpadding` no
   longer appear; the split is 6 `Slice` per step at 1.7 us each and the modulation adds are in
   the `BinaryNg` row (65 calls per replay, 6.8 us mean, 0.44 ms). Every other row is unchanged
   to within 0.5 us (N=768 AGMM 264.2 us, SDPA 1165 us, ff1 433 us, ff2 389 us, qkv 300 us).

12. **The two-row expansion slices a misaligned tile row inside the traced step (sprint-6 lead,
    not built).** `_expand_two_row` cuts row 1 of the `[1,1,2,W]` embedder output with
    `ttnn.slice(rows, [0,0,1,0], ...)`, and a dim-2 start that is not a multiple of 32 takes
    `slice`'s row-major fallback (untilize, RM slice, `TilizeWithValPadding`) -- the same
    round trip fix 1 removed from the blocks, here on two tensors per I2V step (`[1,1,2,4608]`
    and `[1,1,2,768]`, fp32). Running the embedder once per distinct timestep value (two M=32
    MLP passes instead of one) or slicing after a flat reshape removes it; worth ~0.2-0.3 ms per
    I2V step, so low priority.

11. **Small-N AGMM projections as fused matmul + reduce-scatter (sprint-5 lead from Tracy).**
    `attn1.to_out`, `attn2.to_q` and `attn2.to_out` are 2336 x 3072 x 768 per device and cost
    265 us each -- ~41 TFLOP/s per chip, a third of what qkv (110), ff1 (119) and ff2 (132)
    reach -- because each all-gathers the same 2336 x 3072 activation over the TP ring and 768
    output columns cannot hide that behind compute. Together they are 20 % of the step
    (47.6 ms). All three outputs are D-fractured on TP, which is what `ff2`'s
    `MinimalMatmulStridedReduceScatter` produces from an un-gathered K-fractured input, so the
    experiment is a micro-benchmark of that op at (2336, K=768 per device, N=3072 -> 768
    scattered) against the AGMM at the same shape. 100 us saved per call is 18 ms/step (7 %).
    `to_out` also carries the fused addcmul, which the MMRS op already supports for ff2.

    *Measured and dropped (2026-09-28, sprint 6).* `sweep_mm_block_sizes.py` gained
    `mmrs_attn` / `mmrs_attn_noadd` use cases (attention compute config, `math_approx_mode=True`,
    addcmul optional) and the `(M, 768, 3072)` rows; swept at M=2336 on the 4x8 Galaxy
    (`bh_4x8_sp1_tp0`, 2 links, ~350 L1-feasible combos each, kernel JIT cache on
    `/mnt/tt-data/nkira/tt-metal-cache`), with the AGMM `to_out` row re-run in the same session
    as the control:

    | form | best blocking | kernel us | vs AGMM control |
    |---|---|---|---|
    | AGMM `to_out` + addcmul (today) | 12x9, (8, 6, 3, (4, 1)) | **228.8** (229.2 on 2026-09-22) | -- |
    | MMRS + addcmul (`attn1.to_out` candidate) | 12x8, `FusedMMRSConfig(6, 3, 12, 2, 2)` | 225.1 | -3.7 us (-1.6 %) |
    | MMRS, no addcmul (`attn2.to_q` / `to_out` candidate) | 12x8, `FusedMMRSConfig(6, 3, 10, 2, 2)` | 210.7 | -18 us (-8 %) |

    Under the bar set beforehand (> 50 us per call). Scaled by the sweep-to-production ratio
    (229 -> 265 us for the AGMM), the whole switch would be worth ~2.5 ms/step (~1 %) for a
    PCC change and a second weight layout. **The measured reason:** the strided reduce-scatter
    moves the same 2336 x 3072 bf16 partial sums (10.8 MB per device per call) that the
    all-gather moves, and both land at ~41 GB/s effective over the TP ring's 2 links -- the
    three N=768 projections are **fabric-bound, not fill-bound**; ff1 (N=3584) hides the same
    gather behind 5x the FLOPs. Halving the bytes is the lever that follows: bf8 activations on
    the gather input of these three projections (the `all_bf8_lofi` preset intends exactly
    that, but `QuantConfig.activation_dtype` is never applied by `_apply_linear_config`; see
    `~/nkira/Wan2_2_TI2V_5B_sprint7_levers.md`). That is a precision change and belongs with the
    preset, not the bf16 default; at bandwidth-bound 265 us per call it is worth up to ~130 us
    x 180 calls = ~23 ms/step (9 %) under the preset, PCC-gated.

    Kept in the tree, default off: `WanPipelineConfig.small_n_projection = "agmm" | "mmrs"`
    (plumbed like `sdpa_chunk_size_overrides` down to `WanAttention`), the row-parallel
    `forward_fused_addcmul` without addcmul, a `_smallN-mmrs` weight-cache subfolder suffix,
    and the two swept blockings registered under `ttnn.CoreCoord(12, 10)` in
    `_register_5b_matmul_tables`. Not gated end to end (PCC / perf) since it is not enabled;
    the sweep is the only measurement.

13. **Launch-count trims along the block (sprint 6, fix 3).** With fixes 1 and 2 settled, a
    block-pass launches 5 fused norms, 6 matmul programs, 2 SDPAs, 2 `nlp_create_qkv_heads`,
    2 `concatenate_heads` and 1 `BinaryNg` (the cross-attention residual add), plus the
    once-per-block modulation adds. What can go:
    - *Cross-attention residual into `attn2.to_out`* -- built. The fused matmul epilogue is
      `residual + scalar * (xW + b) * gate` with `gate` allowed as a `[1, D/tp]` row broadcast
      (checked in both the AGMM and the MMRS validators), so a constant row of ones
      (`WanAttention.residual_ones_gate`, created in `__init__` so it exists before any trace
      capture) turns it into the plain residual add and the separate `BinaryNg` (25 us + a
      launch, 60 per step) disappears. Not bit-exact: the fused form rounds once at the
      epilogue where the old path rounded the matmul output to bf16 and then added in bf16.
      PCC-gated; measured below.

      *Gates (2026-09-28):* transformer PCC **100.0000 / 99.9901 / 99.9902 %** (from 99.9893 /
      99.9894: the single rounding is slightly kinder), `test_cfg_hoist_ti2v_5b` still 34/34 at
      0.0 (both its paths use the fused add), trace modes bit-identical, 14B 4x8 ring block test
      99.9958 %. *Measured, 720p T2V, mean of 3, 2cq, same host:* denoise **9.776 s** (9.781 /
      9.768 / 9.779, spread 0.14 %), total **10.837 s** (spread 0.04 %), against fix 1's 9.787 /
      10.835 s, and against a same-hour control of the fix-1 files (9.780 / 10.883 s) it is
      **-0.04 % / -0.4 %: inside the run-to-run spread**, not the ~2.3 ms booked. The removed
      program cost ~25 us + a launch, but the fused epilogue reads the 3.6 MB residual inside an
      op that is already bound by the ring bytes it moves, so the saving comes back as AGMM
      time: the 6-block Tracy capture with the fusion (`s6_f3_blocks6`, 4 traced steps, 0
      drops) ran `execute_trace` at **49.24 ms per replay against 48.41 ms** for the fix-1
      capture taken the same way two hours earlier. **Dropped (operator's call, 2026-09-28):** a
      numerics change (PCC moves, preset PCC fell to 99.886 % until `cross_attn_out` went to
      HiFi2) for no measurable speed. The code is in the branch history for reference
      (`WanAttention.residual_ones_gate`, the `attn2(..., addcmul_residual=, addcmul_gate=)`
      call); the ones-gate mechanism itself works and is the way to fuse an ungated residual if
      a later kernel makes the epilogue free. The capture's ops report needed a second
      post-process pass (the first hit a 90 min watchdog in the host-side csvexport step; the
      retry with a 3 h budget took ~70 min; `reports/2026_09_29_00_06_07/`). Per replay per
      device, fix 3 vs fix 1: **46.02 ms kernel and 316 programs vs 45.86 ms and 328**: the 12
      residual adds are gone (`BinaryNg` 0.44 -> 0.14 ms) but the N=768 AGMM row went from
      264.2 to 268.7 us mean over its three call sites (+13 us on the fused `attn2.to_out`,
      i.e. the epilogue's residual read is not hidden) and the ring SDPA read 1188 vs 1165 us
      (run-to-run). Net +0.16 ms kernel per replay: the trim is fully absorbed. The sprint-6 tip
      is fix 1, so the `s6_f1_blocks6` table in section 6 is the tip's Tracy table.

      *Preset interplay (2026-09-28):* under `all_bf8_lofi` the fused `attn2.to_out` inherits the
      preset's LoFi / no-fp32-acc compute config, and the transformer PCC fell from 99.9651 % to
      **99.8863 / 99.8848 %** (scalar / per-token): the residual add now happens inside a LoFi
      epilogue instead of a separate bf16 add. Also, like `self_attn_out`, `cross_attn_out` must
      keep bf16 weights under the presets (the ternary kernel asserts residual and weight tile
      formats match). The preset therefore pins `cross_attn_out` to bf16 weights **at HiFi2 with
      fp32 accumulate** (section 7.7), which the bandwidth-bound projection should absorb for
      free; PCC and 720p x3 under that setting are in section 7.7 / 9.
    - *Head split / merge* -- cannot go without a kernel change. q and k already leave
      `dit_fused_distributed_rmsnorm` in `[1, H_local, N, 128]` heads layout, so what remains is
      one `nlp_create_qkv_heads` on V and one `concatenate_heads` per attention (4 per
      block-pass, 11-22 us each, ~4.4 ms/step). `ring_joint_scaled_dot_product_attention`
      validates 4-D `[B, H, N, E]` TILE inputs (`ring_joint_sdpa_device_operation.cpp:474-532`,
      padding only on the sequence dim) and so does the cross SDPA, and a TILE `[1, N, H*E]`
      -> `[1, H, N, E]` permute is a real data movement, not a view. Left as is; an SDPA that
      consumes the concatenated-heads layout directly, or a V projection that emits heads
      layout, would remove 240 launches per step.
    - *`+1` fold into the table rows* -- not done (bf16/fp32 rounding change for ~60 tiny adds
      per step, ~1.2 ms; see item 3).

10. **Same-hour controls are cheap and worth it.** The hoist's 3-run mean beat the recorded
    sprint-4 mean by 2.3 %, and a single control run of the sprint-4 file in the same hour
    (10.473 / 11.535 s) sat 0.4 % above that recorded mean, so the improvement is 2.6 % against
    the control. `run_control.sh`-style swaps of one file in the checkout take four minutes and
    settle the drift question before it is asked.

---

## 8. Things that will waste your time if you do not know them

- **The CLIP gate is a prompt-similarity check, not a quality gate.** Its threshold is
  calibrated to the test's default prompt. A visibly excellent cyberpunk render scored 34.24 and
  "failed" the 36.00 threshold purely for using different wording, while Teja's own prompt
  scores 40.38. Never use it to judge a change. `WAN5B_CLIP=0` disables it.
- **Two VAE gates cannot gate VAE shortcut work.** `test_wan_decoder_production_blocking` and
  `test_wan_decoder_chunked_consistency` both build `is_residual=False`, so neither contains a
  `WanDupUp3D`. The whole VAE unit suite uses the **A14B** VAE (`VAE_MODEL_NAME`), so the 5B
  residual decoder has no unit coverage at all — only `test_vae_chunk_pcc_ti2v_5b` and the
  pipeline tests reach it.
- **The VAE noise floor of 2.99% is stale.** It was calibrated when the section was 2.46s; the
  ~1s section now spreads 15-20% run to run. Recalibrate before judging any VAE change.
- **`pytest.ini` pins `timeout = 300`.** Long device runs need `--timeout=0`. A run killed by
  pytest-timeout during a cold JIT cache warmup looks exactly like a code failure.
- **Pre-commit's black reformats and then fails the commit.** The commit does not land; re-stage
  and re-run. Check `git log`, not the tail of the hook output.
- **`-k "and not i2v"` deselects everything**, because `ti2v` contains `i2v`. Use node ids.
- **`pgrep`/`pkill -f` match their own command line.** An `until` loop polling
  `pgrep -f 'pytest.*performance_wan'` spins forever matching itself.
- **full-T VAE decode OOMs at 81f** (`bank_manager.cpp:462`), so `vae_t_chunk_size=7` is
  justified beyond the 121f case that motivated it.
- **Teja's `test_pipeline_ti2v_5b` fails on `assert pipeline.transformer.ffn_dim`.**
  Pre-existing and unrelated to any of this: `ffn_dim` is set on `WanTransformerBlock`
  (`transformer_wan.py:56`), never on `WanTransformer3DModel`, identically at `8d596eb242d`.
  The asserts before it pass, so the 121f pipeline builds and warms up fine.
- **`flow_shift` is the 14B value (12.0)** overriding this checkpoint's own 5.0, so quality
  comparisons against upstream are currently invalid. A controlled A/B found 5.0 visibly crisper.
- **480p is out of distribution** for this checkpoint and looks soft; do quality work at 720p.
- **The 5B weights are not in `~/.cache/huggingface`.** They live in a colleague's HF cache on
  the NFS mount; run device tests with `HF_HOME=/mnt/tt-data/teja/hf`. Without it diffusers
  starts a 32 GB re-download into the local cache, which on this box's root disk ends in
  `ENOSPC` a few minutes later and looks like a test crash. Do **not** add `HF_HUB_OFFLINE=1`
  as a guard: diffusers' `from_pretrained` calls the model-info API before touching the cache
  and fails outright in offline mode.
- **`pytest` is not on PATH** until `source python_env/bin/activate`.
- **A LoFi hang leaves chip 0 unable to init FW** (`Device 0 init: failed to initialize FW!`)
  even after the process is killed and `fuser` shows the devices free. The only fix that worked
  is `tt-smi -glx_reset_auto` (about 6 minutes on this Galaxy); check the firmware preconditions
  first. Run new quant presets through the 20 s transformer PCC test under a watchdog before
  anything longer.
- **`pgrep -f` on the pytest node id finds the wrapper, not the device holder.** The device is
  held by a child python (`fuser /dev/tenstorrent/*`); `kill -TERM` on the parent leaves it
  alive. Kill the fuser pid, then confirm `fuser` is empty.
- **The root disk is ~12 GB from full and both Tracy and the matmul sweep fill it** (section
  7.1, 7.8). `generated/profiler/.logs/` alone reaches several GB per capture; delete
  `profile_log_device.csv` and `tracy_profile_log_host.tracy` there after each attempt. Since
  sprint 5 run Tracy with `-o /mnt/tt-data/nkira/profiler/<tag>` and the wasm `traces` dir is a
  symlink onto NFS (`build/profiler/build_wasm/traces`), so disk is no longer the limit.
- **`python -m tracy` says `No available port found`** on this host because its port probe binds
  to `gethostbyname(hostname)` = 10.81.14.43, an address the box does not own. Pass `-t 8086`.
- **A 32-chip Tracy op capture is bounded by host RAM, not by marker drops** (section 7.8).
  Every device marker of every profiled program is held on the host at the end-of-run read
  (~1650 programs per 720p step x 120 cores x 5 RISCs x 32 chips); ~20k programs exceeded the
  566 GB host and was OOM-killed, twice. Keep a capture under ~10k programs (`WAN5B_GAP_BLOCKS`),
  set `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1` (the GUI push is a second copy, one zone per
  marker), and check `free -g` first. `ttnn.ReadDeviceProfiler(mesh)` mid-run takes 4-5 minutes
  per call on this Galaxy, so draining is not a way around it.
- **VAE decode spreads 1-8 % run to run** at ~0.95 s; do not read a VAE delta under 10 % as
  real from a single run.
- **`ttnn.slice` / `ttnn.chunk` on a TILE tensor is only a tile copy when the start is
  tile-aligned on both of the last two dims** (`slice.cpp`: `rm_only` unless `begins[-1] % 32
  == 0 && begins[-2] % 32 == 0`). Anything else silently untilizes, slices in row-major and
  re-tilizes every output a tile consumer touches (`TilizeWithValPadding`, 58 us per tiny
  tensor here). That was the whole AdaLN chunk cost (section 7.9) and it still exists in
  `_expand_two_row` (section 7.12). Chunk on the feature axis at tile multiples, never on a
  padded row axis.
- **Derived device constants must exist before the trace is captured.** Anything built lazily
  on first use from a Parameter (the split AdaLN tables) must be created in the load path, not
  in the traced function: created under capture it either needs a JIT compile mid-capture or
  gets recorded into the trace and re-executed every step. Rebuild them on reload
  (`deallocate_weights` drops them; the cache is keyed on the Parameter data's identity).
- **The harness of a worktree-isolated agent session refuses `fuser` globs, `sed -i` with
  variables and any compound shell line** ("cannot be shown not to be git"). Put device checks,
  file syncs, control swaps and run chains into small script files and `bash` them
  (`~/.claude/jobs/<job>/tmp/*.sh` in sprint 6: `devfree.sh`, `sync_files.sh`, `run_control.sh`,
  `run_perf_batch.sh`, `run_sweep.sh`).

---

## 9. Validation

Everything below was re-run green at the sprint-4 tip (`3d34a070c13`, 2026-09-25 unless noted);
the rows dated 2026-09-26 were run at the sprint-5 tip (`58bd9badfce` and after).

| gate | result |
|---|---|
| `test_dup_up3d_ti2v_5b` | 12/12, `max_abs_diff == 0.0` at production shapes (2026-09-22) |
| `test_vae_chunk_pcc_ti2v_5b` | PCC **1.0**, max_abs_diff 0.0 (2026-09-22) |
| `test_transformer_wan_ti2v_5b` | PCC **100.0000 / 99.9893 / 99.9894%** (2 pre-existing skips) after the CFG hoist; scalar re-run at the tip 2026-09-25: 99.9893% |
| `test_cfg_hoist_ti2v_5b` | `max_abs_diff == 0.0` for cond, uncond and combined at 720p geometry; re-run 2026-09-26 with the per-block modulation hoist passed explicitly: still 0.0 / 0.0 / 0.0 |
| `test_trace_modes_ti2v_5b` | nonblocking and 2cq **bit-identical** to blocking, compute grid 12x10 with two queues (re-run green 2026-09-26 at `58bd9badfce`) |
| `test_transformer_wan_ti2v_5b` after the modulation hoist (2026-09-26) | 100.0000 / 99.9893 / 99.9894 %, unchanged |
| `test_pipeline_performance_ti2v_5b` 720p / 480p, `_i2v` 720p | 3/3 pass each under the 2cq default (section 1 has the means) |
| same three, sprint-5 tip with the modulation hoist (2026-09-26) | 3/3 pass each under the 2026-09-22 gates; means in section 1 (720p T2V 10.198 / 11.23 s, I2V 10.580 / 12.90 s, 480p 4.842 / 5.49 s); plus one 720p T2V control run of the sprint-4 file the same hour (10.473 / 11.535 s) |
| gates recalibrated to the sprint-5 means + 30 % (`ti2v_5b_metrics`, `ti2v_5b_i2v_metrics`, 2026-09-26) | 720p 0.2 / 13.3 / 1.2 / 14.6 s; 480p 0.2 / 6.3 / 0.75 / 7.2 s; I2V 0.2 / 1.7 / 13.8 / 1.2 / 16.8 s; one run per geometry re-run under them, all pass: 720p T2V 10.174 / 11.195 s, 480p 4.844 / 5.523 s, I2V 10.569 / 12.845 s (denoise / total) |
| Teja's 121 f `test_pipeline_ti2v_5b_generate`, sprint-5 tip (2026-09-26) | passes; CLIP mean 40.69 (min 39.84 / max 41.21) vs 36.00; previews `/home/ttuser/wan5b_s5_hoist_t2v_720p_121f_{first,mid,last}.png` (mp4 export needs `imageio`, absent from the venv) |
| same, `WAN5B_QUANT_CONFIG=all_bf8_lofi`, 720p T2V | 3/3 pass, PCC 99.9651 / 99.9651%, CLIP 41.34 |
| same, `WAN5B_QUANT_CONFIG=all_weights_bf8`, 720p T2V | 2/2 pass, PCC 99.9885 / 99.9885%, CLIP 40.20 |
| `test_step_gap_ti2v_5b` | passes; 1.37 ms/step host-only gap on the blocking path (section 7.5) |
| `test_ti2v_5b_i2v_math` | 20/20 (2026-09-22) |
| I2V E2E frame-0 vs seed | PCC **0.9984** (2026-09-22) |
| Teja's 121f `test_pipeline_ti2v_5b_generate` | passes; CLIP mean 40.38 (bf16), 40.20 (bf8), 41.34 (bf8 LoFi) vs 36.00 |
| `wan2_2_ti2v_5b_demo.py` T2V / I2V | 81 f 720p mp4 each, warm traced 11.57 s / 13.76 s (2026-09-23 / 09-24) |
| **Sprint 6, fix 1 (AdaLN split layout, 2026-09-28)** `test_cfg_hoist_ti2v_5b` | passes; 34 comparisons (cond, uncond, combined_step, 6 tensors x 2 blocks, 2 norm_out; scalar and two-row timestep) all `max_abs_diff == 0.0` |
| same, `test_transformer_wan_ti2v_5b` | 3 passed, 2 skipped: 100.0000 / 99.9893 / 99.9894 %, unchanged |
| same, `test_trace_modes_ti2v_5b` | nonblocking and 2cq bit-identical to blocking, grid 12x10; 270.0 / 261.6 / 261.4 ms/step over its 8 steps |
| same, 14B `test_transformer_wan.py` 4x8 ring (`test_wan_transformer_model[short_seq]`, `test_wan_transformer_inner_step`) | both pass, PCC 99.9886 % (the model construction runs the eager split-table creation at D/tp = 1280) |
| same, `test_pipeline_performance_ti2v_5b` 720p / 480p, `_i2v` 720p | 3/3 pass each under the sprint-5 gates; means 720p 9.787 / 10.835 s, 480p 4.438 / 5.137 s, I2V 10.260 / 12.523 s; same-hour control of the sprint-5 file 10.201 / 11.262 s (section 1, 7.9) |
| same, 6-block Tracy capture `s6_f1_blocks6` | 0 dropped markers; 45.86 ms kernel / 328 programs per replay per device (was 48.15 / 404); `execute_trace` 48.4 ms (section 6) |
| **Sprint 6, fix 3 (cross-attention residual fusion, measured and dropped, 2026-09-28)** | PCC 100.0000 / 99.9901 / 99.9902 %; cfg-hoist gate 34/34 at 0.0; trace modes bit-identical; 14B block test 99.9958 %; 720p 9.776 / 10.837 s vs same-hour control 9.780 / 10.883 s (-0.04 %); Tracy `s6_f3_blocks6` `execute_trace` 49.24 ms per replay vs 48.41 (section 7.13) |
| **Sprint 6, `all_bf8_lofi` on the tip (final preset definition, 2026-09-29)** `test_transformer_wan_ti2v_5b` | 3 passed: 100.0000 / 99.9600 / 99.9600 % (sprint-5 preset: 99.9651 %; section 7.7) |
| same, `test_pipeline_performance_ti2v_5b` 720p x3 | denoise 8.217 / 8.258 / 8.274 s; totals 9.666 (failed: VAE 1.33 s > 1.2 s gate, a decode outlier) / 9.316 / 9.458 s |
| same-hour control: `all_bf8_lofi` with the sprint-5 preset file (`cross_attn_out` bf8 / LoFi) on the tip | 100.0000 / 99.9651 % -- reproduces the sprint-5 value, so the 0.005 pp is the `cross_attn_out` change (section 7.7) |
| Sprint 6, Teja's 121 f `test_pipeline_ti2v_5b_generate`, bf16 tip (2026-09-29) | passes; CLIP mean 40.69 (min 39.84 / max 41.21), identical to sprint 5 (bit-exact change); eager 18.87 s; previews `/home/ttuser/wan5b_s6_t2v_720p_121f_{first,mid,last}.png` |
| Sprint 6, `wan2_2_ti2v_5b_demo.py` T2V 81 f, bf16 / `all_bf8_lofi` (2026-09-29) | warm traced 10.85 s / 9.32 s; mp4 via the new OpenCV fallback (`imageio_ffmpeg` is absent from the venv); side-by-side `/home/ttuser/wan5b_s6_ab_bf16_left_bf8lofi_right.mp4` for the visual verdict |

The VAE rewrite is **bit-exact**, not merely within PCC — it is pure data movement, so the gate
asserts exact equality against the original implementation rather than a correlation floor.

Check the box is free before every run (`fuser -v /dev/tenstorrent/*` empty, `tt-smi -s` AICLK
0x320); all 32 chips are claimed by every run, so a second job collides.
