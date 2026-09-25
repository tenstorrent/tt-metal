# Wan2.2 TI2V-5B — I2V enablement and optimization (nkira)

Everything done on `nkira/wan2.2-5B-i2v` (sprint-4 tip on `nkira/wan2.2-5B-i2v-step3`) since
branching from Teja's bring-up at `8d596eb242d`. Companion to `Wan2_2_TI2V_5B.md`, which stays the bring-up doc.

Hardware throughout: one Blackhole Galaxy, 4x8, SP=8 axis1 / TP=4 axis0, Ring, FSDP off.
All timings are 40 steps, warm-traced, and every figure below is a **mean of 3 invocations** —
the per-run `Std` the perf test prints is a single sample, not a spread.

---

## 1. Results

**Current tip (sprint 4, 2026-09-25, host u13-43, mean of 3, 81 f / 40 steps, warm-traced).** The
default pipeline now runs the trace on two command queues non-blocking (section 7.5); the
`all_bf8_lofi` row is opt-in (`WAN5B_QUANT_CONFIG=all_bf8_lofi`, section 7.7) and is the fastest
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

**Denoise (~91% of the run)** is genuinely compute-bound, unlike the VAE:

| callsite | ~share |
|---|---|
| `ring_joint_sdpa @ attention_wan.py:457` | ~28% |
| `MMRS (ff2) @ linear.py:697` | ~27% |
| `AGMM @ linear.py:437` | ~17% |
| `chunk @ transformer_wan.py:209` | ~13% (host-dominated) |
| norms | ~11% |

A full read of the denoise path found **no ROW_MAJOR<->TILE padding pathology anywhere in it**,
so there is no second `WanDupUp3D` to find. Remaining wins are incremental.

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
   decimals (100.0000 / 99.9893 / 99.9894 %). The per-block AdaLN modulation (the
   `scale_shift_table + temb` add and six-way chunk in every block, twice per step) is still
   duplicated: hoisting it means threading 30 x 6 modulation tensors through the block API,
   and the traced steady state pays only its device time, which is small. Perf effect of the
   half that landed: see section 1 (expected <= 1 %).
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
   free. `tracy_summarize_ops.py` is written and untested on a real capture. Options: symlink
   `generated/profiler` (and/or the kernel cache) onto `/mnt/tt-data`, or free ~40 GB on root.
   Not a code blocker; the per-step device time is already known from the blocking
   `execute_trace` measurement (258.5 ms of 266.4 ms, item 5).

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
  `profile_log_device.csv` and `tracy_profile_log_host.tracy` there after each attempt.
- **VAE decode spreads 1-8 % run to run** at ~0.95 s; do not read a VAE delta under 10 % as
  real from a single run.

---

## 9. Validation

Everything below was re-run green at the current tip (`3d34a070c13`, 2026-09-25 unless noted).

| gate | result |
|---|---|
| `test_dup_up3d_ti2v_5b` | 12/12, `max_abs_diff == 0.0` at production shapes (2026-09-22) |
| `test_vae_chunk_pcc_ti2v_5b` | PCC **1.0**, max_abs_diff 0.0 (2026-09-22) |
| `test_transformer_wan_ti2v_5b` | PCC **100.0000 / 99.9893 / 99.9894%** (2 pre-existing skips) after the CFG hoist; scalar re-run at the tip 2026-09-25: 99.9893% |
| `test_cfg_hoist_ti2v_5b` | `max_abs_diff == 0.0` for cond, uncond and combined at 720p geometry |
| `test_trace_modes_ti2v_5b` | nonblocking and 2cq **bit-identical** to blocking, compute grid 12x10 with two queues |
| `test_pipeline_performance_ti2v_5b` 720p / 480p, `_i2v` 720p | 3/3 pass each under the 2cq default (section 1 has the means) |
| same, `WAN5B_QUANT_CONFIG=all_bf8_lofi`, 720p T2V | 3/3 pass, PCC 99.9651 / 99.9651%, CLIP 41.34 |
| same, `WAN5B_QUANT_CONFIG=all_weights_bf8`, 720p T2V | 2/2 pass, PCC 99.9885 / 99.9885%, CLIP 40.20 |
| `test_step_gap_ti2v_5b` | passes; 1.37 ms/step host-only gap on the blocking path (section 7.5) |
| `test_ti2v_5b_i2v_math` | 20/20 (2026-09-22) |
| I2V E2E frame-0 vs seed | PCC **0.9984** (2026-09-22) |
| Teja's 121f `test_pipeline_ti2v_5b_generate` | passes; CLIP mean 40.38 (bf16), 40.20 (bf8), 41.34 (bf8 LoFi) vs 36.00 |
| `wan2_2_ti2v_5b_demo.py` T2V / I2V | 81 f 720p mp4 each, warm traced 11.57 s / 13.76 s (2026-09-23 / 09-24) |

The VAE rewrite is **bit-exact**, not merely within PCC — it is pure data movement, so the gate
asserts exact equality against the original implementation rather than a correlation floor.

Check the box is free before every run (`fuser -v /dev/tenstorrent/*` empty, `tt-smi -s` AICLK
0x320); all 32 chips are claimed by every run, so a second job collides.
