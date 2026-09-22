# Wan2.2 TI2V-5B — I2V enablement and optimization (nkira)

Everything done on `nkira/wan2.2-5B-i2v` since branching from Teja's bring-up at
`8d596eb242d`. Companion to `Wan2_2_TI2V_5B.md`, which stays the bring-up doc.

Hardware throughout: one Blackhole Galaxy, 4x8, SP=8 axis1 / TP=4 axis0, Ring, FSDP off.
All timings are 40 steps, warm-traced, and every figure below is a **mean of 3 invocations** —
the per-run `Std` the perf test prints is a single sample, not a spread.

---

## 1. Results

| | before | after | delta |
|---|---|---|---|
| 720p T2V total (1280x704, 81f) | 16.78s | **12.51s** | **-25.4%** |
| 720p I2V total | 18.86s | **14.68s** | **-22.2%** |
| 480p T2V total (832x480, 81f) | 8.87s | **6.98s** | **-21.3%** |
| 720p VAE decode | 4.632s | **0.966s** | **-79.2%** |
| 720p denoise | 12.045s | **11.436s** | -5.1% |

121 frames at 720p T2V: **19.76s** traced (494 ms/step). No prior baseline — newly measured.

Section detail at the current tip:

| Mode | Resolution | Text enc | Image enc | Denoise | VAE dec | Total |
|------|------------|----------|-----------|---------|---------|-------|
| T2V  | 1280x704   | 0.094s   | —         | 11.436s | 0.966s  | 12.51s |
| I2V  | 1280x704   | 0.089s   | 1.512s    | 12.072s | 0.986s  | 14.68s |
| T2V  | 832x480    | 0.090s   | —         | 6.301s  | 0.578s  | 6.98s |

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

   *Swept (2026-09-22), 10 of 11 shapes* — device kernel time, best of ~330-380 L1-feasible combos
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

   At 720p the pattern is one full M block per core (73 M tiles over 12 columns -> M_block 7) with
   K_block 6, and the top-5 combos per shape sit within ~1% of each other, so the winners are not
   noise picks. Cross-attn `to_kv` (512x3072x1536) is still PRE-SWEEP. E2E perf gates not yet
   re-run with the swept table.

   **The sweep fills the disk.** It compiles one program per combo and the kernel JIT cache
   (`~/.cache/tt-metal-cache`) keeps every one: ~118 GB across 587k files for ten shapes, plus
   ~3 GB of profiler capture per shape. Two sweep runs died with `ENOSPC` on this box before
   that was understood. Budget ~15 GB per shape, or prune the sweep-window cache entries
   afterwards (they are unique blockings and never reused).
2. **Hoist the per-step modulation.** `combined_step` calls `inner_step` twice with the **same
   timestep**, so the timestep MLP, patch embed and every block's modulation are recomputed
   identically — ~26,400 op launches per generation, half of them exact duplicates.
3. **Fold `+1.0` into `scale_shift_table`.** `1 + (table+temb) == (table+1) + temb` exactly;
   removes 4,800 ops per generation at zero numerical cost.
4. **Change the AdaLN split layout.** Measured bit-exact on the production shape: current
   `chunk` dim2 TILE ~495us, dim3 TILE flat ~194us (2.5x), dim2 ROW_MAJOR ~130us (3.8x).
   Host-dominated, so this helps untraced runs and trace capture, not the traced steady state.
5. **Trace the UniPC solver step, or stop blocking on `execute_trace`.** The solver runs outside
   the traced region while `execute_trace` blocks, so ~760 host-dispatched launches per
   generation happen with the device idle.
6. **bf8 for the ring-SDPA K/V gather.** ~120GB/device/generation crosses the SP fabric in bf16
   and the `bfloat8_b` path already exists but is only enabled by a `QuantConfig` no 5B pipeline
   applies. Overlapped with compute, so precision-gate it and expect only what the fabric is
   actually binding.

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

---

## 9. Validation

Everything below was re-run green at the current tip.

| gate | result |
|---|---|
| `test_dup_up3d_ti2v_5b` | 12/12, `max_abs_diff == 0.0` at production shapes |
| `test_vae_chunk_pcc_ti2v_5b` | PCC **1.0**, max_abs_diff 0.0 |
| `test_transformer_wan_ti2v_5b` | PCC **100.0000 / 99.9893 / 99.9894%** (2 pre-existing skips) |
| `test_ti2v_5b_i2v_math` | 20/20 |
| I2V E2E frame-0 vs seed | PCC **0.9984** |
| Teja's 121f `test_pipeline_ti2v_5b_generate` | passes, CLIP mean 40.38 vs 36.00 |

The VAE rewrite is **bit-exact**, not merely within PCC — it is pure data movement, so the gate
asserts exact equality against the original implementation rather than a correlation floor.

Check the box is free before every run (`fuser -v /dev/tenstorrent/*` empty, `tt-smi -s` AICLK
0x320); all 32 chips are claimed by every run, so a second job collides.
