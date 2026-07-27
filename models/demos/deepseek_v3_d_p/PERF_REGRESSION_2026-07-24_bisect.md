# GLM-5.2 chunked prefill: +16% per-chunk regression on main (2026-07-24)

Handoff notes for finishing the bisect on an 8x4 (TG / `bh_sc1`) galaxy.
Everything below was derived from CI job logs only — no hardware was used.

## TL;DR

`Blaze - Chunked GLM (code_debug 55k)` per-chunk median went **2.61s -> 3.03s (+16%)**.

The regression landed in a **30-commit window on main**:

```
good: 218b627e304   2026-07-24 08:15 UTC
bad:  ac4e4e6b8bd   2026-07-25 00:14 UTC
```

Runner/infra variation is **ruled out** (proof below). Three candidate commits are
already ruled out on evidence. Two prime suspects remain, both near the end of the
window, so a revert-test on latest main is likely cheaper than a full bisect.

## Anchor measurements

All numbers are the `print_duration_table` chunk medians from the job logs
(the only trustworthy steady-state metric in these jobs — see "Gotchas").

| s/chunk | commit | what | run |
|---|---|---|---|
| 2.58 | `f04035aca5f` | branch `ipotkonjak/ci-glm-chunked-prefill` | [29989806319](https://github.com/tenstorrent/tt-metal/actions/runs/29989806319) |
| **2.61** | `6eede6ddd80`, main base **`6a84cd727e2`** | branch `ipotkonjak/fix-mla-kv-cache-wrapper-callers` | [30075756400](https://github.com/tenstorrent/tt-metal/actions/runs/30075756400) |
| **3.03** | `ac4e4e6b8bd` | main nightly 07-25 | [30146478733](https://github.com/tenstorrent/tt-metal/actions/runs/30146478733) |
| 3.05 | `7106c7678c8` (base `02246470f99`) | branch `pmilojevic/49804-routedexpert-zeros` | [30167266784](https://github.com/tenstorrent/tt-metal/actions/runs/30167266784) |
| 3.03 | `aa04c33e2e7` | main nightly 07-27 | [30241673348](https://github.com/tenstorrent/tt-metal/actions/runs/30241673348) |

The "good" reference run was on a *branch*; its main base commit is:

```
6a84cd727e22cf3742d69a41f262ae39a2b7ad15
2026-07-24 04:15:14 +0000  Universal input support for untilize_with_unpadding (#50383)
```

The 07-25 nightly was already regressed, which is what shrinks the window from
"good..HEAD" (47 commits) down to 30.

## Second, independent reproduction (Chunked Kimi)

`Blaze - Chunked Kimi (code_debug 55k)` changed **shape** at the same boundary:

| chunk | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `6a84cd727e2` (good) | 1.250 | 1.243 | 1.248 | 1.286 | 1.333 | 1.375 | 1.428 | 1.480 | 1.529 | 1.587 | 1.649 |
| `ac4e4e6b8bd` (bad)  | 1.533 | 1.492 | 1.523 | 1.502 | 1.510 | 1.528 | 1.502 | 1.506 | 1.509 | 1.501 | 1.508 |

It went from a healthy rising profile (cost grows with accumulated KV) to **flat
~1.51s**. That is the signature of the run becoming **host-dispatch-bound**: per-chunk
time pins to a constant regardless of KV length. Worth keeping in mind — it points at
per-op host/dispatch cost as much as at any single device kernel.

Kimi is a useful bisect probe because it is ~2x faster per chunk than GLM and the
rising->flat signal is unmistakable.

## Why it is NOT the runners

The good run was on runner pool `ggrtn`, every bad run on `mh8h5`, which looked like
a hard confound. It isn't:

Branch SHA `b48a41448c7` (`ppetrovic/prefill-perf-summary`, base `218b627e304`) ran
Chunked Kimi **twice**, on both sides of the day boundary and **both times on the
`mh8h5` pool**:

- [30114359730](https://github.com/tenstorrent/tt-metal/actions/runs/30114359730) 07-24 18:18 -> 1.241 ... 1.721
- [30152708217](https://github.com/tenstorrent/tt-metal/actions/runs/30152708217) 07-25 09:52 -> 1.244 ... 1.715

Identical within 0.5%. Same code => same numbers, on the "bad" pool. So:
1. infra/runner is not the cause, and
2. **everything up to `218b627e304` (07-24 08:15) is clean** — this is what sets the
   `good` end of the window.

AICLK was 1.35 GHz in both good and bad runs, so it is not a clock/throttling effect.

## Ruled out on evidence

- **`ad81807b191`** — Revert #49473, restoring the BH `Fp8_e4m3` `unpack_tilize`
  face-pair protocol (i.e. undoing a *perf feature*). Tempting, but the 07-23 run at
  **2.58s predates #49473 entirely** (it merged 07-23 16:12), so removing that feature
  cannot cost 16%.
- **`01d5894f4c8`** — `moe_compute` non-tile-aligned fix. The diff is only
  `floor(n/2)` -> `ceil(n/2)` for a BRISC token capacity and two CB sizes; no work
  redistribution. Also this model's MoE goes through
  `ttnn.experimental.deepseek_prefill`, not `moe_compute`.
- **`fd40d4aa7d3`** — profiler read fix for `ttnn.device.setup_fast_dispatch`. This
  test never calls `setup_fast_dispatch` (only
  `tests/ttnn/unit_tests/base_functionality/test_device.py` does).
- **`8a53c48c288`** — `strided_reduce_scatter` semaphore race fix. The model never
  calls `strided_reduce_scatter`.

Also checked and clean: the model code itself is unchanged across the window except
a `.storage` attribute fix in a PCC-only helper and `deepseek_v3.json` / `kimi26.json`
manifests (not `glm52`). `.github/workflows/blaze-models-prefill-tests.yaml` is
untouched, and both runs installed a `Release` wheel. The signpost count is identical
(141570) in good and bad, so the op graph did not change — this is per-op cost.

## Prime suspects

Both are on the shared per-layer path and both landed 13 minutes apart late on 07-24:

1. **`7c1946ed164`** — *Migrate `reduce_scatter_async_minimal` (Ring) to Mux v2 (#50644)*.
   Rewrites `ring_reduce_scatter_minimal_async_writer.cpp`, the `dim_zero` variant, and
   the program factory (fabric mux V1 -> V2). The model calls
   `ttnn.experimental.reduce_scatter_minimal_async` with **Ring** topology per layer in
   four places: `mla.py:812` (q), `mla.py:1015` (o_proj), `moe/tt_shared_expert.py:492`,
   and `mla/indexer.py:306`. Self-described as "small perf bump" — worth confirming it
   is not a perf *loss* on a 78-layer 8x4 ring. Note the mux is only exercised when
   `num_mux_cores_per_direction_per_link != 0`; that condition itself is unchanged by
   the commit.
2. **`b5e275260d8`** — *sparse_sdpa: replace `get_dynamic_runtime_args` with
   `override_runtime_arguments` (#50351)*. Moves per-dispatch patching of buffer
   addresses + `kv_batch_page_offset` off the dynamic-runtime-arg mechanism onto a host
   `GetRuntimeArgs` loop over all `grid.x * grid.y` cores, twice per core, on **every**
   `sparse_sdpa` call. GLM-5.2 hits `ttnn.transformer.sparse_sdpa` per layer
   (`mla.py:1414`). This is a plausible host-side per-op cost increase, which matches
   the host-bound signature above.

Secondary, if both of the above come back clean:

3. **`2837400b3a9`** — *Remove enqueue-invariance from the Metal 2.0 host API (#50966)*
   — generic per-enqueue host path.
4. **`78cff43bb04`** — Metal 2.0 spec factory rename / `CustomProgramSpecFactoryConcept`.
5. **`69faa8e09c3`** — *Enable watcher in Tier 1/2/3 model unit test pipelines* — should
   not touch this workflow, but watcher overhead is large enough to be worth excluding.

## Reproduce

GLM (the primary signal, ~33s/iter, 10 iters):

```bash
export MESH_DEVICE=TG
export LOGURU_LEVEL=INFO
export GLM52_HF_MODEL=/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8
export TT_GLM52_PREFILL_TTNN_CACHE=/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache
export PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k
mpirun --bind-to none --pernode --tag-output bash -lc '
  export OMP_NUM_THREADS=$(nproc)
  python3 -m pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc[blackhole-glm52-mesh-8x4-L78-preload0-chunks_eleven-ten_iters]" -xvs
'
```

Kimi (faster, and gives the rising-vs-flat signal):

```bash
export MESH_DEVICE=TG
export LOGURU_LEVEL=INFO
export KIMI_K2_6_HF_MODEL=/mnt/models/Kimi-K2_6-dequantized
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export PREFILL_TRACE_DIR=/mnt/models/kimi-prefill-cache/vllm-kimi-k26-codedebug-56320
mpirun --bind-to none --pernode --tag-output bash -lc '
  export OMP_NUM_THREADS=$(nproc)
  python3 -m pytest "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc[blackhole-kimi-mesh-8x4-L61-preload0-chunks_eleven-ten_iters-margin5pct]" -xvs
'
```

### Pass / fail criterion

Read the `chunk timing stats` table at the end of the log (medians over 9 iterations,
iter 0 omitted). Per-chunk stddev is ~0.06-0.15s and the good/bad gap is ~0.42s, so a
single run is decisive — no repeats needed.

- GLM: **GOOD <= ~2.65s**, **BAD >= ~2.95s** per chunk.
- Kimi: **GOOD** = rising 1.24 -> 1.72 across chunks 0..10; **BAD** = flat ~1.51.

## Suggested plan

### Step 1 (cheapest): revert-test the two prime suspects on latest main

Two builds, no bisect, and it tests the suspects in today's tree:

```bash
git checkout -b probe/revert-rs-muxv2 origin/main
git revert --no-commit 7c1946ed164 && git commit -m "probe: revert RS Mux v2"
# build + run GLM repro
```

```bash
git checkout -b probe/revert-sparse-sdpa-ora origin/main
git revert --no-commit b5e275260d8 && git commit -m "probe: revert sparse_sdpa ORA"
# build + run GLM repro
```

If either revert restores ~2.6s, that is the culprit. Expect possible conflicts, since
both areas have moved since 07-24; if a revert does not apply cleanly, fall back to
Step 2.

### Step 2: bisect the window

30 commits => 5 builds. Many are Quasar/docs/CI-only and cannot affect this test, but
`git bisect` does not know that, so just let it run.

```bash
git bisect start ac4e4e6b8bd 218b627e304
# for each step: build, run the GLM repro, then
#   git bisect bad     # >= ~2.95 s/chunk
#   git bisect good    # <= ~2.65 s/chunk
```

To skip the commits that cannot plausibly matter and go straight to the interesting
split, probe **`2cba5b07278`** (07-24 18:49) first. It cleanly separates the two
clusters:

- **bad** at `2cba5b07278` => culprit is in `[ad81807b191 .. 2cba5b07278]`, i.e. the
  Metal 2.0 host API / watcher / slice / MiniMax group.
- **good** at `2cba5b07278` => culprit is one of the 8 commits after it, which contains
  **both** prime suspects.

### Window, in first-parent order

```
 1  ad81807b191  09:34  Revert #49473: restore Blackhole FP8_E4M3 tilize protocol (#50989)   [ruled out]
 2  0981c4a9396  09:55  quasar sfpu: non-approx reciprocal
 3  a07ef0b5fa4  10:27  trace manifest validation script
 4  2848c65adea  11:18  [LLK] Convert Blackhole complex SFPU kernels to SFPI (#49926)
 5  bc15d3423c5  11:49  [deepseek_v3_d_p] Fix stale MLA KV-cache callers (#50986)
 6  2a8e0ffd363  13:28  quasar sfpu: softplus
 7  90c0c046c1e  13:39  quasar: unpack tilize tiny tiles
 8  a65e5351a41  13:40  [Quasar] Assert SFPU destination indices
 9  78cff43bb04  13:55  [ttnn] Metal 2.0 spec factory rename (#50942)                        <- secondary
10  32c1f0e882b  14:48  docs
11  8a53c48c288  15:14  strided_reduce_scatter out_ready_sem race (#50794)                    [ruled out]
12  233cd966b25  15:36  quasar sfpu: non-approx rsqrt
13  69faa8e09c3  15:47  Enable watcher in Tier 1/2/3 model unit test pipelines (#48924)       <- secondary
14  93dabaed547  16:14  [ttnn] slice: height-sharded RM cache-hit descriptor rebuild (#50894)
15  a0fb0661887  16:23  [MiniMax M3] pipeline-parallel prefill (#50466)
16  f8c6597a23a  16:37  [Quasar] dispatch engine
17  bc4c859af16  17:04  docs
18  7d8cafb6244  18:24  ttop-delete-allocation (CI)
19  3650b83f8b3  18:28  Runner failure scan (CI)
20  cebd1d45a19  18:40  [LLK][Quasar] memory clobber in vector_load/store asm
21  2837400b3a9  18:45  Remove enqueue-invariance from Metal 2.0 host API (#50966)            <- secondary
22  2cba5b07278  18:49  [upsample] read barrier                                               <- PROBE HERE
23  01d5894f4c8  19:24  [moe_compute] non-tile-aligned token counts (#50934)                  [ruled out]
24  fd40d4aa7d3  20:57  Fix profiler read w/ setup_fast_dispatch (#51064)                     [ruled out]
25  796b19d3e55  21:25  [Quasar][Profiler] DeviceZoneScopedN
26  b5e275260d8  22:09  [ttnn] sparse_sdpa -> override_runtime_arguments (#50351)             <- SUSPECT
27  7c1946ed164  22:22  reduce_scatter_async_minimal (Ring) -> Mux v2 (#50644)                <- SUSPECT
28  49625428227  22:24  [tt_metal] tt_memory DISCRETE load segment permutation (#49053)
29  f1b96e36079  23:01  Quasar coverage in LLK perf workflow (test only)
30  ac4e4e6b8bd  00:14  TT_METAL_DRAM_BACKED_CQ override on Quasar simulator (#50469)
```

## Gotchas

- **Do not use pytest's "slowest N durations" as a perf signal for these jobs.** Those
  `call` times are dominated by model setup / weight load (e.g. 400-1000s), and they
  swing +-8% run to run. Comparing them across the boundary suggested things like
  "standalone MLA got 8% *faster*", which is noise. Only the per-chunk median table is
  a steady-state measurement.
- **The real-time profiler is active** in these jobs (`[Real-time profiler] Device N
  sync complete ...` in both good and bad logs). Keep it as-is when bisecting so you
  are comparing like with like — but be aware profiler overhead is part of the
  measured per-chunk time, so a profiler-path change could in principle be the culprit.
- The test reuses `TT_GLM52_PREFILL_TTNN_CACHE` / `PREFILL_TRACE_DIR` from `/mnt/models`.
  Leave those pointing at the same paths CI uses, or first-iteration numbers will
  differ (iter 0 is excluded from the medians anyway).
- Per-chunk profile shape matters, not just the mean: GLM is flat in both good and bad
  (sparse top-k attention => KV-length independent), Kimi is rising when healthy.

## Provenance

Derived entirely from GitHub Actions logs of workflow
`.github/workflows/blaze-models-prefill-tests.yaml` (id 312539743), runs 29989806319,
30070767586, 30075756400, 30114359730, 30146478733, 30152708217, 30167266784,
30206035417, 30241673348. No hardware runs were performed, so the two prime suspects
are **unconfirmed** — they are ranked by mechanism and call-site evidence only.
