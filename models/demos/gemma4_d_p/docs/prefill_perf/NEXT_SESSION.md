# Handoff — state as of 2026-09-17 00:45Z

**Read order:** [`README.md`](README.md) (what was asked, what was answered) → this file (state,
open questions, commands, traps) → [`CHUNK_SIZE_ANATOMY.md`](CHUNK_SIZE_ANATOMY.md) (full record,
including two retractions).

## State

* Branch **`kmabee/gemma4-swa-multihop-halo`** @ `23df2b38ba4`, **working tree clean**, no PR
  opened, nothing of this session's work is committed to the repo — it is all in this directory.
* All diagnostic env-var hunks **reverted**, `git diff` is empty. Re-apply from
  §"Diagnostics to re-apply" below. Four were used: `GEMMA4_GLOBAL_Q_CHUNK` and
  `GEMMA4_KV_DTYPE` (+ unused `GEMMA4_RING_GATHER_LINKS`) in
  `models/demos/gemma4_d_p/tt/attention/ring_prefill.py`, and `GEMMA4_NUM_LAYERS` in
  `models/demos/gemma4_d_p/tt/model_config.py`.
  **`ring_prefill.py` does NOT import `os`** — add `import os` when re-applying, or every run
  dies with `NameError` after the model builds (cost 3 wasted runs).
* Board free, kernel + weight caches warm, `/data` had 18T free.
* ~27 run directories under `/data/kmabee/gemma4_runs/`: `floor_c{2048,4096,8192}` (profiled
  chunk-0 per-op), `curve_both_c*` (per-layer-type depth curves), `qchunk_*` (single-layer
  q_chunk ablations), `nlayer_c{8192,2048}_n{60,12,6}` (layer-count differencing),
  `q_e2e_*` (end-to-end q_chunk), `kv_c8192_bfp8` + `kv2_c8192_{bf16,bfp4}` (KV dtype),
  `micro_floor` + `micro_norm` (off-model microbenchmarks).
* **~48 GB of raw profiler logs** in `floor_c*/profiler/.logs/` (4.8 GB `profile_log_device.csv`
  + 5.7 GB `tracy_ops_times.csv` each). The small `reports/*/ops_perf_results_*.csv` are the only
  files the analysis uses — the raw logs can be deleted.

## What is solid (cross-validated, <3%)

| claim | how it was validated |
|---|---|
| The 2.09x at 256k = per-chunk **2.16x** × prefix **1.99x** | measured `a`/`slope`; matches independent 256k runs to 0.9% |
| Prefix term is **100% the 10 global layers** | slope reconstruction to 0.6% / 0.02% / 2.6% at chunk 2048/4096/8192 |
| Sliding layers are **exactly flat** in context | slope −0.0001 ms/index over a full 256k prefill |
| Per-chunk floor is **84% sliding layers** (by count, not per-layer cost) | decomposition closes to 1.1% |
| Floor by op: matmul **43%**, `rms_norm` **26%**, SDPA 14%, **TP collectives only 4%** | three same-branch profiled chunk-0 captures |
| `q_chunk_size` optimum moves with chunk size: **32 / 64 / 128** at 2048 / 4096 / 8192 | measured depth curves, incl. a wrong-direction control |
| q_chunk verified **end-to-end**: −4.6% at 256k (8192, q=128), −8.0% (2048, q=32), TTFT unchanged | whole-model ctx_32k; projections were −4.8%/−6.9% |
| Non-layer cost (embedding, final norm, LM head, inter-layer CCL) is **0.4% of TTFT** | layer-count differencing, linear in N to 5–6% |
| The global SDPA is **NOT bandwidth-bound** | −47% K/V bytes (bfp8→bfp4) ⇒ 0.2% slope change |
| `rms_norm` width-sharding gives **~2x** (not ~8x); precision is not the cost (LoFi 19%) | off-model microbenchmark at the exact shapes |
| Chunk 4096's prefix term is already **within 2%** of 8192's | measured |

## What is NOT understood — pick up here

### 1. ~~Inside `RingJointSDPA` on global layers~~ — ANSWERED 2026-09-17

**It is MAC-throughput-bound on QK^T and PV.** Four causal ablations on the whole model
(`ctx_32k`; the slope is 100% global-layer SDPA):

| ablation | slope | vs base | verdict |
|---|---:|---:|---|
| base (HiFi2, slow exp, k=256) | 11.93 | — | ships |
| K/V bytes −47% (`bfp4`) | 11.81 | **0.998x** | bytes RULED OUT |
| `exp_approx_mode=1` | 12.01 | **1.007x** | softmax RULED OUT |
| **`fidelity=lofi`** | **9.38** | **0.786x** | **DOMINANT** |
| **`fidelity=hifi4`** | **20.94** | **1.755x** | **DOMINANT** |
| `k_chunk=128` | 13.79 | 1.156x | 256 already better |
| `k_chunk=512` | — | — | L1 overflow |

Holds in both occupancy regimes: LoFi is 0.786x at chunk 8192 and 0.764x at chunk 2048. No
precise MAC share is claimed — a `passes·k + c` fit gives 43% on (LoFi,HiFi2) and 76% on
(HiFi2,HiFi4), so it is not linear in passes.

**`DeviceZone` kernel instrumentation is NOT needed.** It was queued to find what starves the
math; the answer is that nothing does — the op *is* the math. Zones would only split QK from PV,
both MACs, changing no decision.

**The follow-on is an ACCURACY question, not a perf one.** LoFi is −9.9% on a 256k prefill but
**triples single-layer RMSE** (0.0059 → 0.0178, PCC 0.9997 → 0.9985) and that compounds over 60
layers. Needs a real eval (perplexity/task metrics); an op-level PCC spot-check is not sufficient.
Also: **HiFi4 is strictly dominated** — no measurable accuracy gain for 1.755x the cost.

*(original text below)*

### 1-orig. Inside `RingJointSDPA` on global layers — the big one
It is **100% of the long-context penalty**. At chunk 8192 it goes **1.32 ms at chunk index 0 →
37.2 ms at index 31** (28x; 89% of the layer). Its internal split — ring gather vs QK vs softmax
vs PV vs sync waits — has **never been measured**, and the gather is *fused inside the op*, so no
op-name attribution can ever see it.

Three routes, increasing cost and fidelity:

* ~~**(a) KV dtype ablation**~~ — **DONE 2026-09-17, and it says NOT bandwidth-bound.**
  bfp8→**bfp4** on the global cache (−47% bytes) moved the prefix slope from 11.83 to **11.81
  (0.998x)**. bf16 could not run — CBs overflow L1 (1 635 456 B vs 1 572 864 B), which is the
  control proving the dtype reaches the op's own buffers. So the op scales with element COUNT,
  not byte count, and "cut the redundant K/V streaming" is dead as a lever. Note the override
  must target the **global** cache only (`row_dim == GLOBAL_PACKED_DIM`): the sliding path
  hard-requires BFP8_B K/V at `ring_joint_sdpa_device_operation.cpp:589`, inside a
  `if (args.has_sliding_window())` block at line 553.
* **(b) tt-npe.** Build it, put it on `$PYTHONPATH`, run `python -m tracy ... --analyze-noc-traces`.
  This is the *only* thing that populates `NOC UTIL` / `DRAM BW UTIL` / `ETH BW UTIL`. Modelled
  from NoC event traces, not measured, but real signal.
* **(c) `DeviceZone` zones inside the kernels — ground truth, and the actual answer.**
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/` — `ring_joint_reader.cpp` and
  `compute/ring_joint_sdpa.cpp`. Zones around the gather wait, QK, softmax and PV. **This is the
  work to hand to whoever owns the op**; it is a rebuild-iterate loop over thousands of lines of
  kernel code.

### 2. ~~~10–20% of TTFT is attributed to nothing verifiable~~ — CLOSED 2026-09-17

**Resolved, and the hypothesis was wrong.** Layer-count differencing (`GEMMA4_NUM_LAYERS`, whole
model at N=60/12/6, `scripts/parse_nlayers.py`) gives `a(C,N) = F + N·L` **linear to 5–6%**:

| chunk | F (non-layer) | L (per layer) |
|---:|---:|---:|
| 8192 | **0.9 ms** (0.4% of TTFT) | 4.029 ms |
| 2048 | **0.5 ms** (0.4% of TTFT) | 2.181 ms |

Embedding + final norm + LM head + inter-layer CCL are **0.4% of TTFT**. The whole 7–11%
discrepancy was the isolated-layer benchmark's per-replay overhead (**1.07x @2048, 1.11x @8192**).
Scale isolated-layer absolutes by that; ratios unaffected. Bonus: the floor is now **70.4 ms**
from whole-model numbers alone vs the 70.6 ms measured excess (0.3%), and the prefix slope per
global layer is 1.188 ms/index confirmed across three layer counts.

*(original text below, kept for the reasoning)*

### 2-orig. ~10–20% of TTFT is attributed to nothing verifiable
`10×global + 50×sliding` from the isolated-layer benchmark **over-counts** the measured `a(C)`:

| chunk | reconstruction | measured `a(C)` | ratio |
|---:|---:|---:|---:|
| 2048 | 140.6 ms | 131.3 ms | 1.07x |
| 4096 | 195.3 ms | 174.2 ms | 1.12x |
| 8192 | 268.3 ms | 242.7 ms | 1.11x |

…**and** the single-layer graph contains no embedding, no final norm, no LM head and no
inter-layer CCL, so the real error is *larger* than the overshoot shows. The isolated benchmark's
per-replay overhead more than hides those.

**Do not** try to close this with a profiled whole-model run without checking the cost first: the
isolated 2-layer capture already wrote a 4.8 GB device log, and a 60-layer × 4-replay run is
~60x more ops. **Better idea, no profiler needed:** run the whole model at **two different layer
counts** and difference them — that gives the true per-layer marginal cost *and* the non-layer
fixed cost with no isolated-layer inflation at all. Check whether `gemma4_d_p` supports a layer
count override first.

> **Done 2026-09-23** (see EXPERIMENTS.md, "Resolved"): `a(L) = 0.54 + 1.503·L` ms. Only ~0.5 ms per chunk is
> outside the layers, and a 6-layer traced tracy capture shows ≥ 96% of each chunk inside op kernels.

### 3. ~~Mechanism of the floor's two biggest items~~ — LARGELY ANSWERED 2026-09-17

Standalone microbenchmarks (`scripts/micro_floor.py`, `scripts/micro_norm.py`):
* **`rms_norm` is width-bound and precision-insensitive.** 170 µs @(256x5376) vs 166 µs
  @(1024x5376) — 4x rows, 0.98x time. **LoFi buys only 19%**, so precision is not the cost.
  **Width-sharding gives ~2x** (best: 6x8=48 cores, block 1x28t, 0.076 vs 0.170 ms), NOT the ~8x
  assumed. More cores is not better — 96 cores loses to 48. `subblock_w <= 3` in fp32 mode.
  => saves ~12.6 ms/chunk, and needs a **block-sharded activation** plumbed through the
  surrounding ops: moderate integration work, not a config flip.
* **Matmul: downgraded.** A naive standalone `M=256,K=N=5376` bfp8 matmul is **243 µs** vs the
  in-model **193 µs** — the model's config is already better than a default. Real gains need
  genuine matmul tuning; rank it below the norm work.

*(original text below)*

### 3-orig. Mechanism of the floor's two biggest items
We know *which* ops and *how much*, not *why*:
* **Matmul** achieves **≥150 GB/s** on weight reads (127 MB/layer/device, bfp8, TP=4) — roughly a
  third of plausible BH peak. 2x the math from M=512→1024 costs only **1.09x** the time, so it is
  saturated at the weight-read bound by M=512. Cause of the low bandwidth unknown.
* **`rms_norm`** takes ~110 µs for one 168-tile-wide row-block at HiFi4 + `fp32_dest_acc_en`.
  Cause unknown — precision? multi-pass? CB thrash? **This decides whether width-sharding
  actually delivers the ~26 ms/chunk.**

Do these as **standalone microbenchmarks at the exact shapes**, not inside a 60-layer build:
`(256 × 5376)` and `(1024 × 5376)` for the norm; `M=256, K=5376, N=5376` bfp8 for the matmul.

### 4. The `q_chunk` win is NOT end-to-end verified
Every `q_chunk` number is a **single-layer** delta. The projected **−4.8% at 256k / −1.9% TTFT**
for q=128 at chunk 8192 is a projection, not a measurement. One `ctx_32k` whole-model run per
config confirms it. Also untested: **16384 and 32768** (measure, do not predict — see below), and
the **4x8 mesh** (rows/device is identical to 8x4 because CP4 halves the head count while doubling
the slab, so the same optima *should* transfer — a cheap falsifiable check).

## Two retractions from this session — read these before trusting any model

1. **"69% movement / 31% math" inside the global SDPA — RETRACTED.** Derived from the chunk-2048
   q64→32 ablation via `cost = depth×(q·m′ + r′)`, which gives `r′ = 142·m′`. The chunk-8192
   q64→128 ablation gives `r′ = 13·m′`. An order of magnitude apart ⇒ no single two-term model
   fits both. **The 2048 ablation changes rows/core AND q_chunk together**, so it cannot separate
   per-unit overhead from "per-row efficiency depends on q_chunk". The **8192 ablation is the
   better design** (rows/core fixed at 128, only units/core 2→1) and gives **per-unit overhead
   ≈17%**.
2. **"`rms_norm` is exactly chunk-invariant (ratio 0.98)" — RETRACTED.** That came from comparing
   the chunk-2048 capture against the **2026-09-09** chunk-8192 capture, where the *identical*
   norm — same shape, same 32 cores — took **99.3 µs instead of 134.2 µs**. Same-branch the ratio
   is **1.22x**, not 0.98x. The mechanism (row-parallel, 8 vs 32 cores) is unaffected and directly
   observed in all three captures; only the tightness of the claim was wrong.

## Traps — do not re-pay

* **Never compare per-op timings across builds.** Two things differed between the 2026-09-09
  capture and this branch: `GatherDeviceOperation` → `GatherCodegenDeviceOperation` (590 → 176 µs),
  and the same `rms_norm` at the same shape and core count (99.3 → 134.2 µs). The second one cost
  a headline claim. A same-shape/same-core-count op is **not** a safe cross-build anchor.
* **An ablation that changes two variables cannot feed a two-term model.** It will produce a
  confident, wrong split. Prefer the variant that holds one quantity fixed, and **always test a
  model on a second ablation before believing it**.
* **`a(C)` is concave, not affine** — per-op intercept solves disagree by 39% (Matmul), 39%
  (RingJointSDPA), 17% (LayerNorm), and Matmul's (4096,8192) intercept (1031 µs) *exceeds* its
  total cost at chunk 2048 (844 µs), which is impossible. **Never extrapolate to C=0.** Use the
  in-range measure `t(2048) − t(8192)/4` ("excess over perfect token scaling") instead.
* **The profiler's utilization columns are EMPTY on this path**: `NOC UTIL`, `MULTICAST NOC UTIL`,
  `DRAM BW UTIL`, `ETH BW UTIL`, `DEVICE COMPUTE CB WAIT FRONT/RESERVE BACK`, and
  `DEVICE KERNEL DURATION PER CORE MIN/MAX`. `PM IDEAL [ns]` is a stub (median 1 ns). They need
  `--analyze-noc-traces` **and** a built tt-npe.
* **`CORE COUNT` cannot verify SDPA occupancy.** The factory assigns kernels to *every* core; a
  core with no Q chunks "is NOT dead — it runs padded handshake iterations
  (`loop_q_count = *_max_q_per_core`)". So it reports ~114 at every chunk size.
* **Never pass `--device-trace-profiler`.** It profiles only trace regions: one row per replay,
  empty `OP NAME`, and it breaks the host/device merge.
* **`python -m tracy` launches a blocking-looking Tracy WASM web-UI server** before generating the
  ops report, so a run looks hung after the test has already passed. Budget ~15 min/run and a
  generous `timeout`; `--process-logs-only` recovers a killed report from `.logs/`.
* **`ninja <target>` does not install** — use `cmake --build build_Release --target install`, then
  check `ls -la build_Release/lib/_ttnncpp.so`. Kernel `.cpp` edits ARE JIT-compiled and do take
  effect, which makes a mixed host+kernel change especially confusing.
* `timeout --signal=INT` does not kill a device-hung pytest; use `timeout -k 10 <s>`.
* `--collect-only` opens all 32 chips — never run it alongside a live job.

## Diagnostics to re-apply

In `models/demos/gemma4_d_p/tt/attention/ring_prefill.py`, in the `if program_config is None:`
block (~line 370) and at the `ring_joint_scaled_dot_product_attention` call (~line 412):

```python
# q_chunk override for GLOBAL layers only. The `q in {64,128}` / `k == 128` allowlist lives
# inside `if (args.has_sliding_window())`, so global layers are unrestricted.
_q_chunk = int(os.environ.get("GEMMA4_GLOBAL_Q_CHUNK", "64")) if not sliding_window_size else 64
program_config = ring_prefill_program_config(
    mesh_device, ccl_manager, head_dim, q_chunk_size=_q_chunk, k_chunk_size=_k_chunk
)
...
num_links=int(os.environ.get("GEMMA4_RING_GATHER_LINKS", ccl_manager.num_links)),
```

`import os` is already present. These are Python-only — **no rebuild needed**.

## Commands

```bash
source /data/kmabee/gemma4_runs/env.sh    # TT_METAL_HOME, HF paths, mesh-8x4 tt_cache
cd $TT_METAL_HOME
cmake --build build_Release --target install   # NOT `ninja <target>`

D=models/demos/gemma4_d_p/demo/text_demo_prefill.py

# whole-model per-chunk times (this is the GROUND TRUTH a(C) and slope(C))
timeout -k 10 2400 ./python_env/bin/python3 -m pytest \
  "$D::test_prefill_long_context_traced[blackhole-readback_final-ctx_32k-chunk8192-text-8x4]" -sv

# per-layer-type depth curves: a() and slope() for a global AND a sliding layer separately
timeout -k 10 2700 ./python_env/bin/python3 -m pytest \
  "$D::test_prefill_layer_perf_chunk_n[blackhole-chunkall-both-sz2048-ctx_256k-8x4]" -sv

# ONE layer type only (use for q_chunk ablations -- half the runtime)
GEMMA4_GLOBAL_Q_CHUNK=128 timeout -k 10 2700 ./python_env/bin/python3 -m pytest \
  "$D::test_prefill_layer_perf_chunk_n[blackhole-chunkall-global-sz8192-ctx_256k-8x4]" -sv

# per-op profile at chunk index 0 (the floor). ~7.5 min device + ~10 min post-process, 10 GB logs
timeout -k 10 3600 ./python_env/bin/python3 -m tracy -r -p -v -o <out>/profiler \
  -m pytest "$D::test_prefill_layer_perf_chunk_n[blackhole-chunk0-both-sz2048-ctx_256k-8x4]" -sv
```

Test-id order is **closest decorator first**: `chunk{idx}-{layer_type}-sz{chunk}-ctx_{n}k-{mesh}`.
`chunk_idx` accepts an integer or `all`; `layer_type` is `global` / `local` / `both`.

## Analysis scripts

**Not in-tree** — they live in `~/debug-docs/gemma4_chunk_size_anatomy-noissue/scripts/` (private repo `kmabeeTT/debug-docs`).

| script | what it does |
|---|---|
| `why_2x.py` | splits the 256k total into its two measured terms; no device needed |
| `occupancy_model.py` | the zero-parameter `depth(C)·q/C` model vs all five measured chunk sizes |
| `parse_curves.py` | fits `a()` and `slope()` per layer type from `[layer_perf_chunk] RESULT` lines: `parse_curves.py <run_tag> ...` |
| `floor_ops.py` | per-op table + affine test across 2–3 profiled captures: `floor_ops.py --csv 2048=<csv> 4096=<csv> 8192=<csv> --region local-chunk0` |
| `project_fixes.py` | projects the levers per chunk size |
| `run_*.sh` | the runners used for each batch (env, tags, gating) |

`--region` is a regex on the signpost name: `local-chunk0`, `global-chunk0`, etc.

## Deployment answer, for reference

| chunk | TTFT | T(256k) | with both levers |
|---:|---:|---:|---|
| 2048 | 131.3 ms | 28.66 s | 110 ms / 24.1 s — **still 1.94x, hard pass** |
| 4096 | 174.2 ms | 17.19 s | 154 ms / 15.9 s — **1.58x better TTFT for 1.28x worse total** |
| 8192 (today) | 242.7 ms | 13.71 s | 218 ms / 12.4 s — **−10% TTFT, −9.5% total, no chunk-size change** |

The lowest-risk win is applying both levers to **today's 8192** — it needs no chunk-size decision.
