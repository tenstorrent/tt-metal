# Running pi0.5 VLM prefill — PCC & profiler (TP=1 / TP=4 / TP=8)

Per-chip device-kernel benchmarking of the VLM prefill (Gemma-2B ×18, 3-camera /
seq=1024). All commands run from the repo root.

## Setup (once per shell)

```bash
cd <tt-metal repo root>
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
# checkpoint (HF cache snapshot):
export PI05_CHECKPOINT_DIR=$HOME/.cache/huggingface/hub/models--lerobot--pi05_libero_base/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd
# production perf flags (head-split, mm-tune, bf8, chunk=1024, num_cameras=3, ...):
set -a; source models/experimental/pi0_5/_bench_runs/pi05_production.env; set +a
unset PI0_VLM_MLP_MINIMAL   # minimal_matmul regresses the TP MLP; bf8-only is the win
```

> If the device is wedged (e.g. `Read 0xffffffff`), reset: `python_env/bin/tt-smi -glx_reset`
> (or `-r`). Always start clean.

## Tests

| | test | mesh / chips | notes |
|---|---|---|---|
| **TP=1** | `test_prefill_tp1_pcc` | 1 chip — `PI0_DEVICE_ID=1` → physical **chip 9** | single-device `PaliGemmaBackboneTTNN.forward_vlm` |
| **TP=4** | `test_prefill_tp4_pcc` | 1×4 — chips **8–11** | `StagePrefillTP4`, bf8 MLP default |
| **TP=8** | `test_prefill_tp4_pcc` + `PI0_TP=8` | 1×8 — chips **8–15** | TP degree derived from mesh |

`T=models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py`

**Checkpoint-free option:** `test_prefill_tp4_perf_dummy` runs TP=4/8 with random
weights (no checkpoint) — for profiling, op shapes/timing are identical to real
weights. It also checks PCC (torch ref on the *same* dummy weights; bar 0.97 since
random weights stress bf8 more than trained). Honors `PI0_TP` / `PI0_VLM_CHUNK_SIZE`
/ `PI0_SKIP_TORCH_REF`. Example: `TT_VISIBLE_DEVICES=8,9,10,11 python_env/bin/python
-m tracy -p -v -r --op-support-count 8000 -m pytest -sq $T::test_prefill_tp4_perf_dummy`

### PCC (correctness)

```bash
# TP=1  (chip 9 — chip 8 is the mmio chip; use DEVICE_ID=1)
TT_VISIBLE_DEVICES=8,9,10,11 PI0_DEVICE_ID=1 \
  python_env/bin/python -m pytest -sq $T::test_prefill_tp1_pcc

# TP=4
TT_VISIBLE_DEVICES=8,9,10,11 \
  python_env/bin/python -m pytest -sq $T::test_prefill_tp4_pcc

# TP=8
TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 PI0_TP=8 \
  python_env/bin/python -m pytest -sq $T::test_prefill_tp4_pcc
```
Expected PCC ≥ 0.99 (bf8 ≈ 0.9935–0.9939; bf16 path 0.995127). Prints
`Prefill TP=… stage PCC vs torch: 0.99xxxx (shape (1, 1024, 2048))`.

### Profiler (device-kernel timing, tracy)

Prefix any PCC command with the tracy wrapper:

```bash
TT_VISIBLE_DEVICES=8,9,10,11 \
  python_env/bin/python -m tracy -p -v -r --op-support-count 8000 \
  -m pytest -sq $T::test_prefill_tp4_pcc
```
CSV lands at `generated/profiler/reports/<TS>/ops_perf_results_<TS>.csv`.

> **Read device 1, not device 0.** On this box, **chip 8 (the mmio chip) has corrupt
> eager `DEVICE KERNEL DURATION`** (bogus ~3.5e12 ns on multi-core ops). Aggregate from
> a **sane device** (TP=4 → device 1; TP=8 → device 1/5/6) and **drop rows with
> `DEVICE KERNEL DURATION ≥ 1e8` ns**.
>
> **Always report the init-EXCLUDED forward.** Slice device-1 ops from the **first
> `MatmulDeviceOperation`** onward — the ~95 ops before it are one-time init
> (**~0.54 ms**: 39× `TilizeWithValPadding` + 40× `Typecast` weight-tilization,
> RoPE-table precompute, embedding setup) that a traced pipeline amortizes. Summing
> *all* device-1 ops over-counts by that ~0.54 ms (e.g. 12.10 ms all-ops = **11.56 ms
> forward** + 0.54 ms init). Parse:
> ```python
> rows=[d for d in csv.DictReader(open(f)) if d["OP TYPE"].strip()=="tt_dnn_device" and d["DEVICE ID"].strip()=="1"]
> i0=next(i for i,d in enumerate(rows) if d["OP CODE"].strip()=="MatmulDeviceOperation")
> fwd=sum(float(d["DEVICE KERNEL DURATION [ns]"] or 0) for d in rows[i0:] if 0<float(d["DEVICE KERNEL DURATION [ns]"] or 0)<1e8)
> ```
> Raise `--op-support-count` for >1 iteration to avoid profiler-buffer overflow.

## Knobs

| env | effect |
|---|---|
| `PI0_TP` | TP degree for `test_prefill_tp4_pcc` (default 4; set 8 with 8 visible chips) |
| `PI0_DEVICE_ID` | single-device chip index for TP=1 (use 1 = chip 9; chip 8 timing is corrupt) |
| `PI0_VLM_CHUNK_SIZE` | prefill seq / MLP chunk (1024 = 3-camera single-chunk; default) |
| `PI0_NUM_CAMERAS` | 3 (prod); seq = 256·N_cam + 256 (1cam→512, 3cam→1024) |
| `PI0_SKIP_TORCH_REF=1` | skip the CPU torch reference (faster profiling; PCC not checked) |
| `PI0_TP4_ATTN_HEADPAR=1` | **opt-in** head-parallel attention (default off — regresses on this Linear fabric) |

## Reference numbers (per-chip **forward**, init-excluded, prod env, bf8 default)

3-camera (seq=1024, `PI0_VLM_CHUNK_SIZE=1024`):

| TP | PCC | forward (init-excl) | all-ops sum (+~0.54 ms init) |
|---|---|---|---|
| TP=1 | 0.9942 | 18.1 ms | — |
| TP=4 | 0.9939 | **11.56 ms** | 12.10 ms |
| TP=8 | 0.9935 | 10.60 ms | — |

2-camera (seq=768, `PI0_VLM_CHUNK_SIZE=768`):

| TP | forward (init-excl) | all-ops sum |
|---|---|---|
| TP=4 | ~8.7 ms | 9.22 ms |

(TP=4 baseline before this session's opt was 20.4 ms.) Numbers drift ±10–20% run-to-run
(CCL/matmul timing jitter on this box) — the init-excluded forward is the canonical metric.
SDPA is the largest single op (~2.16 ms @ seq=1024, O(seq²) → ~0.72 ms @ seq=768) and is
at its compute floor. Head-parallel attention & fused/async CCL are gated by the fabric
(4-node Linear, no ring) — see the design spec under `docs/superpowers/specs/`.

---

# 1×8 full e2e pipeline — vision + prefill + denoise on a single mesh

`Pi0_5GLX1x8Pipeline` (pipeline_1x8.py) runs all three stages on the same 1×8
mesh (chips 8–15): SigLIP DP (3 real + 5 zero-dummy cams), Prefill TP=8,
replicated 5-step denoise. On-device CCL for cross-stage handoff (no host
bounce, no fabric sockets). TIER A precompute eliminates per-step block-mod
matmuls; staged sub-traces give true per-stage traced timing.

## Setup

```bash
cd <tt-metal repo root>
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
set -a; source models/experimental/pi0_5/_bench_runs/pi05_production.env; set +a
export TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PI0_TP=8 PI0_TP4_ATTN_HEADPAR=1 PI0_MLP_BS=1 PI0_MLP_FUSED_RS=0
# camera count (3 = production training spec; 2-cam supported but PCC drops, see notes)
export PI0_NUM_CAMERAS=3
tt-smi -glx_reset   # always start clean
```

`P=models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8.py`

## Tests

| Test | What it does | Gating |
|---|---|---|
| `test_perf_1x8_eager` | One eager `sample_actions`; asserts shape + finite. | Always on |
| `test_perf_1x8_traced` | Captures e2e trace; reports `input_upload / trace_exec / output_readback` over `PERF_ITERS=20` replays + eager per-stage proportions. | Always on |
| `test_perf_1x8_traced_staged` | Captures 3 sub-traces (vision / prefill / denoise) on the same mesh, replays each separately, reports true per-stage traced ms. Trace region bumped to 256 MiB. | Always on |
| `test_perf_1x8_traced_2cq` | 2 command queues: H2D input upload on CQ1 overlapped with compute on CQ0 via `ttnn.record_event` / `wait_for_event`. Closes wall-clock toward the `trace_exec` floor. | Always on |
| `test_pcc_1x8_all_stages` | Vision + Prefill (isolated, same random prefix on both sides) + e2e (matched-seed noise) + estimated denoise = e2e / (vision · prefill). | `PI05_E2E_PCC=1` |

### Run all tests

```bash
# Perf — e2e single-trace breakdown (host overhead split)
python_env/bin/pytest -sq $P::test_perf_1x8_traced

# Perf — per-stage traced breakdown via 3 sub-traces
python_env/bin/pytest -sq $P::test_perf_1x8_traced_staged

# PCC — vision + prefill + e2e + estimated denoise
PI05_E2E_PCC=1 python_env/bin/pytest -sq $P::test_pcc_1x8_all_stages

# 2-cam variant: rerun with PI0_NUM_CAMERAS=2
PI0_NUM_CAMERAS=2 python_env/bin/pytest -sq $P::test_perf_1x8_traced_staged
```

Each test prints a startup `[env]` block listing which production flags are
set — confirms the pi05_production.env was sourced.

### Knobs (1×8 pipeline)

| env | effect |
|---|---|
| `PI0_NUM_CAMERAS` | Real-camera count (1..8); padded to 8 with zero dummies inside the pipeline. |
| `PERF_ITERS` | Timed iters in `test_perf_1x8_traced*` (default 20). |
| `WARMUP_ITERS` | Trace-replay warmup iters before timing (default 3). For tracy, set to 1 to slim the capture. |
| `PI05_E2E_PCC` | Enable `test_pcc_1x8_all_stages` (default off; runs CPU torch ref → slow). |
| `PI05_NUM_DENOISE_STEPS` | Override the 5-step default schedule (10 matches upstream training spec). |

## Profiling (tracy + annotator) — reproduce the perf CSVs

End-to-end recipe for capturing a tracy profile of the 1×8 pipeline and
annotating it into a per-stage, per-layer CSV.

### 1. Capture (1 warmup + 1 timed iter)

`WARMUP_ITERS=1 PERF_ITERS=1` is the minimum that produces a usable canonical
inference in the CSV (any less and the trace allocator's first-replay overhead
contaminates the numbers).

```bash
# Setup (same env as perf runs)
cd <tt-metal repo root>
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
set -a; source models/experimental/pi0_5/_bench_runs/pi05_production.env; set +a
export TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PI0_TP=8 PI0_TP4_ATTN_HEADPAR=1 PI0_MLP_BS=1 PI0_MLP_FUSED_RS=0
export PI0_NUM_CAMERAS=3   # or 2 for the 2-cam capture
export PERF_ITERS=1 WARMUP_ITERS=1
tt-smi -glx_reset

# Tracy run — ~4-5 minutes (most of it is JIT'ing kernels for the first iter)
P=models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8.py
python_env/bin/python -m tracy -p -r -v --op-support-count 100000 \
  -m pytest -sq $P::test_perf_1x8_traced_staged
```

When the run finishes, tracy prints the output path, e.g.:
```
OPs csv generated at: generated/profiler/reports/<TS>/ops_perf_results_<TS>.csv
```

The directory also contains `tracy_profile_log_host.tracy` (Tracy GUI file —
open with the Tracy desktop app for an interactive flame-graph view across
all 8 devices).

### 2. Annotate (add STAGE / LAYER / SUBSTAGE columns)

```bash
CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
python_env/bin/python _bench_runs/annotate_ops_csv_v4.py "$CSV"
# Writes <CSV>_annotated_v4.csv next to the raw + prints the per-stage summary.
```

The annotator's stdout prints the per-call breakdown — this is the canonical
output you should use for headline timing:

```
=== Inference 32 (canonical = trace replay/last) — per-call breakdown ===
  STAGE              rows                count   kernel_ms
  ----------------------------------------------------------------
  prefix_setup       95479..95491        13      0.081
  siglip             95492..96153       662      5.676
  vlm_prefill        96154..96586       433      3.740
  denoise_step_1     96587..97026       440      3.779
  denoise_step_2     97027..97465       439      3.750
  denoise_step_3     97466..97904       439      3.769
  denoise_step_4     97905..98343       439      3.768
  denoise_step_5     98344..98782       439      3.761
  ----------------------------------------------------------------
  TOTAL/inference                              28.323  ← EXCLUDES init
```

### 3. How to read the annotated CSV

The annotator adds 3 columns to every row: `STAGE`, `LAYER`, `SUBSTAGE`.

| Column | Values | Meaning |
|---|---|---|
| `STAGE` | `init_one_time` / `prefix_setup` / `siglip` / `vlm_prefill` / `denoise_step_{1..5}` (clean = canonical timed iter) **OR** with suffix `_warmup{N}` / `_trace_capture` (the other inferences) | Pipeline stage + which inference this op belongs to |
| `LAYER` | `1..18` (prefill / denoise) or `1..27` (siglip) | Transformer layer index within the stage |
| `SUBSTAGE` | `attn` / `mlp` / `head` / `tail` / empty | Sub-region within a denoise step |

> **Important caveat — multi-device labeling.** The v4 annotator was written
> for single-device CSVs. On our 8-device capture it sees `8 chips × 4 iters
> = 32` total inferences and only labels the very last GLOBAL inference as
> canonical (suffixless). Devices 2-7's actual timed iter ends up tagged
> `_warmup{N}` — about **80% of rows look like warmup** even when the per-row
> timing data is real-iter data on a non-zero device. **Don't filter by
> suffix-less STAGE on multi-device CSVs.**
>
> **What to do instead**: trust the annotator's printed per-call summary
> (above) for headline timing; for per-op drill-down, filter on `STAGE` base
> (ignoring suffix) + `LAYER` + `SUBSTAGE` on device 1 (sane; **device 0 has
> corrupt `DEVICE KERNEL DURATION`** per the TP=8 note above), since trace
> replay is deterministic so any iter's row gives identical timing.

> **Additional caveat — CCLs are NOT in the suffix-less canonical rows.**
> The annotator's "canonical inference" is just the last ~1518 rows of the
> CSV by host timestamp (one inference's worth of ops). But CCL ops
> (`AllGatherDeviceOperation`, `ReduceScatterDeviceOperation`) live at
> *stage boundaries* — AllGather at the END of vision, ReduceScatter at each
> prefill MLP layer — not at the tail of the inference. By the time the last
> op of the inference (a denoise-step-5 tail op) runs, the CCL ops finished
> much earlier and are in the middle of the CSV.
>
> Result: filtering on `STAGE == "siglip"` (exact suffix-less match) returns
> the SigLIP encoder ops but NOT the AllGather that ends SigLIP. Filtering
> on `STAGE == "vlm_prefill"` returns the prefill matmuls but NOT the
> ReduceScatters between them. All those CCL ops are tagged with
> `_warmup{N}` suffixes. **To get CCLs, strip the suffix before grouping**
> (see the drill-down recipes below).

### Multi-device caveat — TL;DR

| What you might do | What you actually get | What to do instead |
|---|---|---|
| Filter `STAGE == "siglip"` | SigLIP non-CCL ops, ~80% of total | Filter `STAGE ~ /^siglip/` (regex prefix match) |
| Filter suffixless STAGE only | Tail of inference, no CCLs | Strip `_warmup\d+$` and `_trace_capture$` from STAGE, then group |
| Sum kernel times by suffixless STAGE | Underestimate (missing CCLs) | Use the annotator's stdout summary (`TOTAL/inference 28.32 ms`) which already filtered correctly, OR use the strip-and-group recipe below |

### 4. Useful drill-down recipes

**The strip-and-group pattern.** Multi-device CSVs require stripping the
`_warmup{N}` and `_trace_capture` suffixes from `STAGE` before grouping —
otherwise CCL ops drop out (see the second caveat above).

Per-stage kernel-time breakdown on device 1 (any iter — trace replay is
deterministic, so each iter has identical kernel times; divide by 4 iters):

```bash
CSV=generated/profiler/reports/<TS>/ops_perf_results_<TS>_annotated_v4.csv
awk -F, '
  NR>1 && $7=="1" && $1!="init_one_time" && $22+0>0 && $22+0<1e8 {
    s = $1; sub(/_warmup[0-9]+$/, "", s); sub(/_trace_capture$/, "", s)
    sum[s] += $22; cnt[s]++
  }
  END {
    order = "prefix_setup siglip vlm_prefill denoise_step_1 denoise_step_2 denoise_step_3 denoise_step_4 denoise_step_5"
    n = split(order, arr, " ")
    total = 0
    for (i = 1; i <= n; i++) {
      s = arr[i]
      if (s in sum) { printf "  %-22s %7.3f ms/iter  (%d ops total)\n", s, sum[s]/4/1e6, cnt[s]; total += sum[s]/4 }
    }
    printf "  %-22s %7.3f ms/iter\n", "TOTAL", total/1e6
  }' "$CSV"
```

Top-10 slowest matmuls in `vlm_prefill` on device 1:

```bash
awk -F, 'NR>1 && $7=="1" && $1 ~ /^vlm_prefill/ && $4=="MatmulDeviceOperation" && $22+0<1e8 {
  s = $1; sub(/_warmup[0-9]+$/, "", s); sub(/_trace_capture$/, "", s)
  printf "%s/L%-2s\t%7.1f us\n", s, $2, $22/1000
}' "$CSV" | sort -t$'\t' -k2 -rn | head -10
```

CCL (cross-chip) ops per iter on device 1 — strip the suffix to catch ALL of
them, then divide by 4 iters per device:

```bash
awk -F, 'NR>1 && $7=="1" && $4 ~ /AllGather|ReduceScatter/ && $22+0<1e8 && $22+0>0 {
  s = $1; sub(/_warmup[0-9]+$/, "", s); sub(/_trace_capture$/, "", s)
  key = $4 "/" s
  c[key]++; sum[key]+=$22
}
END {
  printf "  %-50s %12s %s\n", "OP CODE / STAGE", "ops/iter", "ms/iter"
  for (k in c) printf "  %-50s %12.2f %7.3f\n", k, c[k]/4, sum[k]/4/1e6
}' "$CSV" | (head -1; sort -k3 -rn)
```

Expected CCL summary per iter on a sane device (3-cam, commit `38ac051ee68`):
~73 CCL ops totaling ~3.0-3.3 ms of kernel time (AllGather inside vision,
ReduceScatter × 18 in prefill MLPs, plus per-step reductions in denoise).

### 5. Wall-clock vs kernel-time gap

For commit `38ac051ee68`, 3-cam, PERF_ITERS=20:

| Source | Value |
|---|---:|
| `trace_exec` wall-clock (`test_perf_1x8_traced`) | 33.97 ms |
| Sum of `DEVICE KERNEL DURATION` (canonical iter, annotator) | 28.32 ms |
| Gap | ~5.65 ms |

The gap is **NOT** unaccounted CCL — CCL kernel time on each chip
(~3.0-3.3 ms/iter from `AllGather` + `ReduceScatter`) already includes
cross-chip fabric wait. The gap is **trace-dispatcher overhead** (~1-2 µs/op
× ~1500 ops/iter ≈ 2-3 ms) + per-op firmware setup/teardown + L1↔DRAM DMA
between consecutive ops — work that happens between kernel runs and isn't
captured in any single op's `DEVICE KERNEL DURATION`.

## Reference numbers (commit `38ac051ee68`, 2026-06-23)

### Perf — 2CQ replay (`test_perf_1x8_traced_2cq`, PERF_ITERS=20)

| Config | mean ms | min ms | vs single-CQ traced_total | gap to trace_exec floor |
|---|---:|---:|---:|---:|
| 3-cam | **35.30** | 34.94 | −14 ms (−28%) | +2 ms |
| 2-cam | **33.12** | 32.76 | −6 ms (−15%) | +2 ms |

H2D input upload runs on CQ1 overlapped with CQ0 compute; sync barrier +
output readback at end of each iter is the ~2 ms residual that can't hide.

### Perf — per-stage TRACED breakdown (`test_perf_1x8_traced_staged`, PERF_ITERS=20)

| Stage | 3-cam mean | 2-cam mean | Δ (3→2 cam) |
|---|---:|---:|---:|
| input_upload (host) | 14.25 ms | 6.49 ms | — |
| **vision** (SigLIP DP, 8 chips) | **5.01 ms** | **5.00 ms** | 0 (DP pads to 8 cams regardless) |
| **prefill** (TP=8) | **9.62 ms** | **8.26 ms** | −1.36 (1024→768 prefix) |
| **denoise** (5 step, replicated) | **19.37 ms** | **18.51 ms** | −0.86 (shorter KV in cross-attn) |
| output_readback | 1.24 ms | 1.12 ms | — |
| **compute (v+p+d)** | **34.01 ms** | **31.78 ms** | −2.23 |
| traced_total | 49.49 ms | 39.39 ms | — |

### PCC (`test_pcc_1x8_all_stages`, PI05_E2E_PCC=1)

| Stage | 3-cam | 2-cam | Target |
|---|---:|---:|---:|
| vision | 0.999684 ✓ | 0.999685 ✓ | ≥ 0.997 |
| prefill | 0.994639 ✓ | 0.994757 ✓ | ≥ 0.99 |
| denoise (est) | 1.001883 | 1.001869 | — |
| **e2e** | **0.996197 ✓** | **0.996302 ✓** | ≥ 0.95 |

Both cam counts now pass; e2e PCC is essentially identical (~0.996) — both
close to the 28-chip baseline of 0.9988. The fix in commit `38ac051ee68` was
adding position-aware suffix RoPE + expert attention mask to match how torch
ref builds them in `_denoise_forward`.

## Notes

- The reported `denoise` PCC is the multiplicative-drift estimate `e2e /
  (vision · prefill)` (rough proxy; PCC isn't strictly multiplicative). Pure
  isolation would require injecting torch-computed KV into TT denoise (per-layer
  shape/layout conversion — not implemented).
- E2E noise matches via the "seed-around-both" pattern: `torch.manual_seed(SEED)`
  before `pipe.sample_actions` AND before constructing `Pi0_5Model` then calling
  its `sample_actions`. Empirically better than the 28-chip `seed(SEED+1) just
  before each call` pattern on this 1×8 path (verified during dev).
- TIER A precompute (commit 7f11f571159) brought 3-cam denoise from 24.6 → 18.8 ms
  and e2e PCC from 0.986 → 0.992 — by moving 18×N_steps mod-Dense matmuls + the
  final-norm Dense to host, fewer bf16 rounding cascades.
- TIER B (`keep_padded` + phantom mask + `fill_implicit_tile_padding`) was
  ported then reverted: +0.8 ms denoise for +0.003 PCC on this pipeline,
  because the prefix is already tile-aligned (1024 / 768) so the prefix-side
  benefit is zero. See commit notes if you want to revisit.
