# tt-metal — Experiment Log / Working Memory

Reusable notes on optimization experiments so context survives across sessions.
Newest entries on top. Keep commands copy-pasteable.

---

## Environment setup (run before anything)

```sh
cd /home/iguser/rtp/tt-metal
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
```

- Hardware here: **Blackhole quad (BH-QB), mesh 1×4**, 110 worker cores. `/dev/tenstorrent` present.
- Fabric: tests use `FabricConfig.FABRIC_1D`. In a standalone script open the mesh with:
  ```python
  ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
  dev = ttnn.open_mesh_device(ttnn.MeshShape(1,4), l1_small_size=32768, num_command_queues=2)
  ```
  (`open_mesh_device` does NOT take a `fabric_config` kwarg — set it separately.)
- Weights already downloaded under `models/experimental/seamless_m4t_v2_large/weights/seamless-m4t-v2-large/` (`vocoder_v2.pt`, etc.).
- Git: repo root is a git repo, branch `ign/seamless_m4t_v2_large_exps_wip`. `/home/iguser/rtp` itself is NOT a git repo.

---

## Project: SeamlessM4T v2 CodeHiFi-GAN vocoder optimization

### Key files
- Impl: `models/experimental/seamless_m4t_v2_large/tt/tt_code_hifigan.py`
- PCC test: `models/experimental/seamless_m4t_v2_large/tests/pcc/test_code_hifigan.py`
  - gate `PCC_THRESHOLD = 0.99`, `unit_seq = 1024` (max), batch 1.
- Torch reference: `models/experimental/seamless_m4t_v2_large/reference/torch_code_hifigan.py`
- Perf report snapshot analyzed: `vocoder/exp0.txt` (untracked; `tt-perf-report` output of a profiled PCC run).

### How to run the PCC test (the correctness gate)
```sh
python -m pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_code_hifigan.py -x -q -s 2>&1 | grep -iE "PCC @|PASSED|FAILED"
```
Warm-cache run ≈ 5 s; cold (first compile of new kernels) ≈ 25–30 s.

### Model shape facts (unit_seq=1024)
- 5 upsample stages, `upsample_rates=[5,4,4,2,2]`; timeline grows `1280 → 6400 → 25600 → 102400 → 204800 → 409600`.
- Channels halve each stage: 256,128,64,32,16. `resblock_kernel_sizes=[3,7,11]`, dilations `[1,3,5]`.
- Late stages (long timeline, few channels) dominate device time.
- `t_audio` bucketed to 256 (`_VOCODER_TAUDIO_BUCKET`); timelines are multiples of 256, so the stride-1 timeline-bucket pad/trim is a no-op here (not a bottleneck).

---

## Perf report analysis (baseline, from `vocoder/exp0.txt`)

Total device kernel time ≈ **286,153 µs**. Op-to-op gap ≈ 66.4 s but that is almost entirely one-time
JIT kernel compilation (program-cache misses); steady-state gaps are small. **We optimize device time, not the gap** (user said "forget tracing").

Device time by category (stacked report):
- **TM (layout) ~54%**: Slice 23.3% (738 ops), UntilizeWithUnpadding 17.5% (315), TilizeWithValPadding 5.7%, Tilize 3.6%, Concat 2%.
- **DM ~19%**: InterleavedToSharded 12.75% (390), PaddedSlice 2.6%, ShardedToInterleaved 1.7%.
- **Compute ~24%**: Conv2d 19% (block 14% + height 4% + width 1%), Unary 3.3%, Binary 1.8%.
- Only ~19% is actual convolution; ~76% is layout/data-movement glue.

Root cause of the glue: every long conv was surrounded by slice-in / conv / slice-out(halo) / concat +
implicit tilize/untilize, repeated per chunk per conv in the late stages.

Reproduce a stacked report from a profiler CSV:
```sh
tt-perf-report <path>/ops_perf_results_*.csv | grep -A45 "Stacked report"
```

---

## Benchmarking method (use this for A/B — profiler is currently broken here)

Warm-cache, device-synced end-to-end forward, toggled by env flag in ONE process.
Reusable harness saved at **`vocoder/bench_vocoder.py`**: loads real weights, builds `TTSeamlessM4Tv2CodeHifiGan`,
runs one warm `forward` + sync, then median/min of N synced forwards.

```sh
# optimized vs baseline path, same build:
SEAMLESS_VOCODER_CONV1D_DRAM_SLICE=1 python vocoder/bench_vocoder.py   # optimized
SEAMLESS_VOCODER_CONV1D_DRAM_SLICE=0 python vocoder/bench_vocoder.py   # legacy manual chunk loop
```

Per-stage device timing is built in: `VOC_TIMING=1 python -m pytest ...` prints `[VOC-TIMING]` lines
(syncs the device per stage).

**Profiler caveat (IMPORTANT):** `python -m tracy -r -p -o <dir> -m pytest ...` currently CRASHES in
post-processing (`tools/tracy/process_ops_logs.py`, `_enrich_ops_from_*`) on this run — the raw device log
is ~3.9 GB and the host/device op counts mismatch ("Expected N but received M ops"). `--no-runtime-analysis`
doesn't fix it. So for now measure with the synced A/B harness, not tracy. The pre-existing baseline CSV
(`generated/profiler/reports/2026_07_11_09_52_10/ops_perf_results_*.csv`) still works with `tt-perf-report`.

---

## Experiments

### EXP-1 ✅ DONE — Replace manual conv1d chunk loop with device DRAM width-slicing
**Idea:** `ttnn.conv1d` natively supports `slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0)`
(0 = device auto-picks slice count). It slices the long timeline in DRAM and handles halo internally in
ONE op — replacing the hand-rolled Python chunk loop (per-chunk slice-in/slice-out + concat = the 23% Slice bucket).
This mirrors what the conv-transpose upsample path already does with `Conv2dDRAMSliceHeight`.

**Change (all in `tt_code_hifigan.py`):**
- Added `_vocoder_conv1d_dram_slice_enabled()` (default ON; disable with `SEAMLESS_VOCODER_CONV1D_DRAM_SLICE=0`).
- `_conv1d_run` gained optional `slice_config` passed through to `ttnn.conv1d`.
- `_conv1d` long-timeline branch (`seq > _HIFIGAN_MAX_CONV1D_TLEN`) now does ONE sliced conv1d instead of the loop
  (guards: interleave a sharded input to DRAM first; RM interleaved output; `timeline_chunked=True` config).
- Left prewarm/trace path untouched (default-off, out of scope; new path just cache-misses → correct internal weight prep).

**Results (BH 1×4, real weights, unit_seq=1024):**
- PCC: **0.99820** (baseline 0.99827; gate 0.99) — unchanged, passes.
- Warm forward median: **648 ms → 448 ms (~31% faster)**, min 642 → 444 ms.
- Isolation test: sliced conv1d bit-accurate (PCC 0.9999+) on long low-channel shapes.

### EXP-2 ✅ DONE — L1-resident resblock convs (single-shot HEIGHT, keep sharded)
**Idea:** every per-stage conv FITS BH L1 single-shot with HEIGHT sharding (measured to 409600x16 — the
`_HIFIGAN_MAX_CONV1D_TLEN=4096` cap is far too conservative; the real limiter was conv working set, not
activation size — activations are ~120KB/core spread over 110 cores). So run resblock convs single-shot
HEIGHT (no DRAM width-slice), `conv1` keeps output sharded, `conv2` accepts it → hand off in L1 instead of
round-tripping DRAM. Also moves long convs off the inefficient block-sharded auto path (FLOPs eff 2.5%→~25%).

**Change (`tt_code_hifigan.py`):** `_vocoder_resblock_l1_enabled()` (default on, `SEAMLESS_VOCODER_RESBLOCK_L1=0`
to disable) + `_VOCODER_L1_SINGLESHOT_ELEMS` budget (8M elems; all B=1 stages ≤6.55M) + `_vocoder_fits_l1_singleshot`
+ `_resolve_resblock_shard_layout` (returns HEIGHT on long timelines when it fits, else generic resolver).
`_conv1d` runs single-shot when `shard==HEIGHT and stride==1 and fits L1` (not just seq≤4096). `_resblock`
uses the new resolver. Falls back to DRAM width-slice above the budget (B>1, huge timelines).
**Result:** PCC 0.99827 (unchanged), device time 141.0→134.5ms, height-sharded conv weighted FLOPs 20.9→24.7%.

### EXP-3 ✅ DONE — Free the conv_transpose input reshape (RM instead of TILE)
**Idea:** `conv_transpose2d` needs NHWC `[B,T,1,C]`; that reshape is a FREE view on ROW_MAJOR but a full
relayout on TILE (**2.76ms vs 0.01ms**, microbenchmarked). `h` reached the transpose in TILE (conv1d always
returns TILE — it ignores `output_layout=ROW_MAJOR`). Untilize once (TILE→RM, **0.17ms**) so the reshape is free.
**Change:** `to_layout(ROW_MAJOR)` at the top of `_conv_transpose1d_nlc` (before the pad/reshape).
**Result:** PCC 0.99827 (unchanged). ReshapeView 7,658µs(8 ops)→11µs(3). PaddedSlice 7,459→3,761µs (transpose
now reads RM). Device time 134.5→**123.6ms**.

### EXP-4 ✅ DONE — Front-door built in NLC (drop permutes + untilizes)
**Idea:** the embedding/expansion front-door built ``merged = [lang|unit|spk]`` channel-major (BCT) then
permuted back to NLC for conv_pre — 2 BF16 permutes (gather NLC->BCT, merged BCT->NLC) + 3 TILE->RM
``to_layout`` untilizes. Build it **directly in NLC**: gather returns NLC as-is; lang/spk emitted ROW_MAJOR,
reshaped ``[B,C]->[B,1,C]`` (free view) + repeated over time; ``concat(..., dim=2)`` on the channel dim.
**Change:** `_expand_unit_embeddings_gather` returns ``[B,t_audio,E]`` (no final permute); lang/spk embeddings
use ``ROW_MAJOR_LAYOUT``; `_forward_one` merge concats NLC on dim=2 (no permute, no to_layout).
**Result:** PCC 0.99827 (unchanged). BF16 permutes 4->2 ops (302->40µs); device ops 3420->3408. Front-door is
~1% of total so time delta is small (~0.4ms) — this was an op-count/churn cleanup, not a big time win.

### Cumulative results (device time, from `tt-perf-report`, unit_seq=1024)
- exp0 baseline (manual chunk loop): **286.2 ms**, 6303 ops
- exp1 (+DRAM width-slice, EXP-1): 141.0 ms, 3420 ops
- exp2 (+L1 resblock, EXP-2): 134.5 ms
- exp3 (+transpose reshape, EXP-3): 123.6 ms
- exp4 (+NLC front-door, EXP-4): **123.2 ms**, 3408 ops  → **−57% device**, **−37% wall-clock** (638→402 ms).
- PCC held at 0.99827 (gate 0.99) throughout; multi-shape (useq 128/512/1000) all pass unchanged.
- useq=64 PCC 0.975 is a PRE-EXISTING short-seq accuracy bug (bit-identical with all flags off) — not ours.

### Op-count note (why 3408 is still high)
Op count is dominated by the **conv_transpose upsampling**: its DRAM height-slicing emits per-slice
PaddedSlice(450) + SliceWrite(450) + Halo + Move + Untilize + conv. That's the next target — reducing slice
count / restructuring the transpose is where the big op-count + block-sharded-conv(22%, 1.58% FLOPs) win is.
Most ops show ``in0:dram_interleaved`` because activations live in DRAM between ops (read->shard->compute->
write-back); convs show ``block/height_sharded`` (they receive the sharded tensor). Cutting that means keeping
activations resident in L1 across more of the pipeline (done for resblock conv1->conv2; transpose still round-trips).

### Remaining top buckets (exp3) — next candidates, higher risk
- **Conv2d block_sharded 22.4%** = conv_transpose upsampling at **1.58% FLOPs eff**. Biggest single bucket;
  inherent to skinny low-channel transposed convs. Needs a different upsample formulation / shard tuning.
- **InterleavedToSharded 21.7% (94 ops)** — dominated by the conv_transpose + per-stage input sharding, NOT
  the resblocks (those now chain in L1). Hard to cut without restructuring the transpose.
- **UntilizeWithUnpadding 13.2% (19 ops, ~860µs each)** on the biggest tensors.
- Small layout churn in the embedding/expansion front-door (permute/repeat/concat/tilize, ids ~159-212) — each
  <150µs, ~1-2ms total; low priority vs the transpose.
- `_VOCODER_CONV1D_MAX_INTERIOR=49152` L1 cap and HiFi2 (DM-bound, ~1% gain) remain non-levers.

---

## Conventions observed in this codebase
- Optimizations are gated behind `SEAMLESS_*` env flags with a safe default + rollback (follow this pattern).
- Comments in `tt_code_hifigan.py` reference prior optimization rounds ("vocoder5/vocoder6 TM", gather-vs-matmul).
- RM (ROW_MAJOR) outputs are used deliberately to avoid TILE↔RM churn between ops.
