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

### EXP-5 ✅ DONE — Element-budget conv_transpose DRAM slicing (was fixed 128-count)
**Idea:** the transpose sized by *slice count* (`_vocoder_dram_slice_count` → up to 128), giving ~16x more,
tiny slices than L1 needs — each slice paid its own PaddedSlice/SliceWrite/Halo/Move/Untilize/conv, dominating
the vocoder op count. Slice-count sweep on the real model (unit_seq=1024): 128→406ms, 32→354, 16→344, 8→339,
and cap≤4 OOMs L1_SMALL — so 128 was ~16x over-sliced, floor is 8.
**Change:** `_vocoder_dram_slice_count(input_length, in_channels)` now sizes each slice to a fixed element
budget (`_VOCODER_TRANSPOSE_SLICE_ELEMS`=512K; measured L1_SMALL floor ~819K/slice → 1.6x margin), floor 8,
cap 128 — adapts to any input size instead of a fixed count. `SEAMLESS_VOCODER_TRANSPOSE_FIXED_SLICES=1`
restores the legacy formula; `SEAMLESS_VOCODER_TRANSPOSE_SLICE_ELEMS` tunes the budget. Callers pass in_channels.
**Result:** PCC 0.99827 (unchanged). Device ops **3408→1012 (−70%)**, device time 123.2→**99.9ms**. Transpose
block-conv 196→18 ops (27.7→8.3ms, FLOPs eff 1.58→7.5%). PaddedSlice/SliceWrite/per-slice Halo/Move collapsed.

### EXP-7 ✅ DONE — Hoist the shared per-stage leaky_relu(h) (leaky-fusion candidate)
**Idea:** each stage runs ``num_kernels`` (3) resblocks on the *same* stage input ``h``, and every resblock
starts with ``leaky_relu(h)`` — so ``leaky(h)`` is computed 3x redundantly on the stage's biggest tensor.
Compute it once and share (residual still uses the raw ``h``; bit-identical since leaky is deterministic).
**Change:** `_resblock` takes ``first_leaky`` (used for the first conv pair); `_hifi_gan_once` precomputes
``h_leaky = leaky(h)`` once per stage, passes it to all 3 resblocks, deallocs after.
**Result:** PCC 0.99827 (bit-identical). Unary **9,438µs/54 ops → 4,782µs/44 ops** (−49% time — the removed
leakys were the full-size stage-input ones). Device ops 1012→1002, device time 99.9→**95.2ms**.

### EXP-8 ✅ DONE — Feed conv-derived TILE input straight to conv1d (kills the untilize; the EXP-6 "dead end")
**Breakthrough on the untilize/I2S bucket EXP-6 gave up on.** Root of the ~43% (I2S+untilize): between
resblock convs the pattern was ``conv → S2I → UntilizeWithUnpadding(TILE→RM) → I2S → conv``. The untilize was
``_conv1d`` defensively converting the (interleaved TILE) conv output to RM before the next conv — a comment
blamed embedding inputs. **Probed:** ``ttnn.conv1d`` returns interleaved TILE (never sharded — so keep_sharded/
row_major can't help, confirming EXP-6), BUT it *accepts* TILE interleaved ``[1,T,C]`` input directly, PCC-clean
for every resblock shape (k∈{3,7,11}, d∈{1,3,5}, C∈{16,32,64}). So the untilize is unnecessary for conv-derived inputs.
**Change:** ``_conv1d`` gains ``accept_tile_input`` — when the input is interleaved TILE (not embedding), stride-1,
and not bucket-padded, skip the TILE→RM ``to_layout`` and hand TILE straight to ``ttnn.conv1d``. Enabled for both
resblock convs (gated by ``SEAMLESS_VOCODER_RESBLOCK_L1``).
**Result:** PCC 0.99827 (unchanged). UntilizeWithUnpadding 16.3ms/17ops → **gone**; InterleavedToSharded
26.8ms → **7.8ms** (TILE→sharded is far cheaper than RM→sharded). Device ops 1002→927, device time 95.2→**57.0ms**.

### EXP-9 ✅ DONE — Tilize stage input once so resblocks run on TILE (cheap I2S, no residual tilize)
**Idea:** the transpose emits ROW_MAJOR ``h``, but resblock convs want TILE. RM→sharded conv I2S tilizes on
the fly (**815µs** at big stages) and the residual ``add`` re-tilizes the RM residual (~831µs), whereas
TILE→sharded I2S is **~70µs** (~10x cheaper). Tilize ``h`` **once per stage** (right after the transpose) and
share it as residual + first-leaky input across all ``num_kernels`` resblocks, so every downstream conv/add sees TILE.
**Change:** one ``to_layout(TILE)`` on ``h`` in `_hifi_gan_once` before the resblock loop (gated by RESBLOCK_L1).
**Result:** PCC 0.99827. InterleavedToSharded 7.8→**4.1ms** (−48%); leaky 4.8→2.8ms; Tilize buckets 4.5→1.5ms
(the per-stage tilize is cheaper than the residual tilizes it removes). Device ops 927→917, device time 57.0→**48.4ms**.

### Cumulative results (device time, from `tt-perf-report`, unit_seq=1024)
- exp0 baseline (manual chunk loop): **286.2 ms**, 6303 ops
- exp1 (+DRAM width-slice, EXP-1): 141.0 ms, 3420 ops
- exp2 (+L1 resblock, EXP-2): 134.5 ms
- exp3 (+transpose reshape, EXP-3): 123.6 ms
- exp4 (+NLC front-door, EXP-4): 123.2 ms, 3408 ops
- exp5 (+element-budget transpose slicing, EXP-5): 99.9 ms, 1012 ops
- exp9 (+hoist shared leaky, EXP-7): 95.2 ms, 1002 ops
- exp10 (+accept_tile_input, EXP-8): 57.0 ms, 927 ops
- exp11 (+tilize stage input once, EXP-9): 48.4 ms, 917 ops
- exp12 (+L1-full resblock conv chain, EXP-11): **44.9 ms**, **782 ops**
  → **−84% device time, −88% op count, −53% wall-clock (638→297 ms).**
- PCC held at 0.99827 (gate 0.99) throughout; multi-shape (useq 128/512) pass unchanged.
- useq=64 PCC 0.975 is a PRE-EXISTING short-seq accuracy bug (bit-identical with all flags off) — not ours.
- KEY LAYOUT RULE for this model: conv inputs should be **TILE** (TILE→sharded I2S is ~10x cheaper than
  RM→sharded), and ttnn.conv1d accepts interleaved TILE directly (EXP-8). Keep the resblock chain TILE end-to-end.
- NOTE: EXP-6's "dead end" was only the sharded/row_major route. L1-interleaved intermediates still OOM the conv
  circular buffers (2.78MB > 1.5MB L1) — can't keep activations resident in L1 at the big stages.

### EXP-11 ✅ DONE — L1-full resblock conv chain (the EXP-10 "dead end", SOLVED)
**EXP-10 below concluded L1 residency was impossible — that was WRONG.** It failed because the L1-full test
used a large auto ``act_block_h`` AND omitted the slice config. The fix is BOTH together:
``slice_config=ttnn.Conv2dL1FullSliceConfig`` **+** ``act_block_h_override=32``. With those, ``ttnn.conv1d``
runs fully in L1 and **returns a sharded L1 output** — verified L1-fitting for every resblock shape (to
409600x16, k=11) and reshape/leaky preserve the sharding. So conv1 keeps its output sharded in L1 and conv2
consumes it there, eliminating the conv1->conv2 DRAM ShardedToInterleaved+InterleavedToSharded round-trip.
**Change:** `_vocoder_resblock_l1_full_enabled()` (default on, ``SEAMLESS_VOCODER_RESBLOCK_L1_FULL=0`` to
disable); ``l1_full`` param threaded through `_conv1d`/`_conv1d_run` (sets the L1-full slice config); both
resblock convs set ``l1_full=True`` (conv1 keep_sharded, conv2 accept_sharded -> S2I for the add).
**Result:** PCC 0.99827 (multi-shape 128/512 pass). Device ops 917->782 (-135); InterleavedToSharded
4.1->2.5ms (94->49 ops), ShardedToInterleaved 3.9->2.0ms (91->46). Device time 48.4->**44.9ms**.
NOTE: caps at act_block_h=32 (abh=16 hard-crashed the driver); creating a >~10MB tensor directly in L1 also
crashes — always feed DRAM input + L1FullSliceConfig, never allocate the big activation to L1 directly.

### EXP-10 ⚠️ SUPERSEDED by EXP-11 — (original "dead end" reasoning was incomplete)
**Goal:** kill the remaining I2S(8.4%)+S2I(8%) DRAM round-trip around each resblock conv. Reference models
(SDXL resnet `tt_resnetblock2d.py`) keep activations in L1: ``sharded_to_interleaved(x, L1_MEMORY_CONFIG)``
(not DRAM), reshard in L1, and use ``slice_config=ttnn.Conv2dL1FullSliceConfig`` (SliceType::L1_FULL). conv1d
doc confirms: *"inputs already in L1 run fully in L1; DRAM inputs are auto width-sliced through DRAM."*
**Tested L1-full on the vocoder conv shapes (single conv, L1 input):**
  - k=3: OK at 102400x64 / 204800x32, but OOM at 409600x16.
  - k=7, k=11: OOM at EVERY stage ("circular buffers grow to 2.78MB > max L1 1.5MB").
**Why:** a DRAM input is *streamed* (small CB), but an L1-resident input counts against the CB budget, so the
k=7/11 (and biggest-stage k=3) convs can't run L1-full. They MUST DRAM-slice → DRAM-interleaved I/O → I2S/S2I.
The resnet hack works there only because those convs fit L1-full. **Conclusion: the ~16% conv-I/O round-trip is
inherent for the vocoder's k=7/11 long-timeline convs on BH.** Needs a ttnn change (sharded conv output, or a
smaller-CB conv) to remove — not a model-level fix. Best committed state stays exp11 (48.4 ms). Probe: scratchpad/l1full.py.

### Remaining top buckets (exp5) — next candidates
- **InterleavedToSharded 26.9% (26.8ms, 94 ops)** — now #1. Resharding DRAM→L1 for each conv input (the
  activations still live in DRAM between ops). Cutting it means keeping more of the pipeline resident in L1
  across ops (harder — stage activations are large, and eltwise leaky/add run in DRAM).
- **UntilizeWithUnpadding 16.3% (16.3ms, 17 ops)** on the biggest tensors.
- **Unary(leaky) 9.5% + BinaryNg(add) 5.2%** — standalone DRAM eltwise; fusion candidates.
- conv_transpose block-conv still 8.3% at ~7.5% FLOPs (skinny low-channel) — largely inherent.
Most ops show ``in0:dram_interleaved`` because activations live in DRAM between ops (read→shard→compute→
write-back); convs show ``block/height_sharded`` (they receive the sharded tensor).

### EXP-6 ❌ DEAD END — L1-chaining resblock convs (untilize/I2S bucket)
**Goal:** kill the ~43% (I2S 26.9% + untilize 16.3%). The per-conv pattern at big stages is
``conv → ShardedToInterleaved(87µs) → UntilizeWithUnpadding(1019µs) → InterleavedToSharded(815µs) → next conv``:
each conv emits **interleaved TILE**, so the next conv untilizes (TILE→RM, a full DRAM read+write) and reshards.
**What was tried:** (a) `keep_sharded_output=True` to hand conv1→conv2 sharded in L1; (b) `row_major_output=True`
on conv1/conv2 so the chain stays RM (no untilize); (c) sharded leaky/add. **All no-ops or worse.**
**Root cause (probed):** `ttnn.conv1d` with these sharded configs **returns interleaved TILE regardless of
`keep_sharded_output`/`row_major_output`** — and whether `output_layout=ROW_MAJOR` is honored is *shape-dependent*
(RM at k=11/d=5, TILE at k=3/d=1, same T/C). So the conv output layout/sharding isn't controllable at the model
level. Sharded `add` also broke PCC (0.956) due to shard-spec/shape mismatch. Cracking this needs a **ttnn-level
fix** (reliable sharded or RM conv output) or a full sharded-pipeline rewrite with matched shard specs across
every op — beyond safe model tuning. Reverted; exp5 (99.9ms) stands as the current best. Probe scripts in scratchpad.

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
