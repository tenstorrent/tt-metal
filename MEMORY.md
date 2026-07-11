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

### Ideas not yet done (next candidates, higher risk)
- **Keep resblock activations sharded across consecutive convs** to cut `InterleavedToSharded` (~13%) and the
  conv-internal Untilize/Tilize. Bigger restructure; watch PCC margin (only ~0.008 above gate).
- `_VOCODER_CONV1D_MAX_INTERIOR=49152` is an L1 hard cap (65536 OOMs even at low channels) — not the lever.
- HiFi2 vs HiFi4: measured ~1% total gain only (vocoder is DM-bound, not math-bound) and costs PCC margin — not worth it.

---

## Conventions observed in this codebase
- Optimizations are gated behind `SEAMLESS_*` env flags with a safe default + rollback (follow this pattern).
- Comments in `tt_code_hifigan.py` reference prior optimization rounds ("vocoder5/vocoder6 TM", gather-vs-matmul).
- RM (ROW_MAJOR) outputs are used deliberately to avoid TILE↔RM churn between ops.
