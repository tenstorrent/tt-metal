# Bounded per-chunk kernel queue

Prepared CPU-only. `prefill_chunks7.json` contains28 serial jobs for real TP4 weights and captured checkpoint activations, covering rows64,128,256,1024,4096. It snapshots the explicit **pending tiled_conv7 policies**, not final defaults, into `prefill_chunks7_l{0,3}_policy.json`. The ordinary model remains untouched.

| Group | Roles | Families/layouts | Jobs |
|---|---|---|---:|
| 2D controls/adaptation | GDN joint, gateup, down, output; full QKVG, gateup, down, output | inherited2D and bounded tuned2D; actual DRAM-sharded and exact interleaved weight copy |8|
| N-parallel1D | Same eight roles | mcast-in0, interleaved weights; inherited2D/interleaved matched control |8|
| Minimal matmul | Same eight roles | minimal, both DRAM-sharded and interleaved weights; inherited2D/interleaved control |8|
| M-parallel1D complement | Gateup/down, both kinds; rows1024/4096 only | mcast-in1, interleaved weights; inherited2D/interleaved control |4|

Candidate grid8×4, K block4, bounded2D height1, and minimal blocks2/4/8 limit initial CB demand without changing shapes or precision. This is not a full five-family × two-layout × five-shape Cartesian product. The 1D-in0 ordinary factory treats non-global-CB sharded weights as locally resident and would bind a DRAM address into an L1 CB, so that specific combination is adapted to a verified exact setup interleaved copy. It is not launched as a direct DRAM-sharded trial. Other1D-in1 shapes/layouts are deferred, not declared illegal or rejected. Any material native validation/L1 failure needs a smaller block/grid or compatible layout under a new name; this queue stops and preserves it.

The projection probe now uses the installed joint helper's actual prefill options: current grid8×8, output height4, cap4096. Previously its generic cap512/config would have mislabeled the current joint control. The joint input capture is the real input-norm result immediately before the installed projection; source checks confirm flat prefill calls `_norm` before `_prefill_packed_gdn`. M4096 captures B32×S128 from existing B32×S254 recorded activations, so no synthetic repeat or unavailable tokens are substituted. Gate/up captures the post-attention norm; down/output capture their actual `_linear` operands. All four ranks' captured row counts and widths are asserted and recorded. Capture/readback happens only during untimed setup/checks.

Joint timing covers the raw FP32 packed primary+correction matrix and checks6784 meaningful columns; it excludes the corrected4160-column combine. Gate/up timing likewise measures the contiguous packed projection, not paired fused SwiGLU. Wins require a whole-layer comparison through the actual next consumer. Generic inherited controls still include their selected512-row cap/concat; the joint uses its selected4096-row cap. JSON/CSV label those kernel pieces explicitly.

The probe preserves rank PCC>=0.995, finite outputs, changed-input traces and exact first/last replay. Every invocation includes its own arithmetic control, intentionally repeated across jobs for matched provenance. Timings are synchronized warmed trace wall microseconds, not device-profiler operation times. Interleaved copies are setup-only; a selected deployed copy needs capacity accounting or inclusion of conversion cost in a whole-layer measurement.

## Launch and stopping

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_prefill_chunk_queue \
  --manifest models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prefill_chunks7.json
python -m models.autoports.qwen_qwen3_6_27b.tests.run_prefill_chunk_queue \
  --manifest models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prefill_chunks7.json --run --resume
```

The first command validates without importing TTNN/Torch or launching devices. The second runs bounded900-second jobs through existing `run_job.sh`, with source/policy checks before and after each job. Watcher/profiler are unset; timing tracking is0 and skip-program-cache0. The parent must serialize against all other hardware drivers; the queue lock only excludes another instance of this queue.

`D/STOP` is checked between jobs, or select `--stop-file PATH`. It never kills an active job. A nonzero exit, missing output, failed PCC/trace gate, or source change stops all subsequent launches. The queue does not convert a first API/PCC failure into a family-wide rejection. Failed files are never overwritten. Resume accepts only passing receipts with matching commands, file size/mtime and revalidated gates; use fresh names/subsets for repaired or missing rows. Stop-rule device accounting remains parent-owned and authoritative.

Existing wrapper `.command.json/.log` files record exact environment/source/command provenance. New `.queue_receipt.json` files record command, UTC, elapsed time, exit/pass and artifact size/mtime. `prefill_chunks7.csv` aggregates completed/partial case rows with exact result paths and program configs after every job, including failure rows. It does not manufacture pending measurements. Source/config identities are hashed; activation/result artifacts are not individually hashed.

CPU checks cover all planned role shapes and generated block/subblock divisibility; the queue plan validates and both helpers import without TTNN/Torch. No device runs, native compile results, PCC results, latency values or family decisions were produced by this preparation.
