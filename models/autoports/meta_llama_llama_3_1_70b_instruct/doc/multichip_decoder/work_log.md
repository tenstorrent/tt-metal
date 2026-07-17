# Multichip decoder work log

Date: 2026-07-17

## Scope, baseline, and hardware

- Changes are restricted to `tt/multichip_decoder.py`, its model-local test,
  and model-local multichip/context documentation. No full-model or vLLM work
  was started.
- The single-chip baseline is `tt/optimized_decoder.py` at commit
  `35ccb90250c`: BFP4 projections, BF16 activation/norm/KV cache, LoFi math,
  packed gate/up, and real-weight layer-39 correctness.
- Target hardware is all four local Blackhole p300c devices (IDs 0--3), with
  11x10 worker grid and eight DRAM banks/device. Hardware commands were
  serialized with `flock /tmp/tt-device.lock`.
- The unrelated pre-existing modification to
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` was preserved and
  never staged.

## Decision and implementation sequence

1. The compiler `2x2` plan, tensor shapes, collective payloads, cache layout,
   full-context memory, padding, dense-MLP strategy, and alternatives were
   frozen in `mesh_plan.md` before the selected path was written.
2. `Provenance2DDecoder` implemented the compiler-derived mapping. Its packed
   variant measured 1.410 ms prefill and 1.859 ms decode against 3.187/1.853
   ms. Projection reductions plus distributed norm made decode essentially no
   faster, so it was retained only as provenance.
3. `MultiChipDecoder` implemented flat TP4 on a logical `1x4` ring: replicated
   stack boundary/norms, 16 local Q heads, two local KV heads, column-parallel
   QKV and gate/up, row-parallel output/down, and two full-mesh reductions.
   The logical `2x2` ordering was rejected after both ring and linear flat
   collectives failed routing between non-neighbor row-major ranks 1 and 2.
4. The `8x8` prefill grid was aligned to each local eight-bank DRAM weight
   shard. Prefill and decode now share one physical weight copy, avoiding a
   projected 9.63 GB/device duplicate across 80 layers.
5. The initial independent stage review returned `more-work-needed`. Its
   findings are retained in `STAGE_REVIEW_INITIAL.md`: trace inputs/positions
   were fixed, topology was not fully earned, local decode geometry was not
   swept, and peak live memory lacked explicit reserves.
6. `$autofix` remediation added device-resident current-position and RoPE
   tensors, embedding-based dynamic RoPE lookup, on-device increments,
   refreshable page tables/hidden inputs, a 4096-token prefill MLP chunk bound,
   explicit temporary deallocation, ten geometry candidates, five topology
   candidates, and persistent decomposed decode collectives.
7. A direct two-layer composition test now feeds the real TP4 output into a
   second invocation with independent layer-local caches. Watcher stress was
   increased to 100 advancing trace replays.

The fresh `$autodebug` bootstrap requested by `$autofix` could not run in this
environment: the first runner lacked a usable Claude login/PATH and the
fallback bubblewrap sandbox was unavailable. `AUTODEBUG.md` records those
tooling failures. The repair continued from the independent stage-review
evidence and isolated hardware candidates rather than treating the bootstrap
failure as a model blocker.

## Final contract

- Exact target: logical `1x4` ring, `FABRIC_1D_RING`, two links, shared
  `TT_CCL`, BF16 decomposed reduce-scatter/all-gather with persistent L1
  buffers for batch-1 decode.
- Local weights: QKV `[8192,2560]`, output `[2048,8192]`, packed gate/up
  `[8192,14336]`, down `[7168,8192]`.
- Local activation/cache: Q/K/V `[B,16/2/2,S,128]`; paged K/V
  `[blocks,2,64,128]`; residual `[1,B,S,8192]` replicated at stack boundaries.
- Replicated INT32 page tables/current positions and UINT32 RoPE indices are
  refreshable without trace recapture. `ttnn.plus_one` advances both position
  tensors on device.
- Arbitrary logical sequence lengths are preserved. The final test uses 39
  tokens with an internal 32+7 chunk split; the production chunk bound is
  4096 plus an arbitrary final tail.
- Llama 3.1 70B is dense SwiGLU, so no MoE/expert execution is applicable.
- The source audit rejects `from_torch`, `to_torch`, `torch.*`, and calls back
  to single-chip forwards inside the runtime prefill/decode path.

## Geometry and topology selection

Ten real-weight, one-variable geometry candidates all passed PCC >= 0.999855.
The accepted changes were:

| Candidate | Change | Control eager | Candidate eager | Control trace | Candidate trace |
| --- | --- | ---: | ---: | ---: | ---: |
| O2 | output: 8 activation cores, block 8, `per_core_N=8` | 1.521315 | 1.451859 | 0.636551 | 0.630560 |
| G1 | gate/up: 32 activation cores, block 8, `per_core_N=14` | 1.537236 | 1.470207 | 0.636795 | 0.626549 |

QKV retained 16/16/5 and down retained 32/7/8. The other isolated candidates
either regressed eager/trace time or were noise. Exact metrics are in
`geometry_{Q1,Q2,Q3,O1,O2,O4,G1,G3,D1,D3}.xml`.

| Topology candidate | Output PCC | Control eager | Candidate eager | Control trace | Candidate trace | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| persistent BF16 | 1.0000000 | 1.471944 | 1.451630 | 0.620704 | 0.603138 | selected |
| BF8 composite | 0.9999981 | 1.471272 | 1.508579 | 0.620839 | 0.619271 | eager regression |
| one link | 1.0000000 | 1.539673 | 1.495026 | 0.620960 | 0.642511 | trace regression |
| linear | 0.9999993 | 1.481544 | 1.481561 | 0.620872 | 0.629654 | trace regression |
| persistent BF8 | 0.9999981 | 1.543350 | 1.614094 | 0.620942 | 0.601112 | 11% eager regression for 0.34% trace gain |

The topology audit also rejected non-fused hidden-sharded activations: two
large all-gathers plus two reduce-scatters have the same large-payload wire
count as the selected two all-reduces, then add two norm-statistics gathers.
Fused all-gather-matmul and matmul-reduce-scatter are the only credible lower-
movement family, but current validators do not accept the selected DRAM-
sharded program/layout and the minimal fused Blackhole path has open
nondeterminism issue #46181. No correctness-stable fused candidate was
available to measure.

## Canonical correctness, cache, trace, and latency run

```bash
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b/model-00015-of-00030.safetensors \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/meta_llama_llama_3_1_70b_instruct/doc/multichip_decoder/final.xml
```

Result: 5 passed and four intentional opt-in candidate/profiler/watcher tests
skipped. `final.log` is the complete console provenance.

| Check | PCC |
| --- | ---: |
| prefill/decode | 0.9999978879 / 0.9996807609 |
| contiguous K/V | 0.9999931706 / 0.9999924131 |
| stack layer zero/two-layer result | 0.9996259268 / 0.9984936517 |
| paged decode | 0.9996794316 |
| advancing trace outputs | 0.9996876997 / 0.9997024905 |
| advancing trace K/V minimum | 0.9998138782 |
| dynamic page-one output/K/V | 0.9999930420 / 0.9998256458 / 0.9998141212 |

The page table changes from `[[1,0]]` to `[[0,1]]` without recapture. Current
positions and RoPE indices advance from 39 through 41, and a captured replay
at logical position 64 writes physical block one offset zero. The full-context
allocation probe creates local K and V tensors `[2048,2,64,128]` on all chips.

| Path | Single chip | TP4 | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| nonaligned prefill eager | 3.710810 ms | 1.653101 ms | 2.244757x | 0.561189 |
| decode eager | 1.850049 ms | 1.556849 ms | 1.188330x | 0.297082 |
| decode trace replay | 1.844675 ms | 0.598285 ms | 3.083272x | 0.770818 |

## Full-context memory evidence

`context_contract.json` advertises 131072 tokens at batch 1. Persistent
weights plus 80-layer BF16 K/V are 20,661,469,184 bytes/device. Unchunked
full-context prefill would conservatively peak at 23,689,428,992 transient
bytes and cannot fit the 13,517,261,824-byte transient headroom. With the
4096-token chunk bound and explicit lifetimes, the conservative explicit peak
is 10,737,418,240 bytes. Adding 134,217,728 bytes shared metadata,
100,000,000 trace bytes, 536,870,912 CCL bytes, and 1,073,741,824 allocator/
fragmentation bytes gives 33,243,717,888 bytes against 34,178,731,008 observed
capacity, leaving 935,013,120 bytes/device. The largest feasible advertised-
context batch proven in this stage is one.

## Profiler collection and review

Profiler and watcher were never combined. The final clean profiler command was:

```bash
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE="$LLAMA_31_70B_REAL_WEIGHT_FILE" \
  RUN_MULTICHIP_DECODER_PROFILER=1 \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  python -m tracy -p -r \
  -o models/autoports/meta_llama_llama_3_1_70b_instruct/doc/multichip_decoder/tracy_final \
  -m pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  -k profiler_smoke
```

The losslessly compressed merged input to both reports is:

```text
tracy_final/reports/2026_07_17_15_18_41/ops_perf_results_2026_07_17_15_18_41.csv.gz
```

Expand it with `gzip -dk` before rerunning `tt-perf-report`.

Reports were rendered with `tt-perf-report --arch blackhole --no-color`, using
`MULTICHIP_PREFILL`..`MULTICHIP_PREFILL_END` and
`MULTICHIP_TRACE_DECODE`..`MULTICHIP_TRACE_DECODE_END` signpost pairs, `--csv`
for the report CSVs, and `--summary-file` for summaries. Pseudo-TTY output was
captured in `final_prefill_table.txt` and `final_trace_decode_table.txt`.

The trace range covers five replays/230 device ops: 2,938 us device time,
257 us aggregate gaps, 182 GB/s/35.5% aggregate DRAM roofline, and only 5 us
(0.2%) trace-removable gap. Per replay the four matmuls are approximately
41/35/204/108 us (QKV/output/gate-up/down); two RS operations are 21--23 us
each and two AG operations 12--14 us each. CCL is about 12.3% and matmuls
dominate. The prefill range covers 26 device ops/1,018 us and 109 GB/s/21.2%.
Its harness-side gaps are excluded from latency conclusions.

Raw Tracy internals duplicated approximately 278 MB of device/host logs after
the merged operation CSV was generated. Those generated duplicates were
removed before commit; the merged provenance CSV is retained losslessly as
gzip to satisfy the repository's 500 KB artifact limit. Human-readable tables,
report/summary CSVs, and plots remain, and the exact command above reproduces
the raw capture.

## Watcher, failures, and recovery

The first watcher attempt with active-Ethernet instrumentation exceeded the
fabric router's 25,600-byte kernel-config buffer (instrumented binary 27,920
bytes). The supported `TT_METAL_WATCHER_DISABLE_ETH=1` setting leaves watcher
active on all Tensix cores. A development trace captured an unwarmed position
and correctly failed because capture cannot perform JIT host writes; one
active-Ethernet core then required a targeted `tt-smi -r 0 1 2 3` warm reset.
A four-chip mesh-open health smoke passed afterward. No model data were
deleted.

Final command:

```bash
flock /tmp/tt-device.lock env \
  LLAMA_31_70B_REAL_WEIGHT_FILE="$LLAMA_31_70B_REAL_WEIGHT_FILE" \
  RUN_MULTICHIP_DECODER_WATCHER=1 TT_METAL_WATCHER=10 \
  TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=10 \
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  -k watcher_stress \
  --junitxml=models/autoports/meta_llama_llama_3_1_70b_instruct/doc/multichip_decoder/watcher_final.xml
```

Result: 1 passed, 8 deselected; 17 eager decode positions, position 64, and
100 advancing trace replays completed with no watcher assertion, timeout,
stall, or non-finite result. Exact console evidence is `watcher_final.log`.

## Canonical artifact index

- `final.xml`, `final.log`: final ordered single/TP4 gate and properties.
- `watcher_final.xml`, `watcher_final.log`: final watcher stress.
- `geometry_*.xml`, `topology_*.xml`: geometry/topology candidate evidence.
- `tracy_final/reports/2026_07_17_15_18_41/ops_perf_results_2026_07_17_15_18_41.csv.gz`:
  losslessly compressed merged operation provenance.
- `tracy_final/final_{prefill,trace_decode}_table.txt`, adjacent report CSVs,
  summary CSVs, and PNGs: human-readable and machine-readable perf review.
- `AUTODEBUG.md`, `STAGE_REVIEW_INITIAL.md`: remediation provenance.
- `STAGE_REVIEW_FINAL.md`: final fresh `clean-pass`; no required work.
- Earlier `*_correctness.xml`, `tracy/`, and `watcher.log` artifacts retain
  the candidate progression but are superseded by the canonical files above.

## Independent final review

The required fresh `$stage-review` returned `clean-pass` with no required
work. It directly checked code, optimized baseline, dynamic trace lifecycle,
paging/cache layout, nonaligned chunks, two-layer composition, memory
arithmetic/reserves, geometry/topology candidates, canonical final/watcher
logs, merged profiler provenance, and fused CCL validator sources. The
controlled anomaly ledger and residual full-model follow-ups are preserved in
`STAGE_REVIEW_FINAL.md`.

## Commit record

- optimized single-chip baseline: `35ccb90250c`
- stage implementation/evidence commit: appended after clean independent review

Nothing is pushed.
