# Optimized multichip decoder work log

## 2026-07-14: inventory and hardware health

- Starting checkpoint: `e1a3f724877` (Stage 04 checkpoint metadata), with
  implementation commit `683adda7a3d12cc060df9ab3a36f1fd506eef234`.
- Unrelated pre-existing dirty state is excluded from this stage:
  `tt_metal/python_env/requirements-dev.txt`, `.exp_run/`, and `fusion_tests/`.
- `timeout 60 tt-smi -ls --local` showed four Blackhole P150b boards.
- A source-backed `MeshShape(1, 4)` open/close smoke passed with firmware
  19.9.0 and printed `MESH_SMOKE_OK`.
- Stage 04 accepted warmed baseline: sliding/full prefill-128
  2.4044/2.4387 ms and traced decode 0.5268/0.5768 ms.  Stage 05 will remeasure
  the unchanged default in the current environment before ranking candidates.

## Operation-topology audit (starting path)

| Boundary/group | Starting sequence | Material movement/collective | Candidate family to test | Action/evidence |
|---|---|---|---|---|
| attention input/QKV | replicated DRAM residual -> norm -> L1 width shard -> packed local QKV -> L1 interleaved -> head split | input reshard plus QKV sharded-to-interleaved | persistent L1 residual/working shard; packed versus legal split; geometry/precision cross | packed 32-core/block-7 BFP8/LoFi selected after cumulative review rerun; split and other blocks measured |
| attention heads/cache | per-head norm -> RoPE -> paged cache update -> explicit SDPA -> concat heads | DRAM/L1 conversions around head ops | explicit SDPA configs, BFP8 cache, activation/cache layout | q32/k64 exact-exp and BFP8 cache selected after config/cache sweep |
| row-parallel O | local O matmul -> sharded-to-interleaved -> synchronous two-link all-reduce | one hidden-width reduction and layout conversion | fused CCL+matmul, RS carry-forward, async/persistent CCL | persistent L1 BF16 async reduction selected; fused and fractured families measured |
| MLP gate/up | norm -> reshard -> separate gate/up (gate GELU fused) -> multiply | same-input repeated matmuls; working-shard conversion | packed versus tuned separate, phase-specific shard, dtype/fidelity/geometry cross | packed BFP8-output BFP4/LoFi M=1 path selected; safe separate M>1 branch retained |
| row-parallel down | local down -> sharded-to-interleaved -> synchronous two-link all-reduce | one hidden-width reduction and layout conversion | fused matmul-RS, delayed gather/fractured residual, async/persistent CCL | 14-core/block-12 BFP4/LoFi plus persistent native-BFP8 decode reduction selected; prefill stays BF16 |
| layer boundary | replicated BF16 residual into next layer | no separate boundary op, but row reductions restore replication within each block | fractured/sharded residual through next norm/consumer versus replicated family | replicated boundary selected; no inter-layer collective or conversion remains |

The Stage 04 adapted fractured-boundary result (decode 0.2734 vs 0.1616 ms;
prefill-128 0.8545 vs 0.6796 ms for the measured boundary) is accepted only as
the starting comparator.  Stage 05 will retest coherent families after reviewing
current profiler advice and available fused/async APIs.

## Current-run baseline and first candidate families

- The unchanged Stage 04 default was remeasured on the target 1x4 mesh.  Warmed
  prefill-128 was 2.4180/2.1801 ms (sliding/full); traced warmed decode was
  0.52649/0.57565 ms.  Real-weight non-aligned prefill PCC was
  0.999846/0.999757.  Raw output is in `evidence/baseline_latency.log`.
- Casting the two reduction payloads to BFP8 and restoring BF16 after each
  reduction remained correct (prefill PCC 0.999747/0.999674; decode PCC
  0.999879/0.999825) but regressed prefill to 2.7773/2.4567 ms and decode to
  0.52811/0.58001 ms.  The cast/layout work dominates the byte reduction, so
  BF16 communication remains the reference.  See `candidates/ccl_bfp8.log`.
- The first persistent `all_reduce_async` attempt was adapted through the
  tensor-shape API mismatch, then reached a Blackhole fatal because the minimal
  kernel does not accept DRAM input.  `$autodebug` established that this
  overload requires L1 width-sharded input, TP4-volume scratch, and output; the
  DRAM failure is therefore not a rejection.  The exact failures are preserved
  in `candidates/persistent_async_ccl_retry1.log` and `retry2.log`; the coherent
  L1 retry is in the `$autofix` loop.
- The full traced sliding decoder measured every exact divisor in the expanded
  7/8/12/14/21/24/28/42/56/84-core TP-local square-MLP family.  All passed PCC
  0.999938 or better.  Median latency in core-count order was
  0.53015/0.53053/0.52875/0.52237/0.54244/0.52532/0.52829/0.54014/
  0.55179/0.58029 ms.  Fourteen cores is the provisional leader, but its 0.56%
  advantage over the 24-core default requires paired reproduction before
  selection.  See `candidates/mlp_geometry_extended_retry2.log` and XML.

## Persistent collective repair and selection

The first lower-level async attempt was not treated as a rejection.  The
ordered repair loop was:

1. DRAM input reached the Blackhole minimal-all-reduce fatal; preserve log in
   `candidates/persistent_async_ccl_retry1.log`.
2. Adapt input to width-sharded L1 and retry the exact model shape.  Source and
   `$autodebug` showed that the stable intermediate must contain TP4 times the
   reduced shard and use the same core grid.
3. Preallocate two global semaphores and role/slot buffers, request L1 output,
   and perform only the final sharded-to-DRAM conversion.  This passed both
   layer kinds (`persistent_async_ccl_final.log/.xml`).
4. Pool state was moved from decoder ownership to a mesh-scoped module pool so
   all layers reuse two buffers.  Final tests assert exactly the
   `attention_o` and `mlp_down` buffers after warmup.

BF16 persistent decode measured 0.48592/0.53729 ms with PCC
0.999967/0.999896.  Persistent BFP8 measured 0.47667/0.52814 ms, but full-layer
PCC fell to 0.999849, below the accepted full-layer baseline, so BF16 remains
the collective dtype.

## Projection, geometry, SDPA, and precision families

- The full decode trace swept local MLP core counts
  7/8/12/14/21/24/28/42/56/84.  The paired persistent comparison reproduced
  14 cores at 0.477578 ms versus 24 cores at 0.486932 ms.  Prefill remains on
  its independently selected 24-core placement.
- Packed QKV with 32 cores/block 3 initially measured 0.477638 ms.  Split Q/K/V was
  0.555468 ms.  The faster 32-core/block-7 result lost accepted exact PCC;
  block 21 was adapted after its first program-config error but collided with
  L1, so it was not selected.
- Gate/up packed BF16 was 0.488068 ms.  Packed BFP8 was 0.468598 ms at PCC
  0.999948, and the both-layer run reproduced PCC 0.999948/0.999887 and decode
  0.46960/0.51915 ms before the final lifecycle repair.  It remains the M=1
  projection default.
- SDPA k32/k128 measured 0.481713/0.481578 ms versus the q32/k64 path around
  0.479 ms.  Approximate exp was within run noise and did not justify changing
  math semantics.  A full-grid full-attention retry was 0.530308 ms.  Exact
  q32/k64 remains selected.
- Attention BFP4 and KV BFP4 reduced PCC to 0.997274 and 0.996010.  Attention,
  gate/up, and down HiFi2 were all slower.  The missing BFP4 tensor-cache
  suffix encountered by the first precision run was added and the real
  candidates were retried; the first mapping error is not used as a rejection.
- Full-attention O block 16 was retried in the final persistent memory family
  and collided with the retained L1 scratch.  The lower-footprint block 8
  default passes and remains selected.

## Residual layout, fused CCL, and topology family

The Stage 04 coherent fractured family remains the applicable family-level
measurement: RS -> distributed RMSNorm -> residual -> delayed AG -> real
consumer was 0.2734 versus 0.1616 ms at decode and 0.8545 versus 0.6796 ms at
prefill.  It was not benchmarked with an immediate restore to replicated
residual.

The fused MM+RS test was adapted to the exact Gemma decode shape (M=32, global
K=N=5376, local K=1344, BFP8 weights, BF16 activation, two links, output block
7).  Linear fabric hung because the fused program installs Ring kernels even
when passed Linear topology.  Before recovery:

```bash
tt-triage collect --output candidates/fused_mmrs_hang_triage.txt
```

The raw text was losslessly compressed to `fused_mmrs_hang_triage.txt.gz`
for the repository's 500 KB artifact limit.

The triage showed op 3057 running on all four devices with 44 broken
BRISC/ERISC cores.  The host was terminated, then:

```bash
tt-smi -r all
timeout 60 tt-smi -ls --local
```

All four boards returned cleanly.  The isolated Ring retry passed at PCC
0.999963 (`fused_mmrs_ring2.log/.xml`).  Because this made topology potentially
material, the complete Ring graph was measured rather than rejecting from the
isolated result.

Ring+14 produced sliding decode 0.465383 ms but full prefill collided with
persistent L1.  `$autofix` tried the smallest SDPA adaptation (K chunk 64); it
moved the collision to RMSNorm and proved SDPA alone was insufficient.  A
28-core MLP retry reached a Ring dispatch core and was invalid.  The adapted
Ring+K64+MLP24 family freed 40,960 B/core and passed all four ordered tests:
PCC 0.999959/0.999890, prefill 2.437683/2.443858 ms, decode
0.471058/0.521757 ms.  Those early decode samples appeared slightly faster
than the later Linear lifecycle result, so Ring was rerun against the final
current source instead of being rejected from stale arithmetic.  The rerun was
0.477758/0.527858 ms versus Linear 0.474643/0.527828 ms: slower for sliding and
statistically tied for full.  Linear is selected.  Detailed diagnosis and repair accounting are in
`candidates/ring_l1_AUTODEBUG.md` and `ring_l1_AUTOFIX.md`.

Fused all-gather+matmul was initially bounded as part of the fractured family,
but final independent review correctly required the exact primitive and a
real-weight coherent family.  `$autofix` added env-gated TP4 probes to the
generic Blackhole helper without changing its defaults.  The first exact call
failed because output block width 6 was not divisible by subblock width 4; the
Qwen-derived program was adapted to an 8x6 grid with a 1x1 subblock and
retried.  Ring, one link, dimension 3, four kernel transfers, two semaphores,
persistent gathered-input buffers, and DRAM-interleaved AG/matmul storage then
passed sliding O K8192->N1344 at PCC 0.999963534 and full O K16384->N1344 at
0.999964025.  Synthetic down K21504->N1344 at BFP4 reached only 0.993138794,
below the 0.995 gate, so the shape/layout was retained and its weight adapted
to BFP8; that retry passed at PCC 0.999964537.

The model test repacked real O/down/packed-gate-up/QKV checkpoint weights from
local-K/full-N to full-K/local-N.  Its coherent endpoint kept H/TP local through
fused O, distributed post-attention norm/residual/pre-FF norm, fused packed
gate/up, fused down, distributed post-FF norm/residual/input norm, and the next
fused QKV.  It therefore did not reject a lower-movement family by immediately
restoring the old replicated contract.  Source-current 12-replay evidence:

```bash
GEMMA4_MULTICHIP_FUSED_AGMM_COHERENT=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k fused_agmm_coherent_boundary \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/candidates/fused_agmm_coherent_final.xml
```

Both kinds passed: PCC 0.999751962/0.999694822.  Matching endpoint medians
were 0.384409->0.952576 ms sliding (2.478x slower) and
0.413468->1.055626 ms full (2.553x).  The independent Tracy capture and
`tt-perf-report` intervals reproduced 349.77->925.49 us and
384.95->1026.01 us.  Separate fused O/down plus immediate BFP8 gather variants
also lost (O 49.21->105.14 and 72.59->188.84 us; down 72.47->221.11 and
72.85->222.30 us).  Ring is the only applicable fused topology exposed by
this API; the production Linear replicated-residual path remains selected.
Commands, signposts, CSV SHA-256 `543f5f832d3b41e0bbabe74d1d25277e1677ddb0c0af6ebd10c5a9a9fd2b8cd7`,
and raw-log cleanup are in `tracy/fused_agmm_coherent/README.md`.

## Batch-32 lifecycle autofix

The first complete default suite deliberately ran batch-32 after both batch-1
decode traces.  Batch-32 packed gate/up then failed with static-CB high-water
1,163,136 versus persistent L1 at 854,272.  `$autodebug` showed that logical
M=32 compiles a distinct packed N=10,752 DRAM-sharded matmul whose
triple-buffered weight CB is roughly twice either separate N=5,376 projection.
The first adapted `GEMMA4_MC_MLP_TOPOLOGY=separate` retry passed sliding PCC
0.999911, but the following full prefill found the more general issue: retained
low-grid scratch collided with RMSNorm (1,237,760 versus 1,069,760).  Thus the
projection branch alone was not accepted as the fix.

The final repair keeps the M=1 packed graph unchanged, uses separate BFP8
gate/up with bounded DRAM spills for M>1, and reshards both row-projection
partials only at the CCL boundary to the final 24 row-major worker cores.
Attention-O and MLP-down serialize through one shared 57,344-byte/core
physical scratch tensor and two semaphore epochs.  This keeps persistent state
away from low-grid prefill RMSNorm/SDPA/MLP without clearing trace-owned
addresses.  The exact ordered batch-32 run then passed sliding/full PCC
0.999897/0.999881, including full prefill after the sliding trace had made the
pool resident (`evidence/batch32_autofix.log/.xml`).

The mandatory CCL-boundary reshard costs about 1.05%/1.15% versus the earlier
non-coexistent M=1 candidate.  The pre-review default numbers were PCC
0.999948/0.999887, prefill 2.42154/2.22987 ms, and traced decode
0.47464/0.52783 ms.  They improve the current-run Stage 05 starting decode by
9.85%/8.31%; prefill is unchanged for sliding and 2.28% slower for full.

## Final profiler advice closure and accounting

- `tt-perf-report` 1.2.7 was run with advice enabled for sliding/full prefill
  and traced decode.  The four reports and filtered CSVs are under
  `tracy/final/` for the pre-review default; exact commands, source hash, and the capture-wrapper anomaly
  are in `tracy/final/README.md`.
- The decoder rows prove the claimed dtype/fidelity reached runtime:
  BFP8/LoFi attention, BFP4/LoFi packed gate/up and down, BFP8 packed output,
  and BF16 persistent asynchronous reductions.
- Decode advice that reported no output subblock for DRAM-sharded matmuls was
  closed by the role-specific precision-locked geometry sweeps.  This program
  config class does not expose output-subblock fields; all legal larger
  `in0_block_w` divisors and coherent upstream grids are recorded in the
  candidate logs.
- HiFi2 advice was measured independently for attention, gate/up, and down and
  lost to the retained LoFi policies.
- Prefill advice to increase the 24-core MLP grid was retried at 28/42/56
  cores.  All passed PCC >=0.999955 but measured 0.68133/0.71375/0.67339 ms
  versus 0.64982 ms for the final default.
- The first L1 width-sharded prefill attempt was invalid and corrupted K-sliced
  rows (PCC 0.03555).  It was adapted to a legal 8x4 2D block-sharded input
  with owned DRAM-to-L1 reshards before gate/up and down.  The adapted path
  passed PCC 1.0 but measured 0.80624 ms, so DRAM interleaved remains selected.
- For the pre-review profiler, at prompt length 32, mandatory local weights plus BFP8 K/V are 57,614,336
  bytes sliding and 61,966,336 bytes full.  Dividing by 512 GB/s gives
  0.11253/0.12103 ms rooflines.  The same-workload profiler device sums are
  0.43677/0.48967 ms and pre-review host medians are 0.47464/0.52783 ms, leaving
  0.03788/0.03815 ms of traced replay/synchronization overhead.

## Current-source final gate results

- `evidence/final_latency.log`: 4 passed on the final source; headline numbers
  above replace the earlier pre-copyback samples.
- `evidence/final_suite.log`: 12 passed after review remediation, including pool cleanup, both layer kinds,
  non-aligned 1025/1057, batch 32, trace mutation/replay, pool reuse, cache and
  layout contracts, replica equality, and source fallback audit.
- `evidence/exact_context.log`: 2 passed at absolute position 262,143.
- `evidence/watcher_device.log`: 4 passed with `TT_METAL_WATCHER=10`, worker
  asserts enabled, and ETH watcher scoped off because the inherited full
  watcher overflowed the ACTIVE_ETH config buffer.
- `tracy/final_noop_removed/profile_tests.xml`: all four final-default
  profile windows passed.  `tracy/final/` and `tracy/final_selected/` are
  retained as pre-review and pre-no-op evidence respectively.

## Stage-review remediation and cumulative final selection

The first independent review returned `more-work-needed`.  It found that the
documented selection used an unstated stricter accuracy rule instead of the
test suite's declared `PCC_THRESHOLD = 0.995`, that the broad MLP geometry
sweep reverted the selected packed topology, and that the BF16 CCL claim did
not match the profiler's native-BFP8 MLP partial.  `$autofix` treated these as
three independent hypotheses.

- The cumulative QKV block-7 + attention-BFP4 + BFP8-CCL family improved
  decode to 0.45607/0.50402 ms, but full-attention PCC was 0.992103, below the
  declared gate.  The isolated current-topology attention-BFP4 retry proved it
  was the cause: sliding passed at 0.997253, full failed at 0.992305.  See
  `candidates/review_cumulative_low_precision.*` and
  `review_attention_bfp4_final.*`.
- QKV block 7 plus BFP8 CCL passed both kinds at PCC
  0.999802/0.999718 and decode 0.46441/0.51931 ms.  Applying BFP8 to prefill
  hurt it to 2.60662/2.60277 ms, so the candidate was adapted to a coherent
  phase-specific contract: BF16 prefill reductions and BFP8 decode reductions.
- Source inspection established the actual async op contract.  `dtype=` sets
  its output; fabric pages retain the input tensor dtype.  The old default
  therefore already sent a native BFP8 MLP partial while returning BF16.  The
  final decode policy explicitly uses BFP8 input/output for both reductions
  and restores BF16 at the layer boundary; prefill uses BF16 input/output.
- The corrected 7/8/12/14/21/24/28/42/56/84-core sweep preserved packed
  gate/up with BFP8 output and measured both MLP-only and whole-layer traces.
  Fourteen cores won at 0.21669/0.46436 ms.  Twenty-four cores was
  0.22198/0.46786 ms.  The first 7-core block-24 attempt exceeded L1; the
  legal block-12 adaptation passed at PCC 0.999802 but was slower end-to-end
  at 0.46815 ms.  Evidence is in `review_packed_mlp_geometry*`.
- A terminal `release_multichip_decoder_resources(mesh)` API now synchronizes,
  deallocates scratch, clears semaphore/mesh references, and removes the exact
  pool entry before mesh close.  It is idempotent and is covered without
  hardware plus by real fixture teardown after every trace is released.

The promoted no-override default reproduces PCC 0.999802/0.999718.  The final
post-fused-remediation run measured prefill 2.40641/2.20546 ms and traced
decode 0.46381/0.51663 ms.  Against the current-run starting path this is
0.48% faster/1.16% slower prefill and 11.91%/10.25% faster decode.  The small
full-prefill movement is within observed run variation.  The source-current
full suite is 12 passed/48 gated skips, the
advertised-position trace is 2/2, and watcher is 4/4 with no device error.
The source-current final Tracy sums are 0.42830025/0.48162025 ms, leaving
0.03551275/0.03500725 ms host/replay overhead.  Provenance, hashes, advice tables,
and CSVs are in `tracy/final_post_fused_review/`.

Because the selected BFP8 scratch is smaller than the pre-review pool, the
complete Ring14 family was rerun instead of relying on the old Ring24 result.
It passed PCC 0.999802/0.999718 and measured prefill
2.61877/2.16072 ms and decode 0.46376/0.51807 ms.  Versus Linear it gains only
1.17/1.71 us decode and 14.17 us full prefill, all within cross-run variation,
but loses 211.99 us sliding prefill.  Linear remains the coherent default; see
`candidates/ring_final_selected.*`.

## Second rereview finding and final no-op removal

The second fresh review found one remaining materially measurable no-op: the
MLP decode partial was already BFP8, but `_tp_allreduce` typecast it to BFP8
again before the native-BFP8 async reduction.  The helper now typecasts only
when the partial dtype differs from the requested communication dtype.  This
preserves the BF16-to-BFP8 attention conversion and both BFP8-to-BF16 boundary
restores while removing only the MLP BFP8-to-BFP8 operation.

All final gates were first rerun after this source edit, then rerun again after
the exact fused-AGMM test-only remediation.  Final PCC remains
0.999802416/0.999718188; the last reproducible warmed prefill is
2.4064085/2.205460 ms and traced warmed decode is 0.463813/0.5166275 ms.  The
source-current standard suite passed 12 with 48 gated skips,
advertised-position tests passed 2/2 at position 262,143, and the separate
watcher run passed 4/4 without a watcher error.  The earlier post-no-op
profiler passed all four windows, contains no BFP8-to-BFP8 typecast, and has
device-op sums 428.14175/480.92775 us.  The final post-fused-review profiler
also passed all four windows and measured 428.30025/481.62025 us, leaving
35.51275/35.00725 us versus the source-current host medians.  See
`stage_review_rereview.md`, `evidence/final_*`,
`tracy/final_noop_removed/`, and `tracy/final_post_fused_review/`.

## Final validation commands

Standard correctness, stress, non-aligned lengths, pool reuse, and runtime
fallback audit:

```bash
MPLCONFIGDIR=/tmp/mpl LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
pytest -vv -s --tb=short --durations=20 -x \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/final_suite.xml
```

Final warmed prefill and traced warmed decode:

```bash
GEMMA4_MULTICHIP_BENCH=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH pytest -vv -s \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k 'paged_decode_trace_matches_optimized_baseline or warmed_latency' \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/final_latency.xml
```

Advertised-position trace:

```bash
GEMMA4_MULTICHIP_EXACT_CONTEXT=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH pytest -vv -s \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k advertised_context_traced_decode \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/exact_context.xml
```

Watcher is a separate process from Tracy:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
MPLCONFIGDIR=/tmp/mpl LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
pytest -vv -s models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k 'paged_decode_trace_matches_optimized_baseline or sliding_nonaligned_window_wrap_matches_baseline' \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/watcher.xml
```

Profiler capture and `tt-perf-report` commands, exact CSV paths, hashes, and
signposts are recorded in `tracy/final_post_fused_review/README.md`.

## Handoff gates

- Fresh `$stage-review` rereview verdict: `clean-pass`; see
  `stage_review_fused_rereview.md`.
- Stage-owned technical checkpoint: `96d41fdf0c5` (`Optimize Gemma 4 31B
  multichip decoder`).  The follow-up metadata-only commit records this SHA.
  No push was performed.

## 2026-07-17: current-checkout Stage 05 revalidation

- Starting HEAD: `b68b16df75d121dcbe0128d6fa4cea98f993b870` on branch
  `odjuricic/agentic-research/graph-rewrite-skill`.  Pre-existing unrelated
  dirty paths (`tt_metal/python_env/requirements-dev.txt`, `.exp_run/`,
  `fusion_tests/`, full-model artifacts, and `vllm/`) were not touched.
- `timeout 60 tt-smi -ls --local` listed four Blackhole P150b boards.  A
  source-backed `MeshShape(1, 4)` open/close printed `MESH_SMOKE_OK`; firmware
  was 19.9.0.  No reset or recovery was required.
- The starting topology audit and all candidate-family evidence were reread
  before hardware work.  The current measured path reproduces the same
  packed-QKV, SDPA, packed gate-up, persistent async CCL, replicated-residual
  graph as the original clean-pass.  Fresh report op counts are identical to
  `tracy/final_post_fused_review`, so prior exact family rejections remain
  like-for-like rather than stale-source bounds.

Current warmed latency command:

```bash
GEMMA4_MULTICHIP_BENCH=1 MPLCONFIGDIR=/tmp/mpl_stage05_revalidation \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k test_multichip_warmed_latency \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/current_head_latency.xml
```

This passed both layer kinds.  Same-process single-P150 versus TP4 medians
were 2.665372 -> 2.3350725 ms prefill and 1.186321 -> 0.4635775 ms traced
decode for sliding attention; 3.415086 -> 2.4464875 ms prefill and
1.333623 -> 0.5181375 ms traced decode for full attention.

Current correctness/stress/fallback command:

```bash
MPLCONFIGDIR=/tmp/mpl_stage05_revalidation \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/current_head_suite.xml
```

Result: 12 passed and 48 intentional env-gated candidate/profile skips.
PCC was 0.999845810/0.999756572 prefill and
0.999802416/0.999718188 traced decode for sliding/full.  Non-aligned sliding
1,025/1,057 scored 0.999497101/0.999160561; batch-32 decode scored
0.999675961/0.999658173.  The runtime source audit and terminal persistent-pool
cleanup passed.

Advertised-position command:

```bash
GEMMA4_MULTICHIP_EXACT_CONTEXT=1 \
MPLCONFIGDIR=/tmp/mpl_stage05_revalidation \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k test_multichip_advertised_context_traced_decode \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/current_head_exact_context.xml
```

Both layer kinds passed at absolute position 262,143.  KV dtype/layout and
persistent allocations are unchanged, so `doc/context_contract.json` needs no
Stage 05 capacity update.

The reduced profiler used `GEMMA4_MULTICHIP_PROFILE=1` and the four
`test_multichip_profile` windows under Tracy.  `tt-perf-report` 1.2.7 was run
with advice for each `MC_<kind>_<mode>` signpost.  The enriched source CSV hash
is `0454648836ec291e6ee090a10ff1202f7a65d9bfe0ef0016c7bd0f8cc8fa6002`.
Device sums are 1,153.2255/1,418.152 us prefill and
427.9625/481.827 us decode for sliding/full.  Current versus prior clean-pass
device sums and op counts are unchanged within measurement noise; the
full-prefill wall anomaly therefore localizes outside device work.  Advice
remains closed by the retained role-specific geometry, precision/fidelity,
prefill L1/adapted-block, and coherent fused-family experiments.

Watcher command:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
MPLCONFIGDIR=/tmp/mpl_stage05_revalidation \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH pytest -vv -s --tb=short \
  models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  -k 'paged_decode_trace_matches_optimized_baseline or sliding_nonaligned_window_wrap_matches_baseline' \
  --junitxml=models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/evidence/current_head_watcher.xml
```

Result: 4 passed, with no worker/NoC watcher error or assertion.  Ethernet-only
instrumentation remains disabled for the inherited active-Ethernet firmware
buffer limit; this does not disable fabric execution.  Watcher and profiler
were separate hardware runs.

- Compact current artifacts: `evidence/current_head_*`,
  `tracy/current_head/{README.md,source_csv.sha256,*_PREFILL,*_DECODE}`, and
  `perf_accounting.json`.
- Raw Tracy/device intermediates totaling about 720 MB were deleted after the
  enriched CSV and compact evidence were verified; rerunning the recorded
  capture command is the recovery path.
- No model implementation change was selected: the current default reproduces
  the strongest correct topology and decode performance, while every material
  alternative remains rejected by exact current-applicable evidence.
- Fresh independent review: `stage_review_current_head.md`, verdict
  `CLEAN PASS`, no required work.  The reviewer independently checked current
  source/profile hashes, XML/JSON, device sums, current-versus-prior operation
  signatures, runtime/inter-layer contracts, watcher scope, candidate
  applicability, and unrelated-dirty-state isolation.
- The local current-checkout technical checkpoint SHA is recorded below after
  creation.  No push was performed.

Current-checkout technical checkpoint:
`339026e08e3ae8fed55f15bf24205ed050975e07` (`Revalidate Gemma 4 31B
multichip decoder`).  The follow-up metadata-only commit records this SHA.
