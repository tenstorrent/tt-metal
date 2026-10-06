# Remaining optimization audit

Source/evidence snapshot: 2026-09-26, runtime SHA256
`c96cf65fff7ec69e8193a0df502610bce39f8bd5ea2ab2a0bec3dfbeeda4090a`.
This is a CPU/source-only audit against `.agents/skills/optimize/SKILL.md` and
`tech_reports/LLMs/llms.md` section 4. No new device execution or runtime edits.
Numbers below are median warmed traced **host microseconds**, not device times.
Layer 0 is sliding attention; layer 5 is full attention. The PCC bar remains .995.

## Evidence interpretation and cumulative baseline

`actual_text_fixture_manifest.json` records real corpus tokens, pinned checkpoint,
tokenizer and HF source, and actual layer-input tensors. `actual_*` results with
`input_fixture` use these tensors; `real_weights: true` alone also describes the
older Gaussian-input tests. Gaussian tests remain useful for legal configurations,
layout, tracing and diagnostics. Their precision failures cannot veto candidates
that pass the recorded actual inputs.

The static native-SDPA/BFP8-cache baseline in
`actual_native_qkv_r0_k0_layer{0,5}.json` passes 4096/128 at 1374.83/1478.05 us
(minimum decode PCC .9953817/.9958368). Full attention also uses BFP4 expert
gate/up; sliding retains BFP8. Combined native/cache actual 1025/512 tests pass
at .9956262/.9969529 (`actual_text_native_cache8_stress_layer0.json`,
`actual_text_native_cache8_gate4_stress_layer5.json`). These supersede older
Gaussian-only native/cache objections. The factory still defaults to precise
attention, BF16 cache and BFP8 gate/up for both kinds; the baseline is a candidate,
not yet the final default.

The current attention precision probe **preserves LanePartitionQKV**: it typecasts
the existing packed `.weights` tuple and the source prefill weight/FP32 rows.
Actual-native BFP4/BFP8 results therefore cover the packed lane topology. Policy
JSON can still say BF16 because a test wrapper changed weights after construction;
final runtime rows, not that stale label, must establish the selected dtypes.

## Candidate ledger

| Area | Evidence already fulfilled | Remaining decision or required experiment |
| --- | --- | --- |
| Native attention and cache | Actual headline and 512-step continuation pass both kinds; static public methods accept matching BF16/BFP8 cache pairs. Decode SDPA LoFi/HiFi2 and 32/64/110-core grids ran on actual inputs (`actual_native_sdpa_*`). Full-DST sync is the source-backed requirement. | Combine selected changes and run public contracts on the static implementation. Preserve BF16 update input even when destination cache is BFP8. |
| QKV precision | Actual packed BFP4/BFP8 and HiFi4/HiFi2/LoFi trials exist. Sliding BFP4 passes .9950572 at 1302.81 us; full BFP4 fails .9942133, while BFP8 passes .9959558 at 1417.57 us. | Sliding BFP4 has little headline margin: cumulative 512-step evidence is required. Do not infer that passing separate changes compose. |
| QKV packed/separate | `qkv_geometry_l{0,5}_{packed,separate}_{8x4,8x8}_k{11,22}.json` is a legal 16-run comparison, BF16 weights, two terms, lane16, matched BFP8-gate/BFP4-down experts. Packed best 1869.39/1647.12 us; separate best 1875.96/1659.75 us. All passed S33/8 Gaussian. | The family has been tested. If selecting BFP4/BFP8 QKV, repeat only the best legal packed/separate configurations under that dtype/fidelity and actual input. Existing BF16 timings do not settle the new precision policy. Explicit separate `qkv_grid=(8,4)` or `(8,8)`, subblock1 avoids the full-attention auto-grid/subblock4 incompatibility for narrow K. |
| QKV DRAM sharding | Adapted N padding and readers made the family legal. Actual native sliding readers2/K11 passes at 1383.54 us, readers3/K11 1415.32; full readers3/K11 1514.03. These lose to interleaved 1374.83/1478.05. Readers1/K1 also pass and lose. | No need to retry the original invalid padding or development NameErrors. K22 static CB needs 1,737,728/1,872,896 bytes versus 1,572,864 available; full readers2/K11 overlaps live allocation by 68,608 bytes. Lower-weight DRAM remains a conditional candidate if the final profile makes its likely gain material. |
| Attention output projection | Actual BFP4/BFP8 trials pass both kinds. BFP4 .9955892/.9951880 at 1362.33/1441.20 us, versus baseline 1374.83/1478.05. | Cumulative stress and final dtype/fidelity selection remain. Source still uses auto-configured `ttnn.linear`, DRAM output and interleaved weights after head concat. A legal explicit 1D/DRAM candidate and per-role fidelity comparison remain required if the new profile shows this role is material. QKV geometry does not cover this role. |
| Routed expert precision | Active top8 decode uses sparse matmul, `nnz=top_k`, sparse-input down, BF16 output and routing-weighted reduction; no dense all-expert decode. Actual sliding gate BFP4 fails across 11/22/44-core and K11/K22 configurations, including BFP8-down control (.9928476 native). Actual full BFP4 gate passes headline and 512 steps. BFP4 down has actual passing cumulative evidence. | Retain per-kind precision; Gaussian full-gate failures are superseded. Expert activation BFP8 actual headline passes but has negligible measured gain (1373.71/1466.62): include only if cumulative evidence and final timing justify it. |
| Expert geometry and packing | `expert_sweep_{sliding,full}` covers gate BFP8/BFP4, LoFi/HiFi2, 11/22/44 cores, K2/11/22, with BFP8 down. Independent BFP4-down controls compare 22/44/88 cores (`geometry_retry_down*`). `role_separate_experts*` exercises legal separate gate/up, but uses BFP8 gate **and BFP8 down**. | A bounded matched final-policy packed/separate comparison remains, especially full BFP4 gate/BFP4 down. Current separate program is fixed to 22 cores, one N tile/core; 11 cores with two N tiles/core is another legal adapted choice, unlike using 44 active cores for a 22-tile separate width. Compare K11/K22 including activation/slices/down/reduction. Reuse earlier grids as screening; do not rerun the entire Cartesian sweep. |
| Shared MLP | BFP8/LoFi DRAM readers1/2/3, interleaved, and packed/separate were compared (`shared_bf8_*`); packed reader1 won 4190.76 versus separate 4202.69 and interleaved 4227.14 us. Larger K22 became legal with readers2/3, then lost (`geometry_retry_shared*`). Actual sliding gate/down BFP4 fail .9949651/.9946170; full both pass separately with small gains. | BFP8 packed policy has legal topology evidence; no missing initial DRAM attempt. If full adopts BFP4 shared weights, recheck best packed/separate and geometry under that precision, then cumulative stress. Do not require the full old matrix if BF8 remains selected. |
| Dense prefill | Large QKV 2D configs K4/K11 are already legal and measured at S4096 (`headline_dense2d*`). K4 290113.65/334200.09 us versus matched existing path 289409.31/334179.07; K11 is slower. This is an actual adapted experiment, not an API rejection. | Closed for unchanged QKV policy. A newly selected QKV dtype changes this contract: refresh the inexpensive best 2D-vs-current comparison if prefill cost remains material. The knob only modifies QKV; do not describe it as tuning every dense prefill projection. |
| Active prefill experts | Runtime union-of-selected-experts sparsity, inferred nnz, L1 intermediates and explicit configs are implemented. Batch32/64/128 and chunk1024/2048/4096 have whole-layer evidence. Batch128 initial L1 failure was adapted by freeing temporaries and weighting down in place; batch32 wins. Actual prefill gate BFP4 passes both kinds. | Full prefill-down BFP4 lacks an actual-input decision: old maximum-context Gaussian PCC .9944210 is its present rejection. Run an actual-input full-prefill-down4 control before treating BF8 as required. Combine any selected prefill reductions with cache-consuming decode. Do not repeat the already completed batch-size adaptation. |
| Norms/router/movement | Residual sharding and L1 controls were measured. Native SDPA removes the precise attention gather/primitive sequence. Actual all-sharded norms pass at 1338.48/1477.23 us. Router lane16 K11/22/44/88 all pass but lose (1449–1452/1562–1564 us). Composite routing passes at 1357.30/1461.44; logits-BF16 control alone does not help. | Reject the slower lane router; finish cumulative evidence for useful norm/composite choices. Fresh profiling must identify remaining material conversions, especially norm sharded-to-interleaved, native query BF16/DRAM, attention result FP32/head sharding, and output DRAM-to-residual. Do not optimize every conversion without measured cost. |

Correction to historical narrative: the JSONs show **18/36 passing sliding** expert
screen cases (all BFP8 pass, all BFP4 fail), versus 36/36 full. Some prose says both
36-case sweeps passed. Actual sliding BFP4 failures independently support its
rejection, so this correction does not change the precision decision.

## Required remaining closure, in execution order

1. **Choose and validate one cumulative actual-input candidate per kind.** Combine
   native attention/cache with selected QKV/output, expert/shared, router/norm and
   prefill options. Run 4096/128 and 1025/512 through the static runtime, preserving
   the strongest prior correct policy. Keep the fastest passing cumulative path;
   isolated passing probes are not final acceptance. Full prefill-down4 and
   any chosen output fidelity remain focused precision decisions above.
2. **Finish only material matched comparisons.** Under the chosen dtypes, compare
   best legal packed/separate QKV and expert gate/up; use previous grids/K blocks
   to bound the work. A new BFP4 shared policy needs the same small comparison.
   Give material output projection its own explicit configs and adapted DRAM
   candidate. Larger K22 and independent down-grid trials already exist; a fresh
   profile can determine whether further 44/88 gate K blocks are worth adapting
   separately from the down K dimension (22 tiles maximum).
3. **Revalidate changed public cache/runtime contracts.** The old `default_batched`,
   `default_prefix_continuation`, `default_request_reuse` and `default_long_*`
   hashes are `9fb650...`, BF16/precise attention. They do not validate native
   BF8 integration. Cover both layer kinds: B32 with distinct per-slot positions
   and page tables; nonaligned S65 and S1025 tail; continuation starting inside a
   partially occupied 32-token page; request reuse; BF16 compatibility alongside
   selected BFP8 storage; maximum 262144 and near-maximum 262143 context. Existing
   actual S1025/512 already covers a substantial nonaligned native/cache path.
   Source preserves B32 by tracing a per-slot loop, not by treating padded rows
   as active users. Validate mechanics and real-input correctness without turning
   inherited Gaussian-only precision failures into vetoes or lowering thresholds.
4. **Promote and measure the final default.** Rerun final static defaults with
   same-fixture fused and strongest candidate baselines, deterministic replay,
   program-cache/no-host-fallback guards and separate Watcher checks. Produce
   current complete-layer decode/prefill profiler tables and `tt-perf-report`
   advice closure. Verify runtime input/weight dtypes, fidelity, grids, sparse
   nnz and cache dtype; reconcile device time, host time, op gaps and roofline
   from the same run. Initial expert-candidate profiles cannot stand in for the
   current ~1.3–1.5 ms native path.
5. **Refresh capability/accounting records.** `doc/context_contract.json` still
   calls old BF16 tests the final policy and reports BF16 KV bytes. At identical
   page geometry, BFP8 cache tile storage is 1088/2048 of BF16: 1,140,850,688 bytes
   sliding and 570,425,344 bytes full at maximum context. Include retained source
   weights/FP32 rows and any new precision copies in persistent storage. No
   context reduction is authorized. Finish the cumulative policy table and
   independent stage review after the final measurements.

## Items that should not expand the remaining work

- Single-device decoder scope has no CCL, LM head or sampling path; multichip
  collective and token-out checklist items do not apply here.
- Router projection cost in the initial table is 49.258 us multiply +42.044 us
  reduce +2.074 us transpose =93.376 us, excluding gaps. TopK is another24.379 us.
  The approximately252 us reduction was QKV, not router. Current router cost
  requires a current profile; the failed lane-router speedup is already measured.
- Setup-only duplicate QKV FP32 rows are a persistent-memory cleanup opportunity,
  not evidence of per-token transfer. Releasing them is optional unless new
  allocation pressure makes them material; verify source-prefill/fallback users
  first. This audit does not authorize unrelated cleanup.
- No further Gaussian-only math-model investigation is required to justify
  native SDPA, BFP8 cache or full BFP4 experts after their actual-input passes.
  Keep prior AutoDebug reports as diagnostics and clearly label their input type.

## Follow-up: two material operation-count controls

The subsequent source audit found no actual-input evidence for one-term QKV or
lane counts below16. Existing actual runs retain lane16/terms2; terms3 controls
also exist. After establishing a correct cumulative native base, compare
terms1 at lanes16/8/1 on actual inputs, preserving selected weight dtype and
fidelity. Native BF16 query rounding makes this worth testing, but does not
prove K/V/cache/routing equivalence. Old Gaussian lane controls do not close it.

Packed expert decode also retains a standalone accurate GELU followed by mul.
Separate decode and active prefill already use fused input GELU/mul. The same
GELU0 variant and BF16 internal packing boundary support a bounded fused packed
control; `expert_fused_gelu_source.md` and the recorded patch document the exact
source reasoning. The optional switch was subsequently applied atomically after
the parent confirmed hardware idle; default remains false. These are new focused trials, not reasons to redo
the already completed sparse-matmul and prefill geometry matrices.
