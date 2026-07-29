# Optimized decoder work log

Date: 2026-07-28

Model: `CohereLabs/North-Mini-Code-1.0`

Revision: `d11e61a842617a22dc328552fa5bb86231ee4f37`

Functional ready commit: `78dbd88bec7`

## Scope and starting point

The stage started from the completed functional decoder and is restricted to
`tt/optimized_decoder.py`, optimized-decoder tests, and this documentation.
No multichip decoder, full model, generator, LM head, sampling, or vLLM work
was started.

Initial sequence-128 functional wall baselines:

| Kind | Prefill b1 | Prefill b32 | Decode b1 | Decode b32 |
|---|---:|---:|---:|---:|
| dense/full/forced-RoPE | 0.636 ms | 13.758 ms | 0.356 ms | 6.652 ms |
| sliding/RoPE/MoE | 14.908 ms | 147.182 ms | 9.528 ms | 11.122 ms |
| full/no-RoPE/MoE | 14.655 ms | 146.699 ms | 9.524 ms | 11.129 ms |

The topology audit preceded implementation. It identified split Q/K/V,
repeated gate/up input reads, interleaved decode residual boundaries,
unconfigured decode matmuls, default precision/fidelity, dense all-expert MoE,
batch-independent shard assumptions, cache update choices, and prefill
programs as the material opportunities. The action/evidence table is in
`README.md`.

## Implementation and experiment ledger

1. Added an independently materialized optimized decoder rather than routing
   performance tests through the functional methods.
2. Packed QKV and used DRAM-width-sharded weights for legal batch-1 and
   batch-32 decode. Padding to one tile satisfies the in0-height requirement;
   batch 1 was not rejected as illegal.
3. Established an L1 width-sharded residual/norm/attention chain.
   A mixed qkv32/mlp16 attempt failed because its shared normalized shard made
   K-shard 2 incompatible with MLP `in0_block_w=4`; adapting it requires an
   extra reshard, and the 16-core chain wins. Final Tracy evidence still
   contains four required width-shard conversions (about 6.4 us total) at
   operator-family boundaries; coherent 8/12/32-core attempts regressed the
   whole layer and the documentation no longer claims zero reshards.
4. Swept decode geometry at both batches:
   default 0.2756/0.3414 ms; 16-core 0.2628/0.3401 ms; 32-core
   0.2683/0.3311 ms. Subsequent output-projection and precision tuning selected
   coherent 16-core programs at 0.1865/0.2521 ms.
5. Compared packed and separate dense gate/up with identical surrounding
   contracts. Separate wins 0.2535/0.3188 ms versus 0.2628/0.3401 ms.
6. Compared fused and separate paged cache updates after correcting an initial
   fused K/V grid-overlap bug with disjoint grids. Fused wins b1; separate wins
   b32.
7. Crossed dtype, fidelity, and geometry:
   - BFP8/LoFi attention selected.
   - all-attention BFP4/LoFi measured 0.1950/0.2576 ms but the preliminary
     real-weight/random-activation layer-0 PCC 0.988737 failed.
   - BFP4/LoFi dense decode gate/up/down selected with preliminary
     real-weight/random-activation layer-0 PCC 0.998964; authentic propagated
     activation evidence was added in item 13.
   - BFP4 prefill weights missed the PCC bar at seq 31/33/65; phase-specific
     BFP8/HiFi2 prefill weights pass and report 0.5094/4.6603 ms.
8. Compared explicit and framework-selected paged SDPA programs. The latter
   wins 0.1865/0.2521 ms versus 0.1919/0.2545 ms.
9. Added large-prefill 2-D programs. An 8x8 b32 candidate exceeded the
   1.573-MB L1 CB limit with a 1.639-MB request; the adapted 10x10 program
   passes and wins.
10. Added routed sparse MoE:
    - BF16/HiFi2 router preserves top-k.
    - BFP4/LoFi experts failed the preliminary synthetic active-expert route
      at PCC 0.982485.
    - BFP8/LoFi experts pass the preliminary layer 1/4 routes at PCC
      0.998290/0.998173.
    - gate/up 8x3 and down 8x8 are legal on the 110-core Blackhole grid.
      A larger 10x10 down attempt initially requested 70 receivers with only
      64 working cores; it was adapted to the valid 8x8 family rather than
      rejected on the first API error.
    - `in0_block_w` 16/12 improves b1 MoE decode from 0.8557 to 0.7966 ms;
      6x4 remains 0.8553 ms.
11. A sparse batch-32 family remained 20.535–21.896 ms because sparse output
    padding and receiver topology dominate. A device-resident 100-core dense
    expert family measured 3.330 ms with the passing BFP8 policy and
    140.962 ms for b32 prefill. This is a performance win but does not satisfy
    the optimize skill's no-dense-all-expert MoE checklist item. `$autofix`
    was therefore run; see the AutoFix section below.
12. Kept BF16 cache. BFP8 cache passed correctness but measured 0.1871 ms at
    b1 versus selected BF16 0.1865 ms, despite improving b32
    0.2521→0.2448 ms. The primary b1 regression and context-contract churn do
    not justify selection.
13. Closed the first independent review:
    - Crossed coherent 8, direct/output 12/16, and 32-core geometry under the
      selected BFP8-attention/BFP4-dense policy at both batches. Direct 16
      remains the cumulative choice; exact JSON is in `candidates/`.
    - Added branch-proof layer-1/layer-4 active-expert sequence-33 prefill
      tests. Fixed tail padding/concatenation so logical token rows never
      include internal tile padding.
    - Tuned the selected b32 dense-expert family: automatic 100 cores measured
      3.330 ms versus explicit 100/80/64-core candidates at
      3.366/3.390/3.605 ms and packed projections at 3.395 ms.
    - Loaded authentic checkpoint shards and propagated actual token
      embeddings through prior HF layers. Authentic selected PCC is
      0.998001/0.998580 for dense decode/prefill and 0.999311/0.999742 for MoE
      layers 1/4. BFP4 experts are faster and pass those two authentic samples
      at 0.998206/0.999607, but fail the required layer-4 route stress at
      0.982697; gate-only, down-only, and HiFi2 adaptations also fail.
14. Adapted non-aligned multi-user prefill after the serving test exposed two
    invalid candidates:
    - A special fused b32/s1 QKV geometry passed ordinary execution but watcher
      caught an out-of-bounds BRISC runtime argument.
    - A per-user workaround stalled in a 52-core one-row
      `TilizeWithValPaddingDeviceOperation`. Live `tt-triage` captured BRISC
      writers in `cb_wait_front`, tilize math/pack startup waits, and
      downstream non-idle NoC counters.
    - After a hard four-device reset, the token-packed replacement passed
      normally and under watcher at b32/s1 PCC 0.998695 and b2/s33 PCC
      0.998729. `AUTOTRIAGE.md` records the producer/consumer ledger.
15. Re-ran the final cumulative policy at layers 0/1/4, prefill/decode, and
    batches 1/32 with three warmups and ten samples. The twelve
    `candidates/final_verified_*.json` files record the exact cumulative
    policy and sample distributions.

Every candidate JSON records the exact cumulative policy. A validation/API
failure was followed by a compatible layout/grid/padding adaptation before a
candidate family was rejected.

## Correctness, stress, and context

Final non-watcher suite:

```text
pytest -q -s models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
20 passed in 125.08s
```

Coverage includes the three representative layer kinds, b1/b32 trace replay,
authentic dense and MoE weights/activations, branch-proven active experts,
non-aligned seq 1/31/33/65, multi-user b32/s1 and b2/s33 packing, permuted
paged-cache slots at nonzero positions, and 10 deterministic trace replays.
JUnit plus a complete transcript are saved in `artifacts/`.

The optimized long-context probe:

```text
python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_capacity.py \
  --mode decode --context 500000 --batch 32 --layer 0 \
  --json-out models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/context500000_decode_b32.json
```

Result: 32,768,000,000-byte BF16 KV cache allocated; position 499,999 traced
decode passed with finite output in 130.8709 ms. Weight duplicates therefore
do not reduce the functional context contract.

## Performance and profiling commands

Representative benchmark form:

```text
python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --mode decode --batch 1 --layer 0 --sequence 128 \
  --warmups 3 --iterations 10 --json-out <candidate.json>
```

Final profiles used one warmup and one signposted measured iteration:

```text
python -m tracy -p -r --check-exit-code -o <artifact-dir> \
  models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --mode <prefill|decode> --batch <1|32> --layer <0|1|4> \
  --sequence 128 --warmups 1 --iterations 1

tt-perf-report <ops_perf_results.csv> \
  --start-signpost <PERF_PREFILL|PERF_DECODE> \
  --end-signpost <PERF_PREFILL_END|PERF_DECODE_END> \
  --no-color --no-host-ops [--active-experts 8] \
  --csv <filtered.csv> --summary-file <summary>
```

Profiles cover dense prefill/decode b1/b32, sliding-MoE prefill/decode b1/b32,
and no-RoPE MoE decode b1. Advice remained enabled. Exact device-time,
roofline, and wall/device reconciliation is in `README.md`.

The measured implementation methods contain no torch import, from/to-torch,
explicit tilize/untilize, or host fallback. Filtered profiles confirm no host
ops. Device tilize/untilize rows only occur inside TTNN scatter and the
row-major sparsity-mask composite; timings and the required API contract are
recorded in `README.md`.

## Device safety and watcher

Profiler and watcher runs were serialized and never enabled together.
Pre/post `tt-smi` health showed four p300c boards with DRAM status true,
advancing heartbeats, final temperatures of 48.2–54.4 C, and zero
corrected/uncorrected GDDR errors.

```text
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/watcher/final_rereview \
pytest -q -s models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
20 passed in 134.01s
```

The watcher log contains 2,170 lines. A case-insensitive scan for fatal,
assert, invalid-NOC, CB bounds, overflow, sanitizer, timeout, hang, and error
signatures was clean.

Older and intermediate profiler trees totaling about 928 MB were moved to the
desktop trash after selected raw ops/filtered/summary evidence was retained;
the cleanup is recoverable. No user-authored file was removed.

## AutoFix active-expert closure

`AUTODEBUG.md` diagnosed the full token-by-expert sparse output surface and
dynamic receiver loop. `AUTOFIX.md` records three isolated hypotheses:

1. A binary mask made `nnz=256` exact and watcher-clean. After adapting two
   independent 201-MB L1 OOMs to DRAM, it passed PCC 0.997848 but took
   17.831 ms.
2. Packed sparse gate/up passed PCC across 8x3, 8x4, and 8x6 grids plus block
   sweeps. Best was 19.584 ms.
3. Single-card fused `moe_compute` supports North's exact dimensions and its
   active compute takes 1.642 ms traced, but `compute_only` exposes only the
   last two rolling-buffer expert outputs. Full combine requires fabric; the
   adapted 1x1 fabric attempt timed out waiting for a nonexistent peer.

AutoFix therefore found a shared TTNN API limitation. No model-local
active-expert candidate can produce the full routed result while meeting the
11.122-ms functional no-regression bar. The correct, traceable 3.330-ms
device-resident path remains selected at b32; b1 remains truly sparse.

## Optimize checklist

- [x] Functional checks pass against the optimized methods; no measured
  functional fallback.
- [x] Prefill/decode PCC remains above 0.995 for every meaningful layer kind.
- [x] Paged KV cache, nonzero/permuted slots, warmed trace replay, determinism,
  and non-aligned logical lengths are preserved.
- [x] Decode residual/norm/attention/dense layouts were swept coherently;
  four necessary width-shard conversions remain with whole-layer evidence.
- [x] Prefill is DRAM-interleaved with explicit 8x8/10x10 2-D programs.
- [x] Operation-topology and lower-movement audits are recorded.
- [x] Dense layouts/programs were derived from shapes and swept coherently at
  b1 and b32.
- [x] Single-device stage: collective, CCL, persistent CCL buffer, and fused
  matmul-CCL items are not applicable.
- [x] Final defaults reproduce the best correct dense candidate at both
  batches.
- [x] Runtime rows verify BFP8 attention, BFP4 dense MLP, BFP8 sparse experts,
  BFP4 dense experts, LoFi compute, DRAM-sharded weights, and batch-specific
  programs.
- [x] Paged SDPA and cache TTNN composites are retained and swept.
- [x] QKV is packed; dense gate/up packed versus separate was measured and
  separate selected.
- [x] Important memory/program/compute-kernel configs are explicit.
- [x] Core grids, `in0_block_w`, output blocks/subblocks, dtype, fidelity,
  cache update, and memory candidates were swept separately by dominant role.
- [x] Attention BFP4/LoFi and dense/expert BFP4/LoFi trials have real-weight
  correctness evidence.
- [x] DRAM-sharded decode matmuls are used at b1 and b32.
- [x] MoE active-expert execution is selected and branch-proven at b1
  decode/prefill.
  Batch 32 uses the fastest correct device-resident dense-expert path only
  after AutoDebug/AutoFix exhausted three single-device active-output
  formulations; the limitation and performance comparison are evidenced in
  `AUTODEBUG.md` and `AUTOFIX.md`.
- [x] LM head/sampling items are not part of decoder-module scope.
- [x] Roofline/device/wall accounting is reconciled for the same final runs.
- [x] Batch 1 is primary and batch-32 correctness/performance is preserved.
- [x] Stress/repeated-run, watcher, and post-run device health gates pass.

## Independent review and commits

The first independent review returned `more-work-needed`; its exact actionable
findings are preserved in `STAGE_REVIEW_1.md`. All four P1 findings and the
three hard-check gaps have now been addressed with the evidence above. A fresh
stage review is required before the stage can close. No push will be
performed.

Stage implementation/evidence checkpoint: `f77d4e00940` (`Add optimized North
Mini decoder`).

## Second-review AutoFix closure

The second independent review is preserved in `STAGE_REVIEW_2.md`. Its two
findings reopened AutoFix and produced the following additional evidence and
repairs:

16. Expanded real-weight expert validation to layers 1 and 4, batch 1 and 32,
    non-aligned sequence-33 prefill, and cache-consuming traced decode at
    position 33. Selected BFP8 passes all eight rows at PCC
    0.999103–0.999989. Dense BFP4 also passes those natural-activation rows at
    0.997234–0.999941, but remains rejected because the representative
    synthetic active-expert matrix fails the same dense serving family at
    b32/s1 PCC 0.981633 and b2/s33 PCC 0.981610. The final selected policy
    therefore aliases dense and sparse BFP8 expert weights rather than
    allocating duplicate tensors.
17. Authentic b32/s33 exposed non-finite values in token-packed attention
    before MoE. Attention-only isolation proved the >=1024-row 10x10 QKV/O
    program was the source; reducing expert chunks did not help. The
    non-aligned compatibility branch now uses internal 512-row QKV/O chunks
    and preserves arbitrary public logical lengths. Layer-1/layer-4 b32/s33
    prefill now passes at PCC 0.999509/0.999989.
18. Cache-consuming layer-1 b32 decode then isolated a separate RoPE layout
    defect. All prefetched cache blocks were identical across repeated users
    and unrotated Q/K were correct, while the inherited 8x4 rectangular
    sharding corrupted RoPE lanes 8–31. Matching the exact row-wise sub-core
    ordering used by `nlp_create_qkv_heads_decode` restores all lanes and
    traced decode PCC 0.999150.
19. Adapted the batched DRAM-sharded expert kernel through its full contract
    ladder. Blackhole reports eight DRAM banks. An all-128-expert
    Tile(32,32) launch is physically L1-infeasible before firmware/CB
    reservation; the legal eight-expert group was run in BFP8 and BFP4.
    Projection PCC is 0.99986/0.99384. Measured projection, SiLU, multiply,
    mandatory interleave, expert reduction, and group accumulation establish
    an unrouted lower bound of 5.185/4.170 ms, already slower than the selected
    complete b32 layer at 3.400 ms, so the grouped candidate is rejected with
    hardware evidence.
20. Re-ran the twelve final warmed rows after both layout repairs with three
    warmups and ten samples. Mean prefill/decode milliseconds are:
    layer 0 b1 0.513/0.187, b32 4.683/0.252; layer 1 b1 4.696/0.791,
    b32 140.932/3.401; layer 4 b1 4.892/0.794, b32 140.915/3.384.
    Batch-1 decode remains faster than the best correct baseline and batch 32
    remains non-regressed. Exact samples are in
    `artifacts/final_after_review2_*.json`.

Primary closure artifacts are
`artifacts/authentic_bfp8_matrix_clean.xml`,
`artifacts/authentic_bfp4_matrix_clean.xml`,
`artifacts/synthetic_bfp4_dense_expert_rejection.xml`, and
`artifacts/dram_sharded_expert_candidate_full_lower_bound.xml`.

Second-review repair/evidence checkpoint: `2a9f76b6e29` (`Close North Mini
optimized decoder review gaps`).

21. Ran the expanded final suite normally and under watcher after the two
    hardware-discovered layout repairs. Both runs exercised the optimized
    implementation directly and passed all 30 tests: `175.26s` normally and
    `205.13s` with watcher. The final 1,092-line watcher log has no
    fatal/assert/invalid-NoC/CB-bounds/overflow/sanitizer/timeout/hang/tripped
    or kernel-error signature. Post-run `tt-smi -s` reported four healthy
    p300c boards, DRAM status true, ASIC temperatures 48.7–54.9 C, live
    heartbeats, and zero corrected/uncorrected GDDR errors. Final JUnit
    evidence is in `artifacts/final_after_review2_full.xml` and
    `artifacts/final_after_review2_watcher.xml`; the device log is
    `watcher/final_after_review2/generated/watcher/watcher.log`.

## Third-review AutoDebug/AutoFix closure

The third independent review is preserved in `STAGE_REVIEW_3.md`. Its five
findings were diagnosed in `AUTODEBUG_REVIEW3.md` and repaired under the
AutoFix isolation loop:

22. Split sparse and dense expert precision policy. Sparse active experts
    retain BFP8/LoFi. Dense experts now select BFP4/LoFi. The original
    selected-policy matrix passed two active-sparse batch-1 prefill rows and
    six naturally dense rows: layer-1
    prefill/decode PCC 0.999428/0.997995 at b1 and b32; layer-4
    prefill/decode PCC 0.999990–0.999941/0.997234. Matched b32 decode improves
    from about 3.39 ms with BFP8 to about 2.22 ms with BFP4. Synthetic random
    failures are retained as diagnostics but no longer veto equivalent
    authentic evidence.
23. Replaced the selected batch-1 prefill dense-all-expert path with grouped
    active-expert execution. Tile groups of 24 keep gate/up, activation, and
    down intermediates in L1 and perform routing/reduction on device. The
    selected aligned sequence-128 means are 14.191 ms at layer 1 and
    14.264 ms at layer 4, versus functional 14.908/14.655 ms. Non-aligned
    sequence 33 and 128 tests monkeypatch the dense and legacy branches to
    fail if entered and pass sampled full-output PCC 0.99867/0.99871.
24. Replaced the disconnected DRAM-sharded projection proxy with an opt-in
    real full-chain harness: propagated layer-1/layer-4 activations, all 128
    checkpoint experts, router, gate/up, SiLU/multiply, down, routing,
    reduction, residual, trace, BFP4/BFP8, groups 8/16/32/64, and all legal
    block pairs. G8 BFP4 is correct (PCC 0.99981–0.99985) but slower:
    2.818 ms traced versus 1.883 ms for the selected dense expert chain. G16
    executes but is numerically invalid at PCC 0.675–0.687. G32 fails exact
    CB/L1 capacity; G64 requires 393,216 bytes per bank with only 288,000
    available. The family is rejected on compatible full-chain evidence.
25. Added final optimized capacity evidence. Layer 0 completes logical
    context 500,000 in 159,869.562 ms; layers 1 and 4 complete non-aligned
    context 499,999 in 193,125.821 and 347,770.355 ms with finite output.
    Explicit large-M programs were not rejected on their first error: the
    77-MB-per-core CB request was adapted to automatic programming, then
    width-sharded weight incompatibility was adapted with batch-1
    interleaved QKV/O copies. The context contract remains 500,000.
    A final contract audit extended those copies to every batch so large
    multi-user prefill cannot select a missing weight family. With that final
    resident set, batch-32 context-500,000 cache allocation and traced decode
    pass with finite output in 131.105 ms
    (`context500000_decode_b32_review3.json`).
    A separate watcher-enabled repeat is also finite and its 3,248-line log
    has no fatal, invalid-NoC, CB-bounds, overflow, sanitizer, timeout, hang,
    tripped, or kernel-error signature
    (`context500000_decode_b32_review3_watcher.json`,
    `watcher/review3_capacity_final/`).
26. Re-ran the final correctness suite: `30 passed, 16 skipped in 308.39s`.
    The 16 skips are opt-in DRAM candidate cases. The authentic mixed-policy
    matrix is 8/8 passing.
27. Re-ran the same suite under `TT_METAL_WATCHER=10`: `30 passed, 16 skipped
    in 347.109s`. The 2,170-line log is clean for fatal, invalid-NoC,
    CB-bounds, overflow, sanitizer, timeout, hang, tripped, and kernel-error
    signatures. Post-run tt-smi 6.0.0 reports four healthy p300c devices,
    healthy DRAM, zero GDDR errors, and zero thermal trips.
28. Collected a fresh 12-row Tracy matrix from the final source and analyzed
    every raw ops CSV with advice-enabled `tt-perf-report`. Batch-1 active
    sparse matmuls consume 85% of MoE prefill device time and 57–58% of MoE
    decode. Batch-32 dense expert matmuls remain dominant. No measured window
    contains Torch, `from_torch`, `to_torch`, or a host fallback. Raw and
    analyzed evidence is in `tracy/review3_selected/`.
29. Re-ran all 12 warmed wall rows with three warmups and 20 samples. Final
    means are: layer 0 b1 prefill/decode 0.516/0.187 ms and b32
    4.708/0.252 ms; layer 1 b1 14.191/0.792 ms and b32
    139.959/2.220 ms; layer 4 b1 14.264/0.795 ms and b32
    139.855/2.215 ms. Primary batch-1 decode beats the best correct baseline
    and no batch-32 row regresses. Samples and cumulative policies are in
    `candidates/review3_final_runtime/`.

Commands for the final gates:

```text
pytest -q -s --timeout=900 --junitxml=.../artifacts/review3_full.xml \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py

TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=.../watcher/review3_final \
pytest -q -s --timeout=900 --junitxml=.../artifacts/review3_watcher.xml \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py

python -m tracy -p -r --check-exit-code -o <profile-dir> \
  models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --mode <prefill|decode> --batch <1|32> --layer <0|1|4> \
  --sequence 128 --warmups 1 --iterations 1

tt-perf-report <ops_perf_results.csv> --start-signpost <phase> \
  --end-signpost <phase-end> --no-color --no-host-ops \
  [--active-experts 8] --csv filtered.csv --summary-file summary
```

The final 20-sample table, profiler/device accounting, topology decisions,
candidate rejections, capacity evidence, and artifact paths are consolidated
in `README.md`. The optimize checklist above is complete for this
decoder-only, single-device scope.

Rewritten, model-isolated optimization checkpoints:

- `f77d4e00940` — Add optimized North Mini decoder.
- `2a9f76b6e29` — Close North Mini optimized decoder review gaps.
- `c709816ba57` — Log North Mini optimization checkpoint.
- `a9219e58d4e` — Optimize North Mini decoder active expert paths.
- `a53eebe040d` — Close North Mini optimized decoder review 4.

No commit was pushed.

## Independent review 4 AutoFix closure

`STAGE_REVIEW_4.md` returned `more-work-needed`. AutoFix addressed every
required item and both secondary concerns:

30. Retuned the ordinary dense-expert family under the final BFP4/LoFi
    policy, including automatic split/packed and legal explicit 64/80/100-core
    programs with gate/down block widths 4/3, 8/6, 16/4, and 16/12. At
    layer 1, automatic split is 2.218 ms, the best explicit program is
    2.243 ms, and automatic packed is 2.294 ms. The layer-4 control confirms
    2.215/2.244/2.291 ms. Automatic split remains selected and is 1.2%
    faster than the best explicit candidate. This directly tests and rejects
    the final profile's `in0_block_w >= 2` advice under selected precision.
    Evidence is in `candidates/review4_dense_bfp4/`.
31. Added a compatible packed gate/up topology to the real DRAM-sharded
    full-chain harness. It performs one packed batched matmul per G8 expert
    group, then tile-aligned device slices into the existing L1 activation
    chain. Layer 1/4 pass at PCC 0.999849/0.999806 and trace at
    2.828/2.827 ms, versus the selected ordinary expert-chain
    1.887/1.888 ms. The packed alternative is therefore correct but slower.
    Evidence is in `artifacts/review4_dram_packed.xml` and
    `candidates/review4_dram_full_chain_packed/`.
32. Added runtime branch counters to the authentic precision gate. The
    selected matrix proves that batch-1 prefill enters only the active-sparse
    path while the other six rows enter only the dense path. A separate
    forced-dense BFP4/LoFi matrix proves dense execution for all eight rows.
    Both matrices pass 8/8 with the established PCC range
    0.997234–0.999990
    (`artifacts/review4_{selected_mixed,forced_dense_bfp4}_matrix.xml`).
33. Recorded why synthetic active-expert chunk 128 is ineligible despite its
    10.331-ms screening result: one sequence-wide route union collapses
    toward dense execution and lacks equivalent authentic-output evidence.
    The selected chunk 24 retains active-expert semantics and has real
    sequence-33/128 correctness.
34. Regenerated all 12 final Tracy analyses as retained, human-readable
    operation tables plus advice. Each profile directory under
    `tracy/review3_selected/` now contains `human_report.txt` in addition to
    compressed raw CSV, filtered CSV, runtime JSON, and summary CSV/PNG.
35. Re-ran the revised full suite: `30 passed, 16 skipped in 310.391s`.
    The skips are opt-in DRAM candidates
    (`artifacts/review4_full.xml`). A focused watcher run of the final-policy
    forced-dense BFP4 layer-1/layer-4 batch-32 decode rows passed 2/2 in
    27.262s with no watcher fault signature
    (`artifacts/review4_dense_bfp4_watcher.xml`,
    `watcher/review4_dense_bfp4/`). Post-run `tt-smi -s` reported four live
    p300c devices with healthy heartbeats, zero GDDR errors, and no thermal
    trip.

The stage history is isolated directly onto functional checkpoint
`78dbd88bec7`. `git diff --name-only 78dbd88bec7..a53eebe040d` contains only
`models/autoports/coherelabs_north_mini_code_1_0/` paths. The rewritten
model-only SHAs are listed above. No commit was pushed.

## Independent review 5 OPT-015 attempt

36. Added a model-owned `ttnn-advise capture` target for the final dense
    layer-0 attention+MLP decode block. It reuses the established local
    config, synthetic weights, paged-cache/page-table, RoPE, and decode input
    builders; defaults to batch 32 and supports a separate fresh batch-1
    capture. Python compilation, capture-contract import, and the source-only
    optimized-path audit pass.
37. The required fresh-process bootstrap fails before descriptor generation
    or capture because the only visible tt-mlir environment has neither the
    `ttnn-advise` CLI nor the `ttnn_jit` package. The exact output is retained
    in `shard_advise/bootstrap.txt`; `AUTOFIX.md` records the command, current
    DRAM-sharded baseline, pending compiler-candidate comparison, and recovery
    commands.
38. No `report.json` or `final_ir.mlir` was synthesized, no speculative
    compiler layout/program was applied, and no TT hardware or profiler was
    run. OPT-015 remains blocked on the pinned external advisor installation;
    after that setup, the candidate must be extracted from authoritative IR
    and measured against the selected path.
39. Isolated the review-5 batch-32 prefill discrepancy. The 117.903-ms
    artifact is an incomplete, no-PCC three-sample result from an
    unrecoverable pre-commit implementation that incorrectly forced
    `_qkv_prefill` to batch 1. A final-code single-BFP4-family A/B is
    140.177 ms, refuting the extra 612-MiB BFP8 expert family as the latency
    cause. Added default-off explicit M=1024 gate/up and down program
    controls, perf CLI serialization, authentic precision-gate injection,
    and seven passing static geometry contracts for three legal split/packed
    candidates. Exact evidence and the minimal serialized hardware matrix are
    in `PREFILL_GEOMETRY_AUTOFIX.md`; no TT hardware or profiler was used.
40. Hardware selected packed 80/80-core BFP4/LoFi prefill: layer-1/layer-4
    sequence-128 means are 96.844/96.644 ms after three warmups and 20
    samples, versus the previous 139.959/139.855 ms. Authentic sequence-33
    batch-32 prefill passes at PCC 0.99923857/0.99993403
    (`artifacts/review5_packed_prefill_authentic.xml`). Promoted its exact
    geometry and packing as prefill-only defaults while leaving the global
    decode packing flag false and legacy decode programs automatic. Eight
    static phase/geometry tests and the optimized-path audit pass. The packed
    family adds 216 MiB of persistent BFP4 weights, so final default latency,
    profiler/watcher, full correctness, and advertised-context capacity
    revalidation remain parent hardware gates.
41. The plain promoted default reproduces the selected prefill candidate at
    96.750/96.440 ms for layers 1/4 (three warmups, 20 samples). Batch-32
    traced decode remains split and non-regressed at 2.214/2.219 ms.
42. Revalidated the added 216-MiB resident family at the advertised context.
    Batch-32 context-500,000 construction and traced decode are finite for
    layers 1 and 4 at 3.307/132.660 ms
    (`context500000_decode_b32_layer{1,4}_review5.json`). The supported
    context remains 500,000.
43. Collected fresh, separate Tracy profiles for both changed selected rows.
    Layer-1/layer-4 batch-32 prefill contain 231/229 device ops, zero host
    ops, 95.720/94.711 ms device time, and 97.439/96.458 ms profile wall.
    The selected BFP4/LoFi packed/down rows show the intended 80-core
    `in0_block_w=8/6`, subblock `1x5/1x7` configs, which
    `tt-perf-report` marks as good. Raw/filtered tables, readable advice,
    runtime JSON, and summaries are under `tracy/review5_selected/`.
44. Final correctness is `38 passed, 16 skipped in 383.78s`; skips remain
    opt-in DRAM candidates (`artifacts/review5_full.xml`). The final selected
    layer-1/layer-4 batch-32 prefill rows pass 2/2 under
    `TT_METAL_WATCHER=10` with PCC 0.99923857/0.99993403 and no watcher fault
    signature (`artifacts/review5_packed_prefill_watcher.xml`,
    `watcher/review5_packed_prefill/`). Post-run `tt-smi -s` reports live
    heartbeats and zero corrected GDDR errors on all four devices.

The prefill finding from `STAGE_REVIEW_5.md` is fixed. OPT-015 is the sole
remaining gate: AutoFix cannot run the mandatory compiler seed because the
external pinned tt-mlir environment lacks both `ttnn-advise` and `ttnn_jit`.
Per the shard-advisor skill, building that environment inside this model
experiment is prohibited operator setup. Exact failure and recovery commands
are retained under `shard_advise/` and `AUTOFIX.md`.

## Independent review 6 and checkpoint ledger

45. Committed the review-5 AutoFix closure as `c1b26703d85` (`Fix North Mini
    final prefill geometry`). Repository hooks pass, and
    `git diff --name-only 78dbd88bec7..c1b26703d85` contains only
    `models/autoports/coherelabs_north_mini_code_1_0/` paths.
46. `STAGE_REVIEW_6.md` independently confirms that the review-5 prefill
    geometry finding is closed: the promoted default reproduces
    96.750/96.440 ms, authentic PCC passes for both MoE layer kinds, decode
    does not regress, and the fresh profiler, watcher, capacity, and combined
    suite evidence are mutually consistent. The review report is committed as
    `1774f50bf8c` (`Review North Mini optimized decoder closure`).
47. Review 6 returns `more-work-needed` solely for mandatory OPT-015 after
    this ledger correction. The available external tt-mlir checkout is on
    `mvasiljevic/5738-distributed-rmsnorm-rulebook` at `21c1b3bc4a81`, not the
    required shard-advisor revision, and exposes neither `ttnn-advise` nor
    `ttnn_jit`. AutoFix cannot proceed without prohibited in-experiment
    toolchain construction.

The complete later checkpoint sequence omitted by the earlier ledger is:

- `770f70051f9` — Record isolated North Mini stage history.
- `74c95ddaf4f` — Point North Mini docs at final review suite.
- `c1b26703d85` — Fix North Mini final prefill geometry.
- `1774f50bf8c` — Review North Mini optimized decoder closure.

No commit was pushed.

48. Fixed review-6's checkpoint-ledger finding in `8bd1bf9e318` (`Log North
    Mini final review checkpoints`). Fresh `STAGE_REVIEW_7.md` rereviewed that
    exact commit and confirms the ledger finding is closed, the prefill
    optimization remains intact, and no additional model-code or evidence
    defect exists. Its sole remaining P1 is the externally blocked mandatory
    OPT-015 advisor run. The rereview report is committed as `57e7f984220`
    (`Rereview North Mini optimized decoder closure`). No commit was pushed.
