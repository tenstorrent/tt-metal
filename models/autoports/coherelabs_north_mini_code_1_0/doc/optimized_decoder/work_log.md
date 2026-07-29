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
- [x] Runtime rows verify BFP8 attention, BFP4 dense MLP, BFP8 experts, LoFi
  compute, DRAM-sharded weights, and batch-specific programs.
- [x] Paged SDPA and cache TTNN composites are retained and swept.
- [x] QKV is packed; dense gate/up packed versus separate was measured and
  separate selected.
- [x] Important memory/program/compute-kernel configs are explicit.
- [x] Core grids, `in0_block_w`, output blocks/subblocks, dtype, fidelity,
  cache update, and memory candidates were swept separately by dominant role.
- [x] Attention BFP4/LoFi and dense/expert BFP4/LoFi trials have real-weight
  correctness evidence.
- [x] DRAM-sharded decode matmuls are used at b1 and b32.
- [x] MoE active-expert execution is branch-proven at b1 decode/prefill.
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

Stage implementation/evidence checkpoint: `03b1b0078f1` (`Add optimized North
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
