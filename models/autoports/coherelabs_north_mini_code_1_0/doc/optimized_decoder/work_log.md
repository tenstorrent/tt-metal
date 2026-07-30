# Optimized decoder work log

## Scope and provenance

- Model: `CohereLabs/North-Mini-Code-1.0`
- Revision: `d11e61a842617a22dc328552fa5bb86231ee4f37`
- Branch: `skillexp-work-northmini-fresh`
- Functional-stage base: `4e45a256771`
- Pre-stage HEAD: `a2719cfecabce0124a96835b26eb31fb41928e2d`
- Hardware: one Blackhole p300c through a 1x1 mesh; four local p300c boards
  remained discoverable throughout.
- Scope: optimized decoder, its tests/performance harness, and decoder docs.
  No multichip, full-model, or vLLM work was started.

Every canonical final JSON contains argv, branch/HEAD, source SHA256, Git
state, effective dtype/fidelity policy, program geometry, and execution
topology. Canonical records bind to optimized source SHA256
`21d222646cffcb8c09c0fba1b60b0c1f30f117e6dc6a53b49b3f1af340602ecd`.
Historical JSONs whose manually supplied PCC did not describe the exact
timing workload are retained but explicitly marked unbound.

## Implementation and evidence sequence

1. Audited the functional topology. Packed QKV and native paged SDPA were
   already appropriate. The dominant issue was evaluating all 128 experts
   despite top-8 routing.
2. Added an optimized-owned runtime with named construction-time policies.
   Inheritance is limited to static validation/setup helpers; optimized
   prefill/decode do not call the functional runtime.
3. Added on-device routing, active-expert union formation, and
   `ttnn.sparse_matmul`. Exact per-token sigmoid routing weights are retained.
4. Packed decode gate/up weights and replaced two same-input sparse matmuls
   with one projection plus on-device split. Prefill keeps separate
   projections because the large-M split placement collided with reserved
   dispatch cores.
5. Swept BFP4/BFP8/BF16, LoFi/HiFi2/HiFi4, 8/16/24/32-core expert
   geometries, role-specific inner/output/subblocks, packed/unpacked gate-up,
   attention fidelity, cache dtype, and prefill chunk/program strategies.
6. BFP4 experts failed official-weight populated-history PCC at 0.98240.
   Production decode/small-prefill experts therefore use BFP8/HiFi2.
7. BFP8 cache passed official-weight populated history at 0.999373, matching
   the BF16 control at 0.999373. BFP8 is required to preserve advertised
   batch-32/context-500000 capacity once phase-specific BF16 prefill weights
   coexist with the cache.
8. The corrected block/subblock geometry was retried after its first API
   error; the retry stalled beyond 60 seconds. Only the exact process was
   terminated. `tt-smi -ls --local` showed all four devices healthy, so no
   reset was performed.
9. Sparse chunk 1024 regressed large prefill to 646.509 ms at batch 32.
   Optimized-owned dense expert prefill with an 8x8 program reduced the final
   pre-review result to 135.011 ms; the final-source result is 135.064 ms.
   Logical sequence 1025 remains valid at PCC 0.99996.
10. Batch-1 16-core expert geometry was selected for speed. Faster batch-32
    timings from that geometry and a 24/32-core split were rejected. The
    16-core batch-32 gate failed at PCC 0.953752; the 24/32 historical timing
    lacks an exact bound PCC and is not used as correctness evidence. The
    pre-review serving geometry used correctness-safe 32-core, one-tile
    output blocks.
11. A second independent review found insufficient capacity accounting,
    isolated precision controls, batch-specific tuning, and exact
    official-weight batch-32 evidence. Each gap was closed and all final
    performance and Tracy evidence was regenerated.
12. Regenerated the four final latency JSONs and all four Tracy reports from exact
    final source SHA256
    `21d222646cffcb8c09c0fba1b60b0c1f30f117e6dc6a53b49b3f1af340602ecd`.
13. Selected width-sharded residual/RMSNorm at both decode batches. Selected
    DRAM-sharded QKV/O at batch 1 only; the legal batch-32 retry was faster
    but failed correctness.
14. Rejected BFP4 attention after exact official-weight gates failed at PCC
    0.981934/0.980513 for batch 1/32 despite small speed gains.
15. Executed optimized prefill at the exact 500000-token advertised boundary
    and separately asserted expected shape and finite output under watcher.
16. The third rereview found that official dense layer 0 still used generic,
    separate same-input gate/up projections. Packed BFP8 gate/up plus device
    slices, a 64-core 1x3 gate/up program, and a 32-core 1x2 down program
    reduce dense batch-1 decode from 0.263 to 0.207 ms.
17. Followed the profiler's DRAM-sharded down recommendation through corrected
    retries. It regresses batch 1 (0.221 versus 0.207 ms) but improves batch
    32 from 1.498 to 1.004 ms, so selection is batch-aware. The 64-core
    gate/up program further reduces batch 32 to 0.843 ms while remaining
    neutral at batch 1.
18. The fourth review required current-topology precision controls. BFP4/LoFi
    gate/up plus BFP8/LoFi down passes the 0.995 floor at
    0.995275/0.998849/0.998981 and is faster than the BFP8/HiFi2 control at
    prefill and decode, both batches. The lower correct precision is selected.
19. Replaced the generic unpacked comparison with two explicitly tuned
    48-core 8x6 projections at batch 1. They reach 0.19694 ms and beat the
    best correct packed control at 0.19884 ms. Batch 32 retains the native
    packed 64-core 1x3 path at 0.76639 ms.
20. Retried packed K blocks 4/8/16/32 with interleaved input and DRAM output.
    Block 4 passes exact correctness; larger batch-32 candidates hit a hard
    L1 circular-buffer allocation limit after the corrected retry, rather
    than being rejected on their first validator error.
21. Re-ran the exact 500000-token prefill using the frozen final source:
    331.01 s pytest call, 332.92 s elapsed, finite output, watcher clean.
22. Commit hooks removed unused imports and normalized import order. The
    aggregate watcher then passed at `d4665e8a` in 245.36 s. An exact harness
    smoke exposed that `MODEL_ID` had been an intentional public re-export;
    it was restored as an explicit alias. The resulting `21d22264` source
    changes no runtime method. All headline timings and four Tracy profiles
    were regenerated exactly against `21d22264`; the 500k and ordered-trace
    results remain valid by this audited import/export-only equivalence.

## Final-review AutoFix follow-up

The final reviewer correctly rejected the earlier claim that decode
RMSNorm/residual and attention sharding had been ruled out. The common
`RMSNorm1D` and `Attention1D` modules contain legal single-device
width-sharded norm plus DRAM-sharded QKV/O patterns. Four construction-time,
default-preserving candidates now make the missing experiments reproducible:

- `dense_prefill_packed_2d_g8x8`: one BF16 all-expert gate/up matmul followed
  by an on-device split, retaining the selected 8x8 large-prefill program;
- `decode_sharded_residual_chain`: 32-core width-sharded residual and
  `LayerNormShardedMultiCoreProgramConfig`;
- `attention_dram_sharded_chain`: the sharded residual candidate plus
  DRAM-sharded QKV/O weights and common-module decode matmul configs;
- `router_decode_g2_block8_subblock2`: an explicit two-core router program.

Hardware results:

- `decode_sharded_residual_chain`: PCC 0.998264/0.997756 and
  0.692/3.845 ms at batch 1/32; selected.
- `attention_dram_sharded_chain`: batch-1 PCC 0.998264 and 0.678 ms;
  selected there. The third, legal batch-32 geometry reached 3.583 ms but
  failed PCC at 0.982460; rejected for serving batch.
- `dense_prefill_packed_2d_g8x8`: exact batch-32 optimized/HF PCC 0.989842
  against functional/HF 0.988945, but 142.395 ms versus 135.011 ms for the
  split path; rejected. Batch 1 sequence 128 is below the candidate's
  1024-token applicability threshold.
- `router_decode_g2_block8_subblock2`: PCC 0.998225/0.998236; batch-1
  0.710 ms, slower than the selected chain. Its attempted combination with
  the sharded chain reported the actionable batch-fusion requirement; the
  smaller isolated gain did not justify displacing the selected chain.

The `batch1_exact_nnz8` candidate addressed the remaining `sparse_matmul`
runtime-count path and is selected in the production default.
The TTNN contract requires `nnz == count_nonzero(sparsity)` exactly; a wrong
value can deadlock. Therefore this candidate does not infer presence from
sigmoid routing scores, which Blackhole may flush to zero. For one-token
decode it scatters exactly representable ones at the eight unique `topk`
indices, passes `nnz=8` to packed gate/up and down, and retains the original
sigmoid scores for the final weighted reduction. Batch-32 decode and every
prefill path keep `nnz=None` and device-side dynamic inference. Static
contract tests pass. The exact official-weight b1 gate passes at PCC 0.998268 and the warmed
traced mean improves from 0.678852 to 0.604447 ms. Batch 32 retains dynamic
inference because its exact official-weight active union is 87.

The same AutoFix pass resolved exact b32 prefill agreement. The prior
HiFi4/FP32 router candidate directly matched functional output at only
0.982107 PCC. Omitting that explicit compute policy for prefill routing while
retaining the selected M=1024 8x8 expert programs produces 0.999871 direct
PCC, 0.989082 optimized/HF versus 0.988945 functional/HF, and final warmed
means of 13.228/135.064 ms at batch 1/32.
The intentionally retained pre-remediation failure is
`artifacts/final_prefill_b32_direct_correctness.txt`; the passing remediation
is `artifacts/final_autofix_correctness_watcher.txt` and is repeated by the
runtime-equivalent aggregate `watcher_clean_final.txt`.

The rereview also found that dense-layer precision had only synthetic
evidence. Cached official shard 1 includes dense layer 0, so the optimized
test/performance harness now loads it directly. Current-topology BFP4/LoFi
scores 0.995275 prefill and 0.998849/0.998981 decode b1/b32; BFP8/HiFi2
scores 0.999475/0.999343/0.999490. Since the former meets the 0.995 floor and
is faster at all four workloads, production selects BFP4/LoFi gate/up and
BFP8/LoFi down. Official final warmed prefill is 0.582/12.278 ms and decode
is 0.197/0.767 ms, versus functional 0.632/13.687 and 0.356/6.641 ms. A
fresh dense b1 profile records 0.173 ms device operations, 0.029 ms gaps,
0.107 ms matmuls, and 36.4% modeled DRAM roofline. The dense b32 profile
records a 0.766 ms complete window, including packed gate/up and the selected
DRAM-sharded down.

The artifact audit also found that several manually attached PCC values did
not describe the timing workload: notably g16 batch 32 attached the batch-1
PCC despite an exact failed batch-32 PCC of 0.953752, and final prefill batch
1 attached the batch-32 PCC. The performance harness now requires a
repository-local evidence file, SHA256-binds it, and records explicit PCC
scope/status/threshold in `correctness_binding`. Historical measurements are
preserved as historical evidence; contradictory PCC bindings must be cleared
or superseded, never silently rewritten as newly measured results.

## Final gates

- Aggregate watcher: `TT_METAL_WATCHER=10`, 30 passed and five explicit
  opt-in precision/history/context probes skipped in 245.36 s; no watcher
  error.
- Populated-history watcher, including the ordered functional-then-optimized
  mesh-reopen control: both pass; optimized PCC 0.9991298457.
- Dense/full/forced-RoPE, sliding/RoPE/MoE, and full/no-RoPE/MoE are covered.
- Nonaligned logical lengths 1, 31, 33, 65, and 1025 pass.
- Real-weight repeated traced decode produces five bitwise-identical replays.
- Selected real-weight batch-1 decode PCC: 0.9982682247.
- Selected serving batch-32 sparse decode PCC: 0.9974305620.
- Exact real-weight batch-32/sequence-128 prefill: optimized/functional PCC
  0.9998711168; optimized/HF 0.9890819239 versus functional/HF 0.9889451121.
- Optimized dense prefill executes at the exact 500000-token context with
  expected shape and finite output under watcher. Advertised
  batch-32/context-500000 cache plus weights coexist; last-position sparse
  execution passes. Batch-1 last-position execution passes every layer kind.
- Final warmed means: prefill b1 13.227796 ms, prefill b32 135.064102 ms,
  traced decode b1 0.603727 ms, traced decode b32 3.846495 ms. Official dense
  layer-0 means are 0.581544/12.277912 ms prefill and
  0.196636/0.766503 ms traced decode at batch 1/32.
- Final Tracy windows contain no host fallback; profiler and watcher runs
  were separate.

The first final review exposed a real ordered trace corruption in the earlier
topology: after a functional trace and mesh reopen, the old optimized sparse
populated-history replay returned values around `1e34` (PCC -0.034734).
AutoFix isolated the boundary with these passing controls:

- empty-history full decode (PCC 0.998997);
- one `ttnn.sparse_matmul` trace (bitwise capture/replay);
- routing plus the packed sparse-expert composite (bitwise capture/replay).

Unpacked sparse gate/up, DRAM sparse intermediates, dummy trace, explicit
trace regions, cache clearing, retained host-copy sources, garbage
collection, and release synchronization did not repair the old topology.
The final selected sharded/DRAM chain does: the exact ordered external-runtime
control now passes both functional and optimized targets, with optimized PCC
0.9991298457. The obsolete strict-xfail was removed. Final evidence is
`artifacts/final_ordered_trace_reopen_pass.txt`; the `autofix_*` records are
retained as causal investigation history.

## Commands

Watcher:

```bash
source python_env/bin/activate
TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  2>&1 | tee \
  models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/watcher_clean_final.txt

NORTH_MINI_LONG_HISTORY_TRACE=1 TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py::test_optimized_sliding_moe_populated_history_dynamic_trace_replay_matches_reference \
  2>&1 | tee \
  models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/artifacts/final_long_history_watcher.txt

# Ordered external-runtime control: both targets pass after the fixture closes
# and reopens the mesh.
NORTH_MINI_LONG_HISTORY_TRACE=1 \
TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py::test_sliding_moe_populated_history_dynamic_trace_replay_matches_reference \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py::test_optimized_sliding_moe_populated_history_dynamic_trace_replay_matches_reference

NORTH_MINI_NEAR_LIMIT_PREFILL=1 TT_METAL_WATCHER=10 \
pytest --timeout=600 -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py::test_optimized_advertised_context_prefill_executes_with_finite_output
```

Candidate/final performance pattern:

```bash
python -m models.autoports.coherelabs_north_mini_code_1_0.tests.optimized_decoder_perf \
  --mode <prefill-or-decode> --batch <1-or-32> --layer 1 \
  --sequence 128 --candidate <candidate> --real-weights \
  --warmups 10 --iterations <50-for-b1-or-30-for-b32> \
  --json-out <artifact.json>
```

Profiler pattern:

```bash
python -m tracy -r -p -v -m \
  models.autoports.coherelabs_north_mini_code_1_0.tests.optimized_decoder_perf \
  --mode decode --batch 1 --layer 1 --candidate default --real-weights \
  --warmups 1 --iterations 1

tt-perf-report <ops_perf_results.csv> \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --active-experts 8 --no-color --csv tracy/final_decode_b1_rows.txt
```

The equivalent prefill command uses batch 32, sequence 128, and
`PERF_PREFILL`/`PERF_PREFILL_END`.

## Optimize checklist

- [x] Functional PCC and performance baselines established.
- [x] Operation-topology audit recorded before candidate selection.
- [x] Packed QKV retained; packed same-input decode gate/up implemented.
- [x] Precision/fidelity swept separately for attention, experts, cache,
  prefill, and decode.
- [x] Official dense layer-0 BFP4/LoFi versus BFP8/HiFi2 precision, warmed
  prefill/decode at both batches, and representative Tracy evidence recorded.
- [x] Batch-1 and batch-32 candidates measured independently.
- [x] BFP4/LoFi expert kernels and BFP8/HiFi2 expert kernels swept across
  batch-specific 8/16/24/32-core geometries.
- [x] DRAM-sharded decode matmul legality considered; rejected for routed
  experts because that family lacks sparse active-expert inputs, with the
  measured all-expert baseline as movement evidence.
- [x] Official dense DRAM-sharded down retried to a legal M-axis-sharded
  topology at both batches; rejected at b1 and selected at b32.
- [x] Packed dense same-input gate/up compared against equivalently tuned
  separate 48-core projections at batch 1, retained at batch 32, and packed
  K blocks 4/8/16/32 retried with legal interleaved-input/DRAM-output layouts.
- [x] DRAM-sharded QKV/O and sharded norm/residual candidates measured at
  both batches; sharded residual selected at both, DRAM attention selected
  only at batch 1 after the serving PCC gate.
- [x] Packed large-prefill gate/up corrected retry measured at both batches;
  rejected at serving batch as slower, and recorded as inapplicable below the
  batch-1 1024-token total-M threshold.
- [x] Explicit router program candidate measured at both batches and rejected
  against the faster selected chain.
- [x] Role-specific grid, inner-block, subblock, memory, output dtype, and
  compute-kernel policies measured.
- [x] Corrected block/subblock retry performed after the first API error.
- [x] Small and large prefill program/chunk strategies measured at both
  batches; dense large-prefill composite selected.
- [x] Native SDPA and paged cache operations retained and profiled.
- [x] Public nonaligned sequence lengths remain valid.
- [x] Runtime source has no torch conversion or host fallback.
- [x] Current `tt-perf-report` rows inspected and conclusions recorded.
- [x] Warmed before/after prefill and traced decode recorded at batch 1 and
  serving batch 32.
- [x] Primary batch-1 decode wins; serving batch and both prefill batches do
  not regress.
- [x] Repeated/stress coverage is deterministic; the aggregate suite and
  fresh-process and ordered long-history runs are watcher-clean.
- [x] Independent `$stage-review` clean-pass.
- [x] Stage-owned local commits created and SHAs recorded.

## Stage review and commits

- First review verdict: more-work-needed.
- Remediation: source-bound candidate provenance; full correctness floor;
  packed gate/up; populated-history and advertised-context tests; large
  prefill composite; current Tracy evidence; explicit rejection
  evidence for sharding, precision, and program candidates.
- Second review verdict: more-work-needed.
- Remediation: byte-exact capacity accounting and coexistence test; isolated
  cache/expert/attention precision controls; independent batch-1/batch-32
  expert geometry sweeps; exact official-weight batch-32 prefill gate;
  regenerated final latency and Tracy evidence.
- Fourth/final review verdict: **CLEAN-PASS**. Required work: none. Hard-check
  gaps: none affecting the stage gate.
- Implementation/evidence commit:
  `b495df82d00623fac85d585123cd7a1568fc03ed`.
- Provenance-clarification commit:
  `446a47cf5bf1a9b9cd9956a13b366d0ad124cb98`.
- Post-review evidence-log commit:
  `f8b538e1f34b7c839e76d4035136a2018fb4276c`.
- SHA-recording finalization commit: this document's HEAD, reported in the
  stage handoff because a commit cannot contain its own SHA.
- Push: never performed.
