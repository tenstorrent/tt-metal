# Work log

## 2026-07-30

1. Audited the functional operation topology before tuning: packed same-input
   projections, residual movement, cache updates, RoPE, SDPA, prefill program
   shape, and decode weight placement.
2. Implemented a standalone `OptimizedDecoder` and optimized-only correctness
   and performance tests. No functional inheritance or fallback exists.
3. Swept real-weight projection precision/fidelity and KV dtype at batch 1 and
   32. Selected BFP4/LoFi projections and BFP8 paged cache.
4. Swept decode DRAM-sharded weight geometries on core 8/16/32 at both batches.
   Selected core 8 and QKV/output/gate/down widths 12/12/6/16.
5. Compared packed/separate gate-up, direct/interleaved packed split,
   fused/separate cache update, and explicit/default SDPA. Kept the fastest
   correct cumulative topology.
6. Ran AutoDebug/AutoFix investigations for the packed split, long-prefill L1
   exhaustion, head-width-96 RoPE, and batch-32 prefill configs. Only isolated,
   proven fixes were retained.
7. Replaced manual RoPE with fused `rotary_embedding_llama` using an offline
   adjacent-pair Q/K coordinate permutation. This reduced decode from
   `0.459/0.701` to `0.358/0.473` ms at b1/b32.
8. Selected large-prefill block width 2 after b1/b32 testing. Inner-M/N and
   8x10 adaptations ran but failed batch-32 correctness; they were not rejected
   on their first API error.
9. Corrected both perf harnesses to use the same logical context and positions.
   Recollected functional baseline, final performance, every reported
   real-weight sweep, and all four final Tracy windows.
10. Expanded context/cache coverage to page boundaries, multi-user batch 2,
    nonzero 32769, exact 131072, and varied-position b32 cache consumption.
11. The expanded suite found a physical-M grid undercount for batch-2 sequence
    33. Fixed it by counting per-user independently padded sequence tiles;
    focused and full reruns pass.
12. Ran `TT_METAL_WATCHER=10` separately from profiling. Five representative
    optimized tests passed without watcher errors.
13. The fresh rereview requested same-run profiler reconciliation. Added
    device-op time, op gaps, device-plus-gap time, Tracy E2E, and the separate
    non-profiled headline for all final windows.
14. The rereview also required recorded checkpoint activations. Generated
    `activations/layer0_inputs.safetensors` from the tokenizer and checkpoint,
    then paired each decode row with its real 127-token sliding prefix.
15. Recorded nonzero-cache b32 exposed PCC `0.988046` despite passing random
    controls. Fresh AutoDebug and two independent AutoFix hypotheses refuted
    precision, KV dtype, page permutation, cache fill, and cache update.
16. A controlled 2x2 matrix found only canonical manual RoPE plus default
    paged SDPA passes (`0.999963` at BFP8/HiFi2). Neither rollback alone
    passes. Selected those two defaults while retaining fused cache update.
17. Revalidated final BFP4/LoFi projections and BFP8 KV with recorded cache:
    PCC `0.999264/0.998993` and `0.466/0.793` ms b1/b32. Optimized-prefill
    produced caches independently pass `0.998985/0.999005`.
18. The non-aligned b32 prefix setup exposed a packed-MLP L1 limit for groups
    of 16. Device-only groups of eight pass while keeping public batch 32.
19. Removed the rejected fused path's default duplicate decode RoPE
    allocation. The four final row-major tables are shared across phases;
    context remains 131072.
20. Recollected current-source performance, watcher10, and four-window Tracy
    artifacts. Final same-run device+gap/E2E decode is
    `0.518/0.520` ms b1 and `0.801/0.803` ms b32.
21. Final review identified the manual-prefill RoPE permutation class
    (`3.955` ms, 20.87% of b32 device time). Implemented a phase-specific
    fused-prefill/manual-decode candidate with separate QKV/table basis and a
    device-only strided-slice adapter that restores canonical Q/K before
    SDPA/cache.
22. The adapted candidate passes prefill PCC (`0.998572/0.998578`) and
    optimized-prefill cache-consuming decode (`0.998924/0.998949`) at b1/b32,
    but warmed prefill regresses from `1.596/19.670` to `1.626/28.358` ms.
    Rejected it on measured performance, not on its initial tilized-cos API
    error; kept it as the `phase_split_prefill_rope` sweep policy.
23. Regenerated all four human-readable `tt-perf-report` tables without
    `--csv`, retained machine-readable CSVs, and populated the matching console
    logs. Corrected the final block-width, baseline-command, and SDPA
    descriptions.
24. Recollected the full 16-test suite, watcher-10 subset, nonprofiled
    performance, and all four Tracy windows from the exact post-candidate
    source. Final nonprofiled prefill/decode is `1.609/0.466` ms at b1 and
    `19.745/0.794` ms at b32; all correctness and watcher gates pass.

## Final gates

| Gate | Result/artifact |
| --- | --- |
| Like-for-like functional baseline | `final_functional_baseline.log`: 4 passed |
| Optimized correctness/stress | `final_correctness_after_phase_candidate.log`: 16 passed |
| Multi-user regression | `final_correctness_after_phase_candidate.log`: prefill then cache-consuming decode pass |
| Final b1/b32 perf | `final_perf_after_phase_candidate.log`: 4 passed |
| Optimized-prefill cache control | `final_optimized_prefill_cache_after_phase_candidate.log`: 2 passed |
| Watcher 10 | `final_watcher_after_phase_candidate.log`: 5 passed, clean |
| Final Tracy/profile | `final_tracy_after_phase_candidate.log`, `tracy_final/` |
| AutoFix reports | `AUTOFIX.md`, `AUTODEBUG_RECORDED_STATE.md`, independent H1/H2 results |
| Initial independent review | `stage_review_initial.md`: more work needed; all findings addressed |
| Fresh independent rereview | More work needed; recorded-activation, cumulative-evidence, and allocation findings addressed |
| Final independent review | `stage_review_clean.md`: clean-pass, no required work |

## Review-finding resolution

| Initial finding | Resolution/evidence |
| --- | --- |
| Baseline and optimized used different positions | Both use logical position 127; `perf_*context128.log` |
| Stale core-16 and non-inspectable sweep claims | Every row rerun on the cumulative checkpoint path; selected geometry and precision revalidated after the semantic attention rollback |
| Prefill block advice and RoPE composite unaddressed | Block-2 cumulative path and fused RoPE measured; semantic rereview later selected manual RoPE |
| Decode output-subblock advice unexplained | DRAM-sharded config type has no such fields; source locations documented |
| Long-context coverage too weak | Added nonzero 32769, exact 131072, page boundaries, and multi-user test |
| Watcher level too low | `TT_METAL_WATCHER=10` representative run passes |
| Roofline absent | Byte lower bound and final `tt-perf-report` roofline documented |
| Recorded target activations fail b32 | Semantic cache constructed; 2x2 RoPE/SDPA fix selected; final b1/b32 pass |
| Latest source lacked cumulative evidence | Current full suite, watcher, performance, and Tracy recollected |
| Duplicate RoPE allocation absent from contract | Final default shares tables; 100663296-byte total and zero duplicate recorded |
| Phase-specific fused prefill not evaluated | Adapted canonical-cache candidate passes PCC/cache semantics but regresses prefill to `1.626/28.358` ms; rejected with logs |
| Profiler text reports and policy prose stale | Human-readable reports regenerated; console logs populated; block widths, baseline env, and default SDPA prose corrected |

## Artifacts and caveats

- `final_context128_*.log` records the cumulative topology candidates.
- `final_context128_real_precision_*.log` and
  `final_context128_real_geometry_*.log` are inspectable real-weight sweeps.
- `context128_prefill_*.log` records selected and rejected large-prefill
  adaptations.
- `tracy_final/optimized_ops.csv` is the raw final capture; four filtered CSVs,
  summaries, text reports, and plots are retained.
- `autofix_phase_split_prefill_retry.log` and
  `autofix_phase_split_cache_decode.log` record the adapted prefill-RoPE
  candidate's correctness, performance, and canonical-cache decode controls.
- `tt-smi` was unavailable. Hardware commands were timeout-bounded and all
  device-close logs completed.
- Stage implementation/evidence commit:
  `13eebb362f1c02b5ba9b8a5a45769c04ba4d4d3a`.
- Large reproducibility-artifact commit:
  `d0e6957b4669b555fc7dcb0c8cffe9461cd29a07`.
- No push is performed.
