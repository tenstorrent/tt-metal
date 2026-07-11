# Fused decoder work log

Target: `google/gemma-4-31B`

Branch: `odjuricic/agentic-research/graph-rewrite-skill`

Initial draft base: `82ce4e431f2`

Final validation resumed at functional-decoder checkpoint: `5fa49e9fa25`

## Scope

Changed only `tt/fused_decoder.py`, `tests/test_fused_decoder.py`, and
`doc/fused_decoder/`. The existing `doc/context_contract.json` remains valid
and unchanged. No later autoport stage was started.

## Device record

- `timeout 60 tt-smi -ls --local`: one Blackhole P150 visible.
- 1x1 TTNN mesh open/close with `trace_region_size=0`: `MESH_SMOKE_OK`.
- All device commands were serialized. Watcher and profiler evidence came from
  separate processes/runs.
- No reset or recovery was required.

## Fusion loop

1. Audited the functional code and all four baseline `tt-perf-report` tables.
2. Enumerated TTNN dedicated ops and relevant model usage for RMSNorm, SDPA,
   QKV/head transforms, RoPE, cache update, matmul activation, and binary
   post-activation.
3. Added bounded-L1 dedicated sliding prefill head concat, true matmul+GELU,
   and add+scalar fusion.
4. Profiler inspection refuted the first activation syntax: `activation=` was
   recorded as a user request but auto config kept `fused_activation=null`.
   An explicit 11x10 1D multicast config with `in0_block_w=2`, 1x7 subblock,
   Mx7 output block, `per_core_N=7`, and approximate GELU produced a fused
   kernel and removed the unary row.
5. Wrote decode QKV directly to L1, eliminating the DRAM-to-L1 copy.
6. Adapted full K/V fused update to equal non-overlapping K/V core grids.
   Sliding retains separate updates because the fused op cannot express the
   bounded circular-cache contract.
7. Tried direct sharded SDPA output; TTNN rejected it with `Sharded output not
   supported for GQA`. Retained the one required DRAM-to-height-sharded move.
8. Measured packed gate/up and slice-placement alternatives. Packed gate/up
   was slower. The original single-sample slice result was later superseded by
   the paired AutoFix experiment described below.
9. Repeated the topology table after each accepted/rejected pattern. The final
   assessment is in `README.md`; no applicable graph-fusing pattern remains
   unclassified.
10. Audited the inherited context contract and found that rounding a padded
    sliding-prefill tail to a 64-token cache block violated live-slot ownership
    after the 1024-token circular window wrapped. A focused fused regression
    verified the bug: seq 1025 decode PCC was 0.994710, and seq 1057 exposed an
    all-zero live K slot. Restored exact logical-tail writes and reran the same
    experiment: decode PCC 0.997649/0.999271 and direct K/V ownership PCC
    0.999885-0.999911. This conditional path does not execute in the aligned
    seq-128 performance workload.
11. Independent review returned `more-work-needed`. AutoDebug ranked mutable
    trace inputs, long-prefill GELU closure, and evidence provenance as real
    gaps; `AUTODEBUG.md` and `stage_review.md` preserve the findings.
12. AutoFix added the fused mutable-buffer wrapper. Both layer kinds at random
    non-block-aligned and 1023-to-1024 positions consumed changed token/uint32
    RoPE/int32 cache-index buffers, beat stale-output negative controls, and
    remained deterministic; both ordinary and watcher runs passed four cases.
13. AutoFix exhausted long-GELU F4/F2/F1 and C2048/C1024/C128 families. All
    were correct, but normalized fused gate latency was 18.725-20.618 ms versus
    11.196 ms for M4096 B0, so B0 is retained with profiler evidence.
14. A same-process paired device A/B refuted the old one-sample slice ordering.
    Post-projection crop won by 2.512 us median and all six ABBA-cycle means;
    it is now the final runtime path. Temporary instrumentation was removed.
15. The first rereview identified one remaining composition gap at the exact
    context limit. AutoFix added a thin inherited-test wrapper and ran genuine
    distinct-token traced decode at position 262143 against the HF one-query
    oracle. Sliding/full passed PCC 0.999380/0.998937, correct-position RMSE
    beat wrong-position controls, and repeated trace outputs were bitwise
    identical. No implementation change was needed.

## Commands and gates

- Hash-bound final standard suite: `standard_suite_final.log`: 23 passed,
  9 gated candidate/performance/long tests skipped (the two additional skips
  are the env-gated exact-context distinct-token cases).
- Exact context: `long_context_262144_final.log`: 2 passed in 220.15 s.
- Largest non-aligned context: `long_nonaligned_262113_final.log`: 2 passed
  in 222.62 s; sliding prefill/decode parity PCC 0.998834/0.998888 and full
  prefill/decode parity PCC 0.999089/0.998192.
- Genuine advertised-context decode:
  `exact_context_distinct_262144_final.log`: 2 passed in 155.96 s with a
  distinct captured/replayed token, HF reference, wrong-position negative
  control, and deterministic repeated trace for both layer kinds.
- Hash-bound mutable watcher: `watcher_final.log`: 4 passed;
  `watcher_final/generated/watcher/watcher.log` is clean.
- Four final Tracy commands used the single performance nodes in
  `test_fused_decoder.py`; each `profile_command_final.log` records the four
  source/test hashes. Raw CSVs and directly generated filtered reports are
  under `tracy/` with destination-consistent console provenance.
- Final `python -m py_compile`, fused source audit, profiler forbidden-op scan,
  and `git diff --check` passed.

## Correctness and performance

The complete PCC and latency tables are in `README.md`. Headline final values:

| Kind | Prefill | Decode PCC | Traced decode |
|---|---:|---:|---:|
| sliding | PCC 0.999265 at seq 128 | 0.999629 | 2.560 ms |
| full | PCC 0.999233 at seq 128 | 0.999624 | 2.881 ms |

Functional traced-decode baselines were 2.577 ms sliding and 2.911 ms full.
The selected path beats the functional baseline and every correct candidate.
Repository search found the same-family production implementation under
`models/demos/gemma4`, which supplied op idioms, but no same-model, same-stage,
single-P150 fused-decoder profiler reference to compare against.

## Rejected candidates and exact evidence

- Full dedicated head concat: smallest tile still requested 2208512 B on one
  core, above P150's 1572864 B L1 limit. The multi-core structural path passed.
- `activation=` without explicit program config: standalone unary remained;
  candidate CSV retained.
- Packed gate/up: 2.593092 ms and 42 ops versus final 2.560197 ms / 40 ops.
- Slice placement: repeated device timing selected post-projection crop at
  2556.062 us median versus 2558.574 us pre-projection; all six paired cycle
  means favored the selected path and PCC was 1.0.
- Sharded GQA SDPA output: exact TTNN validation blocker; adapted DRAM output
  plus required reshard passed.
- Fused sliding cache update: API lacks `cache_position_modulo`, `block_size`,
  and `num_kv_heads`; using it would violate the advertised circular-cache
  semantics, so it is not expressible in the current op contract.
- Rounded sliding-prefill tails: rejected after the focused wrapped-window
  regression proved cache corruption. Exact logical-tail writes are retained;
  aligned measured prefill still uses the two bulk cache-fill ops.
- Long-prefill fused GELU: F4/F2/F1 real-M4096 candidates were 66.5%-551.4%
  slower; C2048/C1024/C128 normalized to 18.725/18.993/20.618 ms versus
  11.196 ms B0. Exact configs, samples, PCC, and op rows are under
  `candidates/long_gelu/`.
- Pre-add RMSNorm, fused Llama QK RoPE, conv/bias/softmax/reduction patterns:
  semantically inapplicable or already encapsulated by dedicated ops.

## Runtime hashes

- `tt/fused_decoder.py`:
  `941dd1d16b64246111e8402d875cbb7fc1cc6bbf6b33465ff0ba97103df90af6`
- `tt/functional_decoder.py`:
  `2f8a26cbdd8ebb46c8ba478080c963cd288c7936854a7023227ad58d69a144f2`
- `tests/test_fused_decoder.py`:
  `29b99b8c51d8bc3f5052da35f37477267e66215627ece3e100e012071b3eee06`
- `tests/test_functional_decoder.py`:
  `2fd1278ecd6703a62ba6292fa67644c1c7d3e7bc97b4d7877c1b3d9584d2d6be`

## Review and checkpoint

Initial independent `$stage-review`: `more-work-needed`; its findings were
worked through `$autofix`. First rereview found one exact-context distinct-token
coverage gap; that gate now passes through the fused graph. Final independent
rereview: `clean-pass`; no required work remains. See `stage_review_final.md`.

Stage-owned local commit SHA:
`ce88390ebcceb9e8d83af37ed8a166406e360370`. No push was performed.

## Current live-worktree revalidation

On 2026-07-11, the completed stage was revalidated at `8245b883767` before
handoff. The four bound implementation/test hashes still exactly match
`evidence_manifest.md`, and `ce88390e..HEAD` has no change to either decoder
implementation or either decoder test.

- `timeout 60 tt-smi -ls --local`: one Blackhole P150 visible.
- A 1x1 TTNN mesh opened and closed successfully with
  `trace_region_size=0` after setting `LD_LIBRARY_PATH=$PWD/build/lib`.
- The serialized current-suite command
  `pytest -q models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py -s`
  passed 23 tests and skipped the 9 explicitly environment-gated
  candidate/performance/long tests in 104.27 seconds. The active PCC,
  mutable-trace, batch, non-aligned, cache-ownership, determinism, and both
  layer-kind results reproduce the canonical evidence.
- Fresh independent review `stage_review_current.md` re-derived the source and
  gate hashes, profiler totals, functional comparisons, candidate closure,
  watcher state, context contract, and commit scope. Verdict: `clean-pass`;
  no required work.

The live optimized-decoder files and optimized-only `context_contract.json`
additions remain unrelated dirty state and are excluded from the fused-stage
checkpoint. No reset, recovery, or `$autofix` rerun was needed.
