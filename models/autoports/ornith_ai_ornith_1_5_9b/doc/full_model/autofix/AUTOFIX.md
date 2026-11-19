# AutoFix: Ornith full-model request and trace contracts

The diagnosed runtime and numerical failures are repaired and the affected
full-stack correctness gates pass. This report does not stand in for the final
performance, qualitative, or independent stage review records.

## Diagnosis and rejected explanations

The initial 33-token request passed, while a 34-token request with live decode
traces created 20 unsafe program allocations. `AUTODEBUG.md` traced them to eager
prefill programs whose logical shapes and runtime offsets were not prepared.
Both requests padded to 64 rows, so padding alone did not establish an identical
program signature. Garbage collection, trace eviction, broad corruptible scopes,
and precision changes were rejected as repairs for that failure.

Resumed prefill separately failed at starts 32 and 68 for a 513-token prompt.
Canonical whole-request precision profiles and aligned recurrence snapshots
separated chunk-boundary policy changes from cache defects. No BF8 KV rollback,
CCL precision increase, replicated projection, or host computation was retained.

## Repairs and discriminating evidence

| Contract | Minimal repair boundary | Accepted evidence under doc/full_model |
|---|---|---|
| Arbitrary logical prefill with live traces | `prefill_plan.py`, `prefill_ops.py`, `prefill_runtime.py`: finite original-policy profiles, fixed staging, runtime indices, masks and logical slicing; prepare before capture | `serving_prefill_probe.json`, `watcher_trace.json`, `capacity_full_final.json`; unchanged program counts |
| Resumed recurrence and precision | Whole-request horizon profiles; retained aligned FP32 snapshots and bounded overlap | `../../readiness/prefill_resume_check.json`: starts 32, 68 and 512 pass, both wrong-history controls fail |
| First-use cache mutation | Prepare before admission; snapshot and restore the actual owned/external binding | `cache_ownership_full.json`, `external_cache.json` |
| External global pages and inactive rows | Global allocation metadata, native -1 cache sentinel, safe masked RoPE and preallocated recurrence scratch | `batch32_full.json`, `batch_state_probe_b3.json`, `batch_state_probe_b4.json` |
| Async output ownership | Queue owning transfer before overwrite; immutable submission activity mask | `serving_decode_probe_final2.json`: two pending submissions, unchanged active outputs, inactive -1/zero sentinels |
| Common sampler shape and temperature | Format physical 32 rows once and preserve raw temperatures for logical scoped updates | `sampling_transition2.json`, `prefill_generation.json` |
| Greedy token feedback | Common sampler trace writes persistent token input; model trace increments positions; no host reconstruction | `runtime_fallback_probe.json`, `sampled_trace_probe.json`, `sampling_transition2.json` |

The normalized linear and attention probes compare against the prior optimized
layer implementation. Their retained smoke/boundary artifacts and failed
hypotheses are indexed in `doc/full_model/work_log.md`. The repaired generator
preserves the selected decoder weights, fidelity, KV dtype split, CCL policy,
and inter-layer residual layout. Decoder edits are restricted to global cache
ownership/addressing and masked-position handling.

A deliberate unsafe late-allocation positive control still fails under
`TT_METAL_TRACE_ALLOC_TRACKING=1` with program buffers included. Removing that
buffer allows replay. Exact producer-logit acknowledgments remain; no blanket
warmup/capture exemption suppresses the check.

## Validation commands and results

All device commands source `/home/ttuser/ornith-9b/lane-env.sh`, serialize the
four chips, and use bounded timeouts. Exact final commands, source hashes, start
and completion times are in `doc/full_model/final_gate_commands.json`.

- `python -m readiness_check.run_prefill_check ... --reference readiness_aime24_chat.refpt`: top-1 99%, top-5/top-100 100%.
- `python -m readiness_check.run_teacher_forcing ... --reference readiness_aime24_chat.refpt`: traced top-1 98%, top-5/top-100 100%.
- `python -m readiness_check.run_prefill_resume_check ... --lengths 513 --window 128`: every start passes; both negative controls fail.
- `python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_capacity_probe --all-layers --also-context ...`: 262143-token prefill, decode at position 262143, and full 262144-token prefill pass with 820 programs unchanged.
- Focused serving, sampler, batch, external-cache and runtime probes: accepted JSON paths above; failed runs are retained separately.
- Host contract suite: 40 tests pass, `doc/full_model/logs/host_contract_tests.log`.
- Production/probe Ruff checks: pass, `doc/full_model/logs/lint.log`.

## Remaining scope and limits

Maximum-context evidence proves allocation, execution, finite outputs and stable
program reuse; it is not a full-context HF accuracy comparison. Batch 32 is
proved separately at context 512; simultaneous 32-by-262144 occupancy exceeds
physical DRAM. Scheduler prefix caching has the documented recurrent-state
exclusion, while public resumed prefill is validated. Explicit host-logits and
host-greedy compatibility paths are exposed and excluded from device token-out
performance. No device reset, remote push, vLLM implementation or server was used.
