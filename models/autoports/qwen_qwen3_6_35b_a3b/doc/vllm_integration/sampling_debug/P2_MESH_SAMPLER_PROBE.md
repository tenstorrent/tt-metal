# Canonical TP4 sampling diagnostic prepared after native64 acceptance

Source-only preparation; no device execution by this agent. The parent reports native64 all24 fixture PASS and candidate32 token digests identical to baseline32 after installing the rebuilt runtime. This document does not independently certify those hardware artifacts.

`tests/run_vllm_sampling_mesh_probe.py` uses **QwenGenerator._sampling_forward unchanged**, common SamplingGenerator, production SamplerCCL and CollectiveWorkspace. It constructs only the fields needed for that call, so no weights or decoder/cache are loaded. Input shape is `[1,1,32,248320]`, BF16, vocab-sharded across the real 1×4 mesh. Sixty-four distinct leading values per row have known IDs distributed across all four shards. Diagnostic host reads never supply subsequent sampled token feedback. Qwen's device merge, output copy and active seed advancement execute normally.

Acceptance checks four output/seed replicas; eager versus retained trace identity under changed seeds, k/p/inverse-temperature and logits; active rows1/32; inactive token/seed preservation; authoritative request row permutation; rank33–40 selection for top40; and unchanged program entries. Every physical draw width has exactly one retained trace. Raw normalized tokens and digests are written beside JSON. This is a sampler-state check, **not** the adapter's full ownership/remap acceptance gate. Width32 vs64 greedy identity is required. Top20 seeded identity is recorded but not assumed: a changed reduction order may change a rounded probability boundary without changing the target distribution. The conservative top-p support check is not an exact probability-distribution proof.

The common wiring hunk must be applied before running either width (the probe asserts that its opt-in was recognized). With the lane environment active and exclusive healthy devices, prepared commands are:

```bash
source /home/ttuser/qwen36-35b-a3b/lane-env.sh
M=models/autoports/qwen_qwen3_6_35b_a3b
D="$M/doc/vllm_integration/sampling_debug"
# Parent only, after its original-path diagnostic closes:
git apply --include=models/common/sampling/tt_sampling.py "$D/canonical_sampling64_wiring.patch"
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 900 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_sampling_mesh_probe \
 --candidate-width 64 --draw-widths 32,64 --draws 32 --mode acceptance \
 --output "$M/readiness_vllm/sampling64_probe/mesh_acceptance64.json"
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 300 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_sampling_mesh_probe \
 --candidate-width 32 --draw-widths 32 --mode timing \
 --output "$M/readiness_vllm/sampling64_probe/mesh_timing32.json"
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 300 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_sampling_mesh_probe \
 --candidate-width 64 --draw-widths 32,64 --mode timing \
 --output "$M/readiness_vllm/sampling64_probe/mesh_timing64.json"
```

Timing mode refuses profiler/watcher flags. Its measurement queues100 warmed trace replays and synchronizes once; no reads or uploads are timed. Every timing number records physical32 rows, active1/32 rows and the actual profile. Compare identity-permutation `sampler_ms` on equal profile/active shape. The candidate64/native32 case measures whether a retained smaller native draw helps **without a second sampling algorithm or independently owned seed state**. Local top-k64, gather256 and all other canonical work remain identical to candidate64/native64. Compare candidate32/native32 against candidate64/native64 to expose total widening cost; compare candidate64/native32 vs candidate64/native64 to isolate the native-width effect approximately. No sampler timing is headline model t/s/u.

## Why widening can slow existing greedy/top20 requests

Current per-chip vocab width is62080; no vocabulary chunk split is needed. Candidate32 gathers128 values per row (four tiles); candidate64 gathers256 (eight tiles). Local top-k work and gather payload therefore increase even when logical k is1 or20. Native64 performs two accumulator stages for each incoming tile: eight gathered tiles require12 cascade sorts after initial pair sorts. The default32 native merge route remains unchanged. These are source-based risk factors, not a measured latency claim.

A viable optional fast trace keeps **one** max_top_k64 common sampler, shared parameter/seed/token buffers, and captures native32 and native64 signatures. It selects native32 only if every retained physical row has effective k≤32, otherwise native64. Dormant rows still execute native sampling before their outputs are discarded; their preserved parameters must fit the selected width too. It must never clamp40 to32. This retains the canonical algorithm and preserves state ownership, but cannot recover the candidate64 local-topk/CCL cost. Independent width32/64 sampler instances would introduce extra seed/remap authority and are not proposed. This sampler probe preceded production selection. The subsequent production selector and dormant-wide safety correction are audited in `FINAL64_ADAPTER_AUDIT.md`; this earlier probe alone does not cover mixed per-row widths or the adapter transition.

## Serving policy omission repaired in the proposal

`canonical_sampling64_wiring.patch` now also updates all three `tt/serving.py` validation call sites. `_validate_sampling` takes a required `device_top_k_limit`, supplied from **generator.sampler.tt_sampling.max_top_k**. Thus top40 is accepted only by a configured sampler with adequate capacity; unrelated configured32 samplers retain their limit. Host-logit mode remains explicitly independent. No global64 fallback is added. The full patch still passes `git apply --check`; no production Python file was edited by this agent.

`run_sampling_policy_cpu.py` extracts the actual validator AST from a candidate or applied source, tests24 boundary cases (k0/20/32/40/64/65; capabilities32/64; device/host), and verifies all three callers source the bound from the owning sampler. The isolated candidate at `/tmp/qwen_sampling64_serving_candidate.py` passes; JSON is `sampling64_serving_policy_cpu.json`. After applying the proposal, rerun against production source:

```bash
python "$D/run_sampling_policy_cpu.py" --source "$M/tt/serving.py" \
 --output "$M/readiness_vllm/sampling64_probe/serving_policy_cpu.json"
```

CPU fixture artifacts are `sampling_mesh32_cpu.json` and `sampling_mesh64_cpu.json`; Python compile and repository hooks pass. Actual hardware behavior, stochastic row relocation, dual-trace allocation tracking, and measured widening overhead remain unverified by this agent.

## Host fake and capture audit follow-up (2026-11-17 03:14–03:17 UTC)

Added `max_top_k=32` to the existing `run_serving_commands_cpu.py` fake sampler; retained all routing, prevalidation-before-mutation, stale-input, remap, page and prefill assertions. It passes against the current production serving source **and** the isolated proposed64 source at `/tmp/qwen_sampling64_serving_candidate.py` (loaded using an isolated temporary ROOT; neither TTNN import nor production patch application). Other host tests only extract `_neutral_device_features`, not the capacity validator or serving entry points; no additional fake changes are needed. Existing test-file changes made by the parent were preserved.

The probe now matches production `prepare`'s sampler warmup: `sampler.precompile(...)` then `_sampling_forward()` **for each native width before the first capture**. Program entries are recorded after warmup and each capture; a cold capture compilation fails immediately. The final replay check requires unchanged program entries and exactly one capture per requested native width. Repository hooks pass after formatting.

Source audit: all persistent input/output/seed/parameter buffers, common penalty/logprob buffers and CCL semaphores exist before capture. Each canonical call performs two all-gathers, traversing both members of the semaphore ping-pong ring; warmup therefore covers both. Sampler intermediates are explicitly deallocated or local to the call; the only returned persistent token output is supplied by the caller, and logprobs are disabled. Parameter refresh uses host-only staging plus copies to preallocated device tensors. The diagnostic does not create device tensors between captures; it does run eager sampling for correctness after captures, whose temporaries must be freed before replay. Allocation tracking remains enabled in acceptance, with program-cache allocations included and **no acknowledgement/suppression added**. Actual dual-trace allocator safety still requires that acceptance run; this audit does not claim a hardware pass.
