# Gemma4 full model — Stage 06

Full-model TTFT **2115.07ms**; trace-verified batch1 token-out throughput
**49.22 tokens/s/user** at4096 input /128 output / one request.
Measured in `performance.json` with all30 layers and canonical split sampling. Teacher-forcing decode is49.84 tokens/s/user on the separate161-input/100-position
AIME workload, with host token injection; it is not autoregressive performance.
Status: **complete**, independent stage review **clean-pass**
(`stage_review_final.md`).

Target: `google/gemma-4-26B-A4B-it`, revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, four Blackhole P300c ASICs,
1×4 mesh, branch `gemma-4-26b-a4b-it`, starting commit `6b1776a0bc`.

The model loads all30 real decoder layers, scaled token embeddings, final
RMSNorm, tied-weight LM head and HF logit softcap. It uses the accepted Stage05
TP4 attention / hybrid EP4-prefill and indexed TP4-decode experts. Its BF16
replicated residual layout is unchanged; no inter-layer gather is added.
Embedding hidden shards gather once at entry; the LM head emits vocabulary
shards directly into the common sampler.

Decoder policy and rejected alternatives remain in
`../optimized_multichip_decoder/README.md`, `candidate_comparisons.md` and
`residual_contract.md`. No dtype/fidelity/KV/CCL relaxation is selected here.
Both layer kinds retain BFP8 paged storage, with the decoder's existing BF16
prefill intermediate/cache-fill and paged decode contracts. Model-owned CCL
payload buffers are shared only across serial layers on one queue. Private
layer semaphore sets remain. The new router pool shares identical constants
and scratch whose outputs are copied before expert consumption; see
`AUTODEBUG_prefill_l1.md` and `AUTOFIX.md`.

## Public API and ownership

`tt/generator.py:build_generator(model_dir, mesh_device, **kwargs)` returns the
standard Metal readiness Generator. `prefill_forward` accepts explicit tokens,
logical prompt lengths, slots, page table and per-layer cache pairs. The
caller can pass mixed prompt lengths; padding/chunking stays in the public
model path. `decode_forward` accepts tokens, positions, page table and cache;
fixed inactive rows use position−1 and are excluded from RoPE/cache work.
The accepted decoder already processes batch slots serially within each layer;
the inactive branch retains those same TP4 kernels and skips inactive slots.
Active-slot mask or cache identity changes require a new trace binding.

Standalone `generate` owns its cache and tokenizer. `enable_trace` is explicit.
`host_sampling=True` is an explicit greedy, unpenalized test-compatibility
mode; unsupported sampled/penalty policies raise rather than silently change
semantics. Device sampling supports the common top-k/top-p and penalty path. Optional
log-probability requests are rejected before state mutation: both common
implementations share a calculator limited to8/32-device meshes. See
`AUTODEBUG_logprobs.md`; no host fallback is substituted.
Host compatibility is never the optimized measured path. `reset` zeroes standalone-owned cache/state and keeps
buffer/trace identities. Warmed requests with the same prompt shape and default greedy policy reuse
traces; other request signatures release traces before prefill compilation. `teardown` releases traces.
External cache contents remain caller-owned. `max_seq_len` is an explicit
construction allocation setting. Full30-layer262143/262144-token prefill and
final-position decode pass with the full cache; the supported context is262144
(`capacity.json`, `../context_contract.json`).

## Sampling and tracing

See `sampling_comparison.md` for both common implementations. The selected
SamplingGenerator uses32 physical candidates per vocabulary shard with
semantically greedy k1/p0 parameters. Its sampling trace writes directly into
the persistent decode token input. The model trace advances both position
representations on device. Request-seed setup is outside capture; sampled-mode
seed state advances on device. Page-table updates happen only when content
changes. Explicit host sampling and teacher-forcing injection are separate
compatibility paths, reflected in counters.

The focused reduced probe passes token-feedback snapshots, position coherence,
changed/unchanged table handling, cache-zero reset with retained identities,
and repeated generation under trace allocation tracking. See
`trace_watcher.json` and `trace_watcher_fixed.log`, including retained-trace
changed-prompt comparison against fresh capture. MixedB3 with inactive slot1
and B32 now compare prefill and two decode steps against independent B1
caches; minimum PCC0.99999994 (`trace_mixed_slots.json`, `trace_batch32.json`).
Public token outputs use persistent storage, preserving trace allocation safety.

## Current evidence

Fresh AIME24 HF chat-template reference:100 continuation tokens and top100,
`reference_metadata.json`, `hf_reference_retry.log`. Full all-layer prefill:
top1=.96, top5=1, top100=1. Full all-layer traced teacher forcing:
top1=.94, top5=1, top100=1. Both score100 token positions; see `readiness_final.json`
and `readiness_final.log`. This is one AIME prompt, not a full AIME benchmark.

All six shared qualitative prompts have HF and TT completions in
`qualitative_hf.json` and `qualitative_tt.json`. Reading them found coherent
answers without mechanical repetition or wrong-language drift; lexical
differences and same-prefix HF ranks are documented in `qualitative_verdict.md`.
Full-context and full30-layer B32 validation pass (`capacity.json`,
`trace_full_batch32.json`). The reduced profiler report is in
`profile_terminal/`: the sampler trace occupies512.37us of the complete3078.76us
two-layer window and does not dominate. Full-model device time and rooflines
remain unmeasured; reduced profiles are not reported as full-model metrics.
The standard HF/TT autoregressive comparison and degeneracy check pass
(`autoregressive/`, `degeneracy.json`); both outputs were read. Their different
English introductions remain coherent and both truncate at128 tokens.
Independent review returns clean-pass (`stage_review_final.md`).

Exact commands and failed controls are recorded in `work_log.md`.

The headline workload uses exactly4096 tokenized document-continuation tokens
and forces128 generated tokens; it is performance stress coverage, not the chat
quality verdict. TTFT includes request state preparation and decode-trace setup
in the final runner. Token-out decode includes sampler trace and caller token
readback; separate `generation_wall_ms` and `trace_setup_ms` expose setup cost.

Native Watcher repair: guard unused scatter initialization for multicast
all-gather4096-byte pages. Linear/ring focused controls and the original model
Watcher test pass. Full native build is unverified because Docker is unavailable
(`native_build.log`); the changed device kernels JIT-compiled in passing tests.

The accepted layer-stack host-wall lower bound is19.843ms/token; full-model
token-out is20.315ms/token (2.38% higher). This cross-run diagnostic is not a
device-time subtraction (`layer_stack_comparison.json`).

Final sampled/penalty/host-compatibility checks pass under allocation tracking
(`sampling_contract.json`, `sampling_contract_pass.log`). Unsupported TP4
logprob requests preserve trace/token/position state.
