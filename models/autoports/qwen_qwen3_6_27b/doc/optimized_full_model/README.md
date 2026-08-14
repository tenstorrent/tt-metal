# Qwen3.6-27B optimized full model — TP4 Blackhole

## Headline result

| warmed B1, S128/G128, 4x p300c | completed full-model baseline | optimized default split token-out | change |
|---|---:|---:|---:|
| TTFT | 4036.68 ms | **4041.51 ms** | 0.12% slower |
| token-out decode | 17.4934 t/s/u | **17.6371 t/s/u** | 0.82% faster |
| traced teacher forcing | 6.98 t/s/u | **6.98 t/s/u** | separate compatibility path |
| AIME24 prefill top-1 / top-5 / top-100 | 92% / 100% / 100% | **92% / 100% / 100%** | preserved |
| AIME24 teacher top-1 / top-5 / top-100 | 97% / 100% / 100% | **97% / 100% / 100%** | preserved |

The optimized measurement is the generator default: no force-argmax flag, no
host logits boundary, no per-token synchronization/readback, and no
full-vocabulary all-gather. It uses separate nonblocking model and sampler
traces; the sampler writes `tt_out_tok` directly into the persistent token read
by the next model replay. The semantic probe returned the exact global greedy
token 248046 after overwriting seed 123. Evidence:
`artifacts/final_default_split_no_readback_b1_s128_g128.json` and its matching log.

## Target and preserved numerical policy

The target is a 1x4 ring over four Blackhole p300c devices, tensor parallel 4,
`FABRIC_1D_RING`. This stage preserves the selected decoder-stack policy and its
rejection ledger: replicated BF16 inter-layer residuals; BF16 activation and
CCL boundaries; BFP8 paged KV and linear recurrent state; BF16 HiFi2 full
attention QKV/O with HiFi4 SDPA; BFP4 LoFi linear projections and MLPs; BF16
HiFi2 recurrent matmuls; decode weights DRAM-sharded and activations L1
width-sharded. It does not adopt a rejected low-precision projection policy, a
replicated TP stream, or run a datatype frontier search.

The full path remains mesh-native: embedding, mixed 48-linear/16-full decoder
stack, residual boundaries, terminal norm, device-local sharded LM head,
sampler-ready logits, split sampling, token/position/RoPE advance, paged cache,
changed-only page tables, and generator orchestration. Public request state is
explicit (`cache`, page table, position, prompt length, batch/fixed-slot active
state). The Watcher run passed mixed non-aligned S65/S63 prompts, an inactive
row, non-greedy `temperature=.8/top_k=5/top_p=.9`, traced feedback, exact
inactive KV preservation, and reset/reuse:
`artifacts/mixed_slots_split_watcher.json`.

## Greedy sampler repair and performance closure

The prior semantic-greedy shortcut gathered the full 248,320-wide vocabulary
and forced argmax. The generic split path instead sent the 62,080-wide local
vocabulary through a slow single-core TopK (about 9.7 ms). Padding to 65,536 was
also invalid for the optimized factory and regressed the reduced path.

The final path applies an explicit sharded invalid-vocabulary mask, pads each
local shard, performs two 32,768-wide 65-core TopKs, restores chunk-relative
IDs, merges 64 candidates to 32 with device gather, then all-gathers only the
candidate values/indices and invokes the common device sampler. The focused
all-rank probe covers IDs 0, 32767, 32768, and 248063 while deliberately making
invalid padded IDs competitive; no invalid ID survives. Reduced warmed latency
improved from 12.667 ms to 3.451 ms (3.67x).

The Tracy/tt-perf-report slice records the two first-stage TopKs at 169 us each
on 65 cores, the merge TopK at 15 us, candidate gathers at 15/13 us, and device
sampling at 28 us. The two representative terminal LM-head shards are about
598/600 us. The report models 146 GB/s overall, 28.5% of the Blackhole DRAM
roofline for the reduced full-path capture. Exact CSV, human table, console
output, and provenance are under `tracy/split_sampler_reduced/`.

## Decoder-stack lower bound

Optimized multichip layer medians are 0.593794 ms for each of 16 full-attention
layers and 0.899718 ms for each of 48 linear-attention layers. Their sum is
52.6872 ms/token, or 18.980 t/s/u. Final token-out is 56.6985 ms/token; terminal
embedding/norm/LM-head/split-sampling/trace work adds 4.0113 ms, a 7.61% gap
over the layer-only lower bound. This is below the 10–15% split-and-close gate,
and the profile shows no force-argmax, full-vocab gather, generic one-core TopK,
or host boundary dominating the remaining terminal work.

## Correctness, capability, and runtime audit

The exact-revision AIME24 chat-template reference has a non-aligned 161-token
prompt and 100 reference tokens. Fresh all-64-layer results are:

- prefill: top-1 92%, top-5 100%, top-100 100%;
- traced teacher forcing: top-1 97%, top-5/top-100 100%, 5129.57 ms TTFT,
  6.98 t/s/u;
- default split-token-out autoregressive: 100 HF and 100 TT tokens, coherent
  task-relevant English, zero adjacent duplication, no degenerate loop.

The exact-revision six-prompt shared qualitative suite was refreshed on the
final default split sampler (`artifacts/full_model_qualitative_50.json`). Every
TT result is coherent, English, prompt-relevant, and non-repetitive. Three
cases match HF for all 50 tokens; the other first divergences are at tokens 1,
4, and 28 and remain semantically appropriate.

`doc/context_contract.json` retains B1 context 262144, public bounded streaming
prefill through 262144, and the physically bracketed B32 terminal-resident
limit (72192 succeeds, 72256 fails). No capability is reduced here. Existing
maximum-context and streaming artifacts remain applicable because numerical,
cache, residual, and prefill policies are unchanged; the fresh optimized
mixed-slot run revalidates non-aligned generator behavior.

The measured token-out loop issues `execute_trace(..., blocking=False)` then
the sampler trace. Its counters show no token, position, or page-table refresh.
Readbacks in the performance JSON are setup/probe/final reporting only and are
outside the measured replay interval. Static and dynamic fallback evidence is
in `artifacts/runtime_fallback_audit.json`; the measured path has no CPU op,
eager fallback, logits readback, or host sampling boundary.

## Commands and artifacts

Exact commands, failed experiments, Watcher/profiler separation, hardware
health checks, and artifact provenance are in `work_log.md`. Primary evidence:

- `artifacts/baseline_full_force_argmax_no_readback_b1_s128_g128.json`
- `artifacts/final_default_split_no_readback_b1_s128_g128.json`
- `artifacts/candidate_split_two_stage_reduced.json`
- `artifacts/mixed_slots_split_watcher.json`
- `artifacts/degenerate_output.json`
- `artifacts/full_model_qualitative_50.json`
- `autoregressive_final/`
- `logs/run_prefill_check_final.log`
- `logs/run_teacher_forcing_final.log`
- `tracy/split_sampler_reduced/decode_ops.csv`
- `tracy/split_sampler_reduced/decode_perf_report.csv`
- `tracy/split_sampler_reduced/decode_perf_report.txt`

No vLLM integration or datatype-sweep work was performed in this stage.
