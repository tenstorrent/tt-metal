# Gemma-4 26B A4B optimized full model

| P300C 1x4, batch 1 | TTFT | Decode |
| --- | ---: | ---: |
| Inherited full-model: host-visible 128 prompt / 128 generated | 320.841 ms | 23.7629 t/s/u |
| Current reproduced: host-visible 128 prompt / 128 generated | **280.598 ms** | **26.2128 t/s/u** |
| Current reproduced: no-readback token-out, 128 warmed replays | n/a | **28.0151 t/s/u** (35.6950 ms/token) |
| Traced teacher forcing, AIME24 100-token window | 417.800 ms | **25.43 t/s/u** |

The host-visible rows are the requested inherited/current comparison, but the
10.31% decode difference is not claimed as an optimization delta: retained
model/generator candidates were reverted, and cache/JIT/runtime state differs
between sessions. The authoritative optimized result is the current reproduced
number. The serving-loop number measures all 30 layers, final norm, tied
vocabulary-sharded LM head, split Sampling1D greedy sampling, `tt_out_tok`
feedback, and device-side current/RoPE advance. It uses persistent token,
position, cache, and page-table tensors; five warmups precede 128 nonblocking
replays, with zero token readbacks and one synchronization after the window.

The optimized decoder-stack lower bound is 32.3291 ms/token: 25 sliding layers
at 1.070437 ms plus five full-attention layers at 1.113634 ms. Complete token-out
decode is 35.6950 ms/token, 10.41% above that lower bound, leaving 3.3659 ms for
final norm, LM head, split sampler, device position advance, and orchestration.
This is inside the 10–15% closure band.

## Selected full-path policy

- Mesh: 1x4 P300C Blackhole, TP4, `FABRIC_1D_RING`, two CCL links.
- Decoder policy is unchanged from optimized multichip: BF16 replicated
  inter-layer residuals, selected attention/MLP/expert weight formats and
  fidelities, BF16 KV cache and CCL payload, active top-8 sparse experts, and
  persistent async full-attention reductions.
- Embeddings are hidden-sharded then gathered once. The tied BF16 LM head stays
  split into four 65,536-token shards for sampling.
- Greedy stays on device. The shape-faithful sampler comparison selected
  force-argmax (2.3434 ms) over semantic split top-k=1 (10.7359 ms), with
  identical tokens. A scoped full-path profile reports argmax at 1.393 ms,
  vocabulary all-gather at 0.790 ms, and generic top-k at only 0.049 ms; none
  dominates full 30-layer token-out decode.
- Sampled top-k/top-p/temperature/seed traces remain supported and feed their
  result directly into the persistent model token tensor. Sampling-mode changes
  safely release and recapture the active trace set.
- Explicit cache, page-table, prompt-length, position, active-mask, and fixed
  slot state is preserved. Mixed non-aligned 33/47-token prompts, inactive rows,
  changed-only page tables, batch 32, and non-aligned public context support pass.

## Correctness and validation

The refreshed AIME24 chat-template gates pass:

| Gate | Top-1 | Top-5 | Top-100 |
| --- | ---: | ---: | ---: |
| Prefill | 96% | 100% | 100% |
| Traced teacher forcing | 98% | 100% | 100% |

The repository teacher-forcing runner is the shifted-left autoregressive rank
gate: it scores each traced greedy prediction against HF top-100, then feeds the
HF ground-truth token into the next decode step. Its 100% top-5/top-100 result
satisfies the requested rank gate. The separate refreshed 100-token free-running
comparison uses the checkpoint chat template; TT and HF both produce coherent
step-by-step equation setup, and the machine degeneracy audit is clean. All six
shared qualitative prompts were also rerun with rendered IDs and HF controls;
their combined degeneracy audit is clean. The
fallback-raising all-30-layer batch-32 probe and reduced mixed-prompt probe pass.
The public 262,111-token non-aligned full-stack evidence remains valid, so
`doc/context_contract.json` retains the advertised 262,144-token capability.

## Profiler and runtime audit

`profile/full_model_reduced_decode.csv` and
`profile/full_model_reduced_summary.csv.csv` are the signpost-scoped
`tt-perf-report` outputs for one reduced full-path replay (sliding layer, full
layer, terminal head, sampler). The merged device-op sum is 5.585 ms and modeled
DRAM roofline is 15.1%. Matmuls account for 1.236 ms, async all-gather 0.790 ms,
norms 0.771 ms, sparse matmul 0.339 ms, and argmax 1.393 ms. The compact CSV,
text report, PNG, command log, and provenance are retained; the 1.1 GiB temporary
raw Tracy capture was removed after extraction.

Fallback exceptions were enabled on the measured/probe paths. There is no host
logits readback, host argmax, Python token feedback, per-token page-table rebuild,
or per-token synchronization in token-out decode. A watcher-only audit exposed
and fixed two shared direct-fabric all-gather contracts: invalid endpoint
connection acquisition and one-chunk scatter-header initialization. The original
greedy-plus-sampled watcher repro now passes.

See `work_log.md`, `performance.json`, `runtime_fallback_audit.json`,
`qualitative_verdict.json`, `shared_qualitative_suite.json`,
`autoregressive_accuracy_contract.json`, `AUTOTRIAGE.md`, and `AUTOFIX.md` for commands and
artifact provenance.
