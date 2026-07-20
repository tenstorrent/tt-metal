# Falcon3-7B-Base full model

This stage assembles the optimized TP4 decoder into the complete Hugging Face
autoregressive path: hidden-sharded embedding, 28 decoder layers, final RMS
normalization, TP-vocabulary LM head, paged KV ownership, and a Metal-readiness
generator. It adds no vLLM code.

## Selected implementation

- Target: four Blackhole P300c devices as a `1x4` `FABRIC_1D_RING`, TP degree
  four, two links.
- The 28-layer stack reuses the optimized decoder's BFP4/LoFi projection
  weights, 4/4/24/8 QKV/O/gate-up/down core policy, BFP8 attention and MLP
  activations, BFP8 persistent asynchronous CCL, rank-local BFP8 paged KV,
  and two CCL buffers shared by every layer.
- The inter-layer residual stays replicated across the mesh and BF16 TILE,
  width-sharded over a 32-core L1 grid. There is no inter-layer gather,
  reshard, or host boundary.
- The embedding is hidden-sharded and gathered once into that residual
  contract. The final norm remains on device. The untied LM head is split
  into four 32,768-column device shards and evaluated as four 8,192-column
  local matmuls per rank using BFP4/LoFi.
- The public generator accepts non-aligned prompt lengths. It owns 128-token
  kernel padding, prompt masking, 2,048-token page-aligned chunking, chunk
  page tables, cache fill, logical positions, final-row selection, and output
  slicing.
- Low-level prefill/decode accepts explicit KV cache, fixed page table,
  prompt lengths, logical positions, active batch, fixed slots, and inactive
  rows. A 32-slot run with mixed 33/47-token prompts kept all 30 inactive
  cache positions at `-1`; a separate full 28-layer gate then activated every
  slot with non-aligned lengths 33 through 64 and reversed the slot mapping.

## Correctness

The fresh reference is
[`results/aime24_plain_100.refpt`](results/aime24_plain_100.refpt), SHA256
`45637a3d5f7c41146d267fe6eb0678d48e588fb9d3a6d7d3a2ca32838de720fa`.
It contains one 155-token AIME24 prompt, 100 HF generated tokens, and HF
top-100 tokens at every step.

| Official runner | Top-1 | Top-5 | Top-100 | Gate |
| --- | ---: | ---: | ---: | --- |
| `run_prefill_check` | 92/100 | 100/100 | 100/100 | pass |
| `run_teacher_forcing` decode | 93/100 | 100/100 | 100/100 | pass |

The required top-5 bar is 98%; both paths reach 100%. Top-100 is exactly
100% on both paths. Accuracy evidence is summarized in
[`results/accuracy_summary.json`](results/accuracy_summary.json). Fresh raw
canonical-runner logs are
[`run_prefill_check.log`](results/run_prefill_check.log),
[`run_teacher_forcing.log`](results/run_teacher_forcing.log), and
[`run_autoregressive.log`](results/run_autoregressive.log).
Frozen source, runner, artifact, and raw-log hashes are in
[`results/final_evidence_manifest.json`](results/final_evidence_manifest.json).

Falcon3-7B-Base's exact tokenizer metadata has `chat_template = null`, and
`apply_chat_template` raises rather than selecting a template. A fabricated
chat wrapper would change the base-model prompt contract. The fresh AIME24
reference therefore uses the HF tokenizer's plain completion form; the exact
exception and revision are in
[`results/tokenizer_reference_metadata.json`](results/tokenizer_reference_metadata.json).

## Autoregressive qualitative result

`run_autoregressive` generated 100 tokens from both HF and TT. The sequences
match for the first five generated tokens and first diverge at index five.
Both remain coherent, grammatical, relevant English story continuations.
The TT output has no repetitive loop, wrong-language drift, malformed special
token, or premature EOS. The verdict and both texts were read and recorded in
[`results/autoregressive/qualitative_verdict.md`](results/autoregressive/qualitative_verdict.md).

The independent fresh AIME24 HF reference is coherent step-by-step math in
English and is truncated only by the requested 100-token cap, not by drift or
repetition; its decoded text is
[`results/aime24_hf_completion.txt`](results/aime24_hf_completion.txt).

The six-prompt suite retains one weaker greedy result: the machine-learning
haiku repeats a fluent 18-token stanza. A focused 100-token control produced
the exact same sequence in host-eager greedy, traced free-running greedy,
traced greedy after allocator-safe release/recapture, and traced
teacher-forcing on the common prefix. Positions finish at 107, feedback is
direct, and free-running decode performs zero token H2D copies. This rules out
split sampling, trace feedback, reset, page-table, position, and cache-state
handling as causes. A supplemental seeded on-device top-k 8/top-p 0.9/
temperature 0.7 completion is coherent and non-repetitive; it demonstrates the
supported sampled mode without replacing the required greedy artifact. See
[`results/full_model_qualitative_control.json`](results/full_model_qualitative_control.json)
and [`qualitative_verdict.md`](qualitative_verdict.md).

## Sampling and trace contract

Both common paths were audited before selection. `Sampling1D` fits the TP4
local-vocabulary contract and accepts a caller-owned output token buffer.
Actual `TTSampling` exact greedy was also run: it is semantically correct but
first all-gathers the complete vocabulary, and its broader mutable request
state disables the abstraction's internal trace ownership. No model-local
custom sampler was created.

The selected common `Sampling1D` path computes each rank's BF16 local maximum
and local index, packs value plus global index into a tiny FP32 candidate,
uses the common TP4 Linear fallback to gather four candidates, and selects the
winner on device. Against the common
full-vocabulary force-argmax control and an actual `TTSampling` call, all three
produced token `2107`. Split/force/`TTSampling` measured 0.806/1.009/1.009 ms
per call. The selected split sampler is semantically greedy, not a top-k
approximation.

Decode uses two traces: the model trace consumes persistent token, position,
rotary-position, page-table, and cache state; the sampling trace writes its
token directly into the model trace's input. One capture and four replays
generated `[2107, 2027, 4422, 6247, 2303]`; the token input matched the last
sample and current/rotary positions advanced from 128 to 132. An unchanged
page table caused zero host copies, while each intentional changed or expanded
table caused one copy. Replaying after a page remap changed logits by 1.25,
repeating the changed table was exact, and restoring the original mapping
restored logits exactly. Another 128 trace pairs advanced both positions to
256. The normal feedback edge contains no host argmax, full-logit
readback, token H2D copy, per-token page-table rebuild, or untraced sampling.
Python launches the trace pair and may observe returned tokens, but never
feeds those observed values back into decode. Final five-token evidence
records zero token host copies and four device-to-device token restores during
warm-up/capture.

The caller-visible token-out loop reaches **75.411 t/s/u** at batch one on the
standard 128-token prompt/128-token output workload. This includes the token
read that a caller observes, while feedback itself stays device-to-device.
The separately labeled queued device-only trace pair reaches 75.687 t/s/u.
Queued teacher-forcing tracing reaches 75.676 t/s/u; the official readiness
callback, which includes its compatibility overhead, reaches 40.850 t/s/u.
The model trace is 12.416 ms/token and sampler trace 0.794 ms/token, so
sampling is 6.01% of the traced pair and does not dominate. Cold/warm TTFT on
the standard workload is 78.073/25.083 ms. Full measurements are in
[`results/full_model_evidence.json`](results/full_model_evidence.json).

### Performance attribution

The reduced real-weight Tracy run executes the real embedding, decoder layer
0, final norm, untied TP LM head, and split sampler. It measures 4.637 ms
prefill and 1.442 ms/token for three decode trace replays. Across those three
replays, the largest summed device categories are width-sharded matmul
(1,053.72 us), exact-greedy L1 argmax (984.28 us), and local-max reduction
(897.37 us). This reduced graph intentionally magnifies terminal/sampling
cost; in the full 28-layer path, the sampler remains only 0.794 of 13.210 ms.

An independent real-weight depth sweep at 1/2/4/8/16/28 layers and
`max_context=256` fits model-trace latency to `0.2697 + 0.285363 * layers` ms
with R-squared 0.999998. The slope agrees with the inherited isolated selected
decoder's 0.286597 ms, demonstrating clean linear layer scaling. At depth 28,
that short-context harness measures 8.262 ms for the model trace, 0.794 ms for
sampling, and 109.753 caller-visible t/s/u. The standard full-context-shaped
harness measures 12.416 ms for the model trace. Because its 32K cache/page-
table allocation differs from the sweep's 256-token contract, this absolute
difference is recorded without assigning it to a specific op.

For scale reconciliation, an explicitly analytical roofline uses the inherited
decoder report's 292 GB/s modeled per-device bandwidth. The physical BFP4
matrix payload is 989,331,456 bytes/device/token, a 3.388-ms lower bound. The
average BFP8 KV payload over positions 128--255 adds 0.0109 ms across 28
layers; adding the measured 0.794-ms sampler gives a 4.193-ms composite lower
bound versus the measured 13.261-ms caller-visible token. This bound omits
activation traffic, CCL, dependency synchronization, and terminal work and is
not claimed as attainable throughput. The operation audit found no extra host
boundary, inter-layer reshard, or redundant token-out logits gather. Raw
Tracy data and reports are under [`profile/reduced`](profile/reduced); the
complete measured/analytical separation is in
[`results/perf_summary.json`](results/perf_summary.json), with depth data in
[`results/full_model_depth_sweep.json`](results/full_model_depth_sweep.json).

## Context and serving evidence

The advertised batch-1 context remains 32,768 tokens. With all 28 full-model
weights and runtime state resident, the fresh allocation measured
2,231,753,216 bytes/device. Allocating all 28 rank-local full-context K/V
caches adds exactly 499,122,176 bytes/device, for 2,730,875,392 allocated and
27,351,855,616 free bytes/device after the configured trace-region
reservation. Executing the maximum-context path peaks at 2,734,023,168
allocated and leaves 27,348,707,840 free bytes/device. The full capacity
derivation is in [`../context_contract.json`](../context_contract.json).

The public boundary was exercised through all 28 layers with a 32,767-token
non-aligned prompt: it padded to 32,768, filled sixteen 2,048-token chunks,
mapped all 1,024 pages, sampled two tokens, and replayed traced decode at
position 32,767. The final K/V page was nonzero on all four ranks in every
layer (224 cache tensors checked), and current/rotary positions both advanced
to 32,768.
The mixed-prompt lifecycle gate exercised 32 fixed slots, two active rows,
30 inactive rows, trace positions, allocator-safe trace release before
in-place cache reset, persistent input-buffer identity, explicit host-sampling
compatibility, and
2,049/2,079-token mixed prompts across the 2,048-token chunk boundary. A full
28-layer all-active factorial gate separately used prompt lengths 33--64 and
compared exact repeat, reversed slots with logical page rows preserved, same
slots with disjoint remapped pages, and reversed slots plus remapped pages. It
also checked active-16 versus active-32 and representative batch-1 controls.
Every run's split token equals that run's full-logit host argmax; positions,
logits, tokens, all rounded SDPA cache pages, and the prompt-final/decode-update
K/V rows are bit-exact after permutation/remapping. AutoFix found that dynamic
SDPA rounds a three-page sequence to a four-page read window before causal
masking. Mapping only the three logically live pages left the masked tail page
invalid at position 64. The generator now maps and validates the complete
rounded window, including range, alias, and cross-slot ownership checks. No
precision or execution fallback is used. All ran with
`throw_exception_on_fallback=true`; see
[`results/full_context_coverage.json`](results/full_context_coverage.json) and
[`results/full_model_contract_coverage.json`](results/full_model_contract_coverage.json),
plus [`results/full_model_batch32.json`](results/full_model_batch32.json).
The latter also traces seeded top-k/top-p/temperature sampling entirely on
device, records zero token host copies with direct feedback, checkpoints RNG
around warm/capture, and reproduces identical tokens when the same seed reuses
the captured trace.

## Runtime fallback audit

- Model path: all embedding, decoder, final norm, LM-head, TP communication,
  and optimized sampling operations are TTNN device operations on the 1x4
  mesh. There is no single-chip, replicated-model, or host-model fallback.
- Cache ownership: low-level calls accept caller caches and fixed page tables;
  the convenience generator owns an equivalent persistent paged cache. K/V is
  never reconstructed on the host.
- Logit boundary: full logits are gathered only in explicit host-sampling or
  readiness accuracy modes. The measured token-out path never gathers or
  reads full logits.
- Sampling: optimized greedy and stochastic sampling execute on device. The
  explicit `sampling_mode="host"` compatibility path first releases live
  address-bound traces and then returns host logits for tests that require it.
- Reset: address-bound traces are synchronized and released before cache or
  persistent-input fills, preventing allocation while a live trace exists.
  Reset then clears the same cache/input buffers in place and preserves their
  addresses. The next optimized request performs exactly one safe recapture.
  Host-sampling compatibility, cache identity/shape changes, and teardown use
  the same release-before-rebind rule.
- Page tables: stable tables stay device-resident; an actual content change is
  copied once. Every active row must map the full rounded SDPA read window to
  in-range, disjoint physical pages; logical-only mappings are rejected before
  device execution. Shape, cache identity, or active-slot changes release and
  recapture traces rather than silently rebuilding state per token.

The detailed sampler decision and rejected alternatives are in
[`sampler_contract_audit.md`](sampler_contract_audit.md) and
[`rejection_ledger.md`](rejection_ledger.md).

## Watcher safety gate

A full 28-layer run passed with TENSIX Watcher enabled and Ethernet Watcher
disabled because this firmware's active Ethernet image exceeds the configured
Watcher kernel buffer. The run covers cold/warm prefill, selected split greedy,
both full-vocabulary sampler controls, model/sampling trace capture and 128
replays, changed/unchanged/restored page tables, position coherence, direct
token feedback, reset, and teardown. It is safety evidence, not performance
evidence. The initial reduced run exposed two generic async all-gather writer defects:
a one-tile packet incorrectly initialized a scatter header, and an outward
Linear endpoint requested a nonexistent fabric direction before checking that
the direction had targets. Both kernel guards were fixed. The first
complete-stack run then exposed two trace-only issues: Watcher treated the
valid firmware `RUN_MSG_REPLAY_TRACE` state (`0xf0`) as corruption, and the
traced minimal all-gather writer did not reset its per-invocation packet-header
pool. Watcher now validates primary run messages separately from subordinate
sync messages, and the writer calls `PacketHeaderPool::reset()` on every
invocation. A 600-replay CCL-free Watcher control, the exact reduced sampler
regression, and the complete 28-layer/128-replay stack all pass without
asserts, NoC errors, or corruption. Artifacts are
[`results/full_model_sampler_watcher.json`](results/full_model_sampler_watcher.json)
and [`results/full_model_watcher.json`](results/full_model_watcher.json).

## Reproduction commands

Run each hardware command serially on a healthy four-device QuietBox 2:

```bash
python -m models.common.readiness_check.generate \
  --hf-model /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 \
  --prompt-source aime24 --gen-len 100 --top-k 100 \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt

export TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'

python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --hf-model /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/autoregressive

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_evidence.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_evidence.json

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_context_coverage.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_context_coverage.json

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_contract_coverage.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_contract_coverage.json

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_batch32.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_batch32.json

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_qualitative.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --hf-model /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/qualitative_suite \
  --weight-cache-path /tmp/falcon3-full-model-cache

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_qualitative_control.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --model-path /home/mvasiljevic/hf-cache/hub/models--tiiuae--Falcon3-7B-Base/snapshots/bf3d7ed586cb22a921520e2d681a9d3d7642cde8 \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_qualitative_control.json \
  --max-new-tokens 100 \
  --weight-cache-path /tmp/falcon3-full-model-cache

TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_depth_sweep.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt \
  --depths 1,2,4,8,16,28 --iterations 128 \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_depth_sweep.json \
  --weight-cache-path /tmp/falcon3-full-model-cache

TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_sampler_watcher.py \
  --hidden-rows 128 --hidden-workers 2 \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_sampler_watcher.json

TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_evidence.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_watcher.json \
  --weight-cache-path /tmp/falcon3-full-model-cache
```

The reduced profile was collected separately from Watcher:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python -m tracy -p -r -v \
  -o models/autoports/tiiuae_falcon3_7b_base/doc/full_model/profile/reduced \
  models/autoports/tiiuae_falcon3_7b_base/tests/full_model_profile.py \
  --model-dir models/autoports/tiiuae_falcon3_7b_base \
  --reference models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/aime24_plain_100.refpt \
  --output models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/full_model_profile.json \
  --weight-cache-path /tmp/falcon3-full-model-cache

tt-perf-report \
  models/autoports/tiiuae_falcon3_7b_base/doc/full_model/profile/reduced/reports/2026_07_20_02_47_40/ops_perf_results_2026_07_20_02_47_40.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-color \
  --csv models/autoports/tiiuae_falcon3_7b_base/doc/full_model/profile/reduced/prefill_perf_report.csv \
  --summary-file models/autoports/tiiuae_falcon3_7b_base/doc/full_model/profile/reduced/prefill_summary

tt-perf-report \
  models/autoports/tiiuae_falcon3_7b_base/doc/full_model/profile/reduced/reports/2026_07_20_02_47_40/ops_perf_results_2026_07_20_02_47_40.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-color \
  --csv models/autoports/tiiuae_falcon3_7b_base/doc/full_model/profile/reduced/decode_perf_report.csv \
  --summary-file models/autoports/tiiuae_falcon3_7b_base/doc/full_model/profile/reduced/decode_summary
```

## Limitations

- Batch one is the advertised full-context capability. Batch 32 is validated
  for mixed short prompts and fixed-slot decode, not for 32 simultaneous
  32,768-token prompts.
- The exact base-model tokenizer has no chat template, so the reference uses
  its native completion prompt. No instruction/chat behavior is claimed.
- The TT haiku in the six-prompt suite repeats one fluent stanza; a focused
  control proves 100/100-token equality among host eager greedy, traced greedy,
  safe-recapture traced greedy, and teacher-forced traced prediction. Thus the
  repetition is the inherited low-precision greedy model result, not trace,
  feedback, reset, position, or cache corruption. A seeded on-device top-k 8,
  top-p 0.9, temperature 0.7 run is coherent and non-repetitive. The other
  five suite outputs remain coherent, and Fibonacci matches HF for all 100
  tokens. The greedy limitation is recorded, not suppressed.
- Nanobind reports known reference-leak diagnostics at interpreter shutdown;
  device close and all functional gates complete successfully before those
  binding diagnostics.
