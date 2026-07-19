# Full-model readiness report

Status: complete for the repo-local TTNN full-model stage; vLLM integration is
intentionally out of scope.

## Implemented contract

- `tt/model.py` owns sharded embedding, all 32 optimized TP4 decoder layers,
  final RMS norm, split TP4 LM head, invalid-vocabulary mask, paged cache
  allocation/reset, and the selected common Sampling1D implementation.
- `tt/generator.py` owns non-aligned prompt padding, 128-token prefill padding,
  2048-token internal chunking, masking through logical prompt lengths, page
  allocation, chunk page tables, cache fill, positions, output slicing, mixed
  prompts, inactive rows, fixed sampling slots, and request reset.
- Low-level model prefill/decode accept explicit cache, page table, positions,
  prompt lengths, and batch state. The optimized high-level token-out path
  retains token feedback and position increments on device.
- The default cache is paged BFP8; explicit BF16 cache allocation is supported.
  The public logical context remains the HF value of 131,072 tokens.
- Host sampling exists only as the explicit `sampling_mode="host"`
  compatibility path. It releases live traces before eager allocations;
  normal reset instead zeros cache in place and preserves compatible traces.

## Correctness and qualitative evidence

The fresh AIME24 reference uses the exact local HF snapshot and tokenizer chat
template. It contains one 184-token prompt, 100 continuation tokens, and top-100
reference tokens. Its SHA-256 is
`3dc948d1055779df6df70623ce5025ab48318acef6af55e6fc8f72d6852f4a42`.

| Gate | Top-1 | Top-5 | Top-100 | Result |
| --- | ---: | ---: | ---: | --- |
| `run_prefill_check` | 86/100 | 100/100 | 100/100 | pass |
| `run_teacher_forcing` | 86/100 | 100/100 | 100/100 | pass |

The final 100-token HF/TT autoregressive comparison uses the same chat-rendered
AIME24 prompt. HF and TT agree through the first six tokens and first diverge at
token index 6. The TT output is a coherent English mathematical derivation with
no wrong-language drift, adjacent duplication, phrase loop, malformed special
tokens, or premature EOS. Both sides stop mid-derivation only because of the
100-token cap. See `artifacts/autoregressive/qualitative_verdict.md` and
`degenerate_report.json`.

The required shared six-prompt suite also uses the exact chat template and
100-token greedy cap. TT produces a valid haiku, supervised-learning
explanation, story continuation, thermodynamics explanation, French
translation, and Python Fibonacci implementation. All six were read manually;
all are coherent and task-relevant, with no repetition or wrong-language
drift. The machine checker reports zero adjacent duplication and no findings.
Prompt 0 repeats bit-identically across `reset()` with the same trace IDs and
zero unchanged-page-table copies. See
`artifacts/qualitative_suite/qualitative_verdict.md` and
`qualitative_prompt_format.json`.

## Performance

All measurements use exact weights, P300 1x4 `FABRIC_1D_RING`, TP=4, batch 1,
two CCL links, a 100 MB/bank trace region, and fallback exceptions enabled.

| Measurement | Workload | Result |
| --- | --- | ---: |
| Public warmed TTFT | prompt 128 | 20.51 ms |
| Public token-out | prompt 128, generate 128 | 110.45 t/s/u |
| Device-feedback model + split sampler trace | 100 replays | 111.17 t/s/u |
| Teacher forcing | 100 forced decode tokens | 101.50 t/s/u |
| Model trace alone | 100 replays | 8.199 ms/token |
| Sampling trace alone | 100 replays | 0.795 ms/token |
| Sampling fraction | separate trace times | 8.84% |

The accepted 32-layer decoder lower bound is 7.894 ms/token. The full model is
only 3.85% above it. A performance autofix removed two full-context RoPE-table
untilizes per layer by keeping separate shared TILE prefill and row-major decode
tables. The final bounded Tracy decode report contains no full-context RoPE
untilize; its remaining untilize is the small INT32 sampler boundary.

The fresh current-path `tt-perf-report` under `tracy_exact_greedy/` is bounded
by the decode signposts and contains the delivered local reduce/argmax, packed
candidate gather, and device winner-selection ops; it contains no 131,072-row
untilize. Because that diagnostic intentionally uses one decoder layer, local
reduce plus argmax account for 39.4% of its reduced op time. That artificial
ratio is not the production bottleneck result: the exact 32-layer isolation
measures the complete sampling trace at 0.795 ms versus 8.199 ms for the model
trace, or 8.84% of token-out trace time.

The teacher-forcing runner reports a 51.5 s cold TTFT because its single timed
prefill includes first-program construction. This is kept separate from the
20.51 ms warmed primary TTFT and from token-out decode.

## Sampling decision and trace proof

Both common sampler implementations were evaluated against the exact P300 1x4
TP4 contract before token-out selection. Sampling1D was selected because it
directly supports the required fixed batch-32, padded-vocabulary, caller-owned
trace state and explicit two-link split gathers. SamplingGenerator/TTSampling
was rejected by source contract rather than benchmarked as an equivalent path:
with `max_top_k=32` it derives a one-link split gather, and explicit request
seeds require per-token H2D seed updates while disabling its internal trace.
Benchmarking that path would compare a less-optimized or untraced
configuration, which this stage forbids. See `sampler_contract_audit.md`.

Within Sampling1D, both semantically greedy strategies were tested before
token-out measurement:

- Selected: Sampling1D exact local argmax plus one packed FP32 rank-candidate
  gather. It communicates four `(value, global-token)` packets rather than
  full logits, matches host and force-argmax on all 32 adversarial rows, and
  measures 0.974 ms eager per call in the final full-stack evidence.
- Rejected for the canonical path: Sampling1D full-vocabulary all-gather plus
  force-argmax. It returns the same token but measures 1.242 ms per call and
  adds an unnecessary full-logits collective.

Trace evidence proves four decode replays advance both current position and
RoPE position from 184 to 188. `tt_out_tok` equals the last public output and is
the next model-trace input. An unchanged page table causes zero H2D copies; a
changed table causes exactly one. The prompt-128/generate-128 measured request
reuses warmed traces and has 127 replays, one request-boundary token copy, two
position-input copies, one page-table copy, 128 output token-ID reads, zero
sampling-parameter or seed copies, and no per-token input rebuild. The full
evidence starts from a cold trace pair, proves `tt_out_tok` feedback, advances
current/RoPE positions from 184 to 284 over 100 replays, and maps the complete
replay horizon before those writes.

Stochastic `k>1` uses a real request seed for the first visible draw, then
updates the same persistent seed tensor once to `UINT32_MAX`; trace replays
advance the device RNG without per-token seed copies or recapture. A fixed
equal-probability test proves reproducibility, variation, and exactly two seed
copies per request. Greedy `k=1` performs no seed copies.

## Context capacity

Hardware measurement reports 2,534,929,408 persistent bytes/device after full
weights and 4,816,630,784 after adding the complete 131,072-token BFP8 cache.
The cache delta is 2,281,701,376 bytes, exactly matching the tiled BFP8 formula.
After the 800 MB/device trace reservation and page table, raw-allocator
headroom is 28,562,092,032 bytes/device; visible post-cache free space is
28,562,100,224 bytes/device and the largest contiguous span is 3,570,262,528
bytes/bank. The BF16 cache plan also fits with 26,548,826,112 bytes/device
raw-allocator headroom.
No advertised-context reduction is necessary. Exact formulas and the paged
capacity split are in `../context_contract.json`.

The public horizon is the prompt plus `max_new_tokens - 1` processed decode
positions: prefill itself produces the first sampled token. Hardware coverage
runs a real one-layer 131,071-token non-divisible prompt through 64 chunks,
maps all 2,048 pages, and verifies nonzero/hash-stable K/V data in physical page
2,047. A separate exact 131,072-token public `generate(..., 1)` boundary uses
all 64 chunks and exactly 2,048 pages without requesting a nonexistent decode
slot. See `full_model_contract_coverage.json`.

## Reproduction commands

All commands are run from the tt-metal checkout with
`HF_HOME=/home/mvasiljevic/hf-cache` and
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`.

```bash
python -m models.common.readiness_check.generate \
  --hf-model /home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --prompt-source aime24 --aime24-prompt-index 0 --chat-template \
  --gen-len 100 --top-k 100 \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/aime24_chat_100.refpt

python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING

python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model /home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/aime24_prompt.txt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING --max-new-tokens 100 \
  --chat-template --output-dir models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/autoregressive

python models/autoports/meta_llama_llama_3_1_8b_instruct/tests/full_model_evidence.py \
  --model-dir /home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/aime24_prompt.txt \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/full_model_evidence.json \
  --replay-iterations 100

python models/autoports/meta_llama_llama_3_1_8b_instruct/tests/full_model_perf.py \
  --model-dir /home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/aime24_chat_100.refpt \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/full_model_perf.json

pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_full_model.py \
  -m '' --disable-warnings

python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --scope autoregressive --missing-artifacts critical \
  --json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/autoregressive/degenerate_report.json

python models/autoports/meta_llama_llama_3_1_8b_instruct/tests/full_model_qualitative.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model /home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --prompts-file models/common/readiness_check/vllm_prompts.txt \
  --output-dir models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/qualitative_suite \
  --max-new-tokens 100

python models/autoports/meta_llama_llama_3_1_8b_instruct/tests/full_model_contract_coverage.py \
  --model-path /home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/full_model_contract_coverage.json

python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/qualitative_suite \
  --scope autoregressive --missing-artifacts critical \
  --json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/qualitative_suite/degenerate_report.json

python -m tracy -p -r --check-exit-code \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy_exact_greedy \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/full_model_evidence.py \
  --model-dir /home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/aime24_prompt.txt \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/artifacts/reduced_profile_capture.json \
  --replay-iterations 1 --sampler-iterations 1 --override-num-layers 1

tt-perf-report \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy_exact_greedy/reports/2026_07_19_18_40_17/ops_perf_results_2026_07_19_18_40_17.csv \
  --start-signpost 'start FULL_MODEL_REDUCED_DECODE' \
  --end-signpost 'stop FULL_MODEL_REDUCED_DECODE' --raw-op-codes --no-color \
  --csv models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy_exact_greedy/decode_perf_report.csv \
  --summary-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy_exact_greedy/decode_op_summary
```

The final canonical watcher target is
`test_exact_weight_reduced_full_model_watcher_path`; the comprehensive reduced
target additionally covers sampler equivalence, stochastic seed advance, host
compatibility, mixed prompts, inactive rows, non-aligned lengths, and chunking.

## Artifacts and limitations

- `artifacts/full_model_evidence.json`: full-stack memory, sampler, trace, and
  100-replay performance evidence.
- `artifacts/full_model_perf.json`: public prompt-128/generate-128 result.
- `artifacts/reduced_profile_evidence.json`: stable 100-replay one-layer timing.
- `artifacts/reduced_profile_capture.json` and `tracy_exact_greedy/`: fresh
  bounded exact-greedy Tracy capture, signpost-filtered `tt-perf-report`, and
  retained compact op summary.
- `tracy_reduced/`: earlier compact RoPE-topology reports retained for the
  32-layer decoder-overhead investigation.
- `logs/sampler_single_gather_fp32_oracle_fallback_guard.log`: final
  131,072-padded-vocabulary adversarial batch-32 exact-greedy two-link watcher
  oracle with fallback exceptions enabled.
- `logs/full_model_perf_fallback_guard.log` and
  `logs/full_model_evidence_fallback_guard.log`: authoritative guarded public
  performance and 32-layer trace/sampler/capacity evidence.
- `logs/test_reduced_full_model_fallback_guard.log`: final guarded 5/5
  exact-weight reduced suite, including the canonical watcher target.
- `logs/sampler_single_gather_fp32_oracle_v2.log`: historical pre-guard
  adversarial batch-32
  exact-greedy two-link watcher oracle.
- `logs/full_model_contract_valid_two_decode_pages.log`: non-aligned/mixed
  prompt, fixed/inactive slot, reset, page-boundary, and trace-reuse coverage.
- `logs/full_model_contract_real_near_context_final.log`: real 131,071-token
  one-layer cache fill plus exact 131,072-token high-level boundary.
- `artifacts/qualitative_suite/`: six rendered prompts, HF/TT completions,
  per-prompt metadata, degeneration report, and manual verdict.
- `logs/run_prefill_check_single_gather_final.log`,
  `logs/run_teacher_forcing_single_gather_final.log`, and
  `logs/run_autoregressive_single_gather_final.log`: authoritative full-stack
  correctness and qualitative runs.
- `runtime_fallback_audit.md`: runtime boundaries and reset/cache audit.

The optimized target is exactly a four-device P300 ring. Performance evidence
is batch 1, although explicit fixed-slot/mixed-prompt correctness is covered up
to the configured batch. Host sampling is compatibility-only and is not an
optimized measured path. vLLM registration/server work is intentionally left
for the next pipeline stage.
