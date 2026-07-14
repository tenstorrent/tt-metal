# Gemma 4 31B Full-Model Work Log

## 2026-07-14

Selected skills: `$full-model`, `$tt-device-usage`, `$multichip`, `$optimize`, `$qualitative-check`, `$tt-enable-tracing`, `$autofix`, and `$stage-review`. Hardware commands were serialized; watcher and profiler were never mixed.

### Implementation

- Added `tt/model.py`: selective safetensor loading, TP4 BF16 embedding, exact per-kind RoPE, all 60 production `MultichipDecoder` layers, final RMSNorm, tied-value vocab-sharded LM head, softcap, BFP8 paged-cache ownership, arbitrary logical prefill lengths, and explicit low-level state.
- Added `tt/generator.py`: readiness contract, standalone cache ownership, mixed prompt lengths, fixed/inactive slots, explicit host-sampling compatibility, two-trace split sampling, device token feedback, changed-only page-table refresh, and request reset.
- Added `Gemma4GreedyTP4Sampler` custom TT Metal local-winner and pair-reducer kernels. Greedy stays entirely on device; non-greedy keeps the common sampler.
- Corrected common sampling uint32 boundary handling at local vocabulary width 65,536 and FP32 candidate-value broadcast validation for the retained non-greedy path.
- Added static, focused hardware, watcher, performance, and qualitative tests.

### Context and preserved policy

The model retains TP4 Linear/two-link async CCL, attention BFP8/LoFi, MLP BFP4/LoFi, packed M=1 output BFP8, replicated BF16 DRAM residuals, BFP8 cache storage with BF16 decode updates and BFP8 prefill fills, 50 physical-1,024 sliding caches, and 10 physical-262,144 full caches.

Capacity was recomputed from the full wrapper rather than copied from the decoder stage:

```text
weights/device                 10,908,115,456
KV cache/device                 2,789,212,160
weights + KV                   13,697,327,616
all accounted DRAM             27,672,814,984
usable DRAM                    34,225,520,640
margin                          6,552,705,656
supported context                     262,144
```

No context capability reduction was required.

### Runtime-autofix findings

- Replaced a per-physical-device page-table copy loop with one distributed `ttnn.copy`; live `tt-triage` had shown device-divergent copy op IDs during the hang.
- Enforced external KV/page-table co-ownership, cache allocation identity, explicit mutable table generations, and changed-only table copies.
- Made eager decode reject the false `enable_trace=True` path and retained only the explicit traced APIs.
- Added context-end guards before device work and every replay.
- Derived non-default batches correctly; inactive rows now use cache position `-1` and a masked RoPE-position increment.
- Released both traces before reset, prefill, cache clearing, or sampler state changes.
- Fixed first-token sampling when logical batch one uses a physical batch-two trace slot.
- Fixed custom reducer 4-byte DRAM write misalignment, 8-byte/16-byte pair-page stride mismatch, and batch-one output sizing found by watcher.
- Fixed short prompts (`<=32`) where a full-range TTNN slice aliased `hidden`; deallocating the source invalidated the tensor handed to final RMSNorm. The source tensor is now transferred directly for the full-range case. Two sequential short prompts pass normally and under watcher.
- Generalized sampler-ready device-logit prefill to batch rows with independent user/cache ownership; mixed lengths 33 and 17 pass together at batch two without full-logit readback.
- Preallocated the custom sampler gathered-pair output, bound regular all-gather to that output, and added explicit sampler teardown.
- Reran the complete reduced path with source-current watcher: mixed 33/17 prefill, exact greedy sampling, split traces, changed/unchanged page tables, reset, and teardown passed (`doc/full_model_reduced_final_watcher.xml`).

### Sampler comparison

The common paths were tested before the custom implementation:

1. `Sampling1D` exact local width 65,536: TopK approximately 10.625 ms versus approximately 5.15 ms for the reduced model trace; rejected because sampling dominated.
2. Partitioned common TopK: synthetic boundaries passed, but real equal BF16 maxima at global tokens 177 and 192 returned 192; rejected as non-semantic greedy.
3. Force argmax: failed exact sharded boundaries; rejected.
4. `SamplingGenerator`: batch-32 rounding plus incompatible mutable/internal trace ownership; rejected.
5. Native 32/64-width TILE gather, BF16 broadcast, and row-major composite: writer asserts, corrupted candidate values, and replay hang respectively; rejected and reverted.

The custom greedy path returns exact boundary IDs `[0,32767,32768,65535,65536,262143]`, resolves the 177/192 tie to 177, supports batch two, and remains exact across three trace replays. The final post-fix profile flushes setup traffic before four selected replays. `tt-perf-report` gives 299.464 µs median local winner and 0.4205 µs reducer; sampling is 9.68% of reduced device time and does not dominate. Profiled reduced steady decode is 287.02 t/s/u; the LM head is the dominant 56.25%.

Full-stack lower bound:

```text
50 * 0.463813 ms + 10 * 0.5166275 ms = 28.356925 ms/token = 35.2647 t/s/u
measured steady token-out = 29.52798 ms/token = 33.86618 t/s/u
full-model overhead = 1.17106 ms/token = 4.13%; stack-ceiling attainment = 96.04%
```

### Correctness and qualitative evidence

Fresh reference:

- exact revision `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`;
- tokenizer `GemmaTokenizer`, `chat_template=None`, exact plain-tokenizer completion mode;
- AIME24 prompt `[1,149]`, continuation `[1,100]`, top-k `[100,100]`.

Full-stack results:

```text
run_prefill_check:    top1 91/100, top5 100/100, top100 100/100
run_teacher_forcing:  top1 91/100, top5 100/100, top100 100/100
teacher forcing:      TTFT 1793.14 ms, decode 22.79 t/s/u, e2e 16.30 t/s/u
token-out:            TTFT 693.70 ms, decode 24.97 t/s/u, steady 33.87 t/s/u
```

The common 100-token autoregressive prompt produced coherent HF and TT story continuations. They match at the first generated token, diverge at token two, and match 8/100 positions overall; TT has no adjacent repetition, wrong-language drift, or early incoherence.

The six-prompt qualitative run passed the mechanical degeneracy checker. TT is coherent on haiku, story, and Fibonacci prompts. Repeated question-list output for supervised learning and thermodynamics is identical to HF. Both paths show base-checkpoint prompt-corpus autocomplete rather than French translation. Full verdict: `qualitative/verdict.md`.

### Trace evidence

The final 100-token token-out artifact records:

```text
model trace replays        99
token host refreshes        0
full-logit readbacks        0
position host refreshes     2
RoPE host refreshes         2
page-table refreshes        0
synchronizations            3
sampled-token readbacks      1 (final result only)
```

Focused reduced testing also changed a page table once, observed one distributed copy, then repeated the same identity/generation with no copy. The sampler output and next model input are the same persistent tensor.

### Commands

All device commands set `LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH`.

```bash
pytest -q models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py

GEMMA4_31B_FULL_MODEL_RUN_SAMPLER_BOUNDARY=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_tp_vocab_row_materialization_and_sampler_boundaries

GEMMA4_31B_FULL_MODEL_RUN_REDUCED=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_full_model_prefill_split_greedy_and_trace

GEMMA4_31B_FULL_MODEL_RUN_SHORT_PREFILL=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_short_prompt_repeated_generate

TT_METAL_WATCHER=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
GEMMA4_31B_FULL_MODEL_RUN_SHORT_PREFILL=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_short_prompt_repeated_generate

python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/google_gemma_4_31b \
  --mesh-device P150_X4 --fabric-config FABRIC_1D --max-new-tokens 100

python models/autoports/google_gemma_4_31b/tests/run_full_model_qualitative.py
```

### Artifact index

- `run_prefill_check.log`, `run_teacher_forcing.log`: full-stack accuracy and teacher-forcing performance.
- `readiness_aime24_plain.refpt` and `.metadata.json`: fresh exact-revision reference.
- `autoregressive/`: 100-token HF/TT outputs and token metadata.
- `qualitative/`: prompt format, six HF/TT pairs, mechanical checker, and verdict.
- `token_out_no_readback.json`: full 60-layer performance and trace counters.
- `reduced_token_out_custom_greedy_perf.json`, `sampler_profile_summary.md`: sampler performance and rejection ledger.
- `reduced_token_out_final_perf.json`, `perf/final_filtered.csv`, `perf/final_summary.csv`, `perf/final_report.md`: source-current post-fix Tracy/`tt-perf-report` evidence.
- `triage/` and repo-root `AUTOTRIAGE.md`: hang evidence.
- `doc/full_model_*.xml`: focused source-current and watcher reports.

No vLLM implementation or registration work was performed.

### Review and commits

The first independent `$stage-review` returned `more-work-needed`. Its findings were fixed: mixed-prompt device-logit prefill now preserves per-row state; profiler evidence was recollected from the source-current path; the Stage 05 decoder-stack lower bound and physical batch envelope were added; trace-warning documentation was reconciled; and the short-prompt tensor-ownership defect was repaired.

A fresh independent rereview returned `clean-pass` with no required work and no blocking hard-check gaps. The implementation checkpoint SHA will be appended after the stage-owned commit; no push will be performed.
