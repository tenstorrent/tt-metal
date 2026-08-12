# Full-model work log

## Scope and provenance

- Date: 2026-07-20 UTC.
- Branch: `mvasiljevic/model/mistralai-mistral-small-24b-instruct-2501`.
- Starting commit: `90994c530de`.
- Model: `mistralai/Mistral-Small-24B-Instruct-2501`.
- Exact HF snapshot: `9527884be6e5616bdd54de542f9ae13384489724`.
- Hardware: four local Blackhole p300c devices, logical 1x4 mesh, FABRIC_1D.
- Skills applied: `$full-model`, `$tt-device-usage`, `$tt-enable-tracing`,
  `$qualitative-check`, inherited `$multichip`/`$optimize` policy,
  `$shard-advise`, `$autofix`, and `$stage-review` (completed below).
- Scope stops at `tt/model.py` and `tt/generator.py`; no vLLM file, registry,
  server, or serving integration was created.
- The pre-existing unrelated edit to
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` was preserved and
  excluded from stage ownership.

For the commands below:

```bash
MODEL_DIR=models/autoports/mistralai_mistral_small_24b_instruct_2501
FULL_DIR=$MODEL_DIR/doc/full_model
SNAPSHOT=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724
REFERENCE=$FULL_DIR/artifacts/aime24_chat_100.refpt
```

Hardware work was serialized. `tt-smi -ls --local` was run before every major
gate and after interrupted/failed runs; all four boards remained visible and
resettable. Watcher and profiler were never combined.

## Implementation

`tt/model.py` now owns sharded embedding, 40-layer optimized TP4 stack, final
norm, untied vocab-sharded LM head, cache allocation/reset, and canonical
Sampling1D split sampling. `tt/generator.py` supplies lazy bounded-memory safetensor
loading, readiness construction, low-level prefill/decode, high-level
generation, public non-aligned padding/slicing, fixed-slot page tables,
persistent trace inputs, split trace capture/replay, device feedback, reset,
decode-only prefill-weight release, and explicit host compatibility.

`tt/multichip_decoder.py` gained an optional unsigned rotary-position input so
signed cache/SDPA positions can preserve inactive `-1` rows. The batch-1 path
is unchanged. A later mixed-slot gate exposed that batch 3 received a padded
32-row collective result at residual add; logical slicing was generalized to
batches 2–31 before both residual adds. Guards keep the selected batch-1 and
batch-32 graphs unchanged.

Shared readiness helpers gained a 1x4 `P300_QUAD` mesh label, configurable
trace reservation, corrected tokenizer BatchEncoding normalization, explicit
Mistral-regex selection, valid chat-template autoregressive prompting, an HF
attention mask, and generator trace-stat reporting.

## Reference generation

The first invocation found that this tokenizer returns a BatchEncoding from
`apply_chat_template`; the helper iterated the key `input_ids` and failed.
After normalizing the mapping, a warning identified the legacy Mistral regex.
That attempt was discarded and the final artifact was generated with the
corrected regex:

```bash
HF_HUB_OFFLINE=1 python -m models.common.readiness_check.generate \
  --hf-model $SNAPSHOT \
  --prompt-source aime24 --chat-template --fix-mistral-regex \
  --gen-len 100 --top-k 100 --output $REFERENCE
```

Final reference: 327 prompt tokens, 100 deterministic HF continuation tokens,
top-100 shape `[100,100]`, SHA256
`e88a9c2fe1d59448231e5edc4260f306328e4e4fdeef878d05166d2e4d9bbbc9`.
The command log and decoded continuation are retained in
`logs/generate_reference.log` and the artifact manifest.

## Canonical accuracy gates

The readiness CLI originally had no 1x4 label or way to reserve enough trace
space for 40 layers. `P300_QUAD=(1,4)` and `--trace-region-size` were added to
the shared non-vLLM mesh helper. Final prefill command:

```bash
HF_HUB_OFFLINE=1 TT_LOGGER_LEVEL=info \
python -m models.common.readiness_check.run_prefill_check \
  --model-dir $MODEL_DIR --reference $REFERENCE \
  --mesh-device P300_QUAD --fabric-config FABRIC_1D \
  --trace-region-size 200000000
```

Result: top-1 99/100, top-5 100/100, top-100 100/100.

Final teacher-forcing command:

```bash
HF_HUB_OFFLINE=1 TT_LOGGER_LEVEL=info \
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir $MODEL_DIR --reference $REFERENCE \
  --mesh-device P300_QUAD --fabric-config FABRIC_1D \
  --trace-region-size 200000000
```

Final-source result: top-1 97/100, top-5 100/100, top-100 100/100; TTFT
2808.99 ms; callback-inclusive decode 34.78 t/s/u; trace-only decode 54.77
t/s/u; end-to-end 17.68 t/s/u. The callback-inclusive rate includes
prediction observation and explicit forced-token copies; one-time trace setup
is reported separately as 1038.72 ms. The trace reports 99 model and 99 sampler
replays, final position 426, identical sample/feedback buffer, and zero
full-logit readbacks.

## Sampler, trace, host mode, and mixed slots

Reduced real tests use the exact endpoint weights and one exact optimized TP4
layer at context 128/64. Final primary command:

```bash
MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT TT_LOGGER_LEVEL=info \
pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_shape_full_model_and_split_trace
```

This validates a non-aligned seven-token prefill, explicit host compatibility,
split/force-argmax semantic equality, model/sampling trace order, device token
feedback, final position, changed/unchanged page-table copies, reset retention,
and no optimized full-logit readback. The final distinct-common-sampler
comparison is 0.314182 ms Sampling1D split versus 1.266483 ms SamplingGenerator
force-argmax for token 29317, recorded in
`logs/reduced_real_common_samplers_final_source.log`. SamplingGenerator's formatted
params, unseeded state, disabled penalties/log-probs, internal trace, and trace
release are exercised. Sampling1D is selected; the alternative is rejected; no
custom sampler is required. Older reduced-real logs are retained as iteration
evidence and are not the canonical final measurement.

Mixed-state command:

```bash
MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT TT_LOGGER_LEVEL=info \
pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_mixed_fixed_slots_and_inactive_rows
```

The first run failed `Invalid subtile broadcast type` at attention residual add
because the TP collective returned 32 physical rows for logical batch 3. The
implementation sliced padded collective results before both adds for batches
2–31. The first retry reached sampled outputs and failed only because PyTorch
does not implement CPU `ge` for uint32; the test cast was corrected. The final
run passes mixed prompt lengths 7/11, active slots 0/1, inactive slot 2 at
position -1, six nonalias physical pages, traced sample, final positions
`[8,12,-1]`, zero copies for an unchanged table, and one for a changed table.
All failure, adaptation, and final logs are retained.

Both real serving-state gates run with
`ttnn.CONFIG.throw_exception_on_fallback=true`. Repeated identical batch-1
prefill logits are bit-exact, and swapping the two mixed prompt rows swaps the
final logits exactly. The final batch-1 test replays both an unchanged page
table and a changed mapping, proving position advancement and that the changed
physical cache page is consumed while the prior page remains unchanged.

The terminal LM head initially used an eight-K-tile input block and failed the
Blackhole circular-buffer limit (1,782,528 > 1,572,864 bytes). The adapted
block-four DRAM-sharded program passes. Terminal output is padded to the common
sampler's 32 rows only after all decoder layers.

## Full context capacity

The context arithmetic includes BF16 TP endpoints, both optimized decode and
large-M prefill matrix representations for all 40 layers, all 40 BFP8 cache
pairs, shared context tables, Sampling1D buffers, trace terminal tensors, the
actual 200 MB-per-DRAM-bank (1.6 GB/device) trace reservation, and a 1.5 GiB
runtime reserve. A static test recomputes every total, validates the narrative
margin, and checks the page-aligned ceiling.

The physical gate was strengthened from the inherited 100 MB device fixture to
the full-model 200 MB-per-bank trace fixture:

```bash
MISTRAL_SMALL_24B_MULTICHIP_CAPACITY=1 TT_LOGGER_LEVEL=info \
pytest -q -s \
  $MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_full_context_paged_cache_capacity
```

Result: pass with retained prefill weights, 40 decode and 40 prefill matrix
layers, batch 32, max cache length 32768, local K or V shape
`[32768,2,32,128]`, a real batch-32 32-token prefill chunk, current position
32767, and 1,610,612,736 bytes/rank of physical runtime reserve. Steady state
is 31,378,609,024 bytes/rank; post-reserve margin is 1,189,509,248; physical
page-aligned ceiling is 34,464. HF 32,768 remains advertised. The final-source
gate uses distinct signed INT32 cache/SDPA and unsigned UINT32 rotary position
tensors and does not invoke the irreversible prefill-weight release.

The first retained-prefill remediation attempt used a 3 GiB reserve and failed
while allocating its sixth 512 MiB buffer. The allocator reported a
4,072,341,376-byte usable bank, exactly 200,000,000 bytes below the raw
4,272,341,376-byte bank. This proved that `trace_region_size=200000000` is per
bank, not per device, and exposed the prior 1.4 GB accounting understatement.
After correcting the trace cost to 1.6 GB/device and setting the explicit
runtime reserve to 1.5 GiB, the phase-accurate prefill+decode gate passes. The
canonical retained artifact is the passing final rerun.

## Autoregressive and qualitative gate

The first qualitative attempt was stopped after Transformers warned that EOS
equals PAD and the HF helper supplied no attention mask. The helper now passes
an all-ones mask for the unpadded prompt, and a focused source test prevents
regression. Final command:

```bash
HF_HUB_OFFLINE=1 TT_LOGGER_LEVEL=info \
python -m models.common.readiness_check.run_autoregressive \
  --model-dir $MODEL_DIR --hf-model $SNAPSHOT \
  --prompt-file models/common/readiness_check/autoregressive_prompt.txt \
  --chat-template --fix-mistral-regex \
  --mesh-device P300_QUAD --fabric-config FABRIC_1D \
  --trace-region-size 200000000 --max-new-tokens 100 \
  --output-dir $FULL_DIR/autoregressive
```

HF stops on EOS after 58 tokens; TT after 54. TT reports TTFT 6057.47 ms,
197.22 ms trace setup, and 55.9364 post-capture t/s/u across 53 replays. Final
position is 291=`238+53`; feedback shares the sampled-token address; trace
deltas are model/sampler replay 53/53, token copies 3, position copies 6,
page-table copies 1, sampling-param copies 0, and full-logit readbacks 0. The
fixed copies occur only in warmup/capture.

The outputs were read. They are identical through token index 23 and diverge at
24 into semantically equivalent requests for the missing story detail. Both are
coherent English, on topic, nonrepetitive, and end with EOS. Exact texts, token
IDs, and the qualitative verdict are in `autoregressive/`.

The shared readiness prompt suite was then run with the exact tokenizer/chat
template for six prompts and 128-token budgets. HF lengths were
`[16,128,128,128,62,128]`; TT lengths were `[18,128,128,128,62,128]`. Every
output was read and judged coherent, on-topic, nonrepetitive, and in the
requested language; the first TT prompt repeated byte-identically. The exact
checker reports no autoregressive degeneration finding. Its synthetic exact
128-token prompt/128-token generation benchmark reports TTFT 5812.15 ms and
55.0022 t/s/u across 127 post-capture replays, final position 255, no per-token
state rebuilds, and zero full-logit readbacks. Evidence is in
`qualitative_suite/`, `logs/qualitative_suite.log`, and
`logs/check_degenerate_output.log`.

```bash
HF_HUB_OFFLINE=1 python $FULL_DIR/run_qualitative_suite.py \
  --snapshot $SNAPSHOT \
  --prompts models/common/readiness_check/vllm_prompts.txt \
  --output-dir $FULL_DIR/qualitative_suite --max-new-tokens 128

python models/common/readiness_check/check_degenerate_output.py \
  --model-dir $MODEL_DIR --scope autoregressive \
  --missing-artifacts critical \
  --json $FULL_DIR/qualitative_suite/degenerate_output_report.json
```

## Reduced terminal profiler and watcher remediation

The reduced profiler covers one exact layer, final norm, full sharded head,
Sampling1D, and token feedback. Ten trace replays average 1.378589 ms; matmul is
69.21%, TopK 9.06%, Sampling 1.97%, and ManualSeed 1.25%. The inherited
per-layer lower bound scales to 16.592880 ms versus the 17.877457 ms measured
full token interval, leaving 1.284577 ms (7.18%). Compact profiler reports and
the arithmetic are under `profiler/reduced_terminal_trace/` and
`profiler/lower_bound_accounting.md`.

```bash
MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT \
python -m tracy -r -p -v \
  -o $FULL_DIR/profiler/reduced_terminal_trace \
  -m pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_full_terminal_trace_profile
```

The first worker-watcher run isolated a BRISC assert in Sampling1D's linear
all-gather writer. `$autofix` source/evidence audits traced it to the known
endpoint accessor bug: the writer requested an outward connection even when
`valid_targets(direction)` was false. The exact established repair from repo
commit `ff8ced34251` was applied to the shared writer. Both the canonical
Sampling1D-only terminal gate and the complete reduced gate now pass with
watcher, including both common samplers and clean detach on devices 0-3.
Failure, isolation, and fixed evidence are retained in the four
`logs/watcher_*` artifacts. Active-Ethernet inspection stays disabled for the
documented firmware 19.8.0 watcher-region limit; profiler and watcher were
never combined.

```bash
MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_full_terminal_trace_profile

MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_shape_full_model_and_split_trace
```

## Runtime audit and advisor

The source/runtime boundary audit is `logs/runtime_boundary_source_audit.log`.
It classifies all PyTorch, conversion, logits, cache, page, sampling, and reset
boundaries and records rejected single-chip/replicated/host fallbacks.

The repo-local terminal advisor target is `shard_advise/advise_terminal.py`.
The prescribed preinstalled `ttnn-advise` bootstrap failed before capture with
an undefined `ttnn::experimental::moe_compute` symbol in
`libTTMLIRRuntime.so`, proving tt-mlir/current-metal ABI skew rather than a
graph result. Per `$shard-advise`, the toolchain was not rebuilt. Exact command
and traceback are retained in `shard_advise/advisor_failure.log`; inherited
accepted decoder advisor results still apply to all 40 identical layers.

## Static/format gates

The first independent `$stage-review` returned `more-work-needed`. Its findings
were addressed as follows: both full 40-layer AIME gates were rerun after the
last runtime edit; the sampler comparison now uses the distinct common
`SamplingGenerator`; the shared six-prompt suite and exact checker were added;
changed page tables are replayed and verified against physical cache pages; a
reduced terminal profile, lower-bound accounting, and exact 128/128 benchmark
were added; context JSON now separates signed cache/SDPA from unsigned rotary
positions and documents tested batches/padding; representative gates run with
fallback-as-error; and raw watcher evidence was captured, diagnosed, fixed,
and rerun. A fresh reviewer is required below; the first verdict is not treated
as a pass.

The first fresh re-review also returned `more-work-needed` after finding that
the 32K physical probe released the large-M prefill matrices and that the
human-readable margin differed from the structured field by 128 bytes. The
probe now retains/physically models both weight representations for all 40
layers, corrects the 200 MB-per-bank trace cost, allocates 1.5 GiB of additional
runtime reserve, runs batch-32 prefill with full caches resident, and decodes
position 32,767. The structured and narrative margin is now
1,189,509,248 bytes/rank and the machine-checked ceiling is 34,464. The second
fresh re-review independently verified this corrected live set and arithmetic
and returned `clean-pass` with no required work.

Focused contract tests cover policy defaults, readiness signatures, context
arithmetic, fixed-page normalization, signed/unsigned positions, canonical
sampler wiring, optimized host-boundary absence, tokenizer BatchEncoding, HF
attention mask, and teacher-forcing completeness. `py_compile`, JSON parsing,
Black formatting, and `git diff --check` pass. Final consolidated commands and
results are appended after independent review.

## Review and commit log

- Independent `$stage-review`: `clean-pass` after two documented remediation
  cycles; final verdict retained in `stage_review.md`.
- Remediation/re-review: complete; no required work or material hard-check gaps
  remain.
- Stage-owned source/evidence commit: `3d35e46c5b5` (`Add Mistral Small 24B
  full-model TTNN stage`). The generic 500 KB pre-commit size check was skipped
  only for the required raw profiler log and compressed op table; all source,
  formatting, include, and policy hooks passed.
- Push: prohibited and not performed.
