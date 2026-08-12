# Optimized-full-model work log

## Scope and inherited state

This stage starts at full-model commit `37f95c9e2a1` and optimizes only the
complete TTNN model/generator path on the `1x4` Blackhole mesh. It does not add
vLLM integration and does not reopen the datatype Pareto frontier. The inherited
optimized-multichip policy, rejection ledger, residual layout, cache policy,
split-sampling contract, and 32,768-token context contract are mandatory inputs.

Hardware was checked with `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls` before
device work. Devices 0–3 are p300c Blackhole and firmware is 19.9.0. All
device-facing commands were serialized. Watcher and profiler runs are separate.

## Implementation

`FullModelConfig` now exposes bounded LM-head geometry knobs so exact terminal
candidates can be measured without altering endpoint dtype or fidelity. Defaults
remain 8192/10/64/4. `MistralSmall24BGenerator.replay_token_out_window` replays
the canonical model and split-sampling traces without token observation or
state copies and fences only once at the end of a measured window. The returned
sample token is still the persistent device feedback buffer.

The 40-layer benchmark uses fallback-as-error and verifies 128 model replays,
128 sampling replays, zero token/position/page-table/sampling-param copies, zero
caller/full-logit readbacks, one synchronization, and exact device position
advance. Compatibility generation is measured in the same process before the
host-free window.

```bash
MISTRAL_SMALL_24B_OPTIMIZED_FULL_MODEL_BENCHMARK=$SNAPSHOT \
MISTRAL_SMALL_24B_OPT_FULL_LAYERS=40 \
MISTRAL_SMALL_24B_OPT_FULL_STEPS=128 \
pytest --timeout=1800 -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_optimized_full_model_token_out_benchmark
```

Result: after one explicit prefill compile, adjacent warmed TTFT is 57.313770 ms
before and 57.938621 ms after (+1.09%, within run noise). Caller-observed trace
is 54.352821 t/s/u; host-free token-out is 54.451842 t/s/u (18.364851
ms/token). The read after—not inside—the timed window verifies the exact final
token, signed position, RoPE position, page table, and first/last-layer K/V
caches against the caller-observed control.

## Terminal sweep and lower bound

One-layer exact full-terminal runs evaluated input grids, output grids, K block,
and vocabulary split size. A direct 16K-column weight tilize exceeded L1, so the
candidate was adapted through interleaved-DRAM tilize plus the maximum legal
80-input-core/block-1 runtime family. It passes exact correctness and is 0.785%
faster over 256 replays. The block-8 failure was separately adapted with 80
input cores (effective block 2) and is 0.198% faster; 32 output cores is 0.065%
faster. All are below the 1% materiality threshold, so the simpler established
default is retained. `lm_head_candidates.csv` is the compact ledger and
`logs/lm_head_*` retain raw provenance.

The inherited 0.414822 ms/layer warmed result gives 16.592880 ms for 40 layers.
Current one-layer complete-terminal control is 1.747149 ms, leaving 1.332327 ms
terminal work and a 17.925207 ms composed target. The 18.364851 ms final window
is 0.439644 ms (2.394%) above it, so the avoidable-gap trigger is false.

## Accuracy and qualitative commands

```bash
HF_HUB_OFFLINE=1 python -m models.common.readiness_check.run_prefill_check \
  --model-dir $MODEL_DIR --reference $FULL_DIR/artifacts/aime24_chat_100.refpt \
  --mesh-device P300_QUAD --fabric-config FABRIC_1D --trace-region-size 200000000

HF_HUB_OFFLINE=1 python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir $MODEL_DIR --reference $FULL_DIR/artifacts/aime24_chat_100.refpt \
  --mesh-device P300_QUAD --fabric-config FABRIC_1D --trace-region-size 200000000

HF_HUB_OFFLINE=1 python -m models.common.readiness_check.run_autoregressive \
  --model-dir $MODEL_DIR --hf-model $SNAPSHOT \
  --prompt-file models/common/readiness_check/autoregressive_prompt.txt \
  --chat-template --fix-mistral-regex --mesh-device P300_QUAD \
  --fabric-config FABRIC_1D --trace-region-size 200000000 \
  --max-new-tokens 100 --output-dir $OPT_DIR/autoregressive

HF_HUB_OFFLINE=1 python $FULL_DIR/run_qualitative_suite.py \
  --snapshot $SNAPSHOT --prompts models/common/readiness_check/vllm_prompts.txt \
  --output-dir $OPT_DIR/qualitative_suite --max-new-tokens 128
```

Prefill is 99/100 top-1 and 100/100 top-5/top-100. Teacher forcing is 97/100
top-1 and 100/100 top-5/top-100, with 52.562 traced t/s/u. Refreshed free-running
TT output is coherent, on topic, nonrepetitive, English, and EOS-terminated at
48 tokens; HF terminates at 58. TT uses device feedback, reports 54.210 t/s/u,
and performs no full-logit readback.

The shared suite passed six prompts with TT output lengths
`[18,128,128,128,63,128]` versus HF `[16,128,128,128,62,128]`; every output was
read and is coherent, on topic, nonrepetitive, and in the requested language.
Prompt 1 repeats exactly under greedy decode. The degeneration checker reports
no finding. Its compatibility 128/128 measurement is 54.296 t/s/u and remains
separate from the host-free headline.

## Functional, profiler, watcher, and runtime gates

The reduced real mixed-slot gate uses prompt lengths 7 and 11, swaps request
order and proves exact logits, preserves inactive slot `-1`, verifies device
position advance, and demonstrates changed-only page-table copies. The split
trace gate also exercises a non-aligned 7-token prompt. All measured tests set
`ttnn.CONFIG.throw_exception_on_fallback = True`.

```bash
MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_shape_full_model_and_split_trace \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_mixed_fixed_slots_and_inactive_rows

MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT python -m tracy -r -p -v \
  -o $OPT_DIR/profiler/reduced_terminal_trace -m pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_full_terminal_trace_profile

MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL=$SNAPSHOT \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
pytest -q -s \
  $MODEL_DIR/tests/test_full_model.py::test_reduced_real_full_terminal_trace_profile
```

The source/runtime boundary inventory is retained in
`logs/runtime_boundary_source_audit.log`. PyTorch conversion remains confined
to weight loading, request-boundary compatibility, explicit outputs, and tests;
the host-free measured replay window contains no conversion, copy, sampling,
position, page-table, or synchronization boundary inside its token loop.

Both reduced functional tests pass. Separate watcher processes pass the full
terminal window and the split/common-sampler gate, check worker BRISC/NCRISC,
and detach devices 0–3. ETH inspection is disabled, matching the established
firmware-region workaround; fabric remains active. Watcher and profiler were
never combined.

The refreshed ten-replay profiler contains 18,568.35 us merged device time,
or 1,856.835 us/replay. `tt-perf-report` attributes 68.526% to matmul, 10.378%
to all-gather, 6.732% to TopK, 1.467% to Sampling, 0.928% to ManualSeed, and
2.037% to async all-reduce. TopK + Sampling + ManualSeed is 169.475 us/replay
(9.127%), so no sampler-dominance repair is triggered. The report's modeled
DRAM roofline is 42.6%. The compact op CSV, summary CSV, report table, and
selected command/report provenance are under `profiler/reduced_terminal_trace/final/`
and `logs/reduced_terminal_trace_profiler_provenance.log`. Multi-hundred-MB raw
Tracy internals and the 533-KB raw ops archive are intentionally not committed.

## Review and commit record

The first independent review returned `more-work-needed`. Remediation adapted
the dominant 16K/block-8 LM-head candidates instead of accepting first-error
L1 failures, added exact out-of-window token/position/RoPE/page/cache comparison
for the host-free trace, added adjacent warmed TTFT samples, corrected the
non-aligned prompt claim, and classified the superseded 300-second load timeout.
A fresh independent re-review returned `clean-pass` after comparing the entire
original goal against final source, raw logs, profiler tables, context evidence,
and generated texts. Its conclusions are retained in `stage_review.md`.

`logs/full_40layer_128x128.log` is the superseded first harness attempt: pytest's
300-second timeout fired during the normal layer-40 weight load. Fixture teardown
closed the mesh; a bounded health check showed all four devices. The 1,800-second
retry passed and the final correctness/warmed run supersedes both.

The post-review implementation/evidence commit is `d0313954b04` (`Optimize
Mistral Small 24B full-model token-out path`). This follow-up documentation
commit records that SHA. Nothing was pushed.
