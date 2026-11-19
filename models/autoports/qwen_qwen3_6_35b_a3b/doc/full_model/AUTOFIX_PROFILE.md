# Profiler drain repair after v3 marker loss

Status: repair validated by the parent’s serialized v4 hardware capture and full
native conversion/coverage validation; collector exit 0. This agent performed
only source/CPU validation and documentation reconciliation. This change owns
only `tests/run_full_model_performance.py`; no production model, sampler,
profiler implementation, buffer size or conversion code changed.

## Preserved failure and diagnosis

`performance/profile_reduced_v3_midrun/capture.log` records
`GENERATOR_PREPARED` at line 131582, model/sampling captures at 131584/131586,
and 700 dropped-marker warnings at 131587–132286. All three component results
and the workload PASS follow at 132291–132298. The collector correctly rejected
the capture; workload PASS does not make incomplete profiler evidence valid.
The raw directory and prior v1/v2 failures remain preserved.

The earlier `_attention`/`_moe` wrappers covered only the first `seed_request`.
They were removed before `generator.prepare`, warm traced requests, diagnostic
TTFT and the model-only/sampler-only seed requests. The first read after capture
and four warm model/sampler pairs therefore reported a full profiler buffer.
The warning's location identifies the detection interval, not the exact
individual operation that first overflowed it.

`MID_RUN_DUMP=1` enables dump/clear when results are processed; it does not create
a periodic background read. The native processing branch calls
`dumpDeviceResults(true)` at `tt_metal/impl/profiler/tt_metal_profiler.cpp:1050`,
and parsed marker storage is cleared at `profiler.cpp:2524`. Separately retained
C++ analyses (`profiler.cpp:2470`) and Tracy/cache metadata mean this change is
not a proof that every category of host memory is globally bounded.

Later prefill requests do not repeat the exhaustive geometry warmup:
`tt/prefill.py:264` checks the prepared cache signature and returns. They are
still eager work requiring explicit profiling boundaries. The source evidence
supports repairing the missing drains; it does not prove every later request
would have repeated the original 188 GB OOM.

## Narrow harness change

`install_profiler_drains` installs profiler-only wrappers in an `ExitStack`:

- All eager executor `_attention` and `_moe` calls, including later prefills.
- Phase-one `_decode_into_sampler`, `_sampling_forward` and common sampler
  `precompile`. The selected default `precompile` runs one sampling pipeline;
  no new sampler algorithm or configuration sweep is introduced.
- Immediately before `begin_trace_capture` and after `end_trace_capture`, on
  this mesh only. The model and sampling traces retain their original graph.

Every drain first queries `ttnn.is_trace_capture_active(mesh)`, which checks all
command queues. If any capture is active, the wrapper records a skipped drain
and performs neither synchronization nor profiler read. The native query is
exposed at `ttnn/cpp/ttnn-nanobind/device.cpp:740`. Original operations execute
once and their return values are preserved. Capture hooks call the original
begin/end functions once; reads occur outside those calls.

Explicit drains also follow complete seed terminal/sample handoffs, the final
prepare/reset boundary, and each of the four warm model/sampler replay pairs.
The measured decode loops keep their existing per-sample drains outside the
timed samples. Profiling TTFT now explicitly includes eager-component and
handoff drain I/O/synchronizations and is marked ineligible for headline timing.
Unprofiled model, token-out, observed-token and HF teacher-forcing measurement
loops are unchanged.

`finally` closes the `ExitStack` before generator trace release and mesh close.
All temporary wrappers are therefore restored after preparation, capture,
execution or profiler-read exceptions. New diagnostic fields are
`profiler_drain_counts` and `profiler_drains_skipped_during_capture`; memory
observations retain the boundary name and total drain count.

## Buffer choice and verification limits

The collector's 5000-program buffer is retained. V3's warnings form one block
before the component results; there are no further dropped-marker warnings in
the 16-per-component measured replay phases that already drain every replay.
The initial individually drained geometry sweep also precedes the reported
loss interval. This supports using the existing per-component/per-replay bound
rather than inflating buffers and host-memory peaks. The repair additionally
isolates each capture and each warm replay instead of combining them.

This is evidence for the chosen boundary size, not a guarantee for every future
kernel/configuration. V4 subsequently passed the unchanged dropped-marker/partial-capture rejection
and native execution/metadata coverage checks. The collector completed in about
3040 seconds, within its configured limits; this single capture is not a bound
on future raw-volume or conversion cost. The repair does not reduce retained
raw evidence or relax the conversion bound or validator.

## CPU validation

Executed without importing TTNN or opening hardware:

```text
python /tmp/qwen_profile_drains_cpu.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_35b_a3b/tests/run_full_model_performance.py
python_env/bin/pre-commit run --files models/autoports/qwen_qwen3_6_35b_a3b/tests/run_full_model_performance.py
```

The temporary AST-isolated fake test executes the actual drain installer. It
covers repeated eager and later-prefill calls, phase-one model/sampler calls,
two captures with zero synchronization/read events inside capture, pre/post
capture drains, preserved return values, restoration after an eager exception,
and restoration when the first profiler read raises. Syntax and repository
pre-commit checks passed. These CPU checks establish wrapper control flow only;
they do not establish device marker completeness, runtime safety or conversion
success. No retry, reset, profile conversion or active-job modification was
performed by this agent.

## Final serialized v4 outcome

The parent ran `performance/profile_reduced_v4_drains` and reported collector
exit 0 after approximately 3040 seconds. The workload completed PASS in
916.514689 seconds with zero observed marker-drop/source-overflow warnings and
a clean device close. `component_summary.json` records `validated_profile`,
32,896 native executions, and exactly 16 sessions per component on each of four
ranks. `native_program_metadata.json` retains actual operand dtype, shape,
layout/storage and compute/data-movement kernel metadata. Data-movement-only
programs retain empty compute-kernel lists rather than fabricated fidelity.

`summary.log` reports rank 1 model/sampler kernel sums of 1.471036125 and
0.4548093125 ms; sampler fraction is 0.2361608588332001 and does not dominate
this measured reduced kernel work. Scope is real layers 0/3 plus full
embedding/norm/LM head, batch 1/context 1024. Ranks run in parallel; kernel
sums are not wall latency, an additive autoregressive decomposition or an
all-40 headline. Isolated all-40 performance remains pending.

The retained `*_perf_report.console.log` files warn that several operation
names lack `OPERATION_CATEGORIES` labels. Each named operation is present in
the native metadata inventory; execution coverage is separately validated.
These warnings limit attribution-category specificity. They are not a claim
of complete category labeling or a reason to discard actual metadata. V1/v2
and v3 failures remain intact. No additional hardware run, gate, retry or
conversion was performed by this documentation task.
