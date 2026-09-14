# AutoDebug: prefill sampling program buffers and decode trace lifetime

## Verdict

The 2026-09-14 00:52:43 failure is a missing warmup/lifetime boundary in
`QwenGenerator.sample_prefill`. The tracker identifies two persistent program
buffers created while packing prefill logits after an older decode trace was
captured. Those buffers remain alive when the old decode trace is replayed.

Releasing the existing traces **before the first use of each prefill packing
signature** is the correct narrow repair for this observed path. Record the
signature only after packing and sampling succeed. Retain known signatures
across ordinary request resets and trace recaptures, but associate them with the
current cache binding. A newly bound cache must establish its own warm history.

The parent implemented the signature guard using each input's shape, dtype, and
layout. This investigator subsequently added the authorized three-line cache
identity guard and a dedicated host regression. Hardware verification of the
repair remains with the parent; this report does not claim a tracker-enabled
runtime pass.

## Direct evidence

The investigator read `readiness_vllm/server.log` and
`readiness_vllm/sampling_tests.log`. The sampling selection reports two passes,
one skip, then failure in `TestHostOnlyParameters.test_min_p`. EngineCore PID
646033 reports at 00:52:43:

```text
Found 2 device buffer(s) still alive before trace replay.
Buffer 91592: program_cache: ConcatDeviceOperation dim=2 ... DRAM
Buffer 91607: program_cache: TilizeWithValPaddingDeviceOperation
  output_padded_shape=[1,1,32,62080], output_dtype=BFLOAT16
  enough_space_width=true, enough_space_height=false
```

Both allocation backtraces end at the concat expression in
`tt/generator.py:353`, inside `sample_prefill`, called by
`tt/generator_vllm.py:172`. The failure itself occurs later at the first
`execute_trace` in `QwenGenerator.decode_forward`. No Python stack frame holds
either tracked buffer, consistent with their explicit `program_cache` ownership.

The log does not print the packed input count. A new count is a plausible source
of the new program signatures, not a directly observed integer in this log. The
causal finding is the uncovered packing-program allocation boundary, which the
shape/dtype/layout signature guard addresses more precisely than count alone.

The previously passing concurrent and repeated-request controls supplied by the
parent are compatible with this finding: reuse of individual prompt shapes does
not prove every combination of packed prompt results has been compiled.

## Source chain

1. The adapter calls `generator.prefill_forward` separately for each scheduled
   prompt, collects the resulting last-token logits, then calls
   `generator.sample_prefill(outputs)` once for the packed batch.
2. `QwenGenerator.prefill_forward` tracks individual signatures comprising cache
   batch, table shape, slot, start position, prompt length, and all-logits mode.
   It releases traces for new individual prefill shapes and records them after
   success. This guard does not describe the later concat of several outputs.
3. The original `sample_prefill` called concat, pad-to-32, and the canonical
   sampler without a corresponding first-use guard. Several already-known
   single-prompt shapes can therefore form an unseen packed shape while an old
   decode trace is still retained.
4. The rank-4 TILE logits are concatenated along dimension 2. In
   `ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp`, the fallback's
   `build_untilize_rm_retilize_concat` unpads/untilizes the inputs, concatenates
   row-major rows, then invokes `tilize_with_val_padding`. This explains why the
   tilize allocation's backtrace names the same Python concat line, rather than
   the subsequent explicit pad call.
5. Program-cache ownership outlives the local `packed` tensor. Deleting local
   logits or waiting for queued work cannot remove those cached buffers. The
   allocator tracker records later allocations against every active trace
   (`tt_metal/impl/allocator/trace_allocation_tracker.cpp:117–135`). Before
   replay, it filters by buffers still allocated and rejects the surviving IDs.
6. `_release_traces(keep_prefill=True)` releases the decode, sampling, and any
   prefill trace handles. The `keep_prefill` flag preserves prepared prefill
   inputs, not a live trace. New packing programs are then created before the
   next decode capture. They predate that capture, so they no longer violate its
   allocation lifetime.

This identifies the first necessary intervention. The `min_p` test label does
not establish that the sampling algorithm or numerical policy caused these
allocations: both reported owners come from packing before `_sampling_step`.

## Signature scope and cache lifecycle

The accepted signature is:

```python
tuple((tuple(x.shape), x.dtype, x.layout) for x in logits)
```

It distinguishes prompt count as well as changes in each input specification.
For this generator, each last-token logit has fixed model vocabulary width and
interleaved DRAM placement, and the selected sampler mode remains the canonical
split path. The signature need not include token values or prompt text.

Cache binding already releases old trace handles and clears individual prefill
signatures, but an unscoped packing-signature set would survive binding. The
added guard checks object identity at `sample_prefill` entry:

```python
if getattr(self, "_prefill_sampling_cache", None) is not self.cache:
    self._prefill_sampling_signatures = set()
    self._prefill_sampling_cache = self.cache
```

This covers both external `bind_cache` replacement and a new owned cache from
`_ensure_cache`, without adding a trace release to each ordinary reset. It also
works if a new binding has already captured decode before its first packed
prefill sample. The warmed signature is added only after `_sampling_step`
returns successfully; failed sampling must not mark the new signature warm.

Program-cache lifetime remains an explicit assumption. No program-cache clear
call was found in the autoport or its sampler during this inspection. If a future
caller clears the mesh program cache while retaining a generator/cache binding,
it must invalidate warm metadata before reuse. Likewise, supporting arbitrary
logit memory configurations or a different sampler mode would require extending
the signature or establishing a separate warmup boundary.

## Alternatives checked

- **Native concat factory changes with live L1 occupancy:** the newer native
  unaligned concat eligibility code does inspect live L1, and factory choice is
  hashed. However, `concat_tiled_unaligned_program_factory.cpp` requires the
  concat dimension to be the last dimension. This rank-4 dimension-2 packing
  call cannot select that factory. That alternative is refuted for this trace.
- **Tilize changes with L1 availability:** its `enough_space_width/height` flags
  are derived from available L1. The log records width true, height false for the
  wide BF16 row. The current fixed logit specification and memory placement are
  the intended invariant; arbitrary memory-pressure-induced routing changes are
  not proved impossible by a Python signature test. Keep the allocation tracker
  enabled during runtime validation rather than treating metadata as proof that
  no new native program can ever appear.
- **Release traces on every prefill request:** unnecessary for known packing
  signatures and loses the previously validated reuse behavior. The narrower
  first-use boundary should be tested first.
- **Skip program-cache tracking or mark those buffers corruptible:** unsupported
  by the ownership evidence. These are persistent native program resources, not
  disposable outputs. The repair should establish their lifetime before capture.

## Host experiment

`tests/test_prefill_sampling_trace_host.py` extracts the actual `sample_prefill`
method from its Python AST and runs it against stdlib fake tensors and effects.
It imports neither TTNN nor the model. Five tests verify:

1. Unseen signatures release before concat/pad/sampling.
2. Repeated signatures preserve a newly installed decode trace.
3. New counts and dtypes release before new packing work.
4. A new cache clears old signature history and releases before reuse.
5. Failed sampling leaves the new signature unwarmed and retries the release.

Command and observed result:

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/test_prefill_sampling_trace_host.py
# Ran 5 tests in 0.019s: OK
```

An in-memory negative control removed the cache-identity guard from the extracted
method. The cache-rebinding regression produced exactly one assertion failure
and zero errors, proving it detects the difference from the parent's initial
signature-only repair. Syntax compilation and Black formatting passed. Only the
authorized cache guard, dedicated test, and this report were authored by this
investigator; no hardware was executed.

## Required runtime verification

With allocation tracking and program-cache tracking enabled, reuse already-known
individual prompt signatures while varying packed count (for example 1, 2, 3,
4, then 2 again), decoding after each prefill. Require first-use trace release,
successful recapture/replay, and subsequent reuse of the known count. Repeat the
original reduced sampling selection including `test_min_p`, and preserve the
prior concurrent/repeated-request controls. A separate new-cache binding case
must re-establish the guard without relying on the old warm set.

A runtime pass must show no surviving unsafe buffers on replay and normal token
delivery. Host guard tests establish ordering and metadata behavior only; the
parent's live rerun determines whether further allocation signatures remain.
