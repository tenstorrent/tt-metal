# AutoDebug: Stage 02 fused-prefill and evidence gaps

Date: 2026-07-30

Scope: inspection only. No TT hardware was reserved or used, and no
implementation or test file was edited.

## Headline findings

### 1. FusedDecoder prefill MoE definitely still uses standalone GeGLU

This is a dispatch omission, not a speculative profiler interpretation.

`FusedDecoder` overrides `_dense_mlp` and `_moe_decode_single_user`, but not
`_moe_prefill` or `_moe_prefill_chunk`. Normal virtual dispatch therefore
reaches `FunctionalDecoder._moe_prefill_chunk`, which calls the imported
canonical `sparse_expert_prefill`. That is
`models.demos.gemma4.tt.experts.prefill.prefill_forward`; each 32-token chunk
reaches `_process_prefill_chunk`, which executes:

```python
down_input = apply_geglu(gate, up)
```

`models/demos/gemma4/tt/experts/operations.py` defines that operation as two
separate TTNN calls:

```python
activated = ttnn.gelu(gate, fast_and_approximate_mode=True)
result = ttnn.mul(activated, up)
```

Consequently every prefill chunk dispatches a standalone GELU before multiply.
The current host-only fused test cannot catch this: it inspects only
`_dense_mlp` and `_moe_decode_single_user` and expects exactly two occurrences
of `input_tensor_a_activations`.

### 2. The minimal in-scope repair is a model-local prefill expert helper

Do not modify the shared canonical Gemma4 expert module: that would expand
Stage 02 scope and affect other consumers. In `tt/fused_decoder.py`, add a
model-local equivalent of canonical `_process_prefill_chunk` and override
`_moe_prefill_chunk` to call it. Preserve every canonical operation, argument,
layout, shape, allocation, and deallocation except this replacement:

```python
down_input = ttnn.mul(
    gate,
    up,
    input_tensor_a_activations=[
        ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)
    ],
)
```

The safest minimal implementation copies only the chunk body, not
`prefill_forward`: inherited `_moe_prefill` already owns the model's existing
chunking and calls virtual `self._moe_prefill_chunk`. The override should use
the decoder's existing `expert_weights`, `expert_config`, and
`expert_prefill_sparsity`.

API and topology constraints that must remain identical to the canonical path:

- `hidden_states` is `[1,1,chunk_len,H]`; the current path chunks to 32, so
  `group_size=chunk_len/32` and sparse matmul uses tile `[32,32]`.
- Repeat the row-major all-ones prefill sparsity for gate/up, with
  `nnz=num_experts*group_size`; down projection uses the base sparsity,
  `nnz=num_experts`, and `is_input_a_sparse=True`.
- Preserve `_build_sparse_matmul_config(32, intermediate_size)` for gate/up
  and `(32, hidden_size)` for down, DRAM output memory, BF16 dtype, transpose
  and reshape order, routing-weight permute/multiply, expert reduction, and
  final shape.
- Preserve the canonical `fast_and_approximate_mode=True` semantics. The
  existing fused code represents Gemma's tanh GELU as
  `UnaryWithParam(GELU, 1.0)`; hardware PCC must prove that this is equivalent
  on this sparse-prefill shape.
- Keep tensor lifetime unchanged: deallocate `hidden_grouped` only after the
  up projection and deallocate fused `down_input` after down projection.
- Do not opportunistically move intermediates to L1 or reuse decode configs in
  this fix. Prefill computes all experts and has substantially different live
  shapes; such changes need separate L1-capacity and performance evidence.
- Preserve multi-chip behavior if this helper is ever invoked there. The
  present decoder is single-device, so no new CCL logic belongs in the local
  chunk helper.

### 3. Current decode CSV is capture topology, not traced-replay performance

`tracy/decode_capture_sliding_batch1.txt` explicitly says:

- no device architecture was found;
- conversion defaulted to Wormhole;
- no nonzero worker count was found, so it defaulted to 64;
- the resulting rows have zero `Device Time`.

The CSV also begins with the prefill-end signpost. It can document which TTNN
operations were enqueued while the trace was captured, but it cannot support
Blackhole/P300 per-op timing or traced-replay speed. Trace replay executes
retained Metal commands and may emit no new TTNN op rows. The evidence must
therefore keep three different measurements separate:

1. **capture topology:** op list between explicit decode-capture signposts;
2. **replay latency:** warmed host wall time around blocking
   `ttnn.execute_trace`, with device synchronization;
3. **device performance:** Blackhole device-profiler cycle/timestamp evidence
   for replay, if the installed profiler can attribute retained-trace replay.

Passing `tt-perf-report --arch blackhole` can correct advice selection when
architecture metadata is absent, but it cannot manufacture missing worker
metadata or nonzero device timestamps. It does not make the current zero-time
CSV valid performance evidence.

### 4. Final-path evidence is not immutable or complete

Only the current full-attention batch-1 timing artifact records decoder hash
`519ad63d...`; the sliding b1, sliding b32, and full b32 timing files record
`383ef398...`. The watcher and most correctness artifacts predate the final
router folding and/or contain no decoder/test/build hashes. Those artifacts
cannot prove the delivered source.

The next run must first freeze one revision identity and attach it to every
artifact. At minimum record:

- checkout commit plus dirty diff hash (a commit SHA alone is insufficient);
- SHA-256 of `fused_decoder.py`, `functional_decoder.py`,
  `test_fused_decoder.py`, `test_functional_decoder.py`, and the loaded TTNN
  extension/build;
- exact command and relevant environment;
- model/checkpoint revision;
- Blackhole architecture, P300 identity/device IDs, firmware/build identifiers;
- UTC start/end time, test case ID, repeat/warmup counts, and artifact SHA-256.

Generate a manifest last, containing hashes of all immutable run-identified
artifacts. Do not overwrite canonical filenames during retries. README tables
must name the accepted run ID and explicitly supersede older runs.

## Focused experiments

### A. Host-only structural tests

Extend `tests/test_fused_decoder.py` within stage scope:

1. Assert `"_moe_prefill_chunk" in FusedDecoder.__dict__`.
2. Monkeypatch module-local TTNN and sparse matmul/config dependencies; execute
   the override for a 32-token synthetic chunk.
3. Assert operation order and exact sparse-matmul arguments.
4. Assert there is no `ttnn.gelu`, `apply_geglu`, host conversion, or direct
   `FunctionalDecoder._moe_prefill_chunk` call.
5. Assert the prefill multiply has one GELU unary and preserve the existing
   dense/decode assertions.
6. Add a dispatch test proving inherited `_moe_prefill` calls the subclass
   `_moe_prefill_chunk`.
7. Add a manifest validator test that rejects missing hashes, mismatched source
   hashes, duplicate artifact paths, and artifacts from a different run ID.

Suggested static command:

```bash
pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py
```

### B. Hardware experiment order for the stage owner

Run these as separate, provenance-stamped jobs on Blackhole/P300:

1. Microcompare canonical versus local fused expert chunk at sequence 32 with
   identical real weights/inputs/routing. Check shape, PCC, max/mean error,
   determinism, no fallback, warmed latency, and profiler topology.
2. Full decoder PCC for sliding and full attention, prefill and decode, plus
   shared/natural cache views where applicable.
3. Non-aligned lengths `1,31,32,33,63/127,64/128,65/129,1023,1024,1025`;
   paged-cache boundary/tail integrity; representative near-context decode.
4. Eager versus trace-capture output, repeated trace replay determinism at b1
   and b32, and stress/repeated runs.
5. A separate watcher-clean suite after the final source is frozen.
6. Controlled functional-versus-fused performance. Prefer interleaved paired
   runs in the same reservation; use the same inputs, build, warmups, cache
   state, and repeat counts. Require the final source to win, not merely remove
   an op.

Use explicit run IDs rather than the old overwrite-prone destinations:

```bash
export GEMMA4_EVIDENCE_COMMAND='the complete command used for this run'
export GEMMA4_DECODER_IMPL=fused
export GEMMA4_RANGE_DOWNLOAD=1
export TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'

pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py

pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k 'real_weights_prefill_decode or non_aligned or boundary or traced_decode'

TT_METAL_WATCHER=10 pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k 'real_weights_prefill_decode or traced_decode_batch_contract'
```

The actual exact command string, including all repeat/count/profiler variables,
must be written by the harness rather than reconstructed afterward.

## Required profiler matrix

Use sequence/current position 1024 unless the stage contract says otherwise.
Prefill is meaningful at batch 1 in the current harness; decode must cover both
b1 and serving b32.

| Layer | Mode | Batch | Required retained evidence |
|---|---|---:|---|
| sliding | prefill | 1 | Blackhole raw op CSV, bounded signpost slice, nonzero `tt-perf-report` CSV/table |
| full | prefill | 1 | same |
| sliding | decode capture | 1, 32 | capture-topology raw/sliced CSV; label as capture, not replay |
| full | decode capture | 1, 32 | same |
| sliding | traced replay | 1, 32 | warmed host samples plus device-profiler replay evidence or limitation record |
| full | traced replay | 1, 32 | same |

For a properly populated raw CSV:

```bash
tt-perf-report RAW.csv \
  --arch blackhole \
  --start-signpost PERF_PREFILL_CASE \
  --end-signpost PERF_PREFILL_CASE_END \
  --csv REPORT.csv > REPORT.txt
```

Use exact case-specific signpost names. Retain the raw CSV, converter stdout,
converter version/help output, report CSV, and human table. Validate before
acceptance: architecture is Blackhole, worker metadata is nonzero and
plausible, the selected range contains only the intended case, op device times
are nonzero, and report totals agree with raw timestamps.

For decode, add signposts specifically around **trace capture** so the captured
op topology does not accidentally start at a prefill-end marker. Keep replay
host timing in its JSON. If replay emits device rows, retain and report them
separately from capture rows.

If retained-trace replay cannot be represented by `tt-perf-report`, create a
tool-limitation artifact containing tool versions, exact commands, raw output,
the absence of replay op rows, and a minimal eager/capture/replay control. Then
provide the contract-approved equivalent: device-profiler kernel/zone cycle
data around replay plus synchronized host latency. Never substitute capture
device times for replay performance, and never accept a zero-time or
Wormhole-defaulted table as Blackhole evidence.

## Complete final evidence matrix

The final manifest should have an explicit pass/fail row and artifact list for:

- host-only fused topology/dispatch tests;
- real-weight PCC: sliding/full, prefill/decode, required cache variants;
- non-aligned lengths and paged-cache boundary/tail/context checks;
- eager/capture/replay equivalence;
- repeated replay determinism b1/b32;
- stress/repeated execution;
- watcher-clean final-source run;
- functional and fused prefill b1 performance, sliding/full;
- functional and fused traced-decode b1/b32 performance, sliding/full;
- profiler matrix above;
- tool-limitation/equivalent evidence, if needed.

Acceptance should fail closed if any artifact's embedded provenance differs
from the manifest's frozen identity. Existing stale artifacts may remain as
historical data, but must not appear in the accepted final matrix.

## Conclusion

The code-level defect is clear: fused prefill MoE was never overridden and
therefore still executes canonical standalone GELU plus multiply. A local
`_moe_prefill_chunk` override is the smallest stage-scoped repair. Its benefit
and numerical equivalence remain hardware questions. The present profiler and
artifact set cannot answer them because decode capture was mislabeled as replay
performance, Blackhole metadata/device time is absent, the profiler matrix is
incomplete, and most evidence is stale relative to the delivered source.
