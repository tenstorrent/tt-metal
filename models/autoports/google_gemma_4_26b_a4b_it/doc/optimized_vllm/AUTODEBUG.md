# AutoDebug: decode output formatting

Inspection date: 2026-09-27. Scope: Gemma4 stage 10, TP4/DP1, B1/C1,
4096 input / 128 output; keep max_num_seqs=32, max_model_len=262144,
canonical split sampling, selected precision, and nonblocking replay.

## Supported hypothesis

The device-sampling serving path dispatches an avoidable eager device slice
after every pair of model and sampler trace replays. Move that existing slice
into the existing sampler trace, using the same input and output tensors.
This removes one eager dispatch without changing sampling math, trace count,
output shape, or readback ownership. Whether it materially improves serving
throughput remains unproven until the same-harness before/after run.

Direct observations in the pre-change source:

- `tt/generator.py:339-343` allocates a persistent 32-lane sampler/feedback
  tensor and a separate persistent B-lane public tensor plus a B-vector view.
- `_format_tokens` at lines 345-352 copies the leading B tokens into that
  public tensor. `decode_forward` calls it eagerly at lines 439-443 after
  `_replay`. `_replay` submits model and sampler traces nonblocking at
  lines 446-459.
- `models/common/sampling/generator.py:395-401` captures `_run_sampling`;
  lines 312-331 run canonical sampling and then penalty bookkeeping.
  A model-local override that delegates to `super()._run_sampling` and
  appends the public copy therefore captures the copy after sampling.
  Trace-input validation at lines 290-310 protects logits/output identity.
- `_capture` at generator lines 356-364 warms the public slice and sampler
  before model capture. The public destination already exists, so the appended
  slice needs no new runtime allocation. `_bind` releases old traces before
  assigning new input/output buffers.

The inherited 50.7283 serving t/s/u versus 51.6473 standalone t/s/u is contextual
evidence from different prompt/harness paths, not proof that this slice accounts
for the gap. This inspection ran no accelerator workloads or profiling.

## Smallest intervention and ownership constraints

Use a local `SamplingGenerator` subclass that binds the feedback and public
token tensors, delegates all sampling to the canonical implementation, then
copies to the public tensor only when `tt_out_tok` is the bound feedback tensor.
Retain explicit slice warmup, canonical sampler precompile/capture/replay,
seed increments, penalties, dtype, and the existing persistent public view.
Remove the post-replay eager copy only for device sampling. Host sampling in
`_replay` bypasses the sampler and must keep its eager public copy.

Bind tensor references directly rather than storing the owning generator or
a bound generator callback. Rebinding follows the existing trace-release
lifecycle and creates no generator/sampler reference cycle. Serving prefill
passes no output tensor and must not copy into a previous decode binding;
standalone prefill with the explicitly bound feedback tensor may format it.

Do not replace the public tensor with a raw 32-lane tensor and later trim using
`generator.batch`. `generator_vllm.py:292-306` can finalize output later; a
subsequent rebind can change the generator batch. The existing output tensor's
logical shape belongs to its submission. The plugin immediately enqueues
`read_decode_output(async_read=True)` in `async_decode.py:633-645`, records the
CQ0 event in the adapter, and waits only at finalization (lines 673-685).
With the proposed change the queue stays: model replay, sampler replay including
public copy, token readback, event, next decode submission. Keep this ordering,
the nonblocking replay flags, and the returned persistent tensor identity.

A separate formatting trace would preserve semantics but add a third trace
submission; it is a less focused first experiment. Formatting inside the model
trace would run before this step's sampler and return the preceding token.

## Page-table hypothesis: defer

The generator compares each layer table with its saved host clone at
`generator.py:418-430`; adapter `_decode_inputs` constructs a new row view per
layer at `generator_vllm.py:163`. Plugin `_block_tables_per_layer` expands six
groups into layer entries (`model_runner.py:465-505`), preserving shared Python
objects only when already padded. Thus repeated work across group-equivalent
layer tables is plausible, but identity of row views is insufficient to prove
equivalence. The generator also uploads/clones each input entry separately at
bind time, so skipping comparisons solely by current input identity could miss
changes against distinct previous snapshots or device destinations.

No duplicate whole-table equality check was found in the plugin; its reset
signal uses scheduler deltas and row lifecycle instead (`model_runner.py:698-723`,
`async_decode.py:259-264`). Do not skip page refresh just because device feedback
is active: the direct generator contract accepts changed pages in that mode.
The existing reduced adapter test checks changed-page refresh without recapture.
Treat page-table deduplication as a separate hypothesis with separate evidence.

## Focused validation

1. Host regression: device sampling does not run an eager formatting operation
   after replay, while host sampling still does. Capture-order mock verifies
   canonical sampling precedes the public copy within sampler capture and that
   prefill/unbound outputs are untouched. Preserve original sampling return.
2. Run the complete existing host sampling-contract suite, including page growth,
   seed restore, host/device transitions, and padded/interior slots.
3. Parent-owned reduced hardware regression: existing `check_vllm_adapter.py`
   covers nonaligned prompt length 33, stale host token/position feedback,
   async and synchronous reads, external cache identity, unchanged/changed
   per-layer pages, and trace retention. Add two queued decode/read submissions
   before host finalization, and verify persistent tensor/view identity and
   per-submission values. Cover B1 and a padded batch with inactive slots.
4. Parent-owned unchanged-configuration serving before/after run and qualitative
   validation decide whether to retain the performance candidate. No speedup
   claim is supported by host tests alone.
