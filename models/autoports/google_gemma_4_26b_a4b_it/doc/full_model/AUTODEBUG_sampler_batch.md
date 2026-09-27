# AutoDebug: batch-3 sampler subtile broadcast

2026-09-27. Source-only diagnosis; no implementation changes or hardware runs.
Evidence: `trace_mixed_slots.log`, `tests/check_full_trace.py`, current model and
generator, common sampler, binary/top-k/sampling/pad operation source.

## Finding

**The canonical sampler receives logical batch 3 logits but owns logical batch
32 offset and parameter tensors. Physical TILE padding does not reconcile the
logical shapes.** This source diagnosis directly matches the recorded failure;
the proposed repair still needs execution.

The failing test binds three fixed slots, with slot 1 inactive. Model decode
preserves their order and returns local logits `[1,1,3,65536]` for the four-way
sharded 262144-token vocabulary. This is correct raw-logit shape but incomplete
as the current common sampler's input contract.

| Boundary | Logical per-device shape |
| --- | --- |
| Model decode logits | `[1,1,3,65536]` |
| Local `topk(k=32)` indices | `[1,1,3,32]` |
| Four-device gathered indices, then int32 cast | `[1,1,3,128]` |
| `tt_indices_device_offsets` | `[1,1,32,128]` |
| Persistent sampler token output | `[1,1,1,32]` |
| `k`, `p`, `temp`, seeds | `[32]` |

`TTSampling._create_indices_tensors` (`tt_sampling.py:375`) creates offsets
using `max_batch_size`, which is rounded up to at least 32. Top-k output specs
preserve the input logical height (`topk_device_operation.cpp:379`). The add at
`tt_sampling.py:1078` therefore attempts height 32 versus height 3.

`binary_ng_device_operation.cpp:621` explicitly passes **logical** height/width
to `get_subtile_broadcast_type`. Its cases at lines 226–255 accept equal shapes
or singleton row/column dimensions; 32 versus 3 with width 128 is none of those
and throws the exact `Invalid subtile broadcast type` in the log. The error
occurs in eager sampler precompile, before model or sampler trace capture.
This is not evidence of a trace allocator failure, cache failure, or inactive
RoPE failure.

B1 passed because its singleton height can broadcast to 32. B32 has equal
height. B2 through B31 are exposed to the same mismatch; inactivity is not
necessary to reproduce it.

## Smallest complete boundary repair

Normalize **sampler-bound** logits to logical `[1,1,32,65536]`, BF16 TILE,
vocab-sharded over the existing four-device mesh, with finite zero dummy rows:

```python
batch_rows = logits.shape[-2]
if not 1 <= batch_rows <= 32:
    raise ValueError("Sampling requires 1..32 slot rows")
if batch_rows < 32:
    logits = ttnn.pad(
        logits,
        padding=[(0, 0), (0, 0), (0, 32 - batch_rows), (0, 0)],
        value=0.0,
    )
```

Place this at the device-only model/sampler boundary before `_forward` returns
the logits captured as `trace_logits`. Both model warmup and capture must
execute it. Apply the same normalization immediately before standalone
prefill sampling, which runs before trace capture. A small shared helper avoids
divergent prefill and decode conventions.

Keep raw `prefill_forward(..., return_all_logits=True)` and other public
logits-returning interfaces at their documented logical prompt/batch shapes.
Do not globally pad `model.logits`, which also handles arbitrary prompt
lengths. Either retain raw logits in host-sampling debug mode, or slice padded
logits as `[..., :self.batch, :]` before the debug path reshapes/argmaxes them;
its existing `.reshape(self.batch, -1)` would mix rows after unconditional pad.

The existing Gemma4 implementation uses the same boundary:
`models/demos/gemma4/tt/model.py:2139` pads `on_device_logits` to batch 32 with
zero before returning them to the common sampler.

Why this fixes the complete shape contract:

- Both gathered indices and offsets become `[1,1,32,128]`; the add is equal-shape.
- Values, tie-break `_greedy_col`, penalty state, and indices share 32 rows.
- Sampling's C++ validation requires parameter lengths equal to `num_users`
  and output `[1,1,1,num_users]` (`sampling_device_operation.cpp:129–167`). The
  existing `[32]` params and `[1,1,1,32]` output now match directly.
- Greedy and sampled paths use the same row arrangement; the choice of
  temperature/k/p does not repair or change this shape obligation.

`ttnn.pad` is deliberate here. TILE padding calls
`fill_implicit_tile_padding(input, value)` and then a view with the requested
logical shape (`pad.cpp:478`). A plain view/reshape that exposes physical
padding without filling it does not establish valid dummy-row contents.
Do not pad the entire dummy row to `-inf`: tie-break max/absolute/boost and
sampling softmax need a defined finite row. Zero is the mature path's policy.

Do not change semantic greedy `k=1`, physical `max_top_k=32`, mesh distribution,
or switch to force-argmax to avoid this error. Those changes do not fix the
canonical boundary for intermediate batch sizes or sampled requests. Slicing
only the offsets to B would simply leave parameter/output/tie-break shapes
inconsistent. Never add an eager pad to `_replay` while the model trace is live.

## Inactive and padded rows

Keep rows in fixed slot order. Slot 1 in the B3 test remains logical row 1;
append dummy rows only after row B-1. Current model inserts zero hidden rows
for inactive slots and skips their decoder/cache work. Final norm and bias-free
head preserve zero logits for those rows. A zero-logit inactive row can produce
a valid but otherwise irrelevant sampled token, especially in nongreedy mode;
do not require inactive output token 0 unless an explicit output mask enforces
that separate convention.

The existing persistent 32-token buffer remains the sampler output and next
model input. The model consumes only its bound B rows and fixed active mask;
padding the terminal logits must not increase `batch`, allocate more cache
slots, alter active positions, or compact slot IDs. Negative position sentinels
and inactive cache preservation remain separate assertions.

## Smallest verify/refute sequence

1. **Sampler-only exact-shape oracle.** Extend the existing sampler probe to
   logical batches `1,2,3,16,31,32`, keeping TP4, vocab 262144, physical top-k 32,
   and persistent output width 32. Synthetic row r has a distinct known winner
   on a varying vocab shard. First run B3 without normalization to confirm the
   same add failure; then normalize. Assert normalized logical and padded
   shapes, row-r winners on every chip, and unchanged output buffer ID across
   two replay inputs. Allocate all controls before capture.
2. **Sampled-mode shape control.** With normalized B3/B32 logits, use distinct
   finite per-row top-4 candidate sets and `temperature=1, top_k=4, top_p=1`.
   Check each active row's samples belong to that row's candidates, all chips
   agree, and reseeding reproduces the sequence. Do not require consecutive
   stochastic draws to differ. This checks row isolation and sampled-mode
   compatibility, not distribution quality from a tiny sample.
3. **Original failing integration.** Rerun the exact reduced B3 mixed-slot
   test below. It must reach both traces, preserve feedback identity, leave
   slot 1 KV unchanged, and advance active positions once per replay.
4. **Boundary regressions.** Run B1 and B32 variants, then an all-active B2 or
   B31 variant to show this fix is not specific to inactive slots. Compare raw
   active logits before/after normalization; rows `[:B]` should be unchanged.

Proposed existing command, **not run by this investigation**:

```bash
OMP_NUM_THREADS=8 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 timeout 300 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_trace --batch 3 --output models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/trace_mixed_slots_retry.json
```

The reduced integration test's prefill logits are allocated before capture, so
retaining that local does not itself explain this failure. Allocation tracking
should remain enabled for correctness but disabled for comparative sampler
timing because its Python replay wrapper invokes garbage collection.

## Status

Source and stack trace identify the logical batch-height mismatch. No repair
has been applied by this investigation, and no device correctness or
performance result is claimed. Preserve this report as the hypothesis/evidence
boundary; attach the focused oracle and original regression results after the
terminal normalization is implemented.
