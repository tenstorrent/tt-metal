# QUASAR_UPLIFT_REPORT — `ttnn.unsqueeze_to_4D`

**Status: GREEN — uplift not applicable (host-side view wrapper; no device program factory).**

Per the Quasar-uplift recipe (`docs/source/ttnn/ttnn/ai/quasar_porting.md` +
`docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/{metal2_audit,quasar_audit}.md`),
the Metal 2.0 gate and the Quasar-uplift audit apply to ops with a device program factory.
`unsqueeze_to_4D` has none — there is nothing to audit or uplift here.

## What the op is

`ttnn::unsqueeze_to_4D` (`ttnn/cpp/ttnn/operations/core/core.cpp:20`) is a plain host-side
helper, registered in Python via `ttnn/ttnn/operations/core.py`:

- rank == 4 → returns the input tensor unchanged;
- rank > 4 → throws;
- rank < 4 → `ttnn::reshape(tensor, logical_shape.to_rank(4), padded_shape.to_rank(4))`,
  i.e. prepend leading 1-dims.

It is not a `ttnn::prim::` device operation: no `create_program_artifacts`, no
`create_descriptor`/`ProgramDescriptor`, no kernels, no CBs/DFBs, no semaphores.

## What the test actually exercises

`models/experimental/llama32_1b_quasar/tests/graph_ops/test_unsqueeze_to_4D.py` replays the
4 distinct captured call signatures (82 calls total) from the Llama-3.2-1B graph capture:

| case | in shape | dtype/layout | out shape |
|---|---|---|---|
| 00 | [1, 32, 64] | BF16 / TILE, interleaved DRAM | [1, 1, 32, 64] |
| 01 | [1, 32, 2048] | BF16 / TILE, interleaved DRAM | [1, 1, 32, 2048] |
| 02 | [32] | UINT32 / ROW_MAJOR, interleaved DRAM | [1, 1, 1, 32] |
| 03 | [1, 1024, 2048] | BF16 / TILE, interleaved DRAM | [1, 1, 1024, 2048] |

For every case the last dim and (logical) second-to-last dim are unchanged and the memory
config is preserved, so `ttnn::reshape`'s view predicate
(`ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:613`, `this_is_view`)
holds and the call resolves to `PerformView` → `ttnn::experimental::view`
(`ttnn/cpp/ttnn/operations/experimental/reshape/view.cpp`) → `tt::tt_metal::view` — a
**zero-cost, metadata-only TensorSpec change**. No device program is created or enqueued
for any captured signature; the test exercises host-side shape/spec plumbing only.

## Underlying device ops (owned by other audits)

`unsqueeze_to_4D` delegates entirely to **`ttnn::reshape` (reshape_view)**. Only on inputs
that fail the view predicate (last-dim change, sharding/L1 mismatch, tile-padding conflict —
none occur in this capture) does reshape fall through to real device work:

- `ttnn::prim::reshape_view` — `ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/`
  (its own Metal 2.0 / Quasar audit scope);
- the row-major reshape path (`reshape_rm`, same op family).

Any Quasar-uplift work for those paths belongs to the `reshape_view` op audit, not to this
wrapper.

## Files changed

**None.** No source edits were made; this report is the only artifact.

## Parity claim

Trivially satisfied: zero diff ⇒ WH/BH behavior unchanged; Quasar behavior is whatever
`reshape_view` provides.

## Test command (not run this session — hand-off per recipe)

```
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_unsqueeze_to_4D.py -v
```

Run on Quasar per the craqsim runbook environment; on WH/BH as-is for parity. Note that
since every captured case is a zero-cost view, a Quasar pass here validates host-side
tensor-spec plumbing only, not any device kernel.
