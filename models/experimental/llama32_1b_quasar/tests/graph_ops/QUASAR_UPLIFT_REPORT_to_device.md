# QUASAR_UPLIFT_REPORT — `ttnn.to_device`

**Status: GREEN — uplift not applicable (host-side transfer, no device program factory).**

Audited per `docs/source/ttnn/ttnn/ai/quasar_porting.md` and the canonical audits
(`docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/{metal2_audit,quasar_audit}.md`).
No source files were changed.

## What `ttnn.to_device` actually is

`ttnn.to_device` (registered in `ttnn/ttnn/operations/core.py`, bound from
`ttnn/cpp/ttnn/operations/core/core.cpp::to_device`) is a thin wrapper around
`Tensor::to_device` → `tt::tt_metal::to_device` in `ttnn/core/tensor/tensor_ops.cpp`:

- If the tensor is already on device, it is returned as-is (assert: same mesh device).
- Otherwise the host tensor is written to a freshly allocated device buffer via
  `MeshCommandQueue::enqueue_write_tensor` (uniform case) or
  `non_uniform_data_movement::enqueue_write_tensor` (non-uniform mesh case).

Both paths are **command-queue DMA writes handled by the dispatch firmware**. There is:

- no `device/` subdirectory, no `*_device_operation.*`, no program factory
  (neither legacy `create_descriptor` nor Metal 2.0 `create_program_artifacts`),
- no kernels, no CBs/DFBs, no semaphores, no LLKs.

The Metal 2.0 gate and the Quasar-uplift audit (`quasar_audit.md` checks 1 and 2,
plus §7–§12 of the field notes) are defined over program factories and kernels; none of
their subjects exist for this op. Hence "uplift not applicable" rather than a port verdict.

## Does it ever trigger a device op on the way in?

No. `to_device` performs **no layout or dtype conversion**: layout conversion happens on
host beforehand (`ttnn.from_torch(..., layout=...)` / host `to_layout`) or on device
afterwards via a separate `ttnn.to_layout` call (which *does* dispatch tilize/untilize
programs — those are separate ops with their own audits, e.g. `test_untilize.py` /
`test_typecast.py` in this suite). Unlike some frameworks' "to device" helpers, there is
no implicit tilize or typecast inside this call.

## What the test exercises

`test_to_device.py` (generated from the llama32-1b graph capture) covers 4 distinct
captured signatures, 80 calls total — all row-major host→DRAM-interleaved writes with
identical output specs (pure transfer, no conversion):

| case | shape | dtype |
|---|---|---|
| 00 | [1, 128] | INT32 |
| 01 | [1] | INT32 |
| 02 | [1, 1, 1, 32] | UINT32 |
| 03 | [1, 32] | UINT32 |

**Note on UINT32 (field notes §7 "Quasar has Int32, no uint16/uint32 device format"):**
that limitation concerns the *compute* data-format path (unpack/pack formats in kernels).
A memory write of uint32 bytes has no format branch — `to_device` merely forwards the
`DataType` — so per §7 there is nothing to guard here; the concern is owned by whichever
downstream compute op consumes these tensors (in this model they are index/position
tensors consumed as raw pages by e.g. embedding / paged-cache readers).

## Recipe deliverable checklist

- **Files changed:** none. (Nothing tempted a move/rename.)
- **§7–§8 gotchas applied:** none — no kernels or factory exist to apply them to.
  Considered and ruled out: DM self-loop / sync-free DFBs, non-zero-init semaphores,
  `fifo_page_size`, `unpack_modes`, opt_level, hw_config — all N/A (no program).
  The uint32 point above is the only §7 item that even brushes this op, and it is
  explicitly a forward-only case with nothing to guard.
- **Parity claim:** trivially satisfied — zero diff, WH/BH/Quasar all take the same
  (dispatch-owned) path. Any Quasar behavior of `enqueue_write_tensor` is runtime/dispatch
  territory, not op territory.
- **Deferred / follow-ups:** none at the op level.

## Test commands (not run in this session — hand-run these)

```bash
# This op only:
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_to_device.py -v

# One case:
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_to_device.py -k 00_1x128 -v

# On the Quasar emulator (per suite README):
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_to_device.py -m emulator
```

*This report is a review artifact — leave it uncommitted and delete before any merge,
per the recipe's `QUASAR_UPLIFT_REPORT.md` conventions.*
