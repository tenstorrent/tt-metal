# Native runtime fallback audit

Scope: both public forward methods in `tt/functional_decoder.py`, with DEVICE
BF16/FP32 activations/state, comparison mode disabled. Setup (`from_state_dict`,
`_upload`, state construction), HF oracle, host input refresh and result reads
are outside the measured passes. `runtime_audit.json` enumerates all16 reachable
Python methods and32 explicit TTNN calls. Tensor slicing also dispatches TTNN.
No reachable runtime torch/import/from_torch/to_torch/as_tensor call remains.
The harness patches from/to_torch to raise during measured prefill and decode
capture; replay executes the captured device command stream.

Native composite paths inspected at checkout `a0b0bd54e198d81676662b5a2d3a489e29bc9ab9`:

| Path under `ttnn/cpp/ttnn/operations/` | Finding for these inputs |
|---|---|
| `creation/creation.cpp` | Small ROW_MAJOR int32 `full` builds host data and uploads. Removed from runtime; AST audit explicitly forbids `ttnn.full`. |
| `full/full.cpp` | `moreh_full` calls `prim::full` device operation. The padded1×32 int32 construction avoids a separate native small-row buffer overrun; see `FILL_AUTOFIX_DIAGNOSIS.md`. |
| `data_movement/pad/pad.cpp:203` | Host padding is guarded by storage type != DEVICE. Current operands take device primitive/composite branches. |
| `data_movement/repeat_interleave/repeat_interleave.cpp:145` | The path called “fallback” converts device layout, unsqueezes, concatenates, reshapes and restores layout. No CPU numerical fallback or upload. |
| `data_movement/slice/slice.cpp:186` | Host branch applies only to host source. Device slicing uses device kernels or metadata views. Padded position slice has aligned source/destination buffers. |
| `data_movement/reshape_view/reshape.cpp:760` | Position `(1,B)` to `(B,)` preserves the last dimension and uses a device metadata view. Other activation reshapes use device views/data movement. |

Linear/matmul, norms, elementwise/reduction operations, paged cache writes and
SDPA consume device tensors and dispatch device kernels. Program configuration,
shape arithmetic and chunk scheduling run on the host; they do not evaluate
model numerics or transfer intermediate tensors to/from the host.

The first native-fill repair failed Watcher with a64-byte NoC transfer from
an undersized tiny-row buffer. Final `full_contract_padded_fill` passes Watcher
and HF continuation checks; `full_trace_reuse_padded_fill` preserves one trace
across those eager paths. The failed experiment remains in the audit history.
Profiler collection drains occur only between passes and outside signposts.
