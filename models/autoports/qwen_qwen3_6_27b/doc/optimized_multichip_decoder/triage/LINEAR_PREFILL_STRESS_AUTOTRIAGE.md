# AUTOTRIAGE: linear-prefill slice CB overflow

## Diagnosis

- The row-major slice program under-allocates each input circular-buffer page when the last-dimension slice start is misaligned and the unpadded output row is already an exact multiple of the selected CB alignment. The NCRISC reader requests `unpadded_stick_size + misalignment`, but `compute_cb_size()` rounds only `unpadded_row_size_bytes`. In the observed case that produces a 130-byte NoC read into a 128-byte CB page, which Watcher correctly rejects as a circular-buffer overflow.

## Triage Evidence

- `logs/stress_prefill_linear_s128_16_watcher.log` records the first device-side failure on device 3, worker core logical `(0,0)` / virtual `(1,2)`, NCRISC, in `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp`.
- Watcher reports one noc0 DRAM-to-L1 unicast read of exactly 130 bytes targeting local L1 address `0x01d280`, with the explicit reason `NOC transaction overflows a circular buffer`.
- BRISC was concurrently running the matching row-major slice writer and all TRISCs were blank. This localizes the failure to the data-movement slice operation; it is not a matmul, collective, compute-kernel, or fabric wait.
- The process abort and host backtrace through `WatcherDeviceReader` are downstream consequences of Watcher intentionally stopping the invalid transaction. They do not establish a device-health or CCL root cause.
- The full-attention prefill stress passing while the linear-attention prefill stress fails is consistent with the linear path's substantially heavier use of Python-style slices in its convolution and logarithmic affine scan. It does not implicate TP4 itself: the failing kernel is an ordinary local slice reader on only one device/core.

## Source Evidence

- In `ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp`, runtime argument 2 is `unpadded_stick_size` and argument 5 is `misalignment`. The kernel sets:

  ```cpp
  uint32_t read_size = unpadded_stick_size + misalignment;
  ```

  and passes that complete size to `noc_async_read_sharded()` at the CB write pointer.
- When `misalignment != 0`, the kernel waits for that read and moves the requested row left by `misalignment`, retaining `unpadded_stick_size` useful bytes. Therefore the CB producer must have room for the leading alignment bytes as well as the useful row before the move occurs.
- In `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp`, `compute_cb_size()` computes the same `misalignment`, doubles the rounding alignment when it is nonzero, but then calculates:

  ```cpp
  cb_page_size = round_up(unpadded_row_size_bytes, alignment);
  ```

  The bytes actually written by the producer are absent from this expression. Doubling an alignment does not guarantee extra capacity: if `unpadded_row_size_bytes` is already divisible by the doubled alignment, the rounded result remains exactly the useful row size.
- The observed values prove this mismatch concretely. The reader's requested size is 130 bytes. The smallest and model-applicable decomposition is a 128-byte padded output row plus a 2-byte BF16 start misalignment; the host's current formula leaves a 128-byte CB page, while the kernel writes 130 bytes from its page start. More generally, regardless of the element-size decomposition, the kernel request exceeds the allocated page whenever `round_up(unpadded, alignment) < unpadded + misalignment`.
- Producer/consumer ledger:

  | Resource | Producer | Produced/requested span | Consumer | Consumed span |
  |---|---|---:|---|---:|
  | CB 0 reserved page | NCRISC slice reader | `unpadded_stick_size + misalignment` bytes before in-place compaction | BRISC slice writer | `unpadded_stick_size` bytes after compaction |
  | CB page capacity | Host `compute_cb_size()` | `round_up(unpadded_row_size_bytes, alignment)` bytes | Reader reservation/sanitizer | Must contain the producer's full pre-compaction read |

  The reservation count and writer loop can agree while the byte span of an individual reserved page is still too small. This explains why the failure is a sanitizer overflow rather than a producer/consumer page-count deadlock.
- The missing behavior was verified in the current source: neither `compute_cb_size()` nor another host-side guard includes `misalignment` in `cb_page_size`. The kernel also has no alternate scratch region for the prefix bytes.

## Downstream Effects

- Device 3 is the first observed victim because its scheduled slice lands on the vulnerable shape/start combination when Watcher polls. The evidence does not show that device 3, its DRAM bank, or core `(0,0)` is defective.
- The Watcher exception, process abort, and any incomplete later iterations are downstream. No evidence indicates an all-reduce, fabric, dispatch teardown, or recurrent-state synchronization hang.
- Without Watcher, the two excess bytes can overwrite CB-adjacent storage or the next logical page. A non-Watcher pass would therefore not make the operation safe; it could merely hide the invalid access.

## Proposed Fix

- Size the CB page for the reader's maximum pre-compaction write, not only the writer's useful row. The smallest source correction is to include `misalignment` before rounding while retaining the existing alignment policy:

  ```cpp
  const uint32_t cb_page_size =
      tt::round_up(unpadded_row_size_bytes + misalignment, alignment);
  ```

- Keep the runtime stick stride and writer size at the aligned useful-row stride / `unpadded_row_size_bytes`; the additional CB-page tail is capacity for the reader's temporary alignment prefix, not part of the output row.
- Add a focused row-major slice regression whose useful padded row is exactly divisible by the post-misalignment alignment and whose last-dimension start contributes a nonzero element-byte offset. It must run with Watcher and compare output values. The regression should reproduce the 128-byte useful-page plus 2-byte prefix boundary (or an equivalent exact-multiple case), because the existing 5-D misaligned regression only guards page-count divergence and does not guarantee this byte-capacity boundary.
- After rebuilding, verification should include: the focused single-device slice test under Watcher; the exact TP4 linear S128 command with warmup 4 and 16 measured iterations under Watcher/no-ETH; and the existing full-prefill stress as a negative regression. Confirm both Watcher cleanliness and PCC.

## Uncertainty

- The log does not contain TTNN operation history or runtime arguments, so it does not identify which individual Python slice in the linear prefill graph generated the 130-byte request. Capturing operation logging or a reduced repro would identify that call, but it is not required to establish the violated CB-size contract: the kernel name, exact byte count, sanitizer classification, and source equations already prove it.
- The exact CB page size is inferred as 128 bytes from the reported 130-byte request and the model's BF16 slice geometry; Watcher does not print the CB descriptor. A descriptor dump or focused regression should confirm it during the repair loop.
- The proposed patch still needs build and hardware validation. If it exposes a separate `num_read_per_barrier` or cache-hash issue, that would be a second bug; it does not invalidate this byte-span mismatch.
