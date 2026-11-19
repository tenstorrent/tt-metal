# v5 cumulative-sum output-alias investigation

Source-only result: the installed Python APIs cannot express physical input/output reuse for this FP32 ordered scan. No candidate or device probe is prepared, because adding `out=products` would retain the full peak allocation and misrepresent the intended optimization. No production, native, core-harness, or hardware changes were made.

Frozen inspected candidate: `v5_cache_writers_native_norm.py.snapshot`, SHA256 `1a593ec9a2522dd579194d5082ffe5e987a81fc87be287590bdec086e067d455`. Parent reports three passing long streams and 1.284772 ms decode; those are context, not measurements from this investigation.

## Public API versus actual allocation

`cumsum/cumsum_nanobind.cpp` exposes `out`, described as preallocated output. However, `accumulation_common.cpp:99–160` only validates that optional tensor, then calls `ttnn::prim::accumulation(..., std::nullopt, ...)` for every nonempty nonscalar input. The primitive therefore allocates a new prefix tensor. After the operation and postprocessing, the wrapper assigns the new tensor's storage to the optional output's tensor attributes. The adjacent TODO(#37807) explicitly says the operation needs refactoring to apply output directly to `optional_out`.

Consequently `ttnn.cumsum(products, -2, out=products, ...)` is storage rebinding after execution, not an in-place scan. The old input must remain alive during the scan while the new output exists. A separate preallocated `out` does not remove either allocation, and could retain additional allocation. The scalar/empty special case is irrelevant to `[2,1,1024,6208]` products.

The internal C++ `AccumulationDeviceOperation` does support a supplied output: `accumulation_device_operation.cpp:111–138` returns that TensorSpec and buffer. The public wrapper never forwards it. Source search found no Python binding for the internal accumulation primitive, no `cumsum_`, and no alternate public in-place entry. This establishes an API blocker; it does **not** establish that the native kernel is intrinsically unsafe for aliasing. Such a change would require separate native alias, reader/writer, cache-rebinding and trace-address validation outside this model-only task.

## Other exposed cumulative-sum APIs

- `ttnn.moreh.cumsum` aliases `ttnn.moreh_cumsum` in `ttnn/ttnn/operations/moreh.py:22`. Its binding exposes `output`, but `moreh_cumsum.cpp:10–17` forwards it to the same `ttnn::cumsum` wrapper. It therefore has exactly the same fresh-allocation/rebinding blocker. It also does not expose `disable_compensated_sum`; its call leaves the default compensated FP32 scan enabled, unlike this projection's required plain sequential scan. It is not an independent legacy native implementation in this checkout.
- `ttnn.moreh_cumsum_backward` likewise forwards to the same wrapper with reverse order, so it cannot help.
- Experimental DeepSeek `offset_cumsum` validates UINT32, ROW_MAJOR, rank2 expert-count inputs and allocates four DRAM results (`offset_cumsum_device_operation.cpp`). It cannot represent the FP32 product scan or its ordered arithmetic.

## Capacity and exact-order implication

Current PF serializes rows, using two K1024 blocks and exactly the existing block reduction. One `[2,1,1024,6208]` FP32 product or prefix occupies 50,855,936 bytes (48.5 MiB); their overlap is 101,711,872 bytes (97 MiB). Batching two rows without genuine aliasing doubles that overlap to 203,423,744 bytes (194 MiB), before coefficients, scan circular buffers, output slices, or other resident tensors.

Using the previously device-observed 110-bank grid, one row needs 12,416 tiles per buffer, at most 113 FP32 tiles per bank (462,848 bytes). Two rows need 24,832 tiles per buffer, at most 226 tiles per bank; two such buffers alone require 1,851,392 bytes per bank. This is an allocation accounting estimate, not a new allocation experiment. The H-scan factory additionally reserves four input and four output tiles (32 KiB/core for FP32). Real physical reuse would make a two-row experiment worth assessing, but public `out` does not achieve it.

The native H-scan's FP32 direct-to-Dest unpack and `disable_compensated_sum=True` preserve the already-qualified ordered arithmetic. Neither a different association nor compensated summation is proposed as a capacity workaround.

## Prior evidence and bounded handoff

`AUTOFIX_v3_singleton_cumsum_l1.md` already documents the public-output allocation behavior. The subsequent controls are distinct and remain preserved: `AUTOFIX_v3_prefill_cumsum.md` (batched DRAM prefixes exact but slower), `AUTOFIX_v3_prefill_serial_l1.md` (serial L1 complete adapter won), and `AUTOFIX_v4_prefill_weight_l1.md` (split-K transient L1 weight reuse exact but slower). None is reopened here.

A future native ownership handoff could address TODO(#37807), forward an explicit output through the public wrapper, and independently prove safe aliasing for this tile-axis scan. Required proof would include complete FP32/BF16 projection outputs, K1024 block boundaries, all batched rows, peak allocation, stable replay addresses and complete-adapter timing. That native change is outside the authorized model-only work and was not made. There is no remaining model-only output-argument substitution that earns a new device candidate on this hypothesis.

## Source fingerprints

- `ttnn/cpp/ttnn/operations/reduction/accumulation/accumulation_common.cpp`: `0c477d56cd34eb1d921eda4ef81a3d19fd2eb89f741ae703faff998a06cdc840`
- `ttnn/cpp/ttnn/operations/reduction/accumulation/cumsum/cumsum_nanobind.cpp`: `bde9ad6a7fcd59f183ec03d77a7be97303e9c3f50606ba1fd78c4fc7f42f0d03`
- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_device_operation.cpp`: `56ab292c4e6c4026e764806a7db31b791aac9f87832c487679074aacf077490b`
- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp`: `f4fe65a38a5baa5451e67c990187105eac58f6c0959eb2ed80d0ad50d454c7f5`
- `ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.cpp`: `b258a0d0b822638bc3f7d9fb4df24452131e4cb0f0885fa90a51549987582693`
- `ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/moreh_cumsum_nanobind.cpp`: `4a86cd30ed3b562ce999bb57e82bf174cbe83f327a539a61bba566fe78abdc20`
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum/device/offset_cumsum_device_operation.cpp`: `39c2baf7bd1b9519fd035a72bc0b5a94f8195af82722322c4ad0d65e07188242`
