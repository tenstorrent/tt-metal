# Decode-only exact input norm in interleaved L1

Source-only control; no device execution and no live fused-source edits. The root reports outer-layout norm passed all three streams at 1.8245 ms, and the subsequent projection-product L1 control reached 1.636934 ms in a short benchmark. Those are root-reported measurements, not evidence for this norm candidate.

`norm_l1_single_token.patch` changes only the exact input-norm method. Its base source SHA256 is `a4a719ab2be30e88520c45975da82bbda6e824f372be45a64891411c76d2cb77`. It deliberately carries no complete source snapshot, so applying it cannot revert the root's concurrent projection-product L1 work.

## Placement hypothesis

For linear-attention single-token rows, cast the existing input directly to FP32 interleaved L1. Keep the explicitly shaped `ttnn.empty((1,1,1,32))` result in L1, and pass `memory_config=L1` to `fast_reduce_nc`. Every following temporary inherits L1; keep the existing final `to_memory_config(result, original_memory)` return. Gamma remains a DRAM RHS. All arithmetic, dtypes, expression ordering, BF16 round/widen boundaries, reduction schedule, logical shapes, and public cache/context interfaces remain unchanged. Rows greater than one retain the current DRAM prefill path.

The explicit reduction memory argument is essential: unlike the other primitives, `fast_reduce_nc` defaults to DRAM and internally allocates its first-stage output even when the final output is supplied. An L1 `output=` alone would leave that first intermediate in DRAM.

## Source evidence

Paths below are relative to `ttnn/cpp/ttnn/operations/`:

- `eltwise/binary_ng/device/binary_ng_device_operation.cpp`: absent an explicit/preallocated output, output memory inherits input A. Thus the final L1 normalized-input × DRAM gamma result stays L1.
- `eltwise/unary/unary.cpp` and `copy/typecast/typecast.cpp`: output memory defaults to the input's memory configuration. This includes unary-chain round/widen and sqrt/reciprocal.
- `reduction/accumulation/accumulation_common.cpp` and `device/accumulation_device_operation.cpp`: cumsum preserves input/output memory through preprocessing, compute and postprocessing. Validation permits interleaved input without restricting the buffer to DRAM.
- `data_movement/reshape_view/reshape.cpp` and `data_movement/slice/slice.cpp`: omitted memory arguments resolve to input memory. Any necessary tilize/untilize/reshape conversions use that resolved placement; this control does not claim to remove them.
- `creation/creation.cpp`: `ones_like` inherits memory from its reference tensor.
- `experimental/reduction/fast_reduce_nc/fast_reduce_nc_nanobind.cpp`: binding defaults to `DEFAULT_OUTPUT_MEMORY_CONFIG` (DRAM). `fast_reduce_nc.cpp` forwards it to every stage. `device/fast_reduce_nc_device_operation.cpp` accepts interleaved input/output and uses preallocated final output when supplied; no DRAM-only validation blocks L1.

The outer-layout squared tensor occupies 64 FP32 tiles (256 KiB total, distributed interleaved), and the original single-token FP32 vector occupies 64 tiles because its tiled height pads to 32. Many scalar intermediates occupy a tile each. These are allocation sizes, not a measured peak or a proof against L1 fragmentation. Combined trace liveness with the now-L1 projection products needs device validation. Runtime `empty` replaces the op's implicit allocation; no persistent shared scratch or new state lifetime is introduced.

## Validation and limits

CPU AST parsing succeeded for the candidate. Device legality, exact output codes, trace allocation stability and speed remain unverified. Root should first run `tests/input_norm_actual_probe.py` on the applied candidate (51 sensitive positions across all three streams), then the existing three-stream endurance and paired warmed traced benchmark. Require unchanged norm/mean codes, unchanged decoder gates, and a measured improvement before keeping it. A profiler should check that the small norm intermediates actually use L1 and identify any added final copy cost. This is a one-path placement experiment, not a datatype sweep or a context-capacity change.

Command after root applies the patch to its current candidate:

```sh
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.input_norm_actual_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/input_norm_l1_actual_device.json
```
