# Direct QKV integration

Applied only after the parent confirmed hardware idle. Runtime SHA256:
`3c52d46cedcde6473f0b02c4292809a235354c024e4b01c478c1456f8499745b`.
The complete source compiled before one atomic replacement. The new class passes
a source AST check excluding lane multiplication, residual decomposition,
row reduction and host tensor transfers from decode. No device commands were
run by this investigation. `direct_qkv_runtime.patch` retains the applied diff;
do not apply it again.

`qkv_direct_dtype=None` leaves existing defaults unchanged. Setting it to
`ttnn.float32` (JSON override `"float32"`) installs static `DirectQKV` after
the existing projection's weight quantization. BF16 is an explicit control,
not the selected actual-input winner. The wrapper has no test imports,
monkeypatches, source execution or host fallback.

| Path | Decode contract |
| --- | --- |
| Packed interleaved | Selected packed weight, inherited grid/K/subblock/fidelity, one padded M tile, FP32 output in original decode memory. |
| Separate interleaved | One inherited legal program per Q/K/V weight (Q/K for tied-KV), each with M=1 tile; concatenate FP32 outputs. Auto full-attention separate uses subblock1 so narrow K is legal. Explicit grid/subblock controls remain available. |
| Packed DRAM-sharded | Convert the FP32 operand to existing width-sharded input memory; inherited readers/K/output geometry and selected DRAM weight; FP32 width-sharded output; convert to original decode memory and crop padded N. |

All paths duplicate tied K into V only after projection/crop, as before. Prefill
delegates to the existing `LanePartitionQKV` object, preserving dense configs,
quantized phase weights and tied-K behavior. Separate+DRAM keeps the existing
unsupported factory contract. `qkv_lanes` must remain nonzero for setup; the
wrapper bypasses its masks and sums at runtime. Policy reports effective
lanes0/termsnull and a `qkv_decode` object containing backend, actual selected
weight/input dtype, weight memory, program geometry and FP32 output.

## Evidence supporting the candidate

The **test wrapper**, before production integration, passes recorded actual
1025-prefill/512-decode inputs with FP32 direct QKV:

- Sliding: min PCC .995555484296, host median1094.5171us
  (`actual_direct_qkv_float32_base_stress_layer0.json`).
- Full: min PCC .996879320480, host median1186.8371us
  (`actual_direct_qkv_float32_base_stress_layer5.json`).

Direct BF16 fails sliding base and higher expert-precision controls
(.99418344/.99454346); full BF16 passes but is approximately1us slower. These
are real-input decisions. Older Gaussian FP32-matmul failures do not veto the
passing actual-input FP32 path. Production headline/stress, geometry and
DRAM/BFP8 compatibility validation remain owned by the parent; the test-wrapper
results do not yet prove the new static path.

## Separate memory-cleanup opportunity

No allocation cleanup is mixed into this integration. For packed interleaved
QKV, the factory currently typecasts the same original packed weight twice:
once into `LanePartitionQKV.weights[0]`, once into its source `.weight` used by
prefill. When both phases use the same quantized dtype, assigning one newly
quantized tensor to both references is a straightforward later deduplication.
Do not apply that identity substitution to separate weights or padded DRAM
weights: their physical storage/layout differs from the prefill packed tensor.

`BroadcastQKV.__call__` uses `.rows` only for logical M=1. DirectQKV intercepts
that case, while all delegated prefill calls use `.weight`; tied-K prefill has
the same branch. Thus regenerated FP32 broadcast rows and the lane mask are
unused on the direct path. Releasing them needs an ownership/alias check and a
new allocation/context record; retaining them costs setup/persistent storage,
not per-token movement. A single packed BFP8 copy occupies about24.5MB sliding
or27.6MB full, and the corresponding FP32 rows about92.3MB/103.8MB. These are
tile-payload estimates, not measured allocator savings.
