# V6 bounded split-product FMA CPU proof

A model-local arithmetic candidate now matches the CPU single-rounding FMA oracle on **205,814,748 bounded-domain triples** from all 32 heads of the actual layer34/seed154 prefix. The bounded domain tested is `a,b == 0 or 2^-30 <= abs(a,b) <= 2^30`, and `c == 0 or 2^-60 <= abs(c) <= 2^60`. There are zero FP32 bit differences in that subset, including signed zero. A separate deterministic two-million-case randomized/cancellation/midpoint-adjacent experiment also has zero bit differences. This is CPU evidence under separate IEEE round-to-nearest operations, **not a verified SFPU implementation or a proof for arbitrary FP32 inputs**.

[Source](../../../tests/v6_fma_expansion_proof.py), [machine-readable results](v6_fma_expansion_proof.json), and [raw log](v6_fma_expansion_proof.log) preserve the implementation, source/input hashes, exact counts, and actual midpoint triples. The inputs are `v6_chunk64_sfpu_fma_compensated_device_v2.json.cpu_matrices.pt`; the device matrices were not used as arithmetic operands or as an oracle. No hardware, TTNN import, production edit, native edit, or threshold change was performed.

## Algorithm and necessary midpoint correction

The embedded C++ helper performs:

1. Veltkamp splitting with 4097 on each operand, then Dekker reconstruction of the separately rounded product and its residual.
2. General TwoSum of that rounded product and the accumulator.
3. A preliminary rounded sum of the high and low terms.
4. An exact short expansion representing `a*b+c`, using general TwoSum, followed by an expansion subtraction of the preliminary result.
5. A comparison of that residual with the signed half-spacing to the adjacent FP32 value. A strictly larger residual moves to that neighbor; exact midpoint moves only when the preliminary result has odd significand parity. Exact zero has an explicit sign rule.

The `corrected` helper is called **at every increasing-K step**. It does not carry a compensated accumulator across the dot product, which would change the CPU recurrence. `-ffp-contract=off` and volatile rounded add/subtract/multiply wrappers preserve operation boundaries. `std::fma` is used only for the independent CPU oracle and counterfactual controls.

The midpoint step matters in practice: three actual-prefix triples required it. The JSON includes these exact `(a,b,acc)` inputs and results for a future device tile probe. A constructed bounded example is:

```text
a = 1 + 2^-23
b = 1 - 2^-23
c = 16,777,218
preliminary sum = 16,777,220
corrected result = std::fma result = 16,777,218
```

Simply adding a rounded residual would therefore be an invalid replacement. The exact product lies just below one, so prematurely rounding that small negative tail changes the midpoint decision.

## Actual full-matrix results and limits

There are 234,881,024 actual-prefix update triples in total; the 29,066,276 outside the conservative bounded domain are also evaluated and explicitly reported. Those include subnormals and cancellation where the unscaled split-product transform loses tiny terms. Across these unbounded triples there are 60,098 numerical mismatches. This helper must not be described as an exact general FP32 FMA.

| Actual matrix result, all 32 heads | Split-product sequential result versus std::fma | Flush-subnormal FMA control versus std::fma |
| --- | --- | --- |
| Gram, chunks 0 and 1 | Bit exact | Bit exact |
| Transformed V, chunks 0 and 1 | Bit exact | Bit exact |
| Transformed K, chunk 0 | 393 FP32 differences; zero BF16 differences | 836 FP32 differences; 507 BF16 differences |
| Transformed K, chunk 1 | 402 FP32 differences; zero BF16 differences | 792 FP32 differences; 474 BF16 differences |
| Incoming-state contribution, chunk 0 | Bit exact | Bit exact |
| Incoming-state contribution, chunk 1 | 634 bit / 582 numerical differences; zero BF16 differences | 997 bit / 945 numerical differences; 442 BF16 differences |
| Final state-update product, chunks 0 and 1 | Bit exact | Bit exact |

The largest split-product numerical discrepancy in any matrix output is approximately `1.148e-41`; remaining differences are subnormal-scale. These matrix experiments independently start from canonical captured operands. They **do not propagate every modified intermediate through the whole chunk graph**, and exact final update products in this table do not establish exact final cache state for a modified graph. The explicit FTZ control flushes subnormal inputs, accumulator, and FMA result at each step; it is a counterfactual arithmetic experiment, not a simulator of every SFPU operation.

The normal-domain experiment avoids split overflow and product-residual underflow by its conservative input bounds. Overflow, NaN, infinity, broad exponent gaps outside these bounds, and true subnormal rounding are not covered by the claim. In particular the code's `half = (nextafter(r)-r)*0.5` cannot represent half the smallest subnormal spacing; a power-of-two normalization or integer mantissa/exponent fallback would be required to cover that case generally. No such fallback is implemented or authorized as a host runtime operation.

## Model-local device path to test next

All primitive steps have a plausible device expression: separate FP32 add/subtract/multiply, comparisons, bit parity, and adjacent-float stepping. Existing `ttnn.nextafter` invokes binary-ng (`ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp:126`); the Blackhole SFPU helper implements integer-bit stepping. An explicit TTNN graph would be a correctness probe with substantial launch overhead. An inline `SOURCE_CODE` SFPU compute kernel could keep the arithmetic in device vector lanes and avoid that overhead, with fixed expansion slots and predication in place of the CPU helper's scalar zero-elimination branches. FP32 direct-to-destination unpack is required; SRCA/B narrowing would invalidate the proof before arithmetic starts.

The first hardware experiment should compare split halves, product residual, TwoSum residual, and corrected result independently on the saved midpoint triples, representative normal operands, signed zeros, and explicitly unsupported subnormal cases. Ordinary SFPU multiplication/addition rounding and compiler non-contraction must themselves be verified; this CPU experiment does not establish their equivalence. Scratch/register pressure and latency remain open. A passing primitive still needs whole-chunk, nonaligned, continuation, stress, watcher, and performance validation before decoder integration.

Command:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fma_expansion_proof > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_fma_expansion_proof.log
```

## Prepared primitive probe, unexecuted

[The minimal inline SFPU probe](../../../tests/v6_fma_sfpu_primitive_probe.py) is prepared, with [frozen source-only metadata](v6_fma_sfpu_primitive_source_only_v2.json) and adjacent compute/writer snapshots. It emits source by default and opens hardware only with `--run-device`. No hardware was used to prepare it. Python syntax and CPU fixture/graph generation pass; C++ compilation and output mapping remain unverified.

The probe evaluates five boundaries—rounded product, product error, TwoSum high, TwoSum error, and preliminary result—on the constructed midpoint, all three captured actual midpoint triples, a normal cancellation case, signed zero, and an explicitly unsupported subnormal. It loads exact compile-time FP32 constants into SFPU vector lanes, writes/reloads each arithmetic result through FP32 DST scratch to force visible operation boundaries and shorten live register ranges, then exports the first element of one tile per boundary. It deliberately does not yet implement the midpoint correction, general input unpack/dataflow, or GEMM. The generated instruction stream must be inspected to confirm the expected separate arithmetic boundaries. The constant-input design isolates SFPU arithmetic and cannot qualify a full input-unpack path.

Parent-controlled hardware command, **not run by this investigation**:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_fma_sfpu_primitive_probe --run-device --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_fma_sfpu_primitive_device.json
```
