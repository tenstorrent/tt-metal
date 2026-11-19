# Two-BFP8 residual attention projection

This candidate substantially reduces real-weight quantization error while **keeping both stored weight tensors BFP8**. It is much stronger than the preceding column-scaling candidates. It costs a second matrix multiplication and additional weight/output storage; no device correctness or speed claim is made.

CPU evidence: `residual_quantization_cpu.json`. Reproducer: `candidates/residual_quantization_probe.py`. Uninstalled setup/runtime prototype: `candidates/residual_projection.py`. No main-source edits or hardware calls were made for this experiment.

## Exact transformation

For the final `[K,N]` transposed/packed matrix, including its ordinary zero padding:

```text
W0 = BFP8(W)
R  = FP32(W) - dequantize(W0)
W1 = BFP8(R)
y  = BF16(FP32_matmul(x,W0) + FP32_matmul(x,W1))
```

The correction is derived entirely from checkpoint weights. It is not fitted to failing outputs and does not change the model algebra. State layout, cache publication and nonlinear order remain unchanged. This is a higher-effective-precision representation using two BFP8 terms, so stage8 should record both storage and effective precision rather than calling it ordinary single-BFP8 cost/precision.

The residual **must** be computed from the exact first quantized tensor. The setup prototype packs a host BFP8 W0, reads its dequantized values at setup, packs the residual as BFP8, then lets the caller transfer/shard both identically. The host packer's HAL/driver caveat from `AUTODEBUG_column_scaling.md` still applies: run setup only inside the caller's device reservation. Runtime has no Torch/host conversion.

## Real-weight CPU results

The probe uses actual layer0 packed QKV/Z/B/A weights and captured HF/TT input-norm values at steps19/58/99, with the source-derived BFP8 quantizer from the column-scaling probe. CPU matmuls accumulate FP32; they do not simulate device HiFi2 arithmetic.

Weight relative RMS error decreases from **.00777530 to .0000101620**, approximately765x. The corrected BF16 outputs are within approximately40–87x of the unquantized BF16-weight CPU control for nonzero-error groups. B is bit-equal in these samples, which must not be generalized to arbitrary inputs.

| Projection | Single-BFP8 relative output error | Two-BFP8, FP32 sum then BF16 | Two-BFP8, BF16 branch outputs first |
|---|---:|---:|---:|
| Q | .00821139 | .00010452 | .00279299 |
| K | .00846098 | .00009689 | .00290250 |
| V | .00849098 | .00020915 | .00287595 |
| Z | .00804110 | .00018948 | .00287733 |
| B | .00784800 | 0 | .00289077 |
| A | .00826141 | .00013178 | .00301693 |

These errors compare against unquantized checkpoint weights on **the exact same captured TT norm input**. They isolate weight representation rather than conflating input normalization differences.

The two-BFP8 end-to-end projection errors versus captured HF outputs remain approximately .00305–.00346, almost exactly matching the BF16-weight CPU control's .00304–.00346. This is the existing TT input-norm discrepancy, not failed residual correction. Per-step Q errors with corrected weights are .003619/.003441/.002339; the BF16-weight control gives .003617/.003443/.002337. Thus pairing this candidate with faithful input normalization matters. Fixing the state SFPU precision alone does not remove that upstream floor.

The full per-step Q/K/V/Z/B/A rows, including errors against HF and isolated quantization, are retained in the JSON. No full-sequence pass follows from these three local projection checks.

## Minimal legal implementation route

1. Initially correct the entire packed GDN projection to establish the strongest precision control; leave every MLP weight in its original single-BFP8 policy. Use the same normalized activation and existing packed offsets for both terms. This changes neither attention weights to BF16 nor decoder state semantics.
2. Both `ttnn.linear` calls must request `dtype=ttnn.float32`, followed by one `ttnn.add(..., dtype=ttnn.bfloat16)`. Calling the existing `_linear` twice unmodified is insufficient because it packs BF16 internally; the table quantifies that lost benefit. Keep program/compute configs explicit and preserve true FP32 partial accumulation. If device HiFi2 error dominates after quantization correction, measure HiFi4 as a separate fidelity control.
3. Source inspection supports an FP32-output route: `matmul_device_operation.cpp` accepts BF16 or FP32 output; the DRAM-sharded factory derives `output_data_format` from output dtype and sizes output/intermediate buffers accordingly. Its FP32 partials already use Float32 format. This is source-supported legality, not a hardware-tested configuration. FP32 output doubles output bytes; verify L1 capacity and legal subblocks rather than reusing a configuration blindly.
4. Reuse the input shard for both matmuls. If output shards/configs match, sum while sharded and perform the usual final transfer once. Avoid separate reshard/untilize paths for the correction. Check that two live FP32 outputs fit; releasing branch outputs after the sum is a normal lifetime optimization, but must respect trace capture.
5. For a simpler first correctness control, use legal interleaved FP32 outputs and sum; then tune sharding. Do not reject the material optimization after a first API/config error.
6. Once full correction passes, measure selective residuals. QKV correction targets convolution/state direction; Z targets the post-state gate; B/A are tiny and cheap to correct separately. A corrected QKV/Z packed residual plus separate small B/A projections may save correction padding but adds dispatch. Keeping packed zero correction columns still spends matrix bandwidth/compute, so it is a correctness ablation rather than automatically a speed win.

A single wider packed matrix `[W0 | W1]` is another legal algebraic candidate, but doubles output width and requires slicing/adding corresponding halves. Compare it to two well-tuned FP32-output matmuls: fewer launches alone do not prove a latency win, and width-sharded output halves may require extra movement. Keep whichever correct variant actually wins traced decode.

## Costs, validation and limits

The complete packed projection has5120×16512 logical elements. A BFP8 standard tile consumes1088 bytes for1024 values, so one complete term occupies89,825,280 bytes before any additional DRAM-shard padding. The residual adds another such term; two terms are roughly BF16 storage plus block-exponent overhead. This can change capacity in a complete decoder context; measure real allocation and retain the advertised context contract unless a hard physical limit is demonstrated.

Two-BFP8 is a promising precision control, not a free optimization. Full-sequence .995, fused/HF state equivalence, non-aligned prefill, watcher/repeated runs and warmed/traced speed remain required. Especially measure the corrected prefix: residual attention precision should be applied consistently during prefill and decode so the test does not retain the original prefix state error.

Reproduction:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.doc.optimized_decoder.candidates.residual_quantization_probe
```

The CPU probe ran successfully and both candidate helpers syntax-compile. No device probe was run by this subagent.
