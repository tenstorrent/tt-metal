# Endurance normalization hypothesis: CPU substitutions

Inspection and CPU-only analysis of frozen `../mlp_norm_shard_dram.py.snapshot` and `../endurance_state_probe.json.pos*.pt`. No TTNN imports, hardware execution, runtime/test edits, or threshold changes. The captured input tensors are BF16; TT core is FP32 and HF core is BF16. Checkpoint revision is `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.

## Findings

**Output gated normalization is a secondary effect in these captures.** Replacing its numerical boundaries with the exact HF formula on saved TT core and TT Z barely changes attention PCC, and does not remove the strongest attention errors. Using HF core with TT Z instead gives attention PCC >0.999984 at every sampled position. Therefore upstream core differences dominate these attention errors; the output norm's own arithmetic is not the leading isolated repair.

**Input RMSNorm is a concrete earlier mismatch worth isolating on device.** The exact CPU HF formula reproduces all 2048 HF BF16 codes at every sampled position. Actual TT input norm differs in 341–607 codes, relative L2 0.00216–0.00315. At 3979, projecting only TT-normalized input through canonical CPU A/B projections changes 21/32 A codes and 25/32 B codes relative to HF. This establishes that input normalization alone perturbs gate projections; it does not establish that those perturbations cause the complete recurrent trajectory failure.

The parent reports that the new gate-rounding device control reduces seven failures to three (3032/5011/6595). That device result is separate from this memo's frozen original captures. This analysis does not claim to have captured the gate-corrected state.

## Reproduction

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/norm_cpu_probe.py
```

Artifacts: `norm_cpu_probe.py` and `norm_cpu_replay.json` in this directory. The script loads only Torch/safetensors and selected checkpoint weights. It computes Pearson correlations in FP64, reports BF16 code differences, and records per-head core errors. Reconstructing HF input norm and HF attention from captured HF core plus projected HF Z is **bit-identical** to the corresponding captured HF values at all ten positions. This validates the substitution formula and checkpoint selection.

Exact HF output gated norm used here:

1. Round core to BF16, then widen to FP32.
2. Compute FP32 RMS normalization with epsilon from checkpoint config.
3. Round normalized values to BF16; multiply BF16 norm weight, producing BF16.
4. Multiply by `SiLU(z.float())` and round final output to BF16.
5. Apply canonical BF16 output projection.

| Position | Actual TT attention PCC | TT core + TT Z, exact HF gated norm | HF core + TT Z, exact HF gated norm | TT input norm differing codes |
|---|---:|---:|---:|---:|
| 128 | .999901107 | .999886584 | .999991464 | 528 |
| 1024 | .999816417 | .999828986 | .999987137 | 381 |
| 2048 | .999925461 | .999936800 | .999992259 | 400 |
| 3979 | .995789063 | .995783142 | .999986099 | 579 |
| 5011 | .995302578 | .995235815 | .999984498 | 459 |
| 6595 | .995283905 | .995396067 | .999989752 | 341 |
| 7551 | .997780833 | .997795628 | .999986387 | 390 |
| 7668 | .999414191 | .999418531 | .999988468 | 371 |
| 7703 | .999046964 | .999109680 | .999993096 | 607 |
| 8157 | .999690948 | .999718844 | .999991641 | 554 |

The experiment also simulates an idealized current TT norm formula using FP32 core, weighted FP32 RMSNorm, BF16 normalized output and BF16 SiLU before multiplication. It tracks actual TT attention closely; this is not an exact native-kernel emulation. At 5011 and 6595, core head 23 relative L2 errors are respectively 0.35082 and 0.33683. These low-amplitude head discrepancies are amplified by RMS normalization. The core-source analysis belongs to the independent recurrence agent; this memo does not assign their cause.

## Precise next isolated device control

Change **only decode input normalization**, retaining canonical gate rounding, the public BF16 recurrent-cache boundary, and the current convolution/L2/recurrence/output norm. Use an interleaved FP32 SFPU decomposition:

```python
x32 = ttnn.typecast(x, ttnn.float32)
squares = ttnn.multiply(x32, x32)
variance = ttnn.mean(
    squares, dim=-1, keepdim=True,
    compute_kernel_config=self.compute,
)
inv_rms = ttnn.rsqrt(ttnn.add(variance, self.config.rms_norm_eps))
normalized = ttnn.multiply(x32, inv_rms)
result = ttnn.typecast(
    ttnn.multiply(normalized, self.w["input_layernorm.weight"]),
    ttnn.bfloat16,
)
```

The already-uploaded gamma is `1 + checkpoint_weight.float()`. Use interleaved input/output for this diagnostic so reduction does not depend on unsupported width-sharded semantics; restore the projection's existing width sharding afterward. This introduces no host arithmetic in measured decode. The root should verify the actual binding supports the named kwargs and keep the operation's default `fast_and_approximate_mode=False`.

Source basis: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:198–217` explicitly selects accurate FP32 SFPU SUM/AVG on Blackhole when FP32 destination accumulation is enabled and fast/approximate mode is disabled. FP32 binary operations use the SFPU path identified in the main AutoDebug report. Reduction order and rsqrt approximation remain potential tiny differences, so this is a precision-preserving discriminator, not a promise of bit-identical backend results.

**Unweighted native RMSNorm followed by SFPU gamma is not equivalent to this discriminator.** Generic `normalization/layernorm/device/kernels/compute/layernorm.cpp:314–336` performs normalized-input multiplication with `BinaryFpu::Mul` even without gamma. BF16 input also returns BF16 before the external gamma multiplication, unlike HF input norm's final-only BF16 cast. Widening the input would avoid that BF16 output but still expose TF32 source multiplication. The factory's special FP32-preserving finalizer excludes RMSNorm (`layernorm_op_multi_core.cpp:240–247,570`). Consequently this shortcut may improve part of the error but cannot isolate exact HF input-normalization arithmetic.

Run the full fixed 8192-token stream and continuation with unchanged per-token .995 output/contribution gates. An inputnorm-only failure reduction is useful evidence, but a complete explanation needs a matched capture on that repaired source. Do not infer that output-norm rounding is safe to discard permanently merely because its isolated effect is small here; preserve final PCC and deterministic repeated-run gates.
