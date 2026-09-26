# Actual sliding native head normalization

The test-only `probe_optimized_headnorm.py` changes exactly one constructed
decoder attribute: `precise_heads=False`. The runtime file is unchanged.
`--headnorm-mode precise` supplies the original-path control when needed.

`FusedDecoder.from_state_dict` binds `attention.normalize` to the decoder's
method. The probe verifies that binding before changing the attribute.
`OptimizedDecoder.normalize` applies its sharded policy only at hidden width
2816; head width256 falls through to `FusedDecoder.normalize`. With
`precise_heads=True`, that method uses FP32 square/mean/RSQRT/multiply.
With the flag false, it calls native `ttnn.rms_norm` on FP32 input and keeps
the learned Q/K gamma as the same external FP32 multiply. V remains unweighted.
The switch affects **both prefill and decode head norms**. Input/post/common
norm settings, QKV, rotary, cache, SDPA, routing and experts remain the current
production policy.

The earlier expanded-expert sliding profile attributes20 native operations to
head normalization (decode report IDs13–32). This control tests whether the
native normalization boundary passes the current actual-text policy; it does
not infer acceptance from a synthetic result or promise bitwise equality.
Native reduction can narrow operands internally despite FP32 storage, so the
complete PCC0.995 checks remain necessary. Prefill and all128/512 decode checks,
program-cache auditing and deterministic replay remain the original harness's
checks. Output metadata records the switch, external weight dtypes, compute
config, fixture provenance and both source hashes.

Parent-owned serialized commands are in `run_native_headnorm_controls.sh`:
headline4096/128, then stress1025/512 with the existing real checkpoint and
matching text-derived layer0 fixtures. The script stops if the headline fails
and refuses to overwrite existing result JSONs. Invoke from any directory:

```bash
bash /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/run_native_headnorm_controls.sh
```

Only Python compilation, Black formatting and shell syntax are checked by this
author. Device execution and final selection belong to the parent.

## Completed controls

The parent ran both commands on source
`b513a1b40988b33a359acb7d7809696eebf81a7197756e3ec3f943692182ab83`.

| Control | Prefill PCC | Minimum decode PCC | Host median us | Result |
| --- | ---: | ---: | ---: | --- |
| Sliding native heads4096/128 | .999155193385 | .995322952862 | 740.762 | Pass |
| Sliding native heads1025/512 | .999102403653 | .994418854936 | 748.973 | Fail |
| Full all-site hidden norms4096/128 | .999131800182 | .995055561052 | 832.355 | Pass |
| Full all-site hidden norms1025/512 | .999045017098 | .996240662329 | 809.089 | Pass |

The sliding stress failures are exactly position1257 at .994804004648 and
position1459 at .994418854936. All512 checks completed; both runtime audits,
the program-cache guard, and deterministic replay passed. The final assertion
reports PCC failure. This is a completed numerical rejection of the tested
both-phase native-head policy, not an initial API/config rejection. Retain
`precise_heads=True` for sliding.

The historical same-fixture precise result
`router_repair_hifi4_stress_layer0.json` has PCC .999633699034 at1257 and
.999762720080 at1459. Its source is `2a55a067...`, preceding the compact expert
and RoPE-layout integration, so it is **not an exact current-policy A/B**.
No precise-head1025/512 result on the identical `b513a1b...` compact source was
found during this audit. `headnorm_control_summary.json` records hashes,
positions, audits and this evidence limit. Final cumulative validation must
still test the retained precise policy.

## Precision boundary in source

`DecodeAttention.__init__` requests HiFi4, non-approximate math and FP32
destination accumulation. The native layernorm factory retains FP32
intermediate buffers (`layernorm_op_multi_core.cpp:243`), but its ordinary
RMSNorm arithmetic still uses FPU source registers:

- `device/kernels/compute/layernorm.cpp:270` squares with `mul_tiles`.
- The same file at336 scales the original input with
  `mul_tiles_bcast_cols` after the variance reduction and reciprocal square root.
- Factory lines442–466 explain the FP32-to-TF32 Src path and install an
  UnpackToDest precision alias only for Welford **non-RMSNorm** cases.

FP32 storage/accumulation therefore does not guarantee the manual path's
operand precision. The retained precise path computes square and scaling with
FP32 SFPU binary operations and keeps its explicit FP32 intermediate boundary.
Q/K learned gamma remains the external FP32 multiply in both policies. These
are concrete arithmetic differences that can perturb attention and subsequent
expert selection; no intermediate tensors or route IDs were captured in this
control, so the exact amplification point is not claimed as measured.

The retained path's SFPU dispatch is also directly visible in the native CSV:
sliding decode ID14 (square) uses `eltwise_binary_sfpu_no_bcast.cpp`, ID17
(ADD+RSQRT) uses `eltwise_binary_sfpu_scalar.cpp`, and ID18 (scale) uses
`eltwise_binary_sfpu.cpp`, all with `DataType::FLOAT32`. Equivalent triplets
appear for K and V. This confirms the manual path is not merely an FP32 output
label on the same FPU product used by native RMSNorm.
