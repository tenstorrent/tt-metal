# Functional recurrent-state oracle is invalid on the real-text S33 fixture

The caller's new independent HF comparisons establish a defect in the inherited **functional control's prefill state**, not a permitted numerical difference that optimized code should reproduce. Replacing this invalid control for the optimized stage is justified while retaining all .995 thresholds, the fused-stage semantic contract, independent HF checks, and historical failure evidence. It does not resolve or waive the separately failing long random-input decode sequence.

This investigation used existing caller-generated device evidence and CPU/source inspection only. No device was opened and no decoder/test gate was changed by this report's author.

## Independent state evidence

All rows use the exact same real-text embedding fixture `text_activations/text_l0_b32_s33.pt`, B32/S33, layer0, pinned checkpoint, then the fixture's `d` and `d2` tokens. The selected-user artifacts for functional and fused have **bit-identical input tensors and HF state tensors**, directly checked for users0,27,30. JSON provenance records each decoder source, runner and environment. Snapshot hashes are retained in `text_state_oracle_sources.json`.

| Boundary | Functional state vs HF PCC | Functional state relative RMS error | Fused state vs HF PCC | Fused state relative RMS error | Minimum fused per-head state PCC |
|---|---:|---:|---:|---:|---:|
| Immediately after prefill | .0170252 | 10.1776x | .99998763 | .005585 | .99995956 |
| After decode1 | .0631003 | 9.0462x | .99998661 | .006123 | .99995628 |
| After decode2 | .1005722 | 8.5224x | .99998478 | .006746 | .99995051 |

The functional prefix output still has PCC .998459 versus HF, while fused has .999994. A high output PCC cannot certify the full recurrent matrix: the token exposes one projection through q, followed by norm/gate/projection and a residual branch. State errors can remain hidden in poorly observed directions.

This discrepancy already exists **before any decode**. It is therefore not explained by a decode trace, state restoration, different numbers of updates, or the old comparison being taken after one decode. The old optimized-versus-functional state PCC .06303984 in `compact_profile_fixed_bfp8_0.json` matches the regime where functional itself has HF PCC .0631003.

Head24 contributes95.884% of functional's total prefix-state squared error,96.211% after decode1, and96.365% after decode2. For user1/head24, prefix state RMS is30.326 versus HF .258484, relative error117.786x, PCC -.459667. The same failure repeats for users5,9,...29 because the input fixture repeats four prompt streams. This repeat pattern is consistent with the fixture, not evidence of a four-user indexing bug.

Saved selected-user tensor maxima for head24 after prefix:

| User | HF max absolute state | Functional | Fused |
|---|---:|---:|---:|
| 0 | 22.5 | 235.982 | 22.538 |
| 27 | 17.625 | 1448.373 | 17.622 |
| 30 | 20.0 | 391.834 | 20.041 |

The optimized independent-HF run subsequently completed in `text_state_optimized.json`: recurrent-state PCC is .99997263/.99997183/.99997011 at prefill/decode1/decode2, relative RMS error .008995/.009529/.010080. Its minimum per-head state PCC is .99990863/.99992319/.99993844. Thus **both fused and optimized agree with HF above .995 for every measured head**, while the historical functional control fails dramatically. A new full optimized-versus-fused gated runner result is still required; these independent-HF rows do not silently replace it.

## Source-level mechanism and real-input CPU falsifier

`tt/functional_decoder.py::_delta_chunks` constructs a strictly lower triangular matrix N and forms its inverse via:

```text
inv = I + N
repeat 4 times:
    N = matmul(N, N)
    inv = matmul(inv, I + N)
```

The identity `(I-N)^-1=(I+N)(I+N²)(I+N⁴)(I+N⁸)(I+N¹⁶)` is exact when N³²=0. That statement does **not** imply numerical stability: huge intermediate powers can cancel to a small inverse. The function uses generic FP32-source matrix operations with HiFi4/FP32 destination accumulation. The checkout's FP32 matrix unpack path can use TF32 operand precision; see the source references already documented in `AUTODEBUG_sequence.md`.

`tests/analyze_state_inverse.py` is a CPU-only reproducible probe. It loads real layer0 attention weights and the first four unique fixture users, executes installed HF input norm/projections/convolution to obtain q/k/v/g/beta, then constructs the functional 32-token N using its FP32 q/k normalization. It compares the exact float64 triangular inverse to the polynomial in ordinary FP32 and in a deliberately labeled TF32-truncated-operand approximation. It does **not** claim to emulate the device kernel or the small TT upstream rounding differences.

For the independently identified worst user1/head24:

- max beta=.99609375; max |N|=.921297.
- max absolute powers N²/N⁴/N⁸/N¹⁶ are8.954/228.772/7165.353/2196.781.
- The true inverse's largest absolute entry is1.0; the system's condition number is20.544.
- The FP32 polynomial's relative inverse error is .0003064.
- With matrix operands truncated to TF32, relative inverse error rises to2.4096x and max inverse entry to6.5708.

The other three unique users' head24 TF32-polynomial errors are .4763x/.6686x/1.7259x. In contrast, examined ordinary heads7/12/14/20/44 have errors near .00003–.0006, with modest intermediate powers. The observed bad head therefore matches an independent, actual-input mechanism for numerical failure in the functional prefix algorithm. The result argues against an intrinsically ill-conditioned delta recurrence: the actual triangular solve is reasonably conditioned; the chosen matrix-power evaluation is the hazardous part.

This is strong mechanism evidence, not an exact reproduction of the device state's117x error. Capturing the functional device's N/powers/inverse would locate every rounding contribution; it is unnecessary to establish that the functional state is already wrong relative to HF. Do not claim a proven silicon/kernel addressing defect or exact TF32 error model from this approximation.

HF uses a row-wise triangular inverse. Native GDN at Blackhole/chunk32 selects SFPU forward substitution under `AUTO` (`chunk_gdn_math.hpp:878-899`); its state scan also applies the inverse after subtraction to limit cancellation (`:1058-1059`). Those structural differences are consistent with the fused control's independently verified state. No shared op or functional decoder fix is requested in this optimized-only goal.

CPU reproduction:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.analyze_state_inverse
```

Output: `doc/optimized_decoder/text_state_inverse_cpu.json`. The helper imports no TTNN and opens no device. Syntax compilation and this CPU execution passed.

## Exact optimized-stage test-oracle correction

1. When `QWEN_DECODER=optimized`, compare prefill/decode outputs, residual-subtracted contributions and mutable state to the completed **FusedDecoder**, using the identical weights, input, layout-independent readback, page table, and update boundaries. This aligns the oracle with the user's “start from completed fused decoder; preserve fused semantics” contract. Do not relabel historical functional comparisons as fused.
2. Keep `.995` for every currently gated PCC metric. Do not skip state, substitute output PCC for state PCC, special-case head24, clamp state, or exempt real-text fixtures.
3. Independently compare optimized and fused state against HF immediately after prefix and subsequent decode boundaries, as the new state runner does. Compare convolution history with the correct common last-three-token orientation and recurrent state in `[B,48,K,V]`. Record HF's BF16 cache dtype and TT's FP32 dtype; dtype differences do not excuse a failed PCC.
4. Preserve `functional_equivalence` historical records and their original failures. New records should identify the new reference explicitly (for example `fused_equivalence`, with reference source hash), while retaining functional-control evidence as diagnostic history rather than silently deleting or overwriting it.
5. Rerun the actual failed optimized real-text case under the repaired oracle and unchanged HF/output/state thresholds. Preserve deterministic/repeated-run, paged-cache, non-aligned S33 and long-sequence gates. The current long random-input decode failure is independent and remains required work.
6. Record the inherited functional defect as a known historical-control limitation, with this source/mechanism evidence. Do not modify stage1/2 history to say it always passed real-text recurrent-state coverage; this fixture exposes a gap in that coverage.

The repair is supported by three independent facts: functional is demonstrably wrong against HF before decode; fused is correct against HF globally and for every head; and the functional-only inverse evaluation has a matching actual-input numerical failure mechanism. The independent optimized-versus-HF state measurement has now passed; the repaired full optimized-versus-fused runner and unrelated sequence gates still require their own passing results.
