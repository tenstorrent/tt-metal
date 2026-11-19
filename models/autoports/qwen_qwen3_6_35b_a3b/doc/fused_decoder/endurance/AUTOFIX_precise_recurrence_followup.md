# Precise frontend and recurrence: intermediate CPU replay

Source/CPU analysis only; no device calls, TTNN imports, implementation changes, or test edits. Candidate: `../endurance_precise_projection_silu.py.snapshot`. Parent-supplied probes: `../endurance_precise_probe.json.pos{128,129,130,1024,3979,5011,7551,7703,8157}.pt`. Complete local comparisons and input SHA256 values are in `precise_recurrence_cpu_replay.json`.

## Finding

The prior memory-matmul discrepancy is absent from this candidate. Captured accurate-SFPU memory differs from same-input CPU reduction by only 1.18e-7–1.63e-7 relative. The entire same-input recurrence now differs by 1.35e-8–1.72e-8 relative in new state, versus approximately 1e-5 previously. Only 2–6 of 524,288 next BF16 cache values differ per sampled token.

The first ordinary FP32 arithmetic divergence is the memory reduction order. Decay has earlier numerical differences, but these are negligible at 128/129/130/7703 and do not change any sampled BF16 decay values. Subtraction and beta multiplication are **bit-exact on identical captured operands**. The remaining clear semantic difference is fused `addcmul` versus HF's separately rounded multiply and add. Its local BF16 effect is now worth isolating; it must not be confused with the much larger former contraction error.

At position 7703, current Q/K/V and beta exactly match HF; replacing only incoming state with HF state makes the canonical BF16 core bit-exact HF. Incoming-state relL2 is .011774, while the actual core relL2 is .008877. This localizes the present mismatch to accumulated state history, not a substantial error in that token's recurrence. These sparse probes do not prove how the accumulated history first diverged or whether removing fusion alone repairs the endurance run.

## Identical-input intermediate comparisons

Each operation below is evaluated from its **captured device operands**, so preceding errors are not attributed to later operations. Q/K heads are repeated in pairs, and Q uses HF's exact scalar `1/sqrt(128)`. The canonical complete recurrence separately uses the originally captured incoming state and Q/K/V/g/beta. Replaying HF inputs reproduces all nine HF BF16 core and cache tensors bit-for-bit.

| Position | Decay relL2 | Memory relL2 | Separate update vs device relL2 | Complete recurrence new-state relL2 | Complete next-cache BF16 differences |
|---|---:|---:|---:|---:|---:|
| 128 | 1.24e-42 | 1.21e-7 | 1.05e-8 | 1.72e-8 | 3 |
| 129 | 1.32e-42 | 1.36e-7 | 1.04e-8 | 1.52e-8 | 5 |
| 130 | 4.96e-30 | 1.18e-7 | 9.38e-9 | 1.46e-8 | 4 |
| 1024 | 6.73e-11 | 1.29e-7 | 9.31e-9 | 1.54e-8 | 5 |
| 3979 | 1.34e-42 | 1.18e-7 | 8.36e-9 | 1.71e-8 | 4 |
| 5011 | 2.54e-10 | 1.50e-7 | 5.88e-9 | 1.35e-8 | 4 |
| 7551 | 4.78e-42 | 1.20e-7 | 9.03e-9 | 1.35e-8 | 6 |
| 7703 | 3.86e-43 | 1.42e-7 | 1.03e-8 | 1.67e-8 | 2 |
| 8157 | 2.74e-9 | 1.63e-7 | 7.82e-9 | 1.35e-8 | 3 |

Additional intermediate details:

- `decayed_state`: CPU `state*exp(g)` differs in negligible/subnormal values at the early three positions and 7703. Maximum absolute differences there are respectively 4.09e-42, 4.46e-42, 2.52e-29 and 1.30e-42. Largest sampled decay error is 7.45e-9 absolute at 8157; none changes a BF16 value. Signed zero distinctions are excluded from BF16 value-difference counts.
- `memory`: CPU `sum(captured_decayed*k)` differs by at most 2.38e-7 absolute, with 0–2 BF16 values different out of 4096. This is ordinary reduction-scale error, rather than the former effective perturbation around 1e-4 relative.
- `value_minus_memory`: exactly equals CPU `v-captured_memory` at every element in all probes.
- `delta_update`: exactly equals CPU `captured_value_minus_memory*beta` at every element in all probes.
- `new_state`: full-precision product/add rounded once to FP32 is much closer to the device than separate multiply/add: FMA-surrogate error is 3.65e-10–9.68e-10 relative, versus 5.88e-9–1.05e-8 for separate arithmetic. FMA-surrogate/device BF16 values match everywhere except one element at 1024. Thus the source-level fused operation is visible in captured results, though the CPU surrogate is not a complete instruction-level hardware emulator.
- `reduced_output`: CPU `sum(captured_new*q_scaled)` differs by 4.55e-8–1.44e-7 relative and 0–2 BF16 values. The complete canonical same-input recurrence core matches the captured device core after BF16 conversion at the first six sampled positions and differs in one of 4096 values at each of 7551, 7703 and 8157; local differences can cancel across stages.
- Saved `state` and captured `new_state` are FP32 bit-exact at all nine positions. Consecutive input caches at 129 and 130 exactly equal CPU BF16 rounding of the previous captured new state. The cache cast is working, rather than introducing an extra rounding convention or stale-state error.

## Separate-update control

Keep all current precise projection, SiLU, L2, gates, memory reduction and cache behavior unchanged. Replace only:

```python
new = ttnn.addcmul(decayed, k, delta)
```

with:

```python
new = ttnn.add(decayed, ttnn.multiply(k, delta))
```

This directly matches HF's explicit FP32 product followed by FP32 addition. Both operations select SFPU for FP32 operands. With captured device delta/decayed inputs, the CPU separate-update result differs from the complete canonical CPU next-cache result in only 0–3 BF16 elements, instead of the actual device's 2–6. At **7703 it removes both differing cache values**, leaving a bit-exact canonical next BF16 cache in this CPU counterfactual. At 128/129/130 the remaining cache differences are 2/1/1, down from 3/5/4. This predicts the local signature of the control; it is not a measured device result or a claim about the full trajectory.

A serial parent-run one-step check can verify this exact predicted change, followed by a full-stream check if useful. If the stream still fails, the remaining sparse BF16 differences from reduction order and historical frontend differences require cumulative diagnosis; another broad precision change is not supported by these local results.

## Incoming history versus current inputs

| Position | Incoming state relL2 | Actual BF16 core relL2 vs HF | Canonical core with HF incoming state relL2 vs HF |
|---|---:|---:|---:|
| 128 | .010509 | .009068 | .005854 |
| 129 | .010567 | .008487 | .005514 |
| 130 | .010609 | .005649 | .001264 |
| 1024 | .012117 | .012352 | 0 |
| 3979 | .012674 | .010191 | 0 |
| 5011 | .015000 | .040684 | 0 |
| 7551 | .011842 | .009779 | 2.55e-6 |
| 7703 | .011774 | .008877 | 0 |
| 8157 | .011346 | .012330 | 3.15e-8 |

At 128/129/130 the Q/K/V BF16 value differences are respectively `(1714,2234,1888)`, `(1008,1386,852)`, `(832,834,576)` after head expansion. Thus the first decode tokens are not an exact-input HF trajectory even with the new precise current-token projection. This is consistent with prefill convolution history contamination, but that history cause is outside this recurrence-isolation task and requires direct frontend evidence. A diagnostic HF prefix seed must initialize **both convolution and recurrent caches** to separate prefix error from decode accumulation. The parent reports that a prefix-seeded stream still fails; this report does not reinterpret that result without its exact seed implementation.

Current Q/K/V are bit-exact HF at 1024,3979,5011,7703. Only four expanded K values differ at 7551, and one V value differs at 8157. Beta is bit-exact at every sampled position. Current g differences are small FP32 differences. A sparse set of exact tokens does not establish that every intervening token was exact.

## Source references

Repository-relative paths:

- `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance_precise_projection_silu.py.snapshot`, `_recurrent_step`: accurate SFPU memory sum, separate subtraction/beta multiply, fused addcmul, output sum.
- `models/autoports/qwen_qwen3_6_35b_a3b/tests/routing_probe.py:63–102`: intermediate operation capture; each comparison uses the actual captured operands at that point.
- `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:332–365`: FP32 HF recurrence, separate multiply/add at line 360, final BF16 core cast at line 365.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h:46–59`: fused MUL/MAD implementation and FP32 output path.
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:51–66`: FP32 add/subtract/multiply use SFPU.
- `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp:48,105–118`: accurate FLOAT32 reduction contract and default mode.
