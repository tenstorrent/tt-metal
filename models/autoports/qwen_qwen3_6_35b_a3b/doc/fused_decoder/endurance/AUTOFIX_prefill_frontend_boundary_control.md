# AutoFix rsqrt risk and prefill frontend control

Source/CPU only. No device runs or runtime/test edits.

## Matching variance does not prove matching input norm

The Blackhole precise rsqrt implementation is a bit-seeded polynomial plus correction, not an explicit correctly-rounded guarantee (`ckernel_sfpu_sqrt.h:63–88,145–151`; rsqrt delegates to it). Its positive-normal path uses seed0x5f1110a0 and FP32 constants2.2825186/2.2533049.

I emulated this source on CPU over all8320 canonical shifted variances from seed913. The emulation uses FP32-rounded multiplies, FP32 constants, and fused multiply-add at the source's compound multiply/add expressions (computed as FP64 then rounded FP32). This is a **source-level prediction**, not verified TT output: SFPI compilation/contraction and instruction details could differ.

* CPU scalar rsqrt versus CPU vector rsqrt on the same variances:0 differences.
* Predicted SFPU rsqrt versus canonical CPU:1737/8320 differing values, each at most1 ULP, in both directions.
* With canonical mean and the remaining CPU FP32 normalization operations unchanged, predicted rsqrt differences change68 BF16 norm elements across64 rows.
* First affected absolute positions:279,430,803,928,1040,1217. Predicted rsqrt at130 matches CPU, so the previous exact130 control does not rule out this issue.
* Among saved captures, predicted rsqrt ULP differences are0 at128/129/130/1024/3979/5011/7703 and−1 at7551/8157. Those last two do not themselves cross the final BF16 norm boundary in this model.

Minimal next discriminator, **only if current endurance still fails**: upload the canonical shifted variances at279/430/803/928 to a same-input rsqrt microprobe and compare the FP32 inverse plus resulting BF16 norm. An inexpensive batched8320-scalar probe would test the whole predicted distribution. Do not bias rsqrt globally or assume another Newton iteration matches CPU: a more accurate mathematical inverse can differ from CPU's intermediate rounding.

The positive-input emulation sequence, with every assignment FP32 unless specified, is:

```
y = bitcast_f32(0x5f1110a0 - (bitcast_u32(x) >> 1))
xy = x*y
c = (-y)*xy
y = y * fma(c, FP32(2.2533049)+c, FP32(2.2825186))
xy = x*y
residual = fma(-y, xy, FP32(1))
y = fma(residual, y*FP32(0.5), y)
```

## Minimal model-only prefill frontend control

For each existing physical prefill chunk of logical length S, keep `prepare_prefill`'s public-call/physical-chunk plan unchanged. Use the same validated input-normalization operations generalized across S, and produce the packed projection with8 residue-masked BF16 native matrices combined by FP32 SFPU adds. No per-mask bias. Alternatively use16 masks as the stronger single-live-product control. Preserve the existing final packed-output shape[1,S,12352].

Then:

1. Cast the combined raw projection once to BF16. QKV is[1,S,8192], Z[1,S,4096], A/B each[1,S,32]. Keep A/B **before dt_bias** at this cast. Remove the current prefill-only packed bias path for this control.
2. Concatenate BF16 conv history[1,3,8192] and current QKV. Compute four vectorized tap products/additions in FP32 using slices`history[:,j:j+S,:]`; cast preactivation to BF16, widen, run precise FP32 SiLU, cast BF16. Update conv carry to`history[:,S:S+3,:]` only, in existing row-major BF16 storage. This works for aligned chunks and one-token tails without introducing padded tokens into the carry.
3. Produce raw Q[1,S,2048], K[1,S,2048], V[1,S,4096]. For canonical frontend Q/K, reshape to[1,S,16,128] and use the validated explicit BF16 square/sum/epsilon/rsqrt/multiply boundaries. Maintain the canonical duplicated32-head CPU comparison shape when validating. The values sent to the recurrent arithmetic should represent HF's BF16-normalized Q/K.
4. Widen rounded A; add FP32 dt_bias; compute FP32 g. Compute sigmoid from rounded B and round beta to BF16 before widening. Both gates retain[1,S,32]. Z retains the raw projection's BF16 boundary.

### Native flat-GDN wiring and exactness blocker

The existing native flat call accepts TILE BF16 Q[1,S,2048], K[1,S,2048], V[1,S,4096], gFP32[1,S,32], betaBF16/FP32[1,S,32], initial_stateFP32[1,32,128,128], chunk_size32, num_key_heads16, and output_final_state=True. S must be divisible by32; the returned default output is row-major[1,S,32,128], final state[1,32,128,128]. Existing caller transformations can remain unchanged. `output_head_major=True` instead returns TILE[32,S,128], but changing it is unnecessary for this control.

**Do not feed already normalized Q/K as rank3.** The wrapper unconditionally interprets flat Q/K as raw and enables in-kernel normalization; it rejects `use_qk_l2norm=True` rather than exposing a bypass (`chunk_gated_delta_rule.cpp:141,239–243,333–335`). Its normalization computes squared norms with matrix products and fuses scaling (`chunk_gdn_math.hpp:425–466,808–828`), lacking HF's intermediate BF16 boundaries and reintroducing the matrix summation mechanism under investigation. Therefore an exact canonical Q/K frontend cannot use the current all-flat route without native changes.

A model-only canonical-frontend adapter can instead supply **rank4 normalized Q/K[1,S,16,128]**, with either rank4 V[1,S,32,128] or flat V[1,S,4096]. Rank4 Q/K disables the native normalization; flat V is independently selected and requires aligned S. Keep num_key_heads16. The wrapper performs GQA duplication itself.

This preserves canonical frontend inputs, but does not establish exact HF chunk recurrence: `head_split_tile` casts Q/K to BF16, then the wrapper multiplies Q by scale in that dtype (`chunk_gated_delta_rule.cpp:35–46,239–246`). HF widens normalized Q to FP32 before scaling. Native chunk solve and accumulation can also differ. Record pre-native Q/K/V/g/beta separately from native state/output. If those frontend values match but prefix state does not, the remaining discrepancy belongs to native chunk arithmetic; do not keep changing frontend to compensate.

## Public cache boundary

Current `prefill_forward` already rounds incoming recurrent state BF16→FP32 **once per public call** before iterating the plan (`fused_decoder.py:762–787`), then passes `cache_boundary=False` into every internal chunk/tail. Preserve that behavior.

Carry FP32 recurrent state between all physical chunks within that call, including one-token boundary/tail steps. Do not BF16-round every32-token kernel chunk, every plan entry, or every token of a diagnostic sequential replay. Those are implementation boundaries; HF rounds its cache only when a public invocation ends. Existing lazy rounding at the next public entry is numerically equivalent for consumption, while retaining the persistent FP32 buffer. If choosing eager BF16→FP32 storage at public return instead, do it there once and avoid another semantic boundary inside the plan; current outputs use the unrounded new state.

For a canonical-prefix control, compare state at the same public boundary after applying BF16 rounding to both sides. Keep all logical outputs and conv carry for the real S; never advance state for padded tail rows. A sequential recurrence control must use the corrected recurrence with `cache_boundary=False` within the one public prefill call and must be labeled a numerical alternative to HF's chunk64 algorithm, not assumed bit-equivalent to it.
