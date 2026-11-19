# Ordered projection: final BF16 boundary review

This source review and the closed `ordered_projection_device_v2.json` isolate the
standalone candidate's two mismatches to final narrowing, not its FP32 reduction.
The candidate remains a diagnostic; this does not clear the full QSA failure.

The ordinary device `ttnn.typecast(FP32, BF16)` path requests preserved FP32
precision and FP32 destination accumulation in
`ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp`. Blackhole's
`calculate_typecast_fp32_to_fp16b`, in
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:273`,
performs explicit integer round-to-nearest-even before packing:

```text
bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
```

It stores this BF16-exact value in FP32 DST. The subsequent pack cannot change
these ordinary finite values. By contrast, the initial custom kernel directly
packs its unrounded FP32 accumulator. In Blackhole `cpack_common.h:475–496`,
FP32 destination accumulation enables 32-bit DST reads; the optional 10-bit
mantissa setting applies to Float16 or FP8 conversion, not this BF16 case. There
is no equivalent software BF16 RNE step in that direct pack path. Source alone
did not establish the gasket's complete rounding rule.

The device experiment supplies the missing causal evidence. For the 32-row
window beginning at 105568, all 20480 prepack FP32 bit patterns exactly match the
independent CPU ordered reduction. Direct BF16 pack differs at two negative
midpoints:

| Global row / column | Exact FP32 | Literal HF / integer RNE | Direct device pack |
|---|---:|---:|---:|
|105583 / 342|-0.0518798828125|-0.0517578125|-0.052001953125|
|105584 / 328|-0.13330078125|-0.1328125|-0.1337890625|

Both observations select away from zero in direct pack, while RNE selects the
even BF16 neighbor. This is evidence for these two ties, not an exhaustive
statement about every format or exceptional input. Explicit device FP32-to-BF16
typecast matches both CPU integer RNE and literal BF16 at every element. It is
therefore a supported candidate correction for this boundary; full shape,
trace, changed-input and decoder validation remain separate gates.

The earlier copy/FIFO source audit found no accumulator-lifetime defect: scratch
CBs are immutable reader-owned reservations; streaming CB waits/copies/pops
match; BF16 `copy_init` changes address/MOP configuration without clearing
unrelated accumulator tiles; FP32 partials stay in DST until final pack. Exact
prepack equality now independently confirms that these transitions preserve the
observed window's arithmetic.
