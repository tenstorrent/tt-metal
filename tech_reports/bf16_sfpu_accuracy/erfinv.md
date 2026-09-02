# erfinv: exhaustive BF16 accuracy on Blackhole

`ttnn.erfinv`'s Blackhole SFPU kernel was replaced for the BF16 destination-register
case (see `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv_bf16.h`
and the shared evaluator in `ckernel_sfpu_bf16_poly_common.h`).
The figure below shows the pure ULP error of the previous and the new kernel at every
one of the 65,536 BF16 input encodings, measured on Blackhole silicon against a float64
`torch.erfinv` reference:

![erfinv BF16 exhaustive ULP: previous kernel up to 255 ULP, replacement below the 1 ULP gate, all 65,536 encodings](images/erfinv_bf16_ulp.png)

Pure ULP = |FTZ(float64 golden) − device result| / bf16_ulp_spacing(BF16-rounded golden),
the `ttnn-eltwise-op-tester` metric; the numerator flush is keyed on the rounded golden,
matching Blackhole's post-round FTZ. The device hardware model is DAZ on input and
post-round FTZ on output, per
[Handling_Special_Value/special_values.md](../Handling_Special_Value/special_values.md).

| kernel | max pure ULP | mean pure ULP |
|---|---:|---:|
| previous (Winitzki + Newton sqrt) | 255.23 | 168.38 |
| replacement (`x * P3(\|ln(1-x^2)\|)`) | 0.83 | 0.25 |

Special values follow the certified contract: `erfinv(±1) = ±Inf`; `|x| > 1`, `±Inf`
and `NaN` inputs produce `+Inf` (the BF16 conversion pipeline maps NaN payloads onto
infinities — the previous kernel also never produced NaN here); zeros and DAZ'd
subnormal inputs produce exact zeros.

The bound is enforced by
`tests/ttnn/unit_tests/operations/eltwise/test_erfinv_bf16_exhaustive.py`, which sweeps
all encodings in a single tiled tensor and regenerates the raw data for this figure via
`TT_EXPORT_ULP_DUMP`.
