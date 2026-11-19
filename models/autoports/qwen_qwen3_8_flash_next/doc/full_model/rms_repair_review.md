# Independent RMS reduction repair review

Scope: CPU source/arithmetic review of
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h`.
No header edits or device execution by this reviewer. This is not stage acceptance.

The repair addresses two independent row-placement dependencies. Replay slots0–5
reduce each A/B four-vector group as `x0 + (x1 + (x2 + x3))`; the original inline
C/D code used `(x0+x1)+(x2+x3)`. The new C/D sequence matches the replay chain
without changing operand formats, accumulation precision, or the number of adds.

The merged horizontal butterfly subsequently selected A/B/C/D results from
different lane columns. Equal mathematical sums therefore used different FP32
pairings even after fixing the vertical association. A symbolic eight-column
simulation confirms that rotating B/C/D by1/2/3 columns before merging makes
all four stored results use the same tree, modulo exchanging the two operands
of individual additions:

`((p0+p7)+(p5+p6)) + ((p1+p2)+(p3+p4))`.

The stores obtain A/B/C/D from columns0/5/6/7, respectively. The simulation
checks these actual selected columns, not just a hypothetical column0 result
for every group. With no prerotations, all four selected association trees
differ. A CPU FP32 control using the regression's width32 seeded input produces
different original-group sums and identical corrected-group sums. This confirms
that the proposed test data can expose rounding sensitivity. It is not a
substitute for executing the hardware regression.

The NOP before B's prerotation separates the replay's final SFPADD writing
LREG4 from SFPSHFT2 reading it. C/D loads and folds retain their live operands;
C is complete in LREG4 and D in LREG1 before prerotation. The intervening C
rotations separate D's final add from its first rotation. The existing masks,
face-pair traversal, stores, AVG scaling, integer folds and min/max branch
remain intact. New rotations are guarded by `RowFold::FloatAdd`, not by an
FP32-only type predicate: other floating SFPU callers can also change rounding.
BF16 sanity is therefore explicitly included rather than assumed unchanged.

This preserves the merged native reduction. It adds six rotates and one NOP
per16-row face pair; it does not introduce a host path, replay of the model,
or a higher-precision fallback. End-to-end performance remains a required check.

The recorded exact-fixture evidence is consistent with this diagnosis:
`rms_rows_v1.json` and `rms_rows_v2.json` first differ at mean (max delta
5.820766091346741e-11), amplified to a BF16 output delta0.0078125 at row18.
`rms_rows_v3.json` has exact full32 versus4+28 results at every tapped RMS stage.
These artifacts were run by the parent, not this reviewer.

New regression: `tests/ttnn/unit_tests/operations/reduce/test_row_reduction_invariance.py`.
It tests FP32 sum/mean at widths32/2560, an identical operand row in all32 row
positions, distinct-row permutations crossing4-row and16-row boundaries,
nonaligned4+28 continuation, and a FP64 Torch-reference error bound. Additional
cases check exactly representable BF16 sums/means, signed Int32 sums above2^24
with low bits preserved, and FP32/BF16/Int32 max. Exact permutation comparisons
are separate from the2e-5 relative numerical-reference tolerance.

CPU validation: syntax and whitespace checks pass; symbolic lane-tree and
numeric sensitivity controls pass; pytest collection finds all10 cases
without device execution. Device regression remains pending. Run
under the device-usage lane/lease policy after the native build is installed:

```bash
timeout 300 python_env/bin/python -m pytest -q tests/ttnn/unit_tests/operations/reduce/test_row_reduction_invariance.py
```

Review verdict: the arithmetic and lane-mapping changes are consistent with
row-invariant finite FP32 reductions. No source defect found in the changed
branches. Native regression, full-model resumed-prefill/accuracy and performance
gates are still required; NaN payload and signed-zero bit identity are not
claimed by this review.

Parent hardware follow-up: `rms_native_regression.log` reports10passed5.59s
(total bounded job11s), all devices closed normally. Reduced real0/1/3 resume
also passes (`reduced_resume_rms_v5.json/log`). Final all48 accuracy/performance
remain separate gates. The FP32 invariance regression is Blackhole-scoped.
