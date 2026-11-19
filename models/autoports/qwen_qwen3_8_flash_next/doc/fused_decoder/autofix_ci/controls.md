# Current owner controls

Frozen fused runtime source is the final_v1 candidate. No quality threshold was changed.

- `ci_control_functional_v1`: original functional decoder, same synthetic seed1901,
  one-token prefill plus three changed traced decode tokens: pass.
- `ci_control_best_group64_v1`: accurate projection sums, all independent output
  rows merged, other accepted v2 fusions, native convolution disabled: pass.
- `ci_control_best_group64_layout_v1`: above plus recurrence metadata reshapes: pass.
- `ci_control_best_group64_conv_v1`: above plus native convolution, layout rewrite
  disabled: fails decode step1 at exactly0.994683563709259, same as final_v1.

All are separate bounded tracked runs with unchanged test assertions, real target
shapes and deterministic synthetic weights. Real-checkpoint counterparts already
passed short/continuation tests; `real_ci_distribution_conv_v1` tests additional
real-weight inputs. These observations isolate native convolution as the trigger,
without asserting that its missing BF16 boundary is the only numerical difference.

Candidate repair to test after diagnosis: concatenate the four FP32 tap windows,
one broadcast product with setup-packed FP32 taps, accurate sum across the tap
axis, then retain the explicit BF16 convolution boundary and FP32 SiLU/BF16 output.
This avoids changing native library files and preserves stage-owned source scope.
It needs both integrated accuracy and timing evidence; fewer operations alone
will not justify keeping it.
