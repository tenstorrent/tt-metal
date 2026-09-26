# Packed decode GELU/multiply folding

Patch record: `expert_fused_gelu.patch`, against runtime SHA256
`e0691dc70a93b1e2d76d3028f022980f137cec4b0ac2ea944a25da425f19ea22`.
After the parent confirmed hardware idle, the complete source was compiled and
applied with one atomic replacement. New runtime SHA256:
`a31d4834fd86d6a770534de1fec1e7dbf268191b6a430ee0fc6cf3841cd18c82`.
`git apply --check` passed before application. No device commands were run by
this investigation; the parent owns hardware validation.

The patch adds `expert_fused_gelu=False` to the factory and `OptimizedExperts`.
When enabled, only packed decode replaces separate `gelu(Accurate)` and multiply
with `mul(..., input_tensor_a_activations=[GELU, 0.0])`. The activation descriptor
is built at setup. Prefill and separate projections retain their existing path.
`precision_policy.decode_expert_fused_gelu` records the effective choice, including
the already-fused separate path. Existing factory defaults are unchanged.

## Source semantics

There is no GELU-variant mismatch:

- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:537-550` implements
  `GeluVariant::ACCURATE` as exactly `UnaryWithParam(UnaryOpType::GELU, 0.0f)`.
  Fast LUT uses parameter1; tanh uses the distinct `GELU_TANH` operation.
- `unary/common/unary_op_utils.cpp:271-274` maps parameter0 to
  `gelu_tile_init<0u>()` and `gelu_tile<0u>(...)`. Both unary and binary activation
  generation use this utility.
- Expert matmul outputs are BF16 regardless of BFP4/BFP8 weight choices. Unary
  `unary.cpp:57` and binary `binary_ng_program_factory.cpp:1205` both select
  non-FP32 destination mode for these BF16 input/output tensors. On Blackhole,
  `ckernel_sfpu_gelu.h:305-349` therefore selects the same accurate BF16
  piecewise-CDF computation and nearest-BF16 conversion.
- Fusion retains a BF16 rounding boundary before multiply. The binary factory's
  activation intermediate CB uses the input data format (`:1062-1073`), and
  `binary_ng/device/kernels/compute/eltwise_utils{,_sfpu}.hpp` explicitly packs
  the activated operand into that CB before consuming it in the binary op.
  This is not an unrounded FP32 GELU-product substitution.
- `binary/binary.cpp:1264-1288` selects approximate multiply automatically only
  for block-float **inputs**; these inputs are BF16, so the weight dtype does not
  silently change multiplication mode.

Thus source supports a same-formula, same-BF16-boundary fold. It does not prove
device bitwise equality or speed. The expected saving is one op dispatch and a
standalone intermediate tensor/read-write; the fused kernel still performs GELU
and internal CB packing. Run matched actual-input whole-layer headline/stress and
profile the eliminated unary row before selecting the option.

Add `"expert_fused_gelu": true` to the already selected `--default-overrides`
object. No CLI edit is needed. Compare with the same candidate and flag false.
The recorded patch is already applied and should not be applied a second time.

## Related outstanding QKV overhead trial

An audit of all root-level candidate JSONs found no actual-input run with
`qkv_terms=1` or `qkv_lanes<16`. Actual runs use default lane16/terms2, plus four
explicit terms3 controls. `qkv4149_lanes.json` is a Gaussian saved-activation
diagnostic, not an actual-text veto. Terms1 with lanes16,8,1 is therefore a new
material precision/topology trial after a correct cumulative native base exists.
Native query rounding alone does not justify the change: K/V normalization,
cache contents and downstream routing also consume the projection result.
