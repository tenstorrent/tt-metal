# Exact-norm scalar preactivation control

Source-only candidate; no F edits or hardware. Patch `norm_scalar_preactivation_fusion.patch`, based on source SHA256 `40e7f7ecda61242a3e517e57d9cfbfe4e27e4279f8e4221cf89697af8454ce57`. Includes only the following two folds, leaving the root's existing RSUB residual control and recurrent L1 changes intact.

```python
# (high * 2.0) * low
cross = ttnn.multiply(high, low, input_tensor_a_activations=[
    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.0)])
# residual * (reciprocal(root) * 0.5)
correction = ttnn.multiply(residual, root, input_tensor_b_activations=[
    ttnn.UnaryOpType.RECIP,
    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 0.5)])
```

The root correction's existing outer `add(root, correction)` remains separate. No arithmetic reassociation, FMA, precision-policy or placement change is proposed. The first fold removes a standalone scalar-multiply operation; the second removes standalone reciprocal and scalar-multiply operations, replacing them with a per-operand activation chain within the existing final multiply. Extra local-CB preprocessing may erase any latency savings, so this is a device hypothesis.

## Exact reciprocal spelling, binding and approximation

- The enum is **`UnaryOpType.RECIP`**, without parameters. It is not `RECIPROCAL`.
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:158` defines the ordinary `ttnn.reciprocal` through `DEFINE_UNARY_OP(reciprocal, RECIP)`, forwarding the same parameterless enum to unary dispatch.
- `unary_nanobind.cpp:243–254` exposes reciprocal's tensor arguments as input plus optional memory, output tensor and sub-core grids. It has **no `fast_and_approximate_mode` argument**. Adding that flag to the reciprocal call would not be a supported way to match this control.
- The ordinary unary factory explicitly sets `math_approx_mode=false` (`unary/device/unary_program_factory.cpp:532`). BinaryNG creates `ComputeConfigDescriptor` without overriding this member (`binary_ng/device/binary_ng_program_factory.cpp:1511`), and its declared default is **false** (`tt_metal/api/tt-metalium/program_descriptors.hpp:111`). Thus both use precise mode in current source.
- `unary/common/unary_op_utils.cpp:760` lowers RECIP to `recip_tile_init(); recip_tile(...)` in both contexts. `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h` defaults its destination mode to `DST_ACCUM_MODE` and its approximation to the kernel's `APPROX` setting. Both contexts have FP32 destination enabled from the FP32 operand.
- Consequently target Blackhole `ckernel_sfpu_recip.h:386–393` chooses `_calculate_reciprocal_fast_24b_5c_` in both contexts, not the approximate7-bit or BF16 path. Current source shows no approximation-flag mismatch blocking this control. Installed native behavior still needs the root's actual probe.

## Operation-order and boundary proof

`high`, `low`, `residual` and `root` are FP32 tensors. BinaryNG's per-operand preactivation uses intermediate c3/c4 with the **actual input dtype** for SFPU (`binary_ng_program_factory.cpp:1197–1240`); both are FP32 here. This avoids the previously measured BF16 Z/SiLU intermediate-pack issue.

`eltwise_utils_sfpu.hpp` loads an operand into destination, executes its activation list in order and packs the final activated operand to its FP32 CB. The downstream binary op reloads that FP32 result. `unary_op_utils.cpp:467` lowers scalar multiplication to `mul_unary_tile`, and Blackhole `ckernel_sfpu_binop_with_unary.h:28` computes `val * parameter` and stores FP32 destination. The RECIP result is stored to FP32 destination before scalar multiplication reads it, so its FP32 boundary is retained even though the external tensor allocation is removed.

For `(high*2)*low`, high is BF16-rounded and multiplication by exact2 is exact over the finite bounded norm range; the following multiply still consumes that scaled high. For reciprocal scaling, exact0.5 performs the same binary exponent adjustment over the valid norm range. The subsequent multiply still consumes the rounded FP32 reciprocal-times-half. Neither change folds the final multiply into reciprocal math or the outer addition. The later split-product correction and HF inverse target remain unchanged.

## Validation

CPU AST parsing and `git apply --check` passed. No device run or claimed speedup. Root should run `tests/input_norm_actual_probe.py` on51 sensitive rows and require unchanged norm/mean codes, then the unchanged three-seed stream gates and paired warmed traced latency. If a mismatch appears, compare the first fold and reciprocal chain separately on identical captured intermediates; do not waive the code or decoder gates. This patch does not modify context/cache capacity, and preserves the same arithmetic for all sequence lengths.
