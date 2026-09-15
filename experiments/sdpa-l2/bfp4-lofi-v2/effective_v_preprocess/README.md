# Effective LoFi V values and V8 centering

`effective_v_preprocess.py` is an isolated BF16-to-BF16 SFPU bitmask primitive.
It copies a tile, clears the low19 FP32 bits (equivalently low3 BF16 bits), and
packs BF16. This keeps five significant bits for the LoFi logical-right/SrcA
operand. It is truncation toward zero, including negative values, **not RNE**.
The exact nonzero oracle is `BF16_bits & 0xfff8`; the copy/pack path canonicalizes
signed zeros to positive zero, which does not affect the consumed-value mean.
Normal finite BF16 values and zeros are supported; NaN, infinity and subnormals are excluded.
Default BF16 DST, batch4, two input/output CB batches; FP32 DST is an optional
diagnostic. Reader/writer wrappers reuse existing preprocessing dataflow without
modification. Checks print input/raw-expected/actual bit triples before asserting
and never suppress nonzero errors. No device result is claimed before the gate passes.

The parent-run v2 records now confirm this contract in both BF16 and FP32 DST:
`effective-v-finite_bf16-{bf16,fp32}-v2.json` each report zero nonzero-bit
mismatches and eight negative-zero canonicalizations;
`effective-v-zeros-{bf16,fp32}-v2.json` each report zero mismatches against the
canonical-zero oracle and262144 negative-zero canonicalizations. Thus the
original eight wide-exponent discrepancies were signed zeros, not lost
nonzero mantissa bits. These records were inspected locally; the implementation
agent did not execute the hardware tests.

API: `build(device, decoded_bf16, ncores=1, batch=4, fp32_dst=False)` returns
`(preallocated_bf16_output, invoke, actual_cores)`. Each invocation costs one
BF16 read and one BF16 write:4 bytes/value, excluding launch/CB overhead.

## V8 attention integration

New `value_centered_b8_fullchip.py` copies the V4 driver but leaves it untouched.
K/V choices are `b8_b8` or `b4_b8`. MAIN, full FAST and FAST `--denom-only`
remain available. Chunks Q256/K512, D128, double K/V buffers, forwarding chain,
and existing attention kernels are unchanged across its centering controls.

- `--center-mode none` (`off` alias): unchanged RNE5/native-B8 V preparation.
- `original_mean`: device mean of original BF16 V; live FP32 SFPU subtract of
  the actual rounded BF16 mean, per-value RNE5, native B8 pack. Add back original
  rounded BF16 mean after attention.
- `matched_mean`: same packed V8 as original_mean. Decode that actual packed V8
  to BF16, run trunc5, and take the device mean of the resulting effective V.
  The output bias is BF16-RNE(original rounded mean minus effective rounded mean).

RNE5 before shared-exponent B8 packing does not guarantee that the actual
decoded values are SrcA5-representable. Therefore matched mode must mean the
truncated actual packed values, not the decoded B8 values or a fresh host codec.
The truncation tensor is only used for computing the correction; attention
still consumes the original packed V8 tensor. This does not eliminate weighted
quantization error, QK error, or accumulation error.

All preprocessing and BF16 output-add costs are included in combined timing.
Matched mode includes the full V8 decode, extra BF16 truncation pass, and second
device mean. Mean precision remains selectable BF16-FPU/FP32-DST or full-FP32
SFPU; both returned biases are BF16. The matched subtraction and epilogue use
explicit accurate BF16-RNE TTNN operations. There is still intermediate BF16
core-output rounding and final BF16 rounding. Near commonV32, small corrections
can disappear in rounded bias subtraction or final output: the new centered
residual metrics expose this, without gain fitting or dividing by zero for
constant V. Matching a BF16 rounded mean is not matching an exact FP64 mean.

Exact preprocessing checks use actual device bias and packed V8; matched mode
also bit-checks the truncation output. Original BF16 Q/K/V remain the FP64
reference. Source pins include the reference implementation, all local input/
oracle helpers (including imported v1 models), and preprocessing/kernel sources.

## Commands for the device owner

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/effective_v_preprocess.py --label effective-v-normal-v1 --distribution normal
python experiments/sdpa-l2/bfp4-lofi-v2/effective_v_preprocess.py --label effective-v-finite-v1 --distribution finite_bf16
python experiments/sdpa-l2/bfp4-lofi-v2/effective_v_preprocess.py --label effective-v-zeros-v1 --distribution zeros
python experiments/sdpa-l2/bfp4-lofi-v2/value_centered_b8_fullchip.py --label value-b8-matched-smoke-v1 --length 1024 --heads 1 --cores 1 --destination fast_bf16 --denom-only --kv-formats b8_b8 --center-mode matched_mean --distribution normal --sample-rows 1024 --check-preprocess --iters 0
```

Repeat the attention smoke with fresh labels for none/original_mean, then
common_v/constant_v before long timing. Primitive timing can use `--cores 110`
and `--iters 10` on a long input. No device jobs
were run by the implementation agent; Python AST, source-path and unchanged
attention geometry/config checks passed locally.
