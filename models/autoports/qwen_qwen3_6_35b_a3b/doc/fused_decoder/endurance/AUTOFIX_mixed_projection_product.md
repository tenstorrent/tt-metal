# BF16 projection storage with FP32 SFPU products

Source-only investigation; no decoder edits, device execution, or changes to precision policy. The setup-ordered attention weights are BF16 checkpoint values currently stored as FP32. Keeping these same values in BF16 storage does not reduce their numerical precision.

## Supported call

```python
products = ttnn.multiply(
    ordered_weights_bf16,
    coefficients_fp32,
    dtype=ttnn.float32,
    fast_and_approximate_mode=False,
)
```

This is a supported direct mixed-input kernel. It does **not** require a runtime weight typecast. Explicit FP32 output is essential because default output follows the left-hand operand. Explicit `fast_and_approximate_mode=False` is essential for selecting SFPU with mixed inputs; the Python default is True, which would select the FPU for this mixed pair.

The false flag is legal with FP32 output for **MUL**. The FP32-output rejection previously encountered for accurate binary operations is restricted to ADD/SUB/RSUB (`binary_ng_device_operation.cpp:20–48`), not MUL.

## Source chain

All paths below are under `ttnn/cpp/ttnn/operations/eltwise/` unless otherwise stated.

- `binary/binary.cpp:715–775` promotes integer/float pairs only. A BF16/FP32 float pair is passed unchanged. Lines781–851 select explicit output dtype, keep already tiled tensors tiled, and dispatch directly to `prim::binary_ng`; no float-family promotion is inserted.
- `binary_ng/device/binary_ng_device_operation.cpp:51–65` selects SFPU MUL whenever the fast flag is false. Both-FP32 also selects SFPU with the default true flag, explaining why the current path works without an explicit flag.
- `binary_ng/device/binary_ng_program_factory.cpp:1134–1145` enables FP32 destination accumulation whenever either input or output is FP32. This includes BF16×BF16→FP32.
- The same factory, lines1371–1381, configures `UnpackToDestFp32` for both SFPU input buffers. Each input CB retains its own BF16 or FP32 data format; BF16 values widen on unpack rather than through a materialized tensor.
- `binary_ng/device/binary_ng_utils.cpp:453–459` selects generic `mul_binary_tile` for both floating input formats. Its BF16 narrowing is guarded by non-FP32 destination mode; an FP32 output disables that narrowing.
- `binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp:76–113` copies each operand with its own data-format reconfiguration, computes in destination, and packs the FP32 output. No host path appears in this dispatch chain.

The product of two finite BF16-representable values requires at most16 significant bits before exponent-range considerations, so the unchanged FP32 SFPU multiply can represent these products exactly. That mathematical observation and the source path support the hypothesis; actual kernel equality and speed still need the parent-owned device probe.

## Existing repository coverage

- `tests/ttnn/unit_tests/operations/eltwise/test_binary_dtype_validation.py:115–145` explicitly accepts mixed BF16/FP32 MUL in both orders.
- `test_binary_ng_activation_mixed_dtype.py` exercises mixed float formats and row/column/scalar broadcast shapes.
- `test_binary_ng_bcast_fp32_dest_acc.py` tests repeated mixed-dtype broadcast multiplication and the fix ensuring FP32 destination when either operand is FP32. Its comments also document a separate Blackhole BF16-to-FP32 column-broadcast issue with an explicit factory guard. Current projection uses scalar subtile broadcast, not that column case.

These are source/test-contract observations, not tests executed in this investigation.

## Focused parent-owned A/B

Keep the current correctly ordered reduction and all model arithmetic unchanged. During setup create a BF16 copy of `reduce_attention_weight` from the same ordered checkpoint values. For the actual decode shapes `[512,4,32,416]` weights and `[512,4,1,1]` coefficients:

1. Current baseline: FP32 weights × FP32 coefficients → FP32 products.
2. Candidate: BF16 weights × FP32 coefficients, explicit FP32 output and false fast flag.
3. Compare complete FP32 product values/codes, final `fast_reduce_nc` projection values/codes, and repeated replay determinism on representative captured normalized inputs including130/3979.
4. Measure warmed product and complete traced-decode latency without host reads in the measured region. Retain only if final correctness and performance gates pass.

Repeat with the prefill broadcast shapes `[1024,2,32,416]` weights and `[rows,1024,2,1,1]` coefficients, including rows1 and4. This verifies the rank-five outer broadcast separately.

The next independent candidate can remove the coefficient typecast too:

```python
ttnn.multiply(ordered_weights_bf16, coefficients_bf16,
              dtype=ttnn.float32, fast_and_approximate_mode=False)
```

It is also legal by the source contract, but both-BF16 scalar broadcast selects an LLK broadcast path whereas the mixed pair uses the reader broadcast path (`binary_ng_program_factory.cpp:347–380`). Test it separately rather than inferring identical implementation/performance from the mixed case.

## Capacity and traffic

Each padded ordered weight tensor has27,262,976 elements:109,051,904 bytes in FP32 versus54,525,952 in BF16. FP32 product storage remains109,051,904 bytes for decode. Weight-read plus product-write traffic therefore falls from approximately218.1MB to163.6MB, a25% reduction for those two components; total decoder traffic and latency gains will be smaller and must be measured. The prefill ordered weight allocation receives the same static storage reduction.

No reduced activation, KV-cache, recurrent-state or product dtype is proposed. This is exact checkpoint storage plus fused widening at an existing multiply consumer, not a global datatype sweep.
