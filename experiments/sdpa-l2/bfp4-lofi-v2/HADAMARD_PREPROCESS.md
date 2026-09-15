# Device Hadamard preprocessing primitive

[hadamard_preprocess.py](hadamard_preprocess.py) implements:

```python
rotated_bf16, invoke, metadata = build(device, src, block_size=16)
```

Input/output are interleaved-DRAM, tile-layout BF16 `[1,H,N,128]`, with positive
H and N divisible by32. Block sizes16 and128 use precisely the shared signs
and unnormalized transform from the [CPU study](QK_HADAMARD_NATIVE.md).
`oracle(src, block_size)` returns the FP64 mathematical transform; `matrix`
returns its FP64 ±1/0 matrix. `source_files()` lists this primitive and19 total
principal TTNN/kernel/Blackhole-LLK provenance files.

Every `invoke()` executes a **real dense TTNN matmul** on the current device
source, using HiFi4, FP32 destination accumulation, `math_approx_mode=False`
and `packer_l1_acc=False`, with BF16 output. It is not a fast butterfly:
the matrix is128×128 even for block-diagonal H16, and the implementation's
dense arithmetic count includes its zeros. The host only generates/uploads
the fixed constant matrix; no input values are downloaded or precomputed by
`build` or `invoke`.

Heads are flattened into independent rows using tile-aligned storage views.
The 1D multicast-B program uses fixed4×4 output blocks and four-tile FP32
subblocks; per-core M grows with sequence length while CB block storage stays
bounded. Source, matrix and preallocated output aliases remain owned by
`invoke.buffers`. Keep the callback alive until captured traces are released.
No output allocation or constant upload occurs in `invoke()`.

## Numerical and timing diagnostics

The standalone CLI generates normal, outlier, common-Q or common-K controls:

```sh
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v2/hadamard_preprocess.py \
  --label hadamard-h16-normal-new --block-size 16 --distribution normal \
  --heads 2 --length 1024 --iters 7
```

Each standalone timed invocation transforms **both Q and K**; the record labels
the median as `median_pair_ms`. Timing uses blocking device trace replay with
the stated repeat count, excluding one-time constant upload/allocation.
Trace output equality is checked. It is wall-clock trace timing, not a claim
of hardware-cycle-only duration.

Diagnostics report each transform against FP64, the BF16-output rounding-only
floor, and deviation from ideal BF16. Ideal-bit mismatches are diagnostic,
not an assertion: HiFi4/FP32 DST still has FPU alignment effects. The sampled
unquantized-QK diagnostic divides the score scale by h and reports both raw
and row-centered score errors, plus the FP64 identity control. It does not
mistake a quantized-input comparison for original-input SDPA accuracy.

## Integration cautions and validation status

Use the identical T for Q and K, with the SDPA scale divided by h. No V/output
rotation is needed. Both the score exp and online correction exponentials must
use the adjusted scale. Coarse input quantizer checks should compare against
**actual device BF16 transformed values**, while transform error remains a
separate comparison against original inputs.

There is deliberately **no centering** in this primitive. Shared common K/Q
can become much worse after rotation and quantization; see the CPU study's
common-mode failures. The primitive is not safe as an unconditional smoothing
switch. K centering and corrected Q centering remain separate work.

Completed without opening a device: AST validation, exact matrix/butterfly
agreement for both widths, orthogonality, sampled FP64 QK invariance, four
real TTNN program-config constructor checks through H10/N256K, and existence
of all19 provenance paths. Device compilation, transform error and timing
are **pending integration/device execution**. Existing shared and frozen
sources are unchanged.
