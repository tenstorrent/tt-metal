# Source-derived QSA projection reduction order

The pinned backend supports a concrete generic arithmetic candidate: independently
accumulate K blocks of 1024, 1024 and 512 elements, processing each adjacent BF16
pair **high element first, then low**, with a separate FP32 rounding after each
update. Add the three FP32 partials in order and convert the final result to BF16.
The corresponding CPU control matches all **307200 BF16 outputs** across seven
critical and eight independently selected 32-row chunks. This is diagnostic
evidence, not device or full-context acceptance.

## Backend identity and configuration

The recorded [backend report](qsa_cpu_backend_v1.json) and its
[raw verbose log](logs/qsa_cpu_backend_v1.log) identify
`brg_matmul:avx512_core_bf16`, plain `abc` operands, and dimensions
`1x32x2560:1x2560x640`. The installed oneDNN build hash equals the upstream
`v3.10.2` commit `f1d471933dc852f956fd05389f9313c7148783d5`.

For nontransposed input and K = 2560, the AVX512 configuration selects K block
1024, BRGEMM batch size 1 and no K-thread partition. The partitioning alternatives
require transposed input or huge FP32 K and do not apply. Accumulation buffers
are FP32 because output is BF16 and multiple K blocks are needed.
[Configuration source, lines 821–977](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul_utils.cpp#L821).

## Instruction and kernel ordering

Intel specifies two FP32 fused multiply-add updates for each `VDPBF16PS` lane:
the higher BF16 member updates the accumulator before the lower member. Each
update uses nearest-even rounding; input denormals are zeroed and output
denormals flushed, independently of MXCSR. Consequently neither a pre-rounded
pair sum nor a single-rounded three-term sum models this instruction.
[Intel SDM volume 2C, printed pages 5-160–5-161](https://cdrdv2-public.intel.com/929355/326018-093-sdm-vol-2c.pdf#page=166).

The BF16 kernel emits `VDPBF16PS`. Its accumulator register depends on output
row and output vector, not K; increasing adjacent K pairs update that same
accumulator. K-loop unrolling does not create independent K-striped sums.
The kernel initializes accumulators to zero and adds the previous C result
after the current block via FP32 `vaddps` when beta is 1. The final BF16 store
uses `vcvtneps2bf16`.
[Kernel source](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/brgemm/jit_brgemm_kernel.cpp#L2234).

The BRGEMM configuration sets a two-element BF16 reduction step and an
eight-element unrolled body. The outer matmul visits K blocks in increasing
order, using beta 0 for the initial block, beta 1 subsequently and a distinct
512-element tail call. Only the final call applies output conversion.
[BRGEMM blocking](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/brgemm/brgemm_utils.cpp#L817),
[matmul calls](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul.cpp#L719).

For ordinary finite BF16 operands whose products are representable normal FP32
values, each product is exact in FP32. The resulting candidate is:

```text
for (start, end) in [(0, 1024), (1024, 2048), (2048, 2560)]:
    partial = float32(0)
    for k in range(start, end, 2):
        partial = RN32(partial + x[k+1] * w[k+1])
        partial = RN32(partial + x[k]   * w[k])
    accumulated = partial if start == 0 else RN32(partial + accumulated)
result = RNE_BF16(accumulated)
```

## Evidence and limits

The independently run [v3 order control](qsa_projection_orders_cpu_v3.json)
records zero critical, holdout and named-coordinate mismatches for
`block1024_high_low_then_partial_add`. The earlier pair-interleaved hypotheses
and whole-K high/low variants were not exact; their failures remain in the v1/v2
reports. No row, output column, value-dependent correction or expert-selection
rule appears in this candidate.

The subsequent [local selection control](qsa_order_selection_cpu_v3.json)
also reproduces all 131 checked CPU attention outputs and selected sets exactly.
That control uses its documented local-window scope, not uniform replacement
of every historical compressed key. The completed
[full CPU projection sweep](qsa_blocked_projection_full_cpu_v1.json) compares
all 262144 × 640 BF16 projection outputs using the original 32-row call shape:
every value is bit-exact to literal HF (zero mismatches; 839.005 seconds).
This is CPU projection evidence, not full TT decoder acceptance.

This establishes a source-derived order with a bounded same-input numerical
match, not disassembly of the installed JIT or universal equivalence across
backends/shapes. The separate [Q1 source review](onednn_q1_projection_order.md)
explains the 512-element K block selected for singleton calls; its source-bound
shape controls cover Q1 through Q32.
NaN, overflow and denormal behavior must follow the instruction specification
if claimed; ordinary FP32 multiply/add is only equivalent under the stated
operand conditions. The [standalone pack review](ordered_projection_pack_review.md) records exact
FP32 device accumulation on one window and isolates two final BF16 midpoint
differences; explicit device typecast removes them. Broader TT execution, trace
behavior, routing/selection and the complete all-row context gate still need
their own candidate evidence.

[Source provenance](onednn_sources/provenance.json) records upstream commit,
full-source SHA256 values, exact retained excerpt hashes, the official Intel
PDF identity, and hashes of the inspected CPU results/logs/provenance. Four
line-numbered source excerpts retain the original copyright/license header.
No runtime source or device execution was changed by this source investigation.
