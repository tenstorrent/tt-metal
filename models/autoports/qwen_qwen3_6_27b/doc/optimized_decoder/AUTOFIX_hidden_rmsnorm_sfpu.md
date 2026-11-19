# Dense FP32 SFPU hidden RMSNorm candidate

This is a source-only AutoFix candidate for the scaler and precision diagnosis in
`AUTODEBUG_norm_scaler_capture41.md`. Hardware correctness and latency are owned
by the parent optimization run; this note makes no device pass claim.

## Files and contract

- `candidates/hidden_rmsnorm_sfpu.py`: standalone generic-op helper;
  `hidden_rmsnorm_sfpu(x, gamma, epsilon=1e-6, output=None)`.
- `candidates/check_hidden_rmsnorm_mapping.py`: CPU-only address schedule checks.
- `hidden_rmsnorm_mapping_cpu.json`: their results.

The candidate accepts interleaved DRAM TILE input `x=[1,B,5120]`, BF16,
`1 <= B <= 32`, and gamma `[1,1,5120]`, FP32. Gamma must already contain the
effective affine scale (including the model's `1 + weight`, where applicable).
Output is BF16 with x's logical shape. A caller-provided output avoids allocation
and must be separate from x and gamma. This initial version is Blackhole-specific
in its NoC alignment assumptions. It does not replace general prefill RMSNorm.

Every call constructs descriptors from current addresses and registers x, gamma,
and output in `ttnn.generic_op`. It does not embed unregistered buffer addresses
in a static source or reuse a descriptor containing obsolete addresses. There is
no torch conversion, host tensor read, or shared-source change.

## Arithmetic and layout

One core handles one logical batch row. Its reader gathers 160 tiled BF16 face
rows, widens BF16 bits directly to FP32, and packs the 5120 useful values into five
dense 32x32 tiles. Gamma is gathered into the same dense layout. Five squares are
folded in FP32 SFPU, followed by column and row SFPU SUM reductions. Multiplication
by FP32 `1/5120` is a separate operation, followed by epsilon addition and precise
SFPU rsqrt. The output applies inverse RMS then gamma using separate FP32 SFPU
multiplications, explicitly rounds to BF16 with the TTNN typecast SFPU primitive,
and packs once. No FPU reduction, BF16 reduction scaler, or TF32 input conversion
is used. All consumed FP32 CBs select `UnpackToDestFp32`; FP32 destination alone
would not provide this guarantee.

The reduction order differs from a host flat reduction, so matching HF within
FP32 rounding tolerance must be measured. This helper does not claim bit equality
to HF reduction or assume all remaining decoder error comes from RMSNorm.

## NoC safety and storage

Blackhole DRAM reads require 64-byte alignment compatibility. For BF16 x, each
64-byte read covers a pair of source rows at an aligned address; the reader
selects the requested row afterward. Gamma face rows are 64 bytes and naturally
aligned. Scratch is 20 KiB. The dense x and gamma CBs are each 20 KiB; inverse and
zero FP32 CBs are 4 KiB each; dense BF16 output is 10 KiB; writer zero is 2 KiB:
80 KiB of CB allocation per core in total.

Each output writer scatters 32-byte face rows. Blackhole NoC writes require
16-byte alignment, which both source and destination satisfy. Different users'
writes do not overlap. The last logical user's writer separately zeroes padding
rows through row 31 in contiguous face-row ranges. No read-modify-write of
adjacent users' values is necessary. B1 and partial batches have deterministic
padding, and the input's padding contents do not enter the norm.

Source support: `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`
defines the read/write alignment requirements;
`tt_metal/hw/inc/api/compute/compute_kernel_api.h` exposes FP32 SFPU SUM for both
axes; `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h`
places column sums in row zero and row sums in column zero. The final scalar is
therefore tile element zero. `sfpu_normalize_bcast_scalar` consumes that element.

## Source-only validation

Run from the checkout:

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_hidden_rmsnorm_mapping.py
```

Python AST and three inline kernel brace checks pass. Address simulations pass
for B=1,3,16,17,31,32. They check every logical channel's read mapping, complete
dense-tile coverage, read alignment, scatter-write alignment, full output and
padding coverage, and absence of overlapping writes. These checks do not compile
or run the device kernels.

The next required hardware checks are own-input HF comparison at B1/B32 (and an
interior partial batch), changed-input/program-cache reuse, traced determinism,
the 512-step layer sequence, latency, and watcher. Compare directly against the
primitive norm44 control; keep this candidate only if correctness and warmed
latency improve the legal incumbent. A compile/API error is a repair task, not
evidence against the mathematical candidate.

## BF16 staging follow-up

`candidates/hidden_rmsnorm_bf16_staged.py` is a distinct source-only performance
candidate with the same public function signature. Parent-reported norm46 device
evidence for the original helper is approximately 99.9994% BF16 bit equality
(one tie among 163840 values), relative RMS error below 1e-5, and a passing short
decoder run at 3.343 ms versus approximately 3.18 ms for the primitive norm44
control. The long sequence was pending when this variant was written. These
measurements motivate reducing reader work; they do not establish its bottleneck.

The new reader retains its 64-byte aligned source reads, then copies eight
uint32 words per face row into five dense BF16 tiles. It eliminates per-element
integer widening and halves the number of RISC-V loads/stores, with indexing now
performed once per 16 values. CB0 shrinks from 20 KiB FP32 to 10 KiB BF16, reducing
total CB allocation to 70 KiB/core. All gamma, scalar, arithmetic, output, and
per-user semantics stay as above.

BF16 CB0 uses Default unpack, which widens the BF16 values exactly into FP32
destination registers. FP32 CBs retain direct FP32 unpack. Startup uses FP32 CB1;
the compute kernel explicitly reconfigures SrcA and calls copy_init before loading
BF16 x and before switching back to the FP32 zero/inverse/gamma group. The three
members of that group share the same format and direct-unpack policy. No numerical
operation is moved or combined. The source arithmetic remains five squares and
the original fold/reduction sequence, then inverse RMS and affine multiplication.

The CPU mapping test now validates both helper sources and checks every value of
the packed uint32 copy against the original scalar-widen reader mapping for all
six batch sizes. It passes. Device compilation, own-input accuracy, changed-input
reuse, the full sequence, and latency must still be measured for this variant.
