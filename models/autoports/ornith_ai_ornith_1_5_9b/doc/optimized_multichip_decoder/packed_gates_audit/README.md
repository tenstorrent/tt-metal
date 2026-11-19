# Packed gate numerical audit and isolated repair

The failed direct probe remains a failure. No PCC threshold, control gate
comparison, input case or sample was removed. Original candidates and queued
sources were not edited. The isolated repairs below need hardware validation.

## Observed failure

Source run: `../flat_ordered_packed_direct_watcher.log` and matching JSON.
The saved tensor is
`../flat_ordered_packed_direct_watcher.B17.wave7.step5.rank0.output_vs_ttnn.pt`.
Original source snapshots and SHA256s are in `original_sources/` and
`original_manifest.json`; the original evidence files remain in place.

At B17, wave7, step5, L1 input, rank0, the global output PCC is .9999988,
but request14 has two failing heads:

| Value head | FP64 PCC | Candidate maxabs | Control maxabs | Maxabs error |
|---|---:|---:|---:|---:|
| 2 (flat head114) | .82804598 | .01452637 | .01916504 | .009765625 |
| 3 (flat head115) | .71943319 | .00001978874 | .00001448393 | .00001385808 |

Recomputing correlation in FP64 reproduces the failure. Head114 is also too
large to dismiss as a zero-output special case. State min-head PCC .9999987
does not prove that its subsequent projection will agree.

`analyze_cpu.py` reconstructs the exact host RNG sequence without importing
TTNN or opening devices. It saves `reproduced_inputs.pt` (initial state and six
steps of inputs) and `cpu_analysis.json`. Both heads share Q/K head1. Step3
strongly decays the old state; step4 has zero Q/K and only scales the carried
state; step5 beta is approximately 8.0e-11 / 5.4e-13, so the output is almost
entirely a projection of that carried state. A nearly rank-one state and a
cancellation-sensitive projection amplify small changes to individual stored
entries. The independent Torch recurrence also disagrees on these heads, so
it is not evidence that either device path is the high-precision answer.
This explains sensitivity; it does not excuse the failed equivalence gate.

## Confirmed source differences

1. **Sigmoid arithmetic differs despite both outputs being BF16.**
   `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:56` selects 16-bit DST for
   BF16-in/BF16-out standalone sigmoid. The generic candidates enable FP32 DST.
   `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h`
   selects exp_21f plus one reciprocal iteration and a BF16 conversion for the
   former, versus FP32 exp and two reciprocal iterations for the latter.
   A final BF16 pack alone does not make those algorithms identical.

2. **Generic FP32 intermediates are unpacked as TF32 by default.**
   `tt_metal/jit_build/genfiles.cpp:1072` selects TF32 for this FP32-destination,
   B-exponent-format kernel. `tt_metal/jit_build/data_format.cpp:211` only keeps
   Float32 unpack format when a non-default unpack mode is explicitly supplied.
   The packed candidate leaves modes at default for raw a/bias/decay in CB23
   and sum/softplus intermediates in CB25/CB3. The separate TTNN FP32 binary
   path instead selects `UnpackToDestFp32` in
   `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1378`.
   These extra TF32 truncations are not part of the intended gate sequence.
   Conv-gates has the same issue in its bias/decay/sum/softplus CB10–13.

The precise contribution of each difference to this saved failure is not yet
isolated by hardware. Both are concrete departures from the stated control.

## Isolated repairs

New helpers/candidates/probe wrappers are:

- `tests/multichip_flat_ordered_packed_precise{,_candidate,_probe}.py`
- `tests/kernels/flat_ordered_packed_precise/`
- `tests/multichip_conv_gates_precise{,_candidate,_probe}.py`
- `tests/kernels/conv_gates_precise/`

The packed repair sets only CB23 to `UnpackToDestFp32`, then keeps
add → softplus → multiply in one FP32 DST window. Each operation writes an
FP32 result; avoiding intermediate pack/reload does not remove a BF16 rounding
boundary. It removes the unintended TF32 conversions. The final g product is
still packed into CB25 and loaded through TF32 before exp, matching post-loop's
existing g-to-exp path. All recurrence, normalization, post-gating, readers and
writers stay unchanged. CB25 now advances 16 pages/head and CB29 12; every
transaction remains four pages and every producer retains ownership.

The convolution/gates repair preserves its separate FP32 packs and sets
CB10/11/12/13 to `UnpackToDestFp32`. Convolution, history writes and recurrence
are unchanged.

Both repairs explicitly call `sigmoid_tile<VectorMode::RC, false, false>` after
`sigmoid_tile_init<false>()`, selecting the standalone BF16 numerical algorithm.
The kernel's physical DST configuration remains FP32. This is valid on the
current Blackhole backend: `llk_math_eltwise_unary_sfpu_macros.h:16–19` documents
that the macro's DST_ACCUM parameter is retained only for compatibility;
the bound check and destination addressing use runtime state. The macro at
lines43–47 forwards only the selected numerical callable, destination index,
and vector mode. `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h:14`
uses that runtime destination geometry. This target-specific observation
corrects the initial concern that the explicit template would alter physical
DST traversal; it is not an unverified claim for other architectures.

Source diffs are `flat_ordered_packed_precise.diff` and
`conv_gates_precise.diff`. Existing direct probes are reused through process-local
module substitution; their gate comparisons and .995 threshold are unchanged.

## Prepared validation

`../precise_gate_repair_jobs.json` contains eight fresh jobs: direct Watcher,
B6 short, B32 short, and B32 1024-answer for each candidate. Watcher and traced
performance jobs remain separate. Direct probes preserve independent persistent
states and cover nonaligned batches, DRAM/L1, saturation, tiny/zero QK and narrow
worker waves. The convolution probe additionally requires exact history shifts.

Host Python compilation and source/pre-commit API checks passed. No device
job was launched by this audit. Neither isolated repair is selected or claimed
to pass. If it fails, the new failure must be diagnosed with the same unchanged
gates; the original packed candidate is not eligible for selection on its
passing decoder-only evidence.
