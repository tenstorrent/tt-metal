# Precise convolution/gate fusion: current source audit

Scope: CPU inspection only, 2026-11-13. No hardware operations, production edits, queued-source edits, or quality-gate changes. Original failed log/JSON/PT are preserved. `evidence.json` records source hashes, exact common-fragment comparisons, and saved-tensor measurements. Reproduce with:

```sh
python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/conv_gates_current_audit/analyze_cpu.py
```

## Finding and confidence

The candidate convolution matches the current production tiled convolution's source arithmetic and memory contract. It does **not** match the separate-TTNN convolution in the original direct probe at every rounding boundary. The `g` path has an independent, concrete algorithm-selection mismatch: missing `INP_FLOAT32` chooses the BF16 softplus polynomial despite FP32 DST. Thus the original output failure cannot yet be assigned solely to convolution. The queued matched-production probe plus the isolated macro repair are the appropriate discriminating experiments; source inspection alone does not establish their results.

The saved failing output is a real strict-gate failure, not a zero-head PCC artifact. At B13/wave104/step5/rank0, row10/value-head4 has FP64-recomputed PCC **0.988250103**, expected RMS **0.055557464**, RMSE **0.008917581** (16.05% relative), and max error **0.054443359** versus expected max **0.28125**. Only this head is below .995. Its norm is .628561 versus median head norm 5.250532. This is consistent with sensitivity to small upstream perturbations, but the saved output alone cannot prove cancellation or identify the responsible input. Keep the .995 gate.

## Convolution equivalence to production

Paths below are repository-relative, with `M=models/autoports/ornith_ai_ornith_1_5_9b`.

* `M/tt/kernels/tiled_conv/compute.cpp:15–31` and `M/tests/kernels/conv_gates_precise/compute.cpp:49–65`: identical startup, eight input waits, product `raw*tap3`, then three `history[j]*tap[j]` products/adds in the same order, FP32 SiLU, and final BF16 pack. The CPU checker compares these fragments byte-for-byte. Gate preparation runs only after the convolution result is packed and published.
* `M/tt/kernels/tiled_conv/reader.cpp:20–28` and candidate reader `:22–30`: identical CB0 raw, CB1–3 history, CB4–7 tap read addresses and barriers. Candidate's additional raw tiles96/97 and bias/decay reads happen afterward. B1..32 has exactly one padded height tile. Packed width3136 has98 width tiles; its first64 tiles are the same logical 2048 channels as production's sliced width2048 input. This assertion would need revisiting beyond B32; both helpers explicitly reject that range.
* `M/tt/kernels/tiled_conv/writer.cpp:12–19` and candidate writer `:14–21`: identical output write and three history writes. Each core owns one channel tile across all three history taps, with all source reads completed before writes. No neighboring channel or request is read through a different core's tile.
* `M/tt/tiled_conv.py:27–41,77–80` and `M/tests/multichip_conv_gates_precise_wide_modes.py:41–52,86–96`: same CB0–7/16 dtypes, page sizes, 8x8 core assignment, HiFi4, FP32 DST, and non-approximate compute. Candidate full-FP32 unpack overrides only CB10–13; convolution tap CB4–7 retain the production default. Extra CBs use only BF16/FP32 and do not change the exponent-class selection for the common CBs.

These checks establish source-level arithmetic/memory equivalence, not compiler-output or measured bitwise equivalence. The matched probe explicitly tests the latter.

## Why the original TTNN reference can differ

`M/tests/multichip_conv_gates_probe.py:172–177` widens each input to FP32, runs separate multiply/add operations, then FP32 SiLU and explicit BF16 typecast. In contrast, both production and candidate retain the four products/adds in one DST window and directly pack the final SiLU to BF16.

The most concrete input-precision difference is the tap unpack mode. `tt_metal/jit_build/genfiles.cpp:1068–1086` selects TF32 as the default conditional unpack destination for FP32 DST with these BF16/FP32 CBs; `tt_metal/jit_build/data_format.cpp:139–154` maps FP32 source to that destination. The TTNN binary SFPU factory explicitly selects `UnpackToDestFp32` for source/intermediate CBs at `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1371–1381`. The direct probe's odd steps deliberately supply full-FP32 taps, exposing this difference. Even steps use BF16-widened taps, which do not lose additional mantissa bits on TF32 unpack.

There is also a different final conversion path: TTNN typecast explicitly applies RNE with bit masking in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:273–286`, while both tiled kernels use direct BF16 packing. Do not claim these paths are universally bitwise interchangeable. Separate intermediate packing is another boundary to preserve when constructing an exactly TTNN-matched experiment.

At the failed record, candidate convolution versus TTNN/Torch is PCC .999998331, max error .001953125; output global PCC is .999992609 but the head gate fails. Those observations support investigating propagation of the convolution difference, but do not isolate it from `g` yet.

## Independent gate mismatch

The wide-modes helper has no `INP_FLOAT32` define. `ckernel_sfpu_softplus.h:181–202,220–260,410` selects different constants, polynomial, and vector traversal using that macro, independently of `fp32_dest_acc_en`. Production TTNN passes the FP32 LHS dtype to activation code generation (`binary_ng_program_factory.cpp:1103` → `binary_ng_utils.cpp:642` → `unary_op_utils.cpp:1186–1197`), which emits the macro. Candidate's same softplus parameters `(1,1,20)` and FP32 intermediate buffers do not compensate for the absent macro. The residual `g` max error .00003814697265625 with bitwise beta is consistent with this concrete mismatch; it does not establish that no other rounding difference remains.

The candidate already preserves the sum and softplus FP32 packing boundaries and full-FP32 unpack for CB10–13. Its BF16 beta sigmoid is explicitly `sigmoid_tile<VectorMode::RC,false,false>`, and beta is bitwise equal in this failure. Adding `INP_FLOAT32` should not change convolution SiLU or that explicit sigmoid numerical branch: Blackhole `ckernel_sfpu_silu.h:13–34` and `ckernel_sfpu_sigmoid.h` select those branches from template FP32 mode rather than this macro. The sibling agent owns the isolated repair; no duplicate implementation was made here.

## Discriminating validation, unchanged gates

The existing queued `multichip_conv_gates_matched_wide_modes_probe.py` already keeps independent production history/state and compares:

1. Fused convolution versus production tiled convolution **bitwise**, with separate TTNN/Torch convolution PCC checks retained.
2. Candidate recurrence versus production post-loop with production convolution and the candidate's exact gate tensors: state **bitwise**, output global/per-head **>= .995**. This isolates packed source addressing and recurrence/postnorm semantics.
3. Candidate versus production convolution + separate production TTNN gates: output/state global/per-head **>= .995**. This exposes any gate-preparation error.

The original separate-TTNN-convolution result must remain recorded as failed. If identical-input recurrence fails, investigate its reader/offset/rounding separately; convolution sensitivity is not an excuse. If matched separate-gate control fails, compare macro-repaired `g` directly and localize its sum/softplus/multiply boundaries. No further probe is necessary before these already-prepared discriminators run. Full decoder short/1024-answer gates remain required for any selected path.
