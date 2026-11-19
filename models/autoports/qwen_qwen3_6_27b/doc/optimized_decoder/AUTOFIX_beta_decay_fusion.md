# Decode beta / decay preparation fusion

Source-only hypothesis, created after integrated167. No device or latency claim.
The parent serial runner must first pass the exact isolated probe, then all
whole-decoder correctness, repeated-run and traced latency gates.

## Removed topology and preserved arithmetic

Current `_gdn_compact` slices packed B columns16384:16432, runs native BF16
sigmoid, then reshapes to [B,1,48]. It separately slices A16448:16496, adds real
FP32 dt_bias, runs FP32 softplus(beta1,threshold20), multiplies the already
uploaded negative exp(A_log), and reshapes to [B,1,48]. The existing unfused_g
policy remains the reference. No setup exponential is recomputed on device.

`candidates/gdn_beta_decay_fused.py` directly gathers these packed BF16 rows,
real FP32 bias and negative A, and emits beta BF16 plus g FP32 in their existing
[B,1,48] TILE / DRAM layouts. One core owns one user / 32-head column tile,
so B32 uses64cores; B1 uses2cores. All tensor addresses are registered through
GenericOp inputs/outputs and runtime arguments. Optional supplied outputs keep
stable addresses. No public API shape or persistent state changes.

Arithmetic requires care: native BF16 sigmoid does **not** use the FP32 sigmoid
algorithm. `ckernel_sfpu_sigmoid.h::calculate_sigmoid<false,false,8>` uses the
exp_21f path, one reciprocal iteration and explicit BF16 nearest rounding.
The wrapper invokes this unchanged native function with FP32 DST addressing,
so the algorithm matches native BF16 sigmoid while the rest of the kernel uses
FP32 registers. Passing false as the DST-address mode would be incorrect.

Native FP32 softplus selects its polynomial through `INP_FLOAT32`, independently
of the DST width. The inline source defines this before including the native
header and calls `softplus_tile<true>(...,1.0,1.0,20.0)`. All four FP32 input CBs
use UnpackToDestFp32. Add, softplus, and final multiply remain separate SFPU
calls and FP32 stores; no reassociation or new FMA is introduced. Sigmoid and
softplus init are called separately to restore their programmable constants.
Beta is already explicitly rounded by native sigmoid before BF16 packing.

Relevant source:

- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`: native sigmoid default
  ACCURATE, and unary dtype selects preserve_fp32_precision / FP32 DST.
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`: native
  sigmoid template arguments and softplus parameter bits.
- `tt_metal/hw/inc/api/compute/compute_kernel_api.h`: distinction between
  addressing mode and calculate_sigmoid numerical specialization.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h`.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h`.

## Legal data movement

Packed BF16 faces contain32-byte row segments. Each read stages an aligned
64-byte segment and explicitly chooses the user parity half, preventing the
previous odd-user NoC alignment defect. Bias/negative-A row0 faces are aligned
64-byte FP32 reads. RISC widening only shifts32 BF16 bit patterns per operand;
there is no floating point host or RISC arithmetic. Each output gets two whole
tiles/user, so no partial-output NoC write alignment problem. Padded rows are
computed deterministically but not part of the logical comparison contract.

CB total is26KiB/core, with four FP32 inputs, one BF16 beta, one FP32 g and
one FP32 scratch page. The single output tile pair is consumed by the existing
recurrent reader without changing dtype or indexing. This adds no persistent
L1 allocation. Source checks and CPU row/face mapping pass B1/3/31/32;
`candidates/beta_decay_mapping_cpu.json` records the latter. These do not prove
hardware numerical equality or performance.

## Artifacts and commands

- Helper: `candidates/gdn_beta_decay_fused.py`.
- Frozen decoder: `candidates/beta_decay_on167.py`.
- Generator: `candidates/make_beta_decay_candidate.py`.
- Provenance: `candidates/beta_decay_on167_source.json`.
- CPU mapping: `candidates/check_beta_decay_mapping_cpu.py`.
- Exact same-input probe: `tests/run_beta_decay_probe.py`.

The generator modifies only `_gdn_compact` and appends namespaced helper code.
Fusion is enabled only when decode=True. Set `fused_beta_decay=False` for the
unmodified native chain; prefill is always unchanged.

Run under the existing exclusive device wrapper:

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_beta_decay_probe \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_beta_decay_fused.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/beta_decay171_probe.json
```

The probe loads actual dt_bias and A_log, covers B1/3/31/32, three input scales,
positive/negative saturation and softplus threshold boundary values. It records
per-output metrics, exact logical equality, repeated stability, output address
reuse and source SHA256s. Both beta and g must pass exact equality before using
whole-decoder PCC as evidence. Initial compile/API failures warrant repair,
not rejection. Full-tile SFPU sigmoid/softplus on mostly padded rows and small
B's limited active cores may erase the dispatch savings; only warmed traced
whole-decoder timing can decide selection.
