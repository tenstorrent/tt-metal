# AutoDebug: router core state and the first fused-tail RMSNorm

## Verdict

The new paired TP1→TP4 failure is localized to the first shared-branch tail
RMSNorm, after its interleaved-to-sharded conversion. The actual input, including
physical padding, and weight are bit-identical across ranks in the failing
invocation. Its output differs within the tile owned by router core `(1,0)`.
This is strong evidence for investigating that core's norm execution or output
storage. It does **not** yet prove a particular state leak from the router.

This audit did not find a source-proven missing SFPU constant, predicate,
stochastic-rounding, or basic pack-format reset. Several plausible reset fixes
are specifically contradicted by source or cached machine code below. Do not
add them as a claimed fix. The smallest useful native probe is now the norm
receiver's broadcast statistic, first multiply result, and gamma multiply
result, including the boundary between DEST and the packed output.

No hardware, TTNN imports, resets, builds, runtime edits, or kernel edits were
performed by this investigation. This report is its only new file. It continues
the delegated AutoFix/AutoDebug source audit; the earlier auxiliary CLI was
stopped after its missing-`bwrap` environment failure, as recorded in
`AUTODEBUG_indexed_expert.md`.

## Evidence and provenance

- Runtime: `tt/multichip_decoder.py`, SHA256
  `070613ddc32b5cc8a22fd92a64cb541de9a9f27152852f1caad0843d1d06903d`.
- The hardware owner runs the ordinary paired TP1→TP4 graph through
  `tests/diagnose_paired_boundary.py`, retaining one actual boundary at a time.
  The failure command uses layer 5, input length 4096, 128 traced decode steps,
  cache checks, fused tail, hybrid experts, shared geometry 1, grouped MoE
  reduction, LoFi QKV/WO, and BFP8 attention CCL.
- `full_bfp8_paired_attention_ag.failure.json/.pt`: actual attention AG exact
  across ranks while final output differs at indices `[39,47,58,63]` on rank 1,
  maximum `0.015625`, step 33 / position 4129.
- `full_bfp8_paired_postnorm.failure.*`: actual post-attention norm return exact
  at an output failure. This operation precedes the router.
- The paired grouped-MoE checkpoint has exact `shared_reduced` and
  `routed_reduced` at an output failure. This rules out a first logical
  difference in the reduced branch results in that run.
- `full_bfp8_paired_tail_shared.failure.json/.pt`: the actual shared tail
  RMSNorm return first differs on rank 3 at step 12 / position 4108. Seven finite
  values change, maximum `0.125`. CPU mask analysis identifies indices
  `[36,38,42,52,54,60,63]`; six differ by one BF16 encoding step and one by two.
  These are numerical changes, not a large layout permutation.
- Hardware-owner follow-up `full_bfp8_paired_tail_shared_io.failure.pt` and
  `.failure.summary.json`: the same invocation's complete physical
  `32×2816` input is bit-identical before and after i2s across ranks, including
  14,490 nonfinite padding words. Weight is also exact. The norm return differs
  at logical columns `[42,44,60,63]`, rank 0 versus the other ranks, with 24
  changed physical words, all in tile 1. This is the strongest localization.
  No raw tensor values are reproduced in this report.
- Prior physical-input frozen experts plus native router prefix passed on both
  router cores 0 and 1 with matching allocation maps. Native gate alone did not
  reproduce a simple persistent-state defect; the whole-layer context matters.

The preceding boundary controls were separate runs. Only the `tail_shared_io`
control proves input/output equality/difference for the same RMSNorm invocation.
The earlier paired-cleanup failure also means live TP1 Python references are
not a sufficient explanation. Pairedness is an observed correlation.

## Exact consumer path and core ownership

`tt/multichip_decoder.py:1015` implements the fused decode tail. Its first call
normalizes `shared` with `post_feedforward_layernorm_1`, after
`to_memory_config(value, memory)`. It supplies no compute-kernel configuration.
`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:16` therefore selects
HiFi4, `math_approx_mode=true`, and `fp32_dest_acc_en=false`. Output dtype follows
the BF16 input. This differs from the earlier FP32 attention normalization.

For H=2816 and the model's 11×8 row-major width sharding, each core owns one
32-column tile. Core `(1,0)` owns columns 32–63. The program factory enables the
two-stage reduction for this rectangular row-major width-sharded grid:

- `sharded_layernorm_factory_helpers.cpp:196` gives one tile row per all-to-all
  worker, one first-stage worker per grid row, and eight such workers total.
- At `:264`, their grid is `1×8`: the `x=0` column.
- At `:295`, the remaining workers begin at `x=1`.

Thus core `(1,0)` is **not** an allgather/statistic worker. In
`device/kernels/compute/layernorm_sharded.cpp`, `is_allgather_worker` guards the
global reduction and rsqrt at `:426` and `:463`. The core under suspicion instead
uses the received statistic in `mul_bcast_cols_init` / `mul_tiles_bcast_cols`
at `:489`, then gamma in `mul_bcast_rows_init` / `mul_tiles_bcast_rows` at `:535`.
Those are FPU operations. There is no norm activation on this invocation.

A bad global scalar computed by an `x=0` worker would ordinarily affect many
output tiles. The observed tile-1-only mask is more consistent with the local
statistic copy, local FPU operations, packing, or output storage. That is a
prediction for the next probe, not proof that the global scalar is correct.

## Reset hypotheses checked

### 1. Router SFPU programmable constants: not supported

The selected router is the single-block ungrouped top-8 path with sigmoid
disabled and softmax output enabled. It uses `SFPCONFIG` to broadcast through
LREG14 and initializes reciprocal constants internally. Its topk init is
intentionally empty because an earlier constant initialization would be
overwritten by those broadcasts.

The following consumer mechanisms contradict a blanket missing-constant reset:

- `llk_math_eltwise_unary_sfpu_init.h:259` runs common SFPU initialization before
  the per-op callback. The old comment describing once-per-kernel hoisting in
  the lower-level header is not the active Metal control flow.
- `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h:62` resets SFPU config
  with `_init_sfpu_config_reg`, configures the SFPU address modifier, and resets
  counters. This clears the router's index-tracking config on the next SFPU init.
- `ckernel_sfpu_rsqrt.h:29` calls `sqrt_init`; `ckernel_sfpu_sqrt.h:115` explicitly
  initializes every programmable constant used by the selected approximate
  rsqrt (`vConstIntPrgm0` and `vConstFloatPrgm1`). The accurate path initializes
  its third constant as well.
- More directly, core `(1,0)` does not execute rsqrt in this norm. Adding a
  reciprocal-constant reset on that receiver has no source-supported mechanism.

### 2. Router leaves a lane predicate active: refuted for the inspected binary

C++ raw-instruction searches alone are misleading: the selected router's
reciprocal contains an SFPI `v_if`, so the compiler emits predicate cleanup.
CPU disassembly of this existing cached Blackhole math ELF shows:

```text
0x806c: sfpsetcc L3,0x000,0
0x8070: sfpmad   ...
0x8074: sfpmad   ...
0x8078: sfpencc  0x003,10
0x8080: sfpconfig 14,0,0
```

`runtime/sfpi/include/sfpi_constants.h:177` defines immediate 3 as setting both
enable and result, and modifier 10 as immediate enable/result. This restores
all lanes after the conditional reciprocal. There is no later predicate
operation in this specialization. Therefore absence of an explicit C++
`TTI_SFPENCC` in the generic init is not evidence that this gate leaks a mask.

Inspected ELF:

```text
/home/mvasiljevic/.cache/tt-metal-cache/4029278992320149898/kernels/generalized_moe_gate_kernel/2476921738570714601/trisc1/trisc1.elf
SHA256 646a20d36ad35ae854174e1340c432dba540a53bd125ec782eb793666678f58a
```

Its generated arguments confirm `topk=8`, `num_blocks=1`, `softmax=1`,
`enable_sigmoid=0`, and its descriptor is BF16 DEST / full synchronization.
This is a matching cached specialization, not proof from a captured workload
handle that the failure used that exact ELF.

### 3. Router leaves pack rounding, ReLU, or UINT16 format: not supported

The router ends by packing BF16 gate values, reconfiguring the packer to UINT16,
packing indices, then releasing DEST
(`generalized_moe_gate.hpp:242`). There is no explicit cleanup there, but the
next kernel's startup owns these settings:

- `layernorm_sharded.cpp:169` calls `compute_kernel_hw_startup`.
- That startup calls `llk_pack_hw_configure`, `llk_pack_init<Default>`, and
  `llk_pack_dest_init`, not merely a short format reconfiguration.
- `tt_llk_blackhole/common/inc/cpack_common.h:388` zero-builds the pack config
  and programs its source/destination formats. At `:445` it clears L1
  accumulation. At `:448` it zero-builds `PCK_DEST_RD_CTRL`, and at `:468`
  full-writes it, including unsigned/int8/32-bit-read and 10-bit-round controls.
- `configure_pack` at `:576` clears destination-format overrides, writes ReLU
  mode and threshold from the startup argument zero (`:624`), and restores a
  full edge mask and row mapping (`:646`).
- The unpack startup calls
  `configure_unpack_AB<is_fp32_dest_acc_en,false,false,false>`
  (`llk_unpack_common.h:95`). Its write at `cunpack_common.h:844` explicitly
  disables FPU, gasket, and packer stochastic rounding.

The router itself does not enable those stochastic-rounding controls. BF16
one/two-ULP changes make arithmetic/rounding a useful observation, but do not
establish a stale rounding flag. Repeating the same reset after startup would
test timing or ordering as well as state; a passing run alone would not prove
which field was wrong.

The source and installed copies of `compute_kernel_hw_startup.h`,
`cpack_common.h`, `cunpack_common.h`, and `llk_math_common.h` were compared and
are byte-identical. This is stronger than auditing a different installed
header set, but does not replace active-ELF provenance.

### 4. Unreset raw SRC/DEST contents: already weakened, not a new finding

The prior report established firmware/kernel entry ZEROACC/ZEROSRC and default
math format/zero-flag initialization. The norm's elementwise init also restores
the operand-driven zero-substitution state
(`llk_math_eltwise_binary.h:645`). No additional missing reset was established
here. This does not rule out a hardware hazard involving these resources; it
does rule out claiming that the source never initializes them.

## Minimal verify/refute sequence

1. **Completed by the hardware owner:** capture the first shared RMSNorm's
   actual physical input and weight, with its actual return, in one failing
   paired invocation. The reported exact physical input rules out i2s input
   corruption for that invocation. Preserve its NaN/Inf padding bits in any
   frozen reproduction; uploading only logical rows changes this fixture.

2. **Router-core ownership control:** move only the router's persistent
   sharded tensors and native one-core grid from `(1,0)` to another norm
   receiver, such as `(2,0)`, in a diagnostic wrapper. If the first bad norm
   tile follows to columns 64–95, the router/core relationship is much stronger
   than one passing placement. An otherwise-unused core outside all later
   sparse/shared/norm grids, such as the coordinator's proposed `(10,9)`, is a
   subsequent containment control. A pass is a placement workaround, not a
   source-level explanation. Keep original shapes, dtype, math config, and the
   paired preamble unchanged.

3. **Smallest native localization, if input-exact failure persists:** in the
   first shared norm only, compare receiver `(1,0)` against a neighboring
   receiver and across ranks at these points:

   | Checkpoint | Consumer source | What it separates |
   | --- | --- | --- |
   | `dfb_ex_global` before use | `layernorm_sharded.cpp:489` | Statistic multicast/local copy versus later local math |
   | `dfb_im`, before gamma | `:512` pack / `:526` completion | First FPU broadcast multiply versus gamma stage |
   | DEST immediately after gamma multiply | `:544` | FPU result versus pack/store |
   | `dfb_outgamma` after pack completes | `:557` | Packer result versus later output-memory overwrite |

   Record equality/hash or selected-word encoding diagnostics rather than
   dumping full model activations into telemetry. Compare all physical rows,
   retaining the logical-row statistics separately. Instrumentation may mask
   the bug; a passing instrumented run is inconclusive.

4. **Only after a wrong state value is observed:** fix that field's owning
   initializer or the proven ordering edge, then rerun both the focused case
   and original paired BF8 command. If state readback is chosen, read the
   actual core's `PCK_DEST_RD_CTRL`, ReLU config, ALU stochastic-rounding bits,
   source formats and zero-substitution flag after startup/operation init;
   compare expected field values rather than assuming stale defaults. Avoid
   adding a broad “reset everything” sequence or switching FP32 as an
   explanation. A synchronized repeat of the existing reset is a diagnostic
   intervention, not yet a justified production fix.

The hardware owner is conducting the placement and boundary experiments.
No duplicate hardware run was launched by this source audit. A docs-only
change needs no build under the repository's AGENTS.md; all new verification
here was source inspection, cached-ELF disassembly, and byte comparisons.
