# AutoDebug: catastrophic real-weight prefill output

## Scope and confidence

This is a source-only investigation. No TT device was opened and no hardware
command was run. The reported final-output PCC (`0.000324`) and BF16
infinities/huge magnitudes come from the stage work log; this report does not
claim a new reproduction.

The headline finding is a concrete caller/configuration contract mismatch with
strong in-tree corroboration. Hardware still has to confirm the first divergent
tensor because the failing test synchronizes only when it reads the final layer
output.

## Direct observations, kept separate from interpretation

- The functional real-weight layer-39 path passes prefill at PCC `0.9999927`.
- The optimized default stores QKV, O, gate, up, and down weights as
  width-sharded DRAM tensors. `_dram_weight_memory_config()` makes each shard
  exactly `N / dram_cores` columns (`optimized_decoder.py:155-173`), and the
  P300 Blackhole architecture has eight DRAM banks
  (`blackhole_implementation.hpp:93`).
- Every optimized prefill projection uses
  `MatmulMultiCoreReuseMultiCastProgramConfig`, but its `per_core_N` is derived
  from the **compute** grid width (`ceil(N_tiles / grid_x)`) rather than the
  DRAM weight shard width (`optimized_decoder.py:572-595`). The failing grid is
  `(11, 10)`.
- With the model dimensions `hidden=8192`, `intermediate=28672`, QKV width
  `10240`, and eight DRAM banks, the two contracts are:

  | role | N tiles | DRAM shard width (tiles) | current 11-wide `per_core_N` | prior 8-wide `per_core_N` |
  | --- | ---: | ---: | ---: | ---: |
  | QKV | 320 | 40 | 30 | 40 |
  | O | 256 | 32 | 24 | 32 |
  | gate | 896 | 112 | 82 | 112 |
  | up | 896 | 112 | 82 | 112 |
  | down | 256 | 32 | 24 | 32 |

- Repo-standard Blackhole attention and MLP configuration code deliberately
  derives prefill `per_core_N` from `mesh_device.dram_grid_size().x`, not the
  compute grid. It states why: matching the DRAM shard width avoids matmuls
  that "silently give bad PCC" (`models/common/modules/mlp/mlp_1d.py:897-936`,
  `models/common/modules/attention/attention_1d.py:1636-1652`, and
  `models/tt_transformers/tt/model_config.py:715-716,1299-1309,1359-1366`).
- The generic 2-D matmul validator checks `per_core_N == in1 shard width` only
  when sharded operand B is in L1
  (`matmul_device_operation.cpp:1259-1267`). A width-sharded DRAM B bypasses
  that equality check. The program factory separately computes
  `per_core_N_storage = ceil(N_tiles / num_dram_banks)` and then attempts to map
  each worker-width block across DRAM banks
  (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:357-381,1302-1363`).
  Thus the inconsistent 11-wide candidate is allowed to execute, which is
  consistent with the observed silent numerical failure rather than an API
  validation error.
- The earlier 8-wide candidate happened to satisfy the storage-width contract,
  but gate/up `in0_block_w=8` exceeded L1. Moving to 11 columns reduced
  `per_core_N` enough to pass the L1 allocation, while simultaneously breaking
  the DRAM-shard-width invariant. That is a controlled contrast, not ordinary
  BFP4 error.

## Headline finding: the legal-L1 11-wide prefill config is not legal for these DRAM-sharded weights

**Verdict: source-verified contract mismatch; highest-probability root cause.**

The optimized path couples two independently chosen geometries:

1. weight materialization shards N over the eight DRAM banks; and
2. prefill program construction shards N over eleven compute columns.

The in-tree common modules say these widths must agree on Blackhole to avoid
silent bad PCC. They agree for the rejected 8-wide configuration and disagree
for every prefill projection in the current 11-wide configuration. The
catastrophic result therefore should not be attributed to precision until this
contract is corrected or isolated.

The **predicted earliest bad boundary is the fused QKV projection**, not gate or
up: QKV is the first call to `_prefill_linear()` in `prefill_forward`
(`optimized_decoder.py:768-776`) and already uses `per_core_N=30` against a
40-tile DRAM shard. O, gate, up, and down repeat the same defect later. A bad
QKV can corrupt the cache and attention; a bad gate/up value can then be
amplified by SiLU/multiply/down into the observed huge values or infinities.
The current final-output readback cannot distinguish these stages.

### Smallest focused verify/refute experiment

Add a temporary diagnostic test (do not change the production default yet)
that runs one real-weight projection and synchronizes immediately:

1. Use the exact seeded BF16 normalized activation from the failing batch/seq
   contract and the real layer-39 QKV weight.
2. Print/assert logical shape, padded shape, dtype, layout, buffer type, DRAM
   shard width in tiles, and `program_config.per_core_N` before dispatch.
3. Compare finiteness and PCC for:
   - A: current DRAM-width-sharded weight + 11x10 + `per_core_N=30`;
   - B: the same weight values in DRAM **interleaved** + the identical 11x10
     program (removes only the storage-shard contract);
   - C: DRAM-width-sharded weight + a program whose `per_core_N=40` (eight N
     blocks; eleven columns may leave columns idle), with the smallest L1-legal
     `in0_block_w`/subblock settings.
4. Repeat the same three-way control for gate with `per_core_N=82` versus 112
   only after QKV is localized.

Prediction: A is the first catastrophic case; B and/or C are finite with normal
low-precision PCC. If A and B fail identically, the DRAM-shard mismatch is
refuted for that role. If QKV passes all controls, read back O, post-attention
residual, post-attention norm, gate, up, gated multiply, and down in that order;
the first non-finite/low-PCC tensor decides which later role to isolate.

The smallest production intervention implied by a confirming result is to
derive prefill `per_core_N` from the weight's actual DRAM shard width (and allow
unused compute columns), or to keep a separate interleaved prefill weight. If
the bank-aligned block is still too large for L1, tune `in0_block_w`,
`per_core_M`, and output subblocks without changing the storage-width contract.
Add a caller-side invariant so future incompatible configs fail instead of
silently executing.

## Ranked follow-up hypotheses

### 2. Exact BFP4/LoFi + DRAM-sharded/fused-batch 2-D gate/up variant has a target-specific numerical bug

**Verdict: possible only after hypothesis 1 is controlled; generic BFP4/LoFi
blame is refuted.**

The failing gate/up policy is BF16 input, BFP4 weight, LoFi,
`packer_l1_acc=True`, BF16 output, and `in0_block_w=4`. That exact combination
could expose a Blackhole matmul factory/kernel bug, and gate/up are the most
plausible point for finite bad values to be amplified into infinities.

However, BFP4+LoFi itself is a supported Blackhole policy. The Blackhole-only
DeepSeek prefill PCC suite deliberately tests BF16 x BFP4 with LoFi and the same
compute flags on an 11x10 2-D matmul
(`models/demos/deepseek_v3_d_p/tests/pcc/test_deepseek_v3_matmul_pcc.py:34-49,56-95`).
Those tests use interleaved weights and different shapes, so they clear the
dtype/fidelity pair but not this exact DRAM-sharded/fuse-batch geometry.

**Predicted first bad boundary:** gate or up projection, while QKV, attention,
post-attention residual, and post-attention norm remain finite and have
reasonable PCC.

**Focused experiment:** first hold a bank-aligned/interleaved weight contract
fixed. With the same real normalized activation and same program geometry,
compare only:

- BFP4 + LoFi;
- BFP4 + HiFi2;
- BFP8 + LoFi;
- BFP8 + HiFi2.

Read gate and up separately before the fused multiply. A precision candidate
may have moderate PCC loss; any isolated non-finite or orders-of-magnitude
output is a kernel/config bug, not acceptable quantization. Do not use an
all-BFP8 full-layer pass alone because it changes too many boundaries and can
mask the storage mismatch.

### 3. `fuse_batch=True` M accounting differs from the assumed independently padded `[1,32,18,K]` contract

**Verdict: lower probability, not disproved by final-output PCC.**

The optimized code manually computes
`per_core_M=ceil(batch * ceil(seq/32) / grid_y)=4` and sets `fuse_batch=True`
(`optimized_decoder.py:588-594`). For batch 32 and logical seq 18 this assumes
32 physical M tiles and eight active compute rows. Shapes are not reduced or
flattened before the matmul. Common prefill modules more often enter these
configs with an already flattened/standard prefill M representation, so the
exact physical/padded shape consumed by `get_M_dim(..., fuse_batch=True)` must
be checked rather than inferred from the logical shape.

**Predicted first bad boundary:** QKV, with a repeatable row/user/tile pattern
(missing or garbage users/rows) rather than a projection-role-specific error.

**Focused experiment:** after fixing/controlling `per_core_N`, print the TTNN
logical and padded shape and the factory-derived fused M tiles. Compare a single
QKV matmul on `[1,32,18,8192]` against the same padded data explicitly reshaped
to `[1,1,32*32,8192]`, slicing both back to the same 32 users x 18 logical
tokens. Compare per-user finiteness and PCC, not only aggregate PCC. Also compare
logical seq 18 versus 32. If both representations agree, this hypothesis is
refuted.

### 4. The fused `mul(SiLU(gate), up)` exposes or amplifies a bad projection

**Verdict: likely an amplifier, not the earliest cause.**

The functional path uses separate `silu` then multiply; optimized prefill uses
`ttnn.mul(... input_tensor_a_activations=[SILU])`. The common MLP path uses the
same fused form, so its mere presence is not an obvious defect. It can make a
corrupt gate/up tensor catastrophically large.

**Focused experiment:** once gate and up are proven finite, compare the fused
form to separate `ttnn.silu` + `ttnn.multiply` using the same inputs and output
dtype/memory config. If gate/up are already non-finite or huge, do not blame the
consumer.

## Explicit adjudication of requested suspects

### `WormholeComputeKernelConfig` on Blackhole

**Refuted as a bug.** TTNN now has one unified `ComputeKernelConfig`; C++ aliases
both `WormholeComputeKernelConfig` and `BlackholeComputeKernelConfig` to it
(`compute_kernel_config.hpp:21-42`). Python also defines
`BlackholeComputeKernelConfig = WormholeComputeKernelConfig`
(`ttnn/ttnn/types.py:58-60`). Blackhole model code uses the Wormhole spelling.
The name is backward compatibility, not a Wormhole-only lowering choice.

### DRAM-sharded weights with a prefill 2-D matmul

**Legal in principle, wrong as currently paired.** The common attention/MLP
paths intentionally combine DRAM-width-sharded weights with a 2-D prefill
program. The defect is not the family; it is deriving `per_core_N` from eleven
compute columns instead of the eight-bank weight shard. The C++ factory contains
an explicit DRAM-width-sharded reader, which further confirms the family is
supported.

### BFP4 gate/up + LoFi

**Cannot explain infinities as ordinary precision loss.** It is a supported
Blackhole policy and is covered by Blackhole-only 2-D prefill tests. The exact
DRAM-sharded/fuse-batch geometry remains a testable target-specific hypothesis,
but only after the higher-confidence storage/program mismatch is removed.

### Shape, padding, and weight-memory mismatch

- Model dimensions and all K/N dimensions are tile aligned.
- The optimized DRAM helper refuses N that is not divisible by the active DRAM
  shard count, so there is no silent tail padding in these five weights.
- The real state is loaded into an HF layer with `strict=True`, and the same
  tensors pass the functional decoder. A transposed or wrong logical weight
  shape is therefore not supported as the current cause.
- The optimized constructor omitted the functional constructor's explicit
  expected-shape checks (`functional_decoder.py:186-210`). Restoring them is a
  worthwhile guardrail, but it does not explain this repro.
- Logical seq 18 is padded internally. The M formula appears intended to cover
  that contract, but hypothesis 3 gives the exact control needed to prove the
  lowered M calculation.

## Recommended experiment order

1. Single real QKV projection: current DRAM-sharded mismatch versus interleaved
   B and bank-aligned `per_core_N` controls.
2. Boundary ladder through QKV, O/residual/norm, gate, up, fused multiply, down.
3. Only if gate/up are first bad after bank alignment, run the four-way
   dtype/fidelity control.
4. Only if row/user patterns remain, adjudicate `fuse_batch` physical M.
5. Rerun the original full real-weight test after the smallest proven fix, then
   the non-aligned/repeated-run and watcher checks required by the stage.

## Remaining uncertainty

Source inspection proves the mismatch and its direct correspondence to an
in-tree Blackhole silent-PCC warning, but it cannot prove which of the five
projection outputs first becomes non-finite on this device. Because TTNN work is
asynchronous and the test reads only the final output, the boundary ladder is
required before assigning the visible infinities specifically to BFP4 gate/up.
