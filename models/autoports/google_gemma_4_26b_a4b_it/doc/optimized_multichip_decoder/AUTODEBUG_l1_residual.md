# AutoDebug: four-core L1 residual candidate

## Verdict and scope

**Updated after the parent's boundary capture:** corruption is localized to the
final sharded fused add. A source defect in BinaryNg's FPU batching is now the
leading mechanism: it schedules eight tiles with FP32 half-DEST, whose capacity
is four. The public mixed-FP32/BF16 ADD contract accepts the call. Device
verification of this specific mechanism is still pending; see the follow-up
section below. The original inspection and conditional plan are retained as
the diagnosis history.

Original source-only conclusion: the highest-value next experiment was a
same-invocation comparison of the final residual, tail result before and after
resharding, and final add output, checked both across ranks and against the
passing TP4 baseline. Full-attention evidence establishes a second symptom:
replicas can remain identical while output accuracy collapses. The candidate introduces both new norm
geometry and new sharded elementwise/movement kernels; those must be separated.
The present failure does not justify a precision change or a CCL replacement.

This is the delegated fresh-context, source-only AutoFix diagnosis. No hardware,
TTNN imports, implementation edits, builds, or additional agents were used.
Only this report and the stage work-log entry were written. The parent owns
all device experiments. Repository AutoDebug's auxiliary CLI was not launched
because the delegated task explicitly prohibited further agents.

## Evidence

- `residual_l1_sliding.log` and `residual_l1_sliding.failure.json` record layer 0,
  real 4096-token prefill, 128 advancing decode positions, trace, cache checks,
  and `--residual-l1`. The paired TP1 reference completes before TP4 fails.
- Runtime SHA256: `a98c9146eb988b451a4ddc457bf692d7bc87d813f2de56b4b2022a4ebe8c3116`.
  Runner SHA256: `a8d30f2a21b2772bf1b2ed6e39d5092cb0ddc3db95d15843eeef25d47276c788`.
- At the first replay, step 0 / absolute position 4096, rank 1 equals rank 0;
  ranks 2 and 3 each differ in exactly 16 of 2816 logical elements. All values
  are finite; maximum differences are 50.65625 and 68.34375 respectively.
  The report does not contain mismatch indices or intermediate tensors.
- The default policy passes the same original workload (stage work log and
  parent confirmation). The parent also reports passing QKV4/persistent trials.
  These controls do not localize the candidate's first bad operation.
- Warm decode outputs are not read by the runner. Thus the log proves a failure
  observed during trace replay, not that eager decode was correct beforehand.
- Follow-up `residual_l1_full.json`/`.log`: layer 5 completes all 128 advancing
  traced positions and one duplicate replay per position with exact replica
  agreement, then fails the TP1 output PCC assertion. Prefill PCC is
  0.9999453677; first decode PCC is 0.7957091692; minimum is 0.5369554510 at
  decode step 14 / position 4110. Cache PCC spans 0.9999714–0.9999736.
  This run records runtime SHA256
  `d93255befb6cadf28063f8077fb2444b3dce6c769ec5211696b5589d7d3a7a94`
  and runner SHA256
  `3c3be60d6387a968e3f97a6468b417272ab7e924140628bf5a4980bd28febc37`.
  Its source hashes differ from the sliding failure; preserve both provenances.
  The candidate is invalid even when the replica-equality check passes.

Exact reproducer, for the parent only:

```bash
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --layer 0 --length 4096 --steps 128 --trace --check-cache --residual-l1 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/residual_l1_sliding.json
```

The full-attention reproducer substitutes `--layer 5` and output
`residual_l1_full.json`, keeping all other flags. Neither command was run by
this diagnosis agent.

## Source checks and exclusions

1. **The requested shard geometry is internally consistent.** Runner lines
   250–270 specify logical width 2816, four width shards of 704 elements,
   physical height 32, and a `(4,1)` grid. Each shard is 22 tiles wide;
   `block_w=22`, `block_h=1`, `subblock_w=2` divide that geometry exactly.
   Per-core tensor storage is 90,112 bytes for FP32 and 45,056 for BF16. This
   calculation alone does not establish complete L1 capacity or non-overlap.

2. **No late-binding closure cause for this invocation was found.**
   `normalize` is installed only in the final TP4 loop iteration; `decoder`,
   `norm_program`, and `original_normalize` do not change during its forwards.
   Its `weight` is an argument, so later runner locals named `weight` cannot
   overwrite it. Calling a function stored on the instance does not add a
   hidden `self`. The local helper inside `_fused_tail` has a separate scope.

3. **The override changes three full-hidden FP32 norms, not attention head
   norms or the fused BF16 tail norms.** `MultichipDecoder._forward` calls the
   override at input, post-attention, and common norm sites (lines 1151–1156).
   The attention and router wrappers were constructed with the original bound
   normalizer; Q/K head normalization keeps that original function. The router
   receives `normalized=` here, so it does not invoke its stored normalizer.
   The override lacks the original hidden-width guard, but no non-hidden-width
   invocation through it was found in this failing path.

4. **Projection boundaries already convert correctly.** `_Projection.__call__`
   explicitly converts QKV input to interleaved L1 (line 119).
   `GeneralizedRouter.__call__` converts its scaled sharded input before native
   linear (`optimized_decoder.py`, lines 934–937). The candidate explicitly
   converts expert/shared working inputs to interleaved L1 at lines 1160/1166.
   A claim that these matmuls consume an incompatible four-core shard is not
   supported by source.

5. **Final residual and tail storage genuinely change.** The long-lived
   residual is FP32 on four cores. Fused tail norms still use the original
   88-core BF16 layout; line 1207 reshares the combined tail from 88 cores to
   four, then lines 1208–1213 execute FP32 + BF16 → BF16 with a fused scalar
   activation and sharded output. This is a distinct kernel/storage boundary
   from the default DRAM-output tail.

6. **No explicit premature deallocation was found for the residual or new
   normalized tensors.** The residual remains a Python local through the
   final tail. The new normalizer is out of place. A lower-level asynchronous
   lifetime or allocator overlap bug remains a hypothesis, not a demonstrated
   Python ownership error.

7. **The earlier router-core workaround is already active.** The generalized
   gate uses `(10,9)`, outside the new four-core norm grid and original 11×8
   tail grid (`multichip_decoder.py:807`). The previous report about router core
   `(1,0)` is useful history, but is not evidence that this candidate has that
   same defect.

## Prioritized hypotheses and discriminating checks

### Required baseline comparison for full-attention accuracy

Use the original paired preamble and real4096 inputs. Save matching TP4 baseline
and TP4 candidate boundaries at step 0 first, then step 14 if needed. Compare
PCC, maximum error, norm, and finite counts; exact equality across ranks alone
cannot detect the observed full-attention failure. Changes in reduction order
may prevent exact baseline/candidate equality without implying a defect.

Start with post-attention residual, the two reduced branches, pre/post-reshard
combined tail, and final output. These bisect the graph without storing every
intermediate. For the first materially divergent region:

| Region | Focused next boundaries / oracle |
| --- | --- |
| Attention/residual | raw and weighted input norm; reduced attention before/after post norm; residual; compare just newly written decode K/V rows using the actual page table |
| Routing/MLP | common normalized residual; scaled router activation; selected expert IDs and gate values; normalized expert/shared inputs; each reduced branch |
| Tail | same input to each branch norm; combined result before/after reshard; final residual/add/scalar result |

Compute CPU RMSNorm/weight/add/scalar oracles on the captured actual logical
inputs outside capture where useful. In particular, compare the final output
against BF16-rounded `(captured_residual + captured_combined) * layer_scalar`;
this distinguishes bad upstream tensors from bad final arithmetic even when
all replicas agree. Preserve physical padding separately for native replay.

Aggregate cache PCC is dominated by the prefill allocation and must not be
used to prove every decode cache row correct. A good decode-row cache result
would lower the priority of input norm/QKV and move attention output norm,
common norm/router, and the tail ahead. The full workload teacher-forces fixed
inputs; its large error is not explained by autoregressive token drift.

The full deterministic accuracy loss makes four-core norm/weighting semantics
co-equal with the final reshard/add hypothesis below. It does not establish that
the sparse sliding corruption and full loss share one root cause.

### 1. Final reshard/add/output read introduces the first difference

The exactly 16 changed logical elements may be one tile-face row. This is an
inference until the actual column mask is recorded. Large finite sparse
corruption is more compatible with a local store/read/elementwise issue than a
single wrong global RMS statistic, which would normally affect many columns.

**First diagnostic run:** retain actual device tensors for the final residual,
combined tail immediately before line 1207, combined tail immediately after
line 1207, and final result. Log metadata and mismatch indices after replay,
outside capture. Do not recompute a second forward to obtain these boundaries.
Use per-device bit comparisons, including the complete physical 32×2816 tile
payload; do not discard padding when saving a frozen reproducer. Compare the
four-core output through both direct sharded readback and a conversion to DRAM
outside capture. A passing instrumented run is inconclusive because retention
can change allocation and timing.

Interpretation:

- Exact pre-reshard input but differing post-reshard tensor localizes to
  resharding or its output lifetime. Replace only the direct 88→4 reshard with
  88→interleaved→4 as a controlled A/B; preserve dtype and final arithmetic.
- Exact residual and post-reshard tail but differing final output localizes to
  final arithmetic/storage/readback. In a focused replay, feed these frozen
  inputs to exactly `add(residual, combined, dtype=bfloat16, memory_config=M4,
  activations=[MUL_UNARY_SFPU(layer_scalar)])`. Compare a DRAM-output variant
  with the same inputs, then separate the scalar activation only if needed.
- Direct sharded host read differs but DRAM-copy read is exact: inspect output
  readback/page mapping before modifying norm or CCL code.
- Map bad column `c` to four-core owner `x=c//704` and local tile
  `(c%704)//32`; for the 88-core tail, owner is `(c//32%11, c//32//11)`.
  Report whether the 16 columns are one contiguous face half.

### 2. Four-core FP32 norm or sharded weighting first corrupts a residual

The new norm has coherent dimensions, but changes the reduction geometry from
88 one-tile shards to four 22-tile shards and retains the result as sharded
through weighting. Its weight multiply is therefore also a new variant.

**Conditional next probe if the residual is already bad:** retain both raw RMS
output and weighted result at the input and post-attention norm, plus the
common norm output. Compare actual attention all-reduce output as the immediate
post-norm input. For the first bad operation, preserve physical input, exact
weight, dtype, memory config, compute config, and output from that invocation.

**Single-variable controls:** keep four-core residual storage but replace only
the affected norm with `original_normalize(...)` followed by a conversion to
M4. If raw RMS output is exact but weighting differs, keep four-core RMS and
convert only its output to interleaved before the weight multiply. Do not
change fidelity, gamma application, and reduction geometry simultaneously.
Four-core standalone tests without the failing prefix are useful, but a pass
does not refute a prefix-dependent kernel state/lifetime issue.

### 3. Trace-specific reuse or delayed overwrite of a correct tensor

The runner currently never validates its two warm outputs. It preserves stable
token/position tensors, refreshes those exact objects before replay, and executes
the trace after capture. No stale token/position closure or missing first replay
was found. However, new L1 allocations can expose a reuse hazard absent in the
default graph.

**Minimal discriminator:** in the same paired TP1→TP4 run, validate the two
warm outputs and first traced output. Alternatively run the original candidate
without `--trace` as a separate control, retaining length 4096. If both fail,
stop calling it trace-specific. If only replay fails, retain/copy the earliest
bad tensor immediately after its producer and compare it after later operations.
Record buffer addresses, sizes, grids, dtypes, and lifetimes, not just names.
Retention removing the symptom is evidence of sensitivity, not sufficient proof
of the exact owning lifetime edge.

### 4. Earlier collective/local branch divergence

Lower priority because the candidate does not change the collective policy and
the default passes. Layout changes can still alter preceding data or allocation
timing. If both residual and final combined tail are already non-identical,
compare actual attention all-reduce and paired-MoE reduction outputs before
pursuing topology controls. Shared and routed local TP outputs are expected to
differ across ranks; only their reduced replicated results should be asserted
equal. Norm/QKV outputs should be equal before rank-specific weights consume
them. Do not require equality of local head/cache slices or local projection
partials.

## Verification state

Initial source-only checks completed with no retained candidate fix. Preserve
the original failing command, then require output/cache PCC and advancing
traced duplicate-replay validation after any proven fix. This docs-only diagnosis
requires no build.

## Follow-up: final BinaryNg fused add localized

The parent supplied `residual_diag_v2_full.tensors.analysis.json` and its local
raw `.tensors.pt`. This source audit read the analysis JSON; it did not run or
modify the diagnostic. Same-invocation evidence now shows:

- Residual, shared, routed, combined before reshard, and combined after reshard
  have zero maximum difference across all four ranks.
- The 88→4 reshard has zero differing elements and zero maximum error.
- All captured full-hidden norm comparisons against the CPU oracle have PCC
  at least 0.9999998808; they do not explain the large final corruption.
- The final result alone differs across ranks, with maximum differences
  1.703125, 1.703125, and 3.71875 relative to rank 0. Its CPU add/scalar oracle
  PCC is 0.8030700, maximum error 26.8467216.

This demotes earlier norm and reshard hypotheses for the captured invocation.
The fact that an earlier uninstrumented full run had equal replicas is not a
contradiction: its final output was already wrong, and retention changed the
observed distribution of corruption.

### Exact supported API contract

The original call is tiled `[1,1,1,2816]` FP32 residual plus BF16 combined tail,
with BF16 output, all width-sharded across `(4,1)` with `(32,704)` shards, and
post activation `MUL_UNARY_SFPU(layer_scalar)`.

- `binary/common/binary_op_dtype_policy.hpp` explicitly includes FP32 and BF16
  in the mixed floating family. `binary_op_dtype_policy.cpp:78` includes ADD
  among operations supporting mixed float inputs; `binary_op_utils.cpp:52–68`
  accepts this pair.
- `binary_ng_device_operation.cpp:309–408` validates each dtype, the mixed pair,
  device placement, TILE/ROW_MAJOR layout, and sharded/interleaved layouts. It
  has no rule requiring equal float input dtypes or forbidding BF16 output
  from an FP32 input. Shape validation at lines 413–458 accepts these equal
  logical shapes. Output construction at lines 518–548 retains the explicit
  shard specification and requested BF16 dtype.
- `binary/binary.cpp:674–732` forwards the requested output dtype and original
  floating operand types; it does not promote this FP32/BF16 pair to equal
  types. The mixed-integer promotion nearby applies only to DIV/MUL and is
  irrelevant here.

Thus treating mixed floats as an unsupported model call would misstate the
source contract. The suspected violation is internal to the selected kernel.

### Source-supported batching defect

1. Python ADD defaults `fast_and_approximate_mode=True`
   (`binary/binary_nanobind.cpp:1899–1908`).
2. `binary_ng_device_operation.cpp:49–57` selects FPU for ADD with unequal
   FP32/BF16 operands in fast mode. Homogeneous FP32 ADD selects SFPU even in
   fast mode. The fused scalar post activation does not change that selector.
3. `binary_ng_program_factory.cpp:1016–1038` enables multi-tile execution when
   both inputs and output are sharded with no broadcast. It hard-codes eight
   tiles per cycle for the FPU path, with no FP32 capacity check. The three
   equal shard specs here qualify; each core has 22 tiles.
4. Later, the same factory correctly sets `fp32_dest_acc_en=true` whenever any
   operand/output is FP32 (`:1202–1211`). The descriptor at `:1334–1343` passes
   the earlier tile count and this FP32 flag, leaving full-DEST synchronization
   at its default `false` (`tt_metal/api/tt-metalium/program_descriptors.hpp:104`).
5. Blackhole half-DEST capacity halves in 32-bit mode
   (`tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix_types.h:188–194`). The
   compute API independently uses `is_fp32_dest_acc_en ? 4 : 8` tiles per DEST
   unit in `tt_metal/hw/inc/api/compute/tilize.h:395`. This kernel therefore
   has four tiles available per acquired FP32 half.
6. `binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp:47–65`
   performs all `n` binary results and post activations at DEST indices
   `0..n-1` within one acquire/commit, then packs those same indices before
   releasing. With a 22-tile shard, its chunks are 8, 8, and 6: every chunk
   exceeds the four-tile acquired half. It can overwrite or overlap the
   other half while the packer operates there.

This is a concrete static batching/capacity mismatch, not a generic precision
hypothesis. Attribution of the observed device corruption still requires the
matching active descriptor/kernel and an isolated passing control. The mixed
dtype itself is not the sole trigger: any selected FPU path with FP32 DEST and
more than four simultaneously processed tiles has the same capacity concern.

The source explains why the default DRAM-output tail can pass: all-three-sharded
is false, so batching remains one tile. It also explains why an 88-core
one-tile-per-core implementation does not exceed capacity even if its requested
batching cap is eight.

### Discriminating controls and owning fix

The parent is testing homogeneous FP32 inputs/output followed by explicit BF16
typecast. That preserves the four-core residual contract and selects the SFPU
path, which uses two input pairs / four DEST tiles per acquire. A pass is
consistent with this mechanism, but changes dtype selection and kernel family
together.

A smaller model-level discriminator keeps the original FP32/BF16 operands,
BF16 output, sharded layout, and fused scalar, adding only
`fast_and_approximate_mode=False` to final ADD. The validator explicitly permits
this flag for BF16 ADD output (`binary_ng_device_operation.cpp:19–46`), and the
selector chooses SFPU. This still changes kernel family but leaves tensor
storage/dtypes unchanged.

The most direct native verification is a frozen-input operation test followed
by changing only the FPU batch cap from eight to four when FP32 DEST is active.
Keep the original mixed operands, BF16 output, and activation. The owning fix
belongs at factory batch-size selection, using the actual DEST mode before
selecting `num_tiles_per_cycle`; rejecting a documented float pair or assuming
the BF16 output guarantees 16-bit DEST would be incorrect. A C++ change needs
the repository build and focused device coverage before retention. This agent
made no implementation edits or device runs.

Minimal useful regression matrix: one four-core 22-tile-per-core mixed
FP32/BF16 ADD matching this model, eager and repeated trace, with fused scalar
present/absent; compare against CPU and inspect all physical output tiles.
An interleaved output is a useful control. If a native cap fix is attempted,
also cover homogeneous BF16 (eight-tile path should remain available) and
homogeneous FP32 (SFPU path should remain correct).
