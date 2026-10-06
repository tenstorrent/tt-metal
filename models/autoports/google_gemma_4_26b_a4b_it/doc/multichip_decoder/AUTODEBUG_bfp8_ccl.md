# AutoDebug: BF8 attention collective replay

## Verdict

**Current checkpoint:** the first changing logical tensor is the sliced gate
output from indexed expert gate/up projection, on rank 1. The up slice is
unchanged. The corrected gate/up-slice diagnostic fails at step 2 / position
4098 (`bfp8_boundary_retain_slices_v2`). Earlier hidden-only instrumentation
failed with the entire physical expert input bit-identical. The unresolved
region is the gate/up sparse producer or its slice; GELU, down, mixing and
collectives propagate that changed gate value in the observed trace.

All-intermediate retention and raw-GU-only retention suppress the observed
failure over 128 steps. Original-class K88 and 40 ms delay controls still fail.
Frozen original expert execution passes with both zero-padded logical uploads
and exact physical BF16 uploads, including producer NaNs. Thus allocator and
preceding-program context remain material. No production fix is retained, and
no universal BF8 collective limitation is established. Keep the candidate
outside stage acceptance until the original equality gate passes.

Current candidate: moving only the one-core generalized router from(0,0) to
(1,0) passes original-class128-step replay, gate/up-slice instrumentation, and
1024 duplicate comparisons across128 changing positions. Restoring core0 fails
again. A frozen native-gate-prefix A/B with identical allocation history passes at
both core locations; the whole-layer context remains necessary. Broad
numerical/stack/batch acceptance remains pending. No
production change is retained by this investigator. The original failing runner
and output-only diagnostic remain the acceptance reproducers. The experiment
ladder and normalization/comparison corrections are documented below.

This is a fresh, CPU/source-only AutoDebug investigation under AutoFix. No
TTNN import, accelerator access, implementation edit, or hardware experiment was
performed for this initial report. The native replay probe was independently
prepared and run by the hardware owner. Its subsequent evidence is listed below.

## Evidence and scope

- Checkout HEAD: `9a529836fc91b1117a48f6d63ef30455a69cd42d`.
- Runtime SHA256: `20151de9802f92af989cdad8f27cb764c79c8b8fad6c80939f5392edb8575cd4`.
- Runner SHA256: `beabf408847488fa8f9daaadc04c913c5b6bd1de681edfcbc5a3fb565b6128ca`.
- The shared tree already contains extensive stage work. Only this report is
  authored by the initial source-only phase; subsequent authorized diagnostics
  and hardware evidence are explicitly recorded below.
- `sliding_ccl_bfp8.log` ends at
  `tests/run_multichip_decoder.py:308`, `Replay is not deterministic`.
  It contains no first-failing step, changed-element count, delta, or intermediate
  tensors. Do not infer that step zero failed or that errors were small.
  `TP_DONE 1` appears before the failure, followed by FABRIC_1D initialization,
  proving this is the TP4 execution. The unmodified
  `sliding_ccl_bfp8_retry.log` repeats the same TP1 completion and TP4 assertion.
- `sliding_ccl_bf16.json`: layer0, 4096/128, traced, output PCC minimum
  `0.9988541343337518`, exact replay and replica equality passed. Recorded host
  decode median is `735.835 us`.
- `sliding_projections_lofi.json`: FP32 attention CCL control passed, minimum PCC
  `0.9989539324625537`; its earlier runner hash differs, but the runtime hash
  matches.
- `full_ccl_bfp8.json`: layer5, 4096/128, same runtime and current runner hashes,
  traced, minimum PCC `0.9994477563467595`, exact replay and replica equality
  passed. This is a useful passing BF8 contrast with the same CCL decode shape,
  not proof that sliding BF8 is safe.
- `attention_ccl_dtype_audit.md` correctly establishes source-level acceptance;
  its statement that runtime BF8 is unmeasured predates these artifacts.
- Subsequent hardware-owner controls `ccl_replay_bf16_cast.json`,
  `ccl_replay_bf16_precast.json`, `ccl_replay_bfp8_cast.json`, and
  `ccl_replay_bfp8_precast.json` all pass128 seeds with3 duplicate replays per
  seed. Probe SHA256:
  `fb9453754b6bbb29d4f698a0380778fac0c1d4d6f5cbd36230079f395f2620b5`.
  This refutes a universal exact-shape native BF8 replay failure and makes
  real-WO/full-layer boundary localization the next experiment. It does not
  clear the twice-reproduced whole-layer failure.

Original failing candidate, preserving the existing environment launcher:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder \
  --layer 0 --length 4096 --steps 128 --trace --check-cache \
  --fused-tail --hybrid-experts --optimized-shared --shared-geometry 1 \
  --grouped-moe-reduce --qkv-fidelity LoFi --output-fidelity LoFi \
  --attention-ccl-dtype bfloat8_b \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sliding_ccl_bfp8_recheck.json
```

Use the stage's actual Python/environment wrapper in place of `python`. There
is no passing result JSON from the original failed run.

## What the assertion proves

`run_multichip_decoder.py:265-308` warms twice, captures one forward into one
trace ID, copies token/current-position/cache-position before each step,
synchronizes, then performs a blocking replay. `read()` at lines110-115 reads
each rank and asserts exact replica equality. It then performs another blocking
replay without changing inputs and calls `read()` again before comparing the
two host outputs. The reported assertion therefore means both reads passed
replica equality, but the replicated output changed between executions. It is
not a host read-before-completion report or the reported cross-replica failure.

Only one trace is active per mesh, and `y` is the captured output. There is no
sampling mode, trace-cache key, or host token feedback in this standalone layer
test. The trace symptoms for coarse sampling keys and delayed feedback do not
match this path. KV-cache updates do execute again at the same position, so
their idempotence remains a localization boundary rather than an assumption.
The BF8 attention cast is downstream of QKV/cache updates; this isolated layer
uses fixture hidden states rather than its previous output as the next input.

## Lowered path and exact dimensions

The override at `run_multichip_decoder.py:151-154` wraps only TP4
`attention.reduce`. `_LocalAttention.project` at `tt/multichip_decoder.py:124`
calls the inherited WO projection and then that reducer. Decode WO returns FP32
in L1 (`tt/optimized_decoder.py:1319-1367`; construction enables output L1 at
`tt/multichip_decoder.py:594-604`). The actual path is:

`WO FP32 L1 -> typecast BF8 L1 -> DRAM copy -> native Linear RS BF8 -> AG BF8 -> typecast FP32 -> width sharding -> post-attention RMSNorm -> L1 interleaved -> norm weight -> residual/router/MoE -> BF16 output`.

`MultichipDecoder.allreduce` moves to DRAM at lines824-826.
`models/demos/gemma4/config.py:96-127` scatters dim3 on mesh axis1, then gathers
dim3, with one link and Linear topology. Neither call supplies persistent output
buffers or an explicit compute configuration. **Correction from follow-up:**
decode dispatches `OptimizedDecoder.normalize` at
`tt/optimized_decoder.py:786-810`, because `use_sharded_norms=True`,
`sharded_norm_site="all"`, logical row1 and hidden widthH2816. It promotes to
FP32, converts to the shared norm's width-sharded config, runs sharded RMSNorm,
converts back to L1 interleaved and multiplies the weight. The inherited
`tt/fused_decoder.py:155-167` path cited in the earlier dtype audit applies to
the larger prefill rows here. BF8 is promoted in either path, but the decode
consumer is a different program family.

| Quantity | FP32 control | BF16 control | BF8 candidate |
| --- | ---: | ---: | ---: |
| Logical input | `[1,1,1,2816]` | same | same |
| Padded input | `[1,1,32,2816]` | same | same |
| Input/output tiles for RS | 88 / 22 | 88 / 22 | 88 / 22 |
| RS logical output | `[1,1,1,704]` | same | same |
| Bytes per tile | 4096 | 2048 | 1088 |
| Input payload bytes | 360448 | 180224 | 95744 |
| Default FP32 destination accumulation | true | false | false |
| Default workers per direction | 2 | 2 | 2 |
| Tiles per worker per slice | 11 | 11 | 11 |
| Reduction tile groups | 4+4+3 | 8+3 | 8+3 |

The worker heuristic uses input bytes times3/4 and selects two workers below
500000 bytes (`reduce_scatter_common/reduce_scatter_program_utils.cpp:32-98`).
Offsets are `[0,11)` and `[11,22)` (lines266-291). The Linear program's
`tile_granularity` is capped at4 for FP32 destinations, otherwise8
(`reduce_scatter_minimal_async_program.cpp:1172-1186`). The standard fabric
packet capacity is sufficient to reach these caps; record effective parameters
if a probe overrides fabric packet sizing.

Full and sliding attention have different upstream geometry and data: local
sliding Q/KV heads are4/2 with head dimension256; full attention has4/1 with head
dimension512. Both project to H2816, so a shape-only blanket RS/AG explanation
must account for the full-attention BF8 pass.

## Checked and demoted explanations

### Partial groups or 1088-byte pages necessarily read garbage: not supported

The Linear reader reserves and pushes `tile_granularity` but copies only the
valid `num_pages_to_read`. Compute waits/pops full groups but performs and packs
only the valid tiles. The writer waits/pops full groups but transmits only the
same valid count:

- `.../device/kernels/line_reduce_scatter_minimal_async_reader.cpp:240-285`
  and `365-420`;
- `.../device/kernels/line_reduction.cpp:36-60`;
- `.../device/kernels/line_reduce_scatter_minimal_async_writer.cpp:307-359`
  and `397-424`.

For each11-tile worker the BF8 ledger is two balanced8-slot transactions with
8 then3 valid tiles. Five untouched tail slots are neither added nor sent. The
24-slot CB is a multiple of8. Blackhole in-order packing advances by the actual
CB page size, and CB push resets the per-push tile pointer
(`llk_api/llk_pack_common_api.h:71-89`, `llk_io/llk_io_pack.h:79-89`). No inspected
path substitutes a power-of-two shift for1088-byte addressing.1088 is divisible
by64. RS scatter writes explicitly use `page_size` and `2*page_size` at writer
lines259-267; AG derives page counts using division and uses the buffer's
TensorAccessor (`all_gather_async_default_program_factory.cpp:399-423,542-571`).
These facts refute that particular padding/stride story, not every possible
transport defect.

### Reduced accumulation precision alone explains nondeterminism: not supported

The wrapper auto-enables FP32 destinations only for FP32 input
(`operations/ccl/ccl_common.cpp:28-36`). BF8 reduction packs intermediate sums
back into BF8, so it is not merely BF8 wire transport around an FP32 allreduce.
This can change accuracy and routing, but with identical operands and fixed
ordering it does not itself explain changed replay results.

The initial FP32-to-BF8 cast uses preserved FP32 precision and precise BF8 packing
(`operations/copy/typecast/typecast.cpp:32-52`). Native RS config sets only
fidelity and FP32-destination state (`reduce_scatter_minimal_async_program.cpp:
1433-1445`); its default BF8 pack source format is the approximate BFP8 route
(`tt_metal/jit_build/data_format.cpp:329-334`). That difference is a useful
isolation boundary, not a demonstrated pack bug. The Linear compute kernel
performs startup on its actual input/intermediate/output CBs, then one add
configuration; it does not change formats between phases.

### Persistent dtype buffer collision or alternating trace ID: not supported

Neither ordinary collective supplies persistent outputs. `CCLManager` has an
unused buffer-cache member, but this path obtains only semaphores. With grouped
MoE there are exactly two allreduces per decode: attention and grouped MoE.
RS and AG each alternate separate two-set semaphore pools; their indices return
to the same state after a decode. Each RS/AG pair consumes two barrier indices.
Trace replay reuses the recorded addresses; Python indices need not advance
during replay (`models/demos/gpt_oss/tt/ccl.py:39-88`).

The Linear reader resets its global ready semaphore at exit (line428); the
writer performs the startup neighbor barrier and reset (lines223-247).
The final forward-to-backward dependency has a completed write barrier before
its local signal (lines411-419). These do not prove absence of all races, but
no BF8-specific missing reset or output alias was established.

## Focused verify/refute sequence

1. **Reproduce and retain the first failure.** Rerun the unchanged original
   candidate and the matched BF16 control. Record first step, absolute position,
   all per-rank changed counts, maximum delta, and output finiteness before
   aborting. Do not enable `--profile`: that option deliberately skips the exact
   duplicate-replay assertion. A passing retry is a flaky-failure observation,
   not retroactive clearance of the original failure.

2. **Frozen exact-shape collective (now passed).** The hardware owner prepared
   `tests/probe_attention_ccl_replay.py`, using `[1,1,1,2816]` local tensors,
   physical row32, rank-distinct values, the same MeshConfig/CCLManager, Linear
   mesh1x4 and128 seeds. These are the probe invocations (including the BF16
   `--precast` fourth control); results are listed above:

   ```bash
   python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_attention_ccl_replay --dtype bfloat8_b --precast --output /tmp/bf8_precast_replay.json
   python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_attention_ccl_replay --dtype bfloat8_b --output /tmp/bf8_cast_replay.json
   python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_attention_ccl_replay --dtype bfloat16 --output /tmp/bf16_cast_replay.json
   ```

   `--precast` removes the device cast from the trace. Failure with frozen BF8
   inputs implicates collective/replay state; failure only with the cast narrows
   the chain to conversion or its interaction. Passing all controls does not
   refute a real-WO/data-dependent/full-layer interaction. The initial probe's
   FP32 input is DRAM, whereas model WO is L1: a failure-only-in-model result
   also requires an L1 cast followed by the same DRAM copy.

3. **Capture real boundaries at the failing position.** Retain device handles
   for WO FP32, cast BF8, copied DRAM BF8, RS BF8, AG BF8, promoted FP32,
   post-attention output, routes and final output during capture. Read them
   only after each blocking replay. Retaining handles changes allocations and
   potentially timing; preserve an uninstrumented reproducer. Record the first
   changing boundary, physical/logical shape, dtype, memory and per-rank values.
   If WO already varies, inspect QKV, updated K/V cache row, SDPA and concat;
   if AG is stable and downstream varies, inspect norm/router/MoE. Frozen real
   WO values are stronger CCL input fixtures than Gaussian seeds.

4. **One model-local control for the localized mechanism.** If the cast input
   is stable, compare `WO -> BF8 -> BF16 -> existing allreduce` against
   `WO -> BF8 -> existing allreduce`. This retains the initial BF8 quantization
   while removing BF8 RS intermediate packing and transport. If that contrast
   localizes RS, a separate explicit-FP32-accumulation RS control is available
   through the public `compute_kernel_config` argument. It also changes tile
   groups to4+4+3, so a pass alone does not distinguish accumulator numerics
   from scheduling. Do not combine controls into one speculative fix.

5. **Only after a stable BF8 input fails RS:** vary one worker count or hidden
   padding at a time. H3072 gives24 output tiles/12 per worker and still has an
   8+4 tail; H4096 gives32 output tiles/16 per worker with no tail. A padding
   experiment must preserve and slice the original H2816 values and is a probe,
   not an accepted runtime workaround. Run a minimal Watcher check if evidence
   points to state/lifetime/transport, with profiling disabled.

Any retained fix must pass the focused failing probe, the original full
4096/128 sliding command with exact replay, the corresponding full-attention
control, cache checks, and the stage's final stack validation. This report does
not waive nondeterminism or select BF8 from an isolated collective timing.

## Final status

**Unresolved; no source-proven root cause and no implementation patch.** The
failure is sufficient to reject the currently unverified sliding BF8 candidate
for stage acceptance. It is insufficient to reject native BF8 collectives as a
family. The headline and demoted explanations were rechecked against the
concrete kernel loops and the same-hash full-attention passing artifact.

## Authorized AutoFix follow-up: diagnostic runner

After the initial source-only report, the coordinator authorized an isolated
diagnostic module and an exclusive hardware slot. No runtime or original runner
file was changed by this investigator. The module is
`tests/diagnose_attention_ccl_boundaries.py`; its output-only mode calls the
original `MultichipDecoder`, while its diagnostic subclass retains selected
intermediates. It saves first-failure tensors, step/position, changed counts,
finite counts and per-rank equality to JSON/PT artifacts.

At runtime `49cd4b3e47aeca24389fd58c29262556d05ec35cf0f0d80c8077c02af51f88e5`:

- `bfp8_output_replay_diagnostic.json` reproduces TP4 failure at zero-based
  step20, absolute position4116. All four ranks have1114 changed output elements,
  maximum absolute difference0.0625, zero nonfinite values, and exact equality
  between replicas within each replay. Diagnostic source hash is
  `d1ef3125f3878e1a2049bf088571fba92b9021c715d2bbf42f4f48dac6e0bacb`.
- `bfp8_boundary_replay_diagnostic.json` passes all128 steps with all retained
  boundaries equal. Its diagnostic source hash is
  `d3f8f681ae931113f21ff53a5cd01b0d3aed917ce2970ed5a3e5938ec3fe4f87`, preserved
  in `diagnose_attention_ccl_boundaries_v1.py.txt`.
- **V1 caveat:** expanded post-attention normalization enqueued residual
  `x -> FP32` after the post-attention operations; the original nested expression
  enqueues that cast first. V1 therefore changed op order/allocator state as well
  as retained lifetimes and host read delays. Its pass does not prove a lifetime
  root cause and does not validate the BF8 candidate.
- The output-only failure was followed by bounded reset, four-device listing,
  and FABRIC_1D1x4 open/close smoke, all exit0; see
  `bfp8_boundary_reset1.log`, `bfp8_boundary_list1.log`,
  `bfp8_boundary_smoke1.log`. The passing instrumented process closed normally
  and hardware ownership was released.

V2 restores the original nested expression order and helper temporary scopes.
It adds `--retain none|attention|router|moe|all`, `--read-output-only`, and
`--replay-delay-ms` so retention and host read delays can be varied separately.
The selected-policy runtime now takes the requested communication dtype at
construction; v2 passes that dtype directly and does not add a second cast
wrapper. **V2 correction:** the first CPU symbolic check used the inherited
normalizer and missed `OptimizedDecoder.normalize`'s sharded override. The
19-operation match was incomplete and is not proof of runtime equivalence.
V2 `--retain none` passes128, while the original-class output-only control still
fails at step1/position4097 with716 changed output elements per rank, max0.125,
no nonfinite values and equal replicas (`bfp8_boundary_v2_none.json`,
`bfp8_output_v2.json`). Thus the diagnostic's unsharded post-attention norm is a
substantive control; a retention-only explanation is not established.

V3 restores the actual sharded decode normalizer and delegates prefill to the
original method. It retains the FP32 promotion, sharded input, sharded RMSNorm
output and L1 interleaved conversion as separate boundaries. Python compilation,
Black with targetpy310, and `git diff --check` pass. These checks do not prove
device behavior.

### V3 first-changing boundary

The coordinator granted a second exclusive hardware slot. At runtime
`8b59370cda6f4ff88157de123123509036f2e91e8054000c809752e21f933175`, the corrected
diagnostic source
`3a783db8067d5a39cd5646abab6d0aa1d8cccdbdcc6848206492426e06e8b53f` produced
`bfp8_boundary_v3_all.json`, `.pt`, and `.log`. It fails at zero-based step6,
absolute position4102:

| Boundary | Duplicate replay result |
| --- | --- |
| SDPA, WO, BF8 cast, DRAM copy, RS, AG | Exact for each rank |
| FP32 promotion, width sharding, sharded RMSNorm, L1 result, residual | Exact for each rank |
| Raw/centered/rounded router logits, gate values/IDs, full routes | Exact for each rank |
| BF16 expert input | Exact for each rank |
| Local routed output | Rank1 changes2697 elements, max0.01953125; ranks0/2/3 exact |
| Local shared output and reduced shared output | Exact for each rank |
| Reduced routed output | All ranks change2525 elements, max0.03125; replicas equal |
| Final output | All ranks change1891 elements, max0.1875; replicas equal |

All observed outputs are finite. The selected experts are identically
`[78,121,25,91,90,20,53,31]`; selected gate values are identically
`[0.205078125,0.185546875,0.16796875,0.10791015625,0.09130859375,0.08349609375,0.08251953125,0.076171875]`.
Logical expert input and routes are BF16 L1 interleaved; selected indices are
UINT16 row-major L1 interleaved. These logical snapshots do not expose unused
physical padding.

`_HybridExperts.__call__` selects the indexed TP decode object for row1.
`OptimizedExperts._chunk` at `tt/optimized_decoder.py:193-241` performs the
indexed gate/up sparse matmul, split, fused GELU multiply, indexed down sparse
matmul, and final routing-weight matmul. This is the first unlocalized region.
The subsequent BF16 grouped allreduce propagates the changed local contribution
to every replica; attention BF8 RS/AG did not diverge in this captured failure.
Changed routing IDs are also ruled out for this execution.

The native collective probes only compare each rank with its prior replay;
they do not assert cross-rank equality or an arithmetic oracle. Their passing
scope must not be used to clear the coordinator's separately reported stacked
cross-replica failure.

Reset3, final serialized list3 and FABRIC_1D1x4 smoke3 all exit0; four devices are
visible and the smoke closes. Logs are `bfp8_boundary_reset3.log`,
`bfp8_boundary_list3.log`, and `bfp8_boundary_smoke3.log`. Hardware ownership was
released to the coordinator after recovery. No runtime fix is retained.

### Frozen expert probe

`tests/probe_indexed_expert_replay.py` freezes the saved real per-rank expert
input, routes and selected IDs, and loads the same layer's TP4 weights/configs.
Its default records gate/up, activation, down, mix weights and permuted down
boundaries. `--output-only` uses the original expert implementation.

```bash
HF_HUB_OFFLINE=1 timeout 180 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_indexed_expert_replay \
  --fixture models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/bfp8_boundary_v3_all.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/frozen_experts_boundaries.json
```

The instrumented command, with `--read-padding`, and the original expert
`--output-only` control both ran in an exclusive hardware slot and passed all
128 duplicate replays. Results are `frozen_experts_boundaries.json/.log` and
`frozen_experts_output_only.json/.log`. Every rank exactly matches the recorded
reference `routed_local` on the first replay. The instrumented run also finds
all physical tile snapshots repeatable. Both processes exit 0 and close the mesh.
Reupload zeroes padding, so this does not refute a producer-padding or
whole-layer state problem. Both diagnostic tools now support `--read-padding`: after blocking
replay, `ttnn.reshape(tensor, ttnn.Shape(tensor.padded_shape),
ttnn.Shape(tensor.padded_shape))` selects the metadata-only view overload and
allows the physical tiles to be read. Source confirmation is in
`reshape.cpp:556-635` and `tensor_ops.cpp:399-414`; no device kernel is added
inside capture. A later corrected CPU symbolic check uses the actual
`OptimizedDecoder.normalize` and matches all25 forward operations/arguments in
order. Python compilation, Black and `git diff --check` pass for both tools.


### Prepared whole-layer expert checkpoints

The diagnostic's `--expert-boundaries --read-padding` option now retains
indexed gate/up raw output, split gate/up, GELU-multiply activation, indexed
down, permuted down, gathered routing weights, tiled routing weights and final
local mix in the original whole-layer context. A CPU symbolic evaluation of
`OptimizedExperts._chunk` and `BoundaryExperts._chunk` verifies the same operation
sequence and arguments for this indexed, fused-GELU decode branch. The frozen
probe imports the same checkpoint class. `--gate-k-block 88 --output-only` is
also prepared as an independent control that changes only the gate/up K block
from 44 to 88, keeping grid 6x2 and per-core N1. No FP32-accumulation control is
planned: the source audit identified a separate CB-page-size hazard in that
unused mode. See `AUTODEBUG_indexed_expert.md`. These two new controls have not
yet run. No production source changes are retained by this investigator.


The first whole-layer physical-snapshot attempt (`bfp8_boundary_v4_experts`)
exited at step 0 because ordinary `torch.equal` treats equal NaNs as unequal.
Only unused padding in post-attention norm and residual was reported; every
logical boundary and output matched. CPU comparison of the saved snapshots as
INT32 bit patterns proves that both reported physical buffers are unchanged on
all four ranks. This is a diagnostic false positive, not evidence of model
nondeterminism. Physical snapshots now compare their FP32 bit patterns, so
stable NaNs are allowed there; logical comparisons retain their existing
NaN-failure behavior. A CPU check verifies changed NaN payload bits still fail
the physical check and logical NaNs still fail. Serialized reset4/list4/smoke4
all exited 0 before retrying the whole-layer experiment.


### Whole-layer expert retention and K88 results

`bfp8_boundary_v5_experts.json/.log` passes all 128 steps with all expert
intermediates retained and physical tiles checked. Runtime is unchanged
`8b59370c...`; diagnostic source is
`3ab30c9e6947f98b3bd568020e1b385fc9d68338b4fb6d33f86a9b18737ea74e`.
The gate program remains grid 6x2, N1, K44. V3 failed with only the outer expert
boundaries retained; V5 adds internal tensor lifetimes and extra host reads, so
this is evidence for a context-sensitive failure, not evidence of a fix.

The independent original-class `--output-only --gate-k-block 88` control in
`bfp8_output_k88.json/.log/.pt` fails at step 47 / position 4143. Each rank
changes 602 finite output elements by at most 0.03125; within-replay replicas
remain equal. The grid and per-core N remain 6x2 and N1. Removing the gate/up
partial spill/reload phase alone therefore does not fix the original failure.
No K88 production change is retained. Reset5/list5/smoke5 all exit 0 afterward.

The next bounded control is original output-only with a 40 ms host delay before
the duplicate replay. Minimal retention can now select only gate/up raw, hidden,
down, or mix tensors using `--expert-retain`; unselected tensors are not stored
in either the expert's or decoder's checkpoint dictionaries.


`bfp8_output_delay40.json/.log/.pt` fails at step 13 / position 4109 despite
a 40 ms delay before duplicate replay. All four ranks change the same 1017
finite logical elements, max 0.125. This refutes host delay alone as an adequate
workaround for this candidate; it does not identify which internal lifetime
change suppresses the observed failure in V5.


### Minimal retention: first logical difference reaches expert activation

`bfp8_boundary_retain_gu.json/.log` retains only raw gate/up among expert
intermediates and passes all 128 steps, including physical snapshots.
`bfp8_boundary_retain_hidden.json/.log/.pt` instead retains only the fused
GELU-multiply result and fails at step 0 / position 4096. The first logical
change is `expert_hidden` on rank 1: 118 finite elements, max 0.296875. All
eight slots are affected only within columns 0:32; per-slot changed counts
are `[16,16,16,15,13,13,16,13]`, with almost identical masks across slots and
across the two 16-column halves. Raw gate/up was not retained in this run, so
this narrows the region to gate/up sparse matmul, slicing, or fused GELU/mul;
it does not prove the activation itself is faulty.

Both logical and full physical expert-input tiles are bit-identical at this
failure, as are routes and indices. The physical expert input contains 37552
nonfinite values in its unused rows. Earlier attention WO/CCL padding row 30
changes, but that difference disappears before expert input. Local routed
output changes only rank 1 (2450 elements, max 0.0078125), then routed reduction
changes identically across replicas (1994 elements, max 0.0078125), and final
output changes identically (1226 elements, max 0.0625). Logical outputs remain
finite. Serialized reset6/list6/smoke6 and reset7/list7/smoke7 all exit 0.

The first gate/up-slice checkpoint attempt
(`bfp8_boundary_retain_slices.json/.log/.pt`) stops at step 0 solely on varying
physical padding: all logical tensors, including slices/activation/output,
match. Physical-only variation is outside the logical output contract. The
whole-layer diagnostic now records those events and continues until a logical
mismatch, while preserving every physical comparison for the failing step.
Reset8/list8/smoke8 all exit 0 before this corrected retry.

The frozen probe now also accepts `--physical-input`: it restores the upper
16 bits of saved FP32-expanded BF16 snapshots directly into BF16 host storage,
including NaN payloads, uploads whole physical tiles, then applies a metadata-only
view restoring the original logical shape. CPU bit reconstruction is exact for
the saved expert input and routes. This option preserves producer padding while
still changing whole-layer allocator and preceding-program context.


### Gate slice localization and exact physical-input control

`bfp8_boundary_retain_slices_v2.json/.log/.pt` fails at step 2 / position 4098.
The first logical change is gate on rank 1 (173 finite elements, max 0.1875);
up is logically unchanged. Hidden changes 162 values (max 0.2578125), local
routed output 2625 values (max 0.02734375), reduced routed output 2340 values
per replica (max 0.03125), and final output 2091 values per replica (max 0.125).
The output replicas remain equal within each replay. Physical input padding
varies in this execution, while the preceding hidden-only failure demonstrated
a logical change with the entire physical expert input unchanged. Neither
observation justifies a speculative padding fix. Reset9/list9/smoke9 exit 0.

`frozen_experts_physical_output.json/.log` uses the hidden-only failure fixture
with `--output-only --physical-input --read-padding`. It passes all 128 repeats
and closes normally. Readback verifies exact logical and physical input/routes
bits after upload. The first frozen local output equals the recorded reference
on ranks 0/2/3, but differs on rank 1 by 2450 values, max 0.0078125. Those counts
match the saved whole-layer failing pair; exact comparison against the saved
`actual` member was not yet performed. Whole-layer predecessor/allocator state
remains absent from this frozen control.

Current hardware ownership remains exclusive to this investigator; historical
release statements above describe earlier handoffs. The coordinator has
authorized a bounded original-class Watcher control next. No production runtime
or original runner modification is retained.


Watcher initial attempt (`bfp8_output_watcher.log`) does not reach the model:
fabric ACTIVE_ETH kernel configuration is 28464 bytes, above its 26624-byte
limit. It exits 1 and closes devices; no replay JSON or memory-check verdict
exists. The next bounded retry adds only `TT_METAL_WATCHER_NOINLINE=1`, preserving
NoC checks while reducing compiled check code size. Profiler and DPRINT are
explicitly unset, polling interval is 10 seconds, and logs use the dedicated
`bfp8_watcher` directory. Recovery is serialized before retry.


### Watcher result and next controls

`bfp8_output_watcher_noinline.json/.log` passes all 128 original-class
output-only steps with `TT_METAL_WATCHER=10` and
`TT_METAL_WATCHER_NOINLINE=1`. Profiler and DPRINT are unset. No NoC checking or
Ethernet checking is disabled. The Watcher log contains eight completed dumps
and no detected fault; all devices detach and process exit is 0. Runtime is
`8b59370c...`; diagnostic source is
`864d2110a16bb3a2591f3fc19aaa896161ed152e48c4193883aa0ad5d46bd74c`.
This is a scoped passing debug-kernel control, not proof the normal-kernel race
is absent. Hardware was released during subsequent CPU preparation and then
reacquired for the N2/address controls under the coordinator's standing grant.

Source auditing confirms raw GU remains a local until original `_chunk`
returns. Retaining it changes its lifetime through later shared/tail operations
and also through the next warm/capture attention prefix because the retained
dictionary is only cleared on `_chunk` entry. Gate/up slices are independent
L1 buffers; the GU reshape is a view. Ordinary attention collective payload
buffers are DRAM, so direct payload-buffer alias to GU L1 is not established.
See the parallel lifetime audit for source details.

The diagnostic now supports `--log-addresses` (host scalar metadata only),
recording warm1/warm2/capture phases, previous retained dictionaries before
clearing, and raw/reshaped GU addresses. `--expert-retain none` stores no expert
handles. `--gate-n-tiles 2` is a separate geometry control using grid6x1,
per-core/block/subblock N2, with original K44 and down program unchanged.
Neither change has been applied to the production runtime or runner.


`bfp8_output_n2k44.json/.log/.pt` uses original output-only execution with
only the gate/up geometry changed to grid6x1/N2/K44. It fails at step7 /
position4103: 1715 finite output values differ per rank, max0.75; replicas
remain equal within each replay. This geometry is not a verified workaround.
A gate/up-slice N2 checkpoint run is queued to test whether the affected
columns follow the sender's widened 0:64 ownership.


### Sender ownership discriminator

`bfp8_boundary_n2_slices_addresses.json/.log/.pt` reproduces at step0 with
first logical change in gate, now rank2. Changed gate elements occupy exactly
columns0:64 across all8 experts: counts by 32-column tile are
`[126,128,0,0,0,0]`. Gate changes254 finite values, max0.328125; up is exact.
Final outputs change1152 values per rank, max0.1875, and replicas agree.
Expanding the sparse sender worker's ownership from one N tile to two expands
the corrupt columns correspondingly. This strongly favors the sparse producer
on sender worker0 over generic slicing/GELU or a universal CCL format issue.

The same JSON contains221 scalar address records across warm1/warm2/capture.
The capture GU and its reshape both use0x174480; gate0x171fc0, up0x1717c0,
hidden0x170fc0. Retained WO starts0x1727c0, exactly after the gate's per-bank
allocation, with no overlap established. The apparent mix-weight/shared-input
reuse at0x176c80 occurs sequentially after the expert call. These observations
do not prove there is no use-after-free; they refute a direct same-base overlap
among the recorded current tensors. N2 physical expert-input padding is not
bit-stable (1822 changed words, finite max delta0); the earlier N1 hidden-only
run separately established a logical error with bit-stable physical input.

Reset11/list11/smoke11 and reset12/list12/smoke12 all exit0. The next control
is original N1/K44 with only input L1→DRAM before `_chunk`. Source confirms
`to_memory_config` uses the default tiled same-dtype copy: all88 BF16 pages of
2048 bytes are transferred without unpack/repack, preserving unused rows and
NaN bits. Extra program timing and addresses still change, so a pass would
remain a placement control, not proof of a source fix.


The lifetime investigator's CPU bit audit identifies the N2 physical input
changes precisely:1596 zero→positive-infinity and226 positive-infinity→zero
words, entirely unused rows1,2,4,5,6,7,16,17,18,19,21,22,23. The finite-pair
max-difference field of0 does not imply these words are numerically unchanged.
The logical row is exact; all ranks have the same padding transitions.

`bfp8_output_expert_input_dram.json/.log/.pt` fails at step26 / position4122:
1241 finite output values differ per rank, max0.125; within-replay replicas
remain equal. Moving only the expert input to DRAM is therefore not a fix.
The original N1/K44 gate/up program, weights and activation dtype are unchanged.
A sliced-boundary DRAM-input control is next, while source work checks whether
the preceding one-core generalized router leaves state on sparse sender core0.


`bfp8_boundary_input_dram_slices.json/.log/.pt` also fails at gate, step6 /
position4102, now rank0. All126 changed gate values lie in columns0:32
(max0.1015625); up is unchanged. Hidden changes123 values; final output changes
1302 values per replica (max0.1875), all finite. Input placement therefore
does not remove the worker0 gate pattern. This demotes an explanation confined
to interleaved L1 reads.

The next native-supported control moves the generalized router's one-core grid
from(0,0) to(1,0), along with bias, index, output and output-index buffers during
setup. The native op takes its worker grid from the input shard and validates
other tensors against it; there is no origin-core requirement. The original
sparse N1/K44/L1-input configuration is restored for this control. This tests
preceding core state separately from the sparse multicast sender's fixed core0
role. No production change is retained.


### Router placement contrast and stress

`bfp8_output_router_core1.json/.log` moves only the native generalized gate's
shard grid and four persistent buffers to(1,0). It passes128 original-class
steps with original N1/K44/L1 expert input. Restoring core0 immediately afterward
(`bfp8_output_core0_postcontrol.json/.log/.pt`) fails at step62 / position4158:
1727 finite values per replica change, max0.125. The negative control confirms
the prior failure is still present. Reset15/list15/smoke15 all exit0.

`bfp8_boundary_router_core1_slices.json/.log` passes128 steps with gate/up/hidden,
router IDs/values and physical input checkpoints plus scalar address records.
No physical-only variation is observed. `bfp8_output_router_core1_stress.json/.log`
then passes exactly1024 duplicate comparisons, eight per position across128
distinct input/position updates, with the original decoder/expert methods.
All passing processes close normally. This justifies a model-local placement
candidate, not a proven low-level reset defect or a completed accuracy gate.

The prepared frozen-prefix control allocates both core0 and core1 gate inputs
and persistent output buffers in identical order in both processes, uploads
exact saved physical expert input/routes/IDs, and changes only which native
gate executes immediately before original expert computation. The expert still
consumes frozen routes/IDs; the prefixed gate cannot change its inputs. Results
will also compare the first output to both members of the original failure pair.


### Final investigator handoff

`frozen_physical_native_gate0.json/.log/.reference.pt` and corresponding
`native_gate1` artifacts both pass128 duplicate replays. They allocate both
prefix placements in identical order; recorded prefix address maps are equal
across processes. The first logical and full physical outputs are bit-identical
between the two controls and exactly match the saved whole-layer `actual`
member. The saved `reference` member differs on rank1. This identifies which
replay value the isolated chain produces; it is not an independent arithmetic
oracle. Native gate alone is insufficient to reproduce the whole-layer failure,
so no simple persistent-reset source defect is established.

Hardware is explicitly released to the coordinator after both processes exit0
and close all devices. No production runtime, original runner or C++ source
was edited by this investigator. All experiment JSONs record runtime
`8b59370c...` for the current-policy phase. The separate diagnostic scripts are
Python-compiled and Black-checked; no C++ build is required. Exact replay/replica
checks were never relaxed. Physical-only padding variation is recorded but
does not replace the logical output contract.

The coordinator owns the remaining acceptance work: validate the integrated
model-local router-placement candidate against original numerical/reference
accuracy and all replay gates, and run full-attention/stack/batch/cache controls
and paired latency measurements. No performance or broad-correctness claim is
made from the instrumented or frozen runs. The tested placement is(1,0), with
all generalized-gate buffers on that same valid one-core grid.


Coordinator integration checkpoint: the production runtime is now
`070613ddc32b5cc8a22fd92a64cb541de9a9f27152852f1caad0843d1d06903d`. The coordinator moves only generalized-router memory and its four
persistent buffers to(1,0), immediately after router construction. Baseline
and original runner are unchanged. Ordinary4096/128 accuracy, cache, replay and
paired performance validation is underway. This is a model-local workaround
pending those gates, not a claimed source-level root-cause repair. All preceding
diagnostic observations retain their recorded old runtime8b59370c provenance;
the changed constructor allocation order requires fresh ordinary-run evidence.


Coordinator's first ordinary integrated run now passes:
`sliding_router1_ccl_bfp8.json` at runtime070613dd, real4096/128, has minimum
output PCC0.998745334, cache PCC0.999996658, exact repeated-trace equality and
replica equality for128 steps. Host TP1/TP4 decode medians are825.761/737.703us
and prefill medians221490.578/93316.212us. These are paired host measurements
from the ordinary runner, not a causal speed claim against the old failed
BF8 path. Full attention, stack, batch, maximum-context and Watcher acceptance
remain with the coordinator.

The diagnostic now defaults `--router-core-x` to unspecified (use current
production placement); explicit `--router-core-x 0` restores the old placement
for future negative controls, and `--router-core-x 1` selects the workaround.
This avoids silently treating the newly integrated core1 default as a core0
control. Prior artifact commands/source hashes retain their original meaning.
