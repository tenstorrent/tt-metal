# AutoDebug: full-attention replica failure after router placement

## Initial verdict

The new artifact proves a TP4 decode replica-equality failure, not its producer.
`full_router1_ccl_bfp8.log` contains `TP_DONE 1`, then the TP4 fabric setup and
`run_multichip_decoder.py:315 -> read(y):124 -> Replicas differ`. This is the
first read of a decode step, before its duplicate replay. The log does not give
the step index or distinguish finite rank differences from identical NaNs.
No JSON was written. The process exited 1 and closed normally; there is no hang.

This is a distinct symptom from the prior sliding-layer finite duplicate-replay
failure. Do not assume the same producer, or an inherent BF8 collective limit.
The sliding ordinary core1/BF8 run now passes numerical, cache and 128 replay
gates. Full BF8 previously passed before the placement change, while a mixed
stack BF8 full-attention replica failure was independently observed earlier.
Those contrasts motivate localization, not a causal conclusion.

## Source and policy provenance

- Production runtime: `070613ddc32b5cc8a22fd92a64cb541de9a9f27152852f1caad0843d1d06903d`.
- Ordinary runner: `06f0a0573dbd14e852be4a27c88b72faa610a51961b90cf33365025747898581`.
- Real layer 5, 4096 prefill tokens, 128 requested decode positions, traced TP4
  on physical Blackhole 1x4 Linear. Router native gate and its persistent buffers
  are placed on logical core (1,0). No new implementation change is made here.
- Constructor `full_attention_ccl_dtype=None` falls back to
  `attention_ccl_dtype` (`multichip_decoder.py:624-627`), so the existing
  diagnostic `--layer 5 --dtype bfloat8_b` selects BF8 correctly.
- Existing diagnostics preserve the actual width-sharded post-attention norm,
  provide original-class output-only mode, finite counts, per-rank duplicate
  comparisons, per-replay replica equality, and physical padding snapshots.
- Physical padding may vary without violating the logical contract. NaNs in
  logical output remain failures. Native synthetic CCL controls previously
  checked per-rank repeatability only, so they do not clear replica equality.

A fresh xhigh delegated source audit was attempted for this new symptom, but
the collaboration tool rejected it because the agent thread limit was reached.
The prior external AutoDebug CLI also had a documented bubblewrap startup
failure. This bounded new-evidence report therefore follows AutoFix's serial
fallback. It was written before new diagnostic implementation edits or new
hardware experiments. The coordinator has completed reset/list/smoke recovery
and explicitly handed exclusive hardware to this investigator.

## Minimal verify/refute sequence

1. Run the unmodified ordinary layer5 BF16 control with the integrated router
   placement. This distinguishes a full-layer placement regression from a
   BF8-sensitive path and records full numerical/cache checks.
2. Run the existing original-class output-only layer5 BF8 diagnostic to record
   the exact first failing position, output finite counts and within-replay
   rank differences. Preserve both replay snapshots.
3. Run corrected outer boundary retention for layer5 BF8, keeping the same
   math and configs. Expected equal-replica boundaries include AG, normalized
   attention/residual, router, expert input, grouped reduced outputs and tail.
   Local WO/RS/expert partials legitimately differ across ranks. Identify the
   earliest boundary whose replicated contract fails.
4. If reduced outputs are exact replicas but final output diverges, add only
   tail norm/add checkpoints. If AG first differs, freeze exact full-layer WO
   values and test native cast/RS/AG with cross-rank and finite checks. If a
   local expert first changes across duplicates but replicas remain equal after
   reduction, use the existing expert slice ladder rather than restarting the
   long C++ source audit.
5. Test one source-backed intervention at the localized boundary. Any candidate
   must pass the original ordinary full-layer numerical/cache/replay check and
   adjacent sliding/stack/batch gates before production acceptance.

Every failed multichip experiment is followed by serialized reset/list/smoke.
Capture live triage before terminating an actual stall. No runtime/original
runner edits are authorized here without coordinator agreement.

## First controls

`full_router1_ccl_bf16_control.json` passes the unmodified ordinary4096/128
run on runtime070613dd/runner06f0: min output PCC0.9994762093738185 and minimum
cache PCC0.9999715022296233; trace equality and replicas pass.

The coordinator then authorized failure-only runner diagnostics and startup
source hashing. Runner c531a9a2f00db1eb9b2135f9b4397cbdf5a388db50776c17b6d4c670899b36ff
adds no forward/trace/device operations, preserves strict comparisons, records
phase/step/first-or-repeat and finite/NaN/Inf counts, and writes compact
`.failure.json` before raising. Existing profiler read placement is unchanged.

`full_router1_bfp8_failure_detail.failure.json` reproduces at zero-based step41,
absolute position4137, first decode replay. All2816 values on all4 ranks are
finite. Rank1 differs from rank0 in6 elements, max0.0029296875; ranks2/3 exactly
match rank0. This is true finite replica divergence, not identical NaN payloads.
The process closes normally; serialized full_bfp8_reset1/list1/smoke1 all exit0.

The next existing boundary run checks expected replica contracts explicitly
(AG, replicated norms/residual/router/inputs, reduced outputs and final output),
while allowing local WO/RS/expert partials to differ across ranks. Physical-only
padding changes remain diagnostic evidence rather than a logical failure gate.

`full_bfp8_outer_boundaries.json/.log` passes all128 positions and all checked
replicated boundaries; no physical variation is recorded. This retained run
changes allocation lifetimes and cannot clear the ordinary failure. Original-
class output-only TP4 diagnosis is the next reproduction control.

## Paired context controls

`full_bfp8_output_only.json` passes128 positions using the original decoder class
in the standalone TP4 diagnostic. `full_bfp8_ordinary_tp4.json` then passes the
ordinary runner with only `--tp 4` added: exact replicas and duplicate replays
all128 positions. TP4-only has no TP1 accuracy oracle (`passed: null`).

`full_bfp8_ordinary_paired2.failure.json` repeats the ordinary paired failure at
step65 / position4161, repeat replay: rank3 has8 finite changed elements vsrank0,
maximum0.0078125; allother ranks exact. It closes normally. This gives3 ordinary
paired failures versus standalone TP4 passes; it does not yet prove why.

Two source hypotheses were refuted before experiments: actual Gemma4 MeshConfig
allreduce (models/demos/gemma4/config.py96-133) has no forced deallocation in this
no-padding path; the inspected forced-deallocation helper was a different class.
Also both ordinary and diagnostic runners delete each prefill output, so that
claimed lifetime difference does not exist. Full-layer dtype overrideNone
correctly falls back to the explicit BF8 attention dtype.

The next isolated control is `tests/diagnose_paired_cleanup.py`, a snapshot of
ordinary runnerc531 with only TP1 device references nulled and gc.collect() before
TP1 mesh close. TP1 still executes all numerics and checks; TP4 setup is unchanged.
This tests delayed destruction of prior-device objects/decoder cycles across the
close/reopen boundary. No production fix is implied by a single passing control.

`full_bfp8_paired_cleanup.failure.json` fails atstep12 / position4108, repeat
replay: rank2 has8 finite changedvalues, maximum0.00390625. Before TP1 close,
gc.collect reports5662 collectedobjects. This refutes cleanup as a sufficient
fix; no production lifetime change is kept. Both failed paired controls recovered
with full_bfp8_reset{2,3}/list{2,3}/smoke{2,3}, serialized exit0.

`tests/diagnose_paired_boundary.py` now snapshots ordinaryrunnerc531 and wraps
actual existing TP4 methods, retaining one selected boundary; it adds no math
operations. The first control retains attention.reduce's returned AG tensor,
and reads that tensor before each existing outputread. Previous warm checkpoint
references clear before the next forward's prefix allocations. This avoids the
broad diagnostic's copied reductions/normalizations and prior-warm retention.
TP1 runs unchanged. Optional post_attention_norm and grouped MoE reduced probes
are dormant, similarly wrapping original methods.

`full_bfp8_paired_attention_ag.failure.json/.pt` reproduces atstep33 / position4129,
repeat replay. Retained actual attentionAG is exactlyequal across allranks; final
output differs onlyrank1,4 finiteelements max0.015625, indices39,47,58,63 (tile1).
This rules out attentionAG replica divergence as the immediate cause in this
reproduced case; later postnorm/router/expert/grouped reduction/tail remain.
Native AG retention alone does not mask this failure. Pairedboundary runnerhash
09b0d2586f60b706a1c7cec35256445a07ab92f5fc06e9af84b1199247105217.

`full_bfp8_paired_postnorm.failure.json/.pt` reproduces atstep0 / position4096,
repeat replay: actual post-attention normalize return is exact acrossallranks;
output differs rank1 atindices33,39 only (tile1), max0.0078125, allfinite.
Thus current firstdivergence is later than postattention normalization.
The subsequent paired probe checks actual grouped MoE reducedoutputs.
Recovery reset4/list4/smoke4 all0; reset5/list5 complete0, smoke5 next.

`full_bfp8_paired_moe_reduced.failure.json/.pt` reproduces step18 / position4114,
repeat. Shared and routed reducedoutputs exactlymatch allranks; finaloutput
rank2 differs in6 finitevalues, max0.005859375, indices43,49,54,55,59,60, again
allwithin tile1. Firstdivergence is downstreamof groupedreduction or in the
parallel residual path. Next wrappers retain actual tail RMSNorm returns in
source order, addingno math. Version9009c38b8802c5b7a252c8adc342e718fc6cb5792bee411cb4961eb27745ee32
snapshots ordinarysetup and has dormant `tail_shared`, `tail_routed`, `tail_combined`
probes, which scope a Python ttnn.rms_norm wrapper only inside the actual
original `_fused_tail` call. Previous snapshot09b0 is paired_boundary_v1.py.txt.
Smoke5 also exit0; current failedrun recoveringreset6/list6/smoke6.

`full_bfp8_paired_tail_shared.failure.json/.pt` firstdetects replica difference
at the actual first shared-branch tail RMSNorm return, step12 / position4108,
first replay: rank3 differsin7 finiteelements, max0.125, indices36,38,42,52,54,60,63.
Allwithin tile1, matchingthe current generalizedgatecore1 location. Finaloutput
also differs tile1. This localizes the first observed badboundary before
remaining tailnorm/add, but a same-run inputcheckpoint is needed to distinguish
shardedcopy/input corruption from RMSNorm computation. Next `tail_shared_io`
retains sharedreduced, actual shardedinput, existingweight, and RMSNormreturn.
Failure-only snapshots include fullphysicaltile views (metadataonly reshape).

`full_bfp8_paired_tail_shared_io.failure.json/.pt` reproduces immediately step0,
first replay. The sharedreduced and exact RMSNorm shardedinput fullphysical
[1,1,32,2816] tiles are bit-identical acrossallranks, including14490 nonfinite
paddingwords, and weight is exact. RMSNorm logicalreturn differs4values at
42,44,60,63 betweenrank0 andranks1/2/3, max0.0625. Fullphysicalreturn differs24words
allwithin tile1. CPU bitcomparison saved in `.failure.summary.json`.
This refutes i2s/input replica divergence and localizes creation to firsttail
RMSNorm computation/outputstorage.

A placement control is now source-supported: move the generalizedgate fromcore1
toanother normalizationreceiver core2 and test whether firstbadcolumns follow
32:64→64:96; then try(10,9), outside norm11x8, expertGU6x2/down11x8, shared11x4,
routerprojection4x1 andSDPA8x8 mathgrids. Some transpose/movement ops may useall110
cores, so(10,9) is not claimed globallyunused. Moves onlyrouter memory andits four
persistentbuffers during setup; dtype, math, inputs, configs are unchanged.

Coordinator prioritized core(10,9) original-output-only paired control before
core2 ownership test; core2 remains unrun. Allfailedruns through tail_shared_io
recovered with reset/list/smoke{6,7,8}, all exit0. No Watcher/profiler or C++
changes have been introduced for this new issue. Runtime0706 remains unchanged.

## Diagnosis handoff

The final experiment ledger is [AUTOFIX_full_router1_bfp8.md](AUTOFIX_full_router1_bfp8.md).
Core (10,9) passes paired output-only, the same RMSNorm input/output probe, and
1024 duplicate comparisons per TP. Restoring core (1,0) fails; reallocating all
four buffers while restoring core (1,0) also fails, at output column49.
The smallest candidate patch is `router_dedicated_core_candidate.patch`.
No production runtime edit was made by this diagnostic agent. All recovery
commands through set12 exited0; hardware was released to the coordinator for
integration and ordinary accuracy/performance, stack and adjacent validation.
A specific native register/race cause remains unproven.
