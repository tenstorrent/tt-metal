# Datatype sweep work log

Stage 8 starts at clean main/e03236d2, following optimized-full-model clean-pass.
Target Qwen/Qwen3.6-35B-A3B revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7.
Hardware: four Blackhole p300c devices, mesh 1 × 4, FABRIC_1D.
Device holders: none at startup; bounded tt-smi listing passes (logs/device_list.log).
No resets, no vLLM integration, no push. Installed enabled tt-autodebug 0.1.8 and
tt-model-bringup 0.1.24 verified from active config/plugin skill inventory.
Environment: python_env and ../lane-env.sh, package readiness runtime on PYTHONPATH.

## Contract

Full-model top-1 ≥ 90%, top-5 ≥ 98%, top-100 = 100% on every evaluated reference.
AIME24 chat template reference has 100 generated tokens; add natural-text teacher
forcing and P1 long reasoning. Serial traced teacher forcing ranks candidates;
repeat baseline twice to establish spread. P1 defaults: ≥ 1,000 paired positions,
delta NLL ≤ max(2 × SE, 0.004), positive-delta KL × 1,000 ≤ 5; baseline needs higher
precision or HF control (HF KL × 1,000 ≤ 4). BFP4 must include same-group LoFi.
Advertised context remains 262,144 aggregate tokens; fixed slots 1–32.
KV candidates need recomputed context arithmetic and short/B32 controls.
Final selected policy needs context smoke or changed-KV maximum acceptance,
nonaligned support, resumed-prefill, qualitative suite, token-out benchmark,
plots, independent review and stage-owned local commits.

## Initial baseline refresh

Refreshed baseline run 1: tests.run_full_model_readiness --mode teacher, timeout 2400,
tracking 0 / skip 0, no watcher/profiler; log logs/baseline_teacher_1.log.
Source remains stage 7 during the initial baseline runs. A CPU-only precision
audit and reference-script preparation run independently; no other device jobs.

## Initial refreshed evidence and policy plumbing

Original-source AIME teacher forcing passes 98/100 top-1, 100/100 top-5 and top-100;
callback-path TTFT 3,867.19 ms, traced decode 46.4253 t/s/u. This includes sampling and
readback and is kept separate from resident-logit ranking measurements.
`baseline_teacher_1.json` and its provenance identify the original source.
CPU HF natural-text preparation completes in 13 s; 768 actual continuation positions
are in `references/natural_text.refpt` with full provenance. No HF generation
ran concurrently with a timed device measurement.

Full-model policy resolution now supports direct checkpoint uploads and actual
runtime dtype/fidelity assertions; no candidate changes through environment.
Twelve coarse policies live in configs/. Existing local optimized policy is the
canonical baseline; no separate Qwen3.6 canonical model exists under models/demos.
BFP4 expert GU/down and grouped EP, shared packed/down each include LoFi and
HiFi2. Shared routing scalar stays BFP8/HiFi2. Sensitive ordered coefficient
scans/state remain BF16/FP32, and validated BF16 residual/activation boundaries
remain explicit. BFP8 CCL applies only to BF16 partials; FP32 partials remain.

First policy-plumbed baseline aborted during layer 3 construction: the runtime
summary asserted a linear-only `scan_router_weight` on a full-attention layer.
The original failed artifact/log are `baseline_bfp8.json`/logs/baseline_bfp8.log.
Moved that assertion into the linear-attention branch; no math changed. Device
closed normally; no reset. Rerun evidence will use a fresh suffix. Hook formatting
changes occurred before this retry; no benchmark is ranked from the failed run.

The baseline v2 reaches 98% AIME prefill/teacher and 97.9167% natural-text prefill,
with 100% top-5/top-100. Its source sidecar records the imported implementation.
During CPU audit, corrected generator's new precision import to an absolute
package import, required by readiness's standalone module loader. This changes
import compatibility only; the already-constructed v2 model uses the identical
resolved policy. Subsequent CLI/P1 runs exercise the absolute import.

Baseline v2 completes in 331 s across all 40 layers. AIME prefill/teacher 98/100, natural prefill
752/768 = 97.9167% top-1, natural teacher 754/768 = 98.1771% top-1 and 767/768 = 99.8698%
top-5; top-100 = 100% everywhere. Resident-logit teacher repeats 47.827395 / 47.829974
t/s/u (spread 0.002579); warmed 161-token resident-logit TTFT 3,724.806 / 3,815.288 ms.
Separate 100-token AIME callback timing: 46.4975 t/s/u. Exact commands, actual
per-layer tensors/configs, trace counters and source sidecar accompany the raw
artifact. Baseline process 2 and final selected remeasurement remain planned to
check process-to-process spread rather than relying only on within-process noise.

At 06:04 UTC CPU HF reasoning generation runs concurrently with reduced KV
correctness only. These probes make no speed claim; all ranked candidate
measurements wait for the HF process to finish. Prompt caps remain 6,144, thinking
on, model-card sampling and generated-token presence penalty 1.5. No TT timing
from this concurrent interval enters the Pareto frontier.

## KV-cache controls and capacity

The serialized reduced real-layer 0/3 controls cover 32 identical users with
independent pages, prompt lengths 32 and 33, and three forced traced decode steps.
Both BF16 and BFP8 controls pass trace-allocation tracking and stable program-cache
checks. `kv_probe/comparison.json` records minimum per-user PCC 0.99984617,
aggregate PCC above 0.995, exact BF16/BFP8 top-1 agreement, and same-input user
consistency. Raw tensors remain local in `kv_probe/{bf16,bfp8}.pt`; these large
binary diagnostics will be referenced rather than committed.

`context_candidates.json` and the appended datatype capacity section in
`../context_contract.json` preserve the advertised 262,144 aggregate tokens and
32 fixed slots. At that capacity, tiled BFP8 KV occupies 1,426,063,360 bytes per
chip versus 2,684,354,560 bytes for BF16: 1,258,291,200 bytes saved. This is
capacity arithmetic, not a new maximum-context acceptance result. A selected
changed-KV policy will run the one maximum-context acceptance; an unchanged-KV
selection will run the required 32K–64K smoke.

The original warmed token-out benchmark now records the normal default
construction's actual precision summary and selected-artifact hash. Its
measurement loop and workload are unchanged. This makes the post-selection
number directly auditable without replacing the optimized-full-model benchmark.

## Preparatory independent policy audit

The CPU-only audit found that P1's sampling introspection would miss this
model's `generate(**kwargs)` sampling support. P1 therefore uses the external
pinned HF reasoning traces already being generated, with explicit model-card
sampling; it does not claim those texts came from the TT reference. Separate
P1 kwargs use context 8,192; the coarse sweep's 2,048-token kwargs are not reused.
The installed P1 runner's `--only` path does not aggregate unvisited cached
candidates despite its help text. Archive each real candidate result, then run
the final selected subset without `--only` to aggregate genuine cached scores.

Full BF16 experts do not fit, but partially BF16 expert layers plausibly do.
The proposed stronger reference uses BF16 dense/head weights with HiFi4 and a
balanced subset of BF16 expert layers. Each promoted expert layer adds
377,487,360 bytes/chip for indexed TP and grouped EP representations. Exact
source payload arithmetic gives 460,144,640 bytes/chip for the dense/head uplift,
including padded DRAM-sharded copies and the preserved shared routing scalar.
Twenty promoted layers project 460,801,024 bytes free at full capacity; 21
project 83,313,664; 22 exceed the historical prepared capacity by 294,173,696.
At this point in the investigation, these were projections awaiting actual
full-capacity allocation and short-input preparation census. The measurements
recorded below supersede that pending-fit status. No long-context generation
or speed measurement was run for this reference-fit check. Coarse timing was
held at the between-job STOP marker.

The audit confirmed material policy propagation and recommended combining
passing switches after the coarse matrix. A failed shared-BFP4 candidate will
also test restoring shared down to BFP8. Indexed-only expert-GU candidates must
retain P1 decode scoring because prefill uses the unchanged grouped EP weights.
Shared scalar protection preserves BFP8/HiFi2; its separate geometry is validated
by full-model results rather than claimed bit-identical by construction.

Schema validation now rejects the grouped EP kernel's known unsupported FP32
destination accumulation and packer accumulation flags before device allocation.
All generated candidates already used the legal false/false values; this changes
validation only, not their numerical paths.

## Local implementation checkpoint

Local commit `a8fdbbc11b9bf58e597e5d4003dec559e3a50a99` records the precision
plumbing and datatype harnesses. Normal commit hooks pass; the hook log is
`logs/plumbing_commit.log`. This is an implementation checkpoint, not stage
closure or a selected policy. No push was performed. Final evidence, selected
default, and independent stage review remain pending.

The initial even-layer 20-expert BF16 full-capacity allocation/preparation
passes, leaving 453,690,880 bytes/chip. The exact required MoE dispatch/combine
live-buffer lower bound is 142,344,192 bytes; adding one more complete BF16
expert layer would leave only 76,203,520 bytes. Thus the 21-layer promotion
cannot fit this path, independent of other positive allocations. See
`memory_fit/precision_payload_and_peak.json`. This is a physical lower-bound
argument, not a measured transient peak. The balanced 20-layer control was
then prepared separately, covering 15 linear and 5 full-attention layers,
including layers 0 and 39; its completed result follows.

The balanced 20-layer reference passes full-capacity allocation and short
preparation too, with exactly 453,690,880 bytes free per device and normal device
closure. Its artifact is `memory_fit/accuracy_bf16_experts20_balanced.json`.
P1's fitting reference is now `accuracy_bf16_experts20_balanced` (context 8,192
for the text probe, independently qualified at full advertised capacity).
The between-job STOP marker is removed; the existing serial coarse queue starts
when both CPU HF reasoning traces finish. The first HF trace uses 6144 tokens
and is not degenerate under the installed runner's tail-repetition check.


## Default resumed-prefill baseline diagnostic — running

The baseline is now running `tests/datatype_resume_check.py`, which delegates
to the installed checker without changing its scoring or acceptance criteria.
The current run uses all 40 layers, context 4,096, explicit safe baseline
precision, the canonical natural book, and the installed defaults: length
2,560, block 32, tile 32, and a 256-position window plus the final position.
Resume starts are 32, 68, 2,528, and 2,559. The first two cases run both
`no_history` and `fresh_suffix` negative controls. The unchanged thresholds are
mean PCC ≥ 0.99, minimum PCC ≥ 0.98, top-1 agreement ≥ 0.98, and exact
last-position top-1 agreement.

```bash
timeout 7200 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_resume_check \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --mesh-device P300x2 --fabric-config FABRIC_1D \
  --build-kwargs '{"max_seq_len":4096,"precision_config":"baseline"}' \
  --text-file models/ttt_compat/tests/tale-of-two-cities.txt.bz2 \
  --diagnostics-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/baseline_resume_diagnostics \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/baseline_resume_check.json
```

The running log is `logs/baseline_resume_check.log`. If a case fails, the wrapper
retains fresh/resumed TT logits, exact input tokens, scored positions, runtime
precision, and the original gate result in `baseline_resume_diagnostics/`.
No pass is claimed while the run is active. Earlier HF resume controls at
lengths 257 and 1,025 do not establish behavior for these new 2,560-token cases;
any failure requires the matching HF split and per-position margin comparison.
The eventual selected policy must rerun this default gate through normal
selected-config construction. This diagnostic run is concurrent only with CPU
HF reasoning preparation and makes no performance claim.

## Selected qualitative check — prepared, not run

After selection, reuse the pinned HF greedy controls only after the shared
six-prompt suite, model revision, tokenizer, rendered chat prompts, and generation
settings have been verified. The existing full-model runner performs these
checks and uses the model's normal default precision construction. A fresh
output directory preserves the earlier full-model and optimized-full-model
controls:

```bash
timeout 3600 python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/run_tt_controls.py \
  --mode suite --context 1024 --max-new-tokens 128 \
  --hf-controls models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_greedy_outputs.json \
  --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/qualitative
```

The six prompts have 18, 25, 33, 19, 24, and 20 tokens; context 1,024 comfortably
covers each 128-token continuation. The HF control is revision
`345d467b117d1fd5eda9b879a5988de9ae5f47a7`, `Qwen2Tokenizer`, chat mode with
checkpoint-default thinking, greedy decoding, seed 0, and no presence penalty.
The runner retains HF-control, suite, readiness-runner, and principal model
source hashes, checks exact prompt token IDs, and requires stable trace IDs and
program-cache counts across all six prompts. The selected artifact and policy
source hashes must also be linked to this run, because the existing qualitative
metadata does not independently record the full runtime precision summary.

Generation alone leaves `GENERATED_REVIEW_REQUIRED`: read every actual HF/TT
completion and record a paired verdict. The 128-token cap and absence of EOS
are not quality failures. A suspected content degradation needs at least ten
blind matched samples per side with model-card sampling before it can reject
the policy. No qualitative pass or stage completion is claimed here.


## 07:26 — strongest fitted reference and resume investigation

The balanced20 plus layer35 EP/down BF16 policy fails full-capacity preparation at the MoE combine allocation. The requested 67,108,864-byte buffer requires 8,388,608 contiguous bytes/bank; 14,679,744 bytes/bank are free but the largest block is only 7,999,936 bytes. This is allocator fragmentation, not a claim that total free DRAM is smaller than the request. Devices close normally. Raw rejection: `memory_fit/accuracy_bf16_experts20_partial.json`.

Restoring layer35 indexed down to BFP8 while retaining its BF16 EP weights passes full-capacity allocation, prefill preparation lengths32/33/161/2048, and trace capture, with264,947,200 bytes free/device. `accuracy_bf16_experts20_ep` is the strongest fitted evaluated reference in this monotone promotion family; no claim is made that every possible allocator packing/subset was searched. P1 candidate kwargs now use it. Full-model scoring remains pending.

The baseline default resumed-prefill gate completed all4 cases: start32 passes, starts68/2528/2559 fail PCC; top-1 and applicable negative controls all pass. Exact failed logit pairs are retained. Matched pinned HF BF16 controls reproduce PCC failure at68/2528 but pass2559. This is not a blanket waiver. AutoDebug completed inspection and no offset/cache-ownership bug was established. AutoFix is active, with a full-depth capture harness being prepared to localize TT-specific singleton/prefix differences. See `AUTODEBUG_resume.md`, `AUTOFIX_resume.md`, and `hf_resume_control/summary.json`. Ranked timing is paused; the pending queue now includes two current-source baseline repeats and the stronger reference.

## 07:47 — localized resume numerics; policy experiments running

The full-depth capture reproduces both original logit tensors bitwise. All 240
retained recurrent/conv rank comparisons are exact before the public cast; this
statement does not cover full-attention KV bytes. Target 2559 first differs at
layer 0 attention, then grows through MoE. Common prefix position 2558 first
differs after layer 1 MoE. Evidence: `resume_localize/summary.json` and
`resume_localize/L2560_s2559/statistics.json`.

Same-input attention controls find cached512 and narrow512 identical, K1024
versus K512 differences at most 9.536743e-7, and SiLU SFPU/LUT/CPU gated outputs
bit-identical on all four ranks. Same-input MoE controls find padding-dependent
output with SR enabled and exact padding invariance with SR disabled. Indexed
and grouped expert arithmetic remain distinct. CPU router controls explain
expert-set changes at the largest later MoE drops from the captured input
changes. These are causal diagnostics, not full-model acceptance or performance
results. Exact artifacts: `attention_localize/L2560_s2559_l0/result.json`,
`moe_localize/l0_suffix2559_v2/result.json`, and
`resume_localize/L2560_s2559/router_sensitivity.json`.

The first MoE diagnostic failed only on a host reshape of a physical 32-row
buffer. The corrected diagnostic retains all rows and extracts logical row 0,
matching production. Its failed artifact is retained in `moe_localize/l0_suffix2559/`.
All device jobs closed normally; no reset was used.

Commands (usual lane environment, four chips serialized, bounded `wait_for`):

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_resume_localize --start 2559 --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/resume_localize/L2560_s2559
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_attention_localize --captures-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/resume_localize/L2560_s2559 --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/attention_localize/L2560_s2559_l0
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_moe_localize --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/moe_localize/l0_suffix2559_v2
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_resume_variants --variants baseline fp32_boundary no_sr fp32_boundary_no_sr grouped_singleton fp32_boundary_no_sr_grouped_singleton --starts 2559 --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/resume_variants/s2559_v1
```

The final command is running. It applies uniform diagnostic overrides to public
prefill calls, records actual override counts and absence of decode replays,
and checks the unchanged baseline against saved original logits. Padding is
confined to MoE; attention/cache lengths stay unchanged. No production source
has changed, and no passing policy is claimed yet. A promising policy needs the
complete default resume gate and all datatype acceptance checks.

### 2026-11-16 08:25 UTC — remaining resume diagnostics

The second factorial remains active under its original bounded launcher. FP32
boundary carry fails starts 68 and 2528; no-SR passes 2528 but fails 68 with
minimum PCC 0.565063; their combination fails both. FP32 plus grouped singleton
fails 2559 (PCC 0.985081). No production policy has changed. The CPU matched-HF
no-SR audit rejects a blanket HF-outlier explanation; exact comparisons are in
`resume_variants/no_sr_s68_hf_comparison.json`. Next is a legal q32/k32 bounded
attention experiment with recorded cache page/alignment assertions. Devices
remain serialized; no reset or forced termination has occurred.

### 2026-11-16 08:28 UTC — evidence tooling and next precision control

The second factorial closed normally. No-SR plus grouped singleton also fails
2559 (PCC0.987118940). The q32/k32 bounded-attention diagnostic is running
under timeout3600/wait3720, log `logs/resume_attention_k32_v1.log`.

`tests/datatype_make_configs.py` now accepts a baseline JSON, version suffix,
and destination, refuses differing existing outputs before writing, and leaves
identical files untouched. All15 original generated configs remain byte-identical;
CPU checks cover resolver validation and overwrite protection. The report accepts
an explicit baseline ID and counts actual group instances in its safer-tie rule:
ten full-attention groups and one terminal head, with resolved layer exceptions.
A synthetic versioned-baseline/tied-policy CPU check passes. No runtime math or
measured result was changed by these tooling fixes.

`diagnostic_precision/accuracy_bf16_dense_ep40/proposal.json` prepares a paired
BF16 dense versus BF16 dense+all40 grouped-expert-storage control. The only paired
policy change is expert_ep BFP8→BF16; its7,549,747,200B/chip payload increase is
known, but fullcontext allocator fit is unmeasured. Indexed experts remain BFP8,
shared scalar BFP8/HiFi2, and native grouped internal activations remain BFP8.
This is an AutoFix hypothesis, not a selected config or a new P1 reference.

### 2026-11-16 08:37 UTC — runtime and reference identity

Future candidate results bind exact reference bytes and actual mapped TT ELF
libraries plus native source/build state. `tests/datatype_runtime_identity.py`
uses compact native tree/diff/submodule identities and loaded ELF hashes, with
no TTNN import in its API. A read-only CPU inspection of the active diagnostic
process maps succeeded; five mapped TT libraries are captured in
`resume_variants/attention_k32_no_sr_v1/native_runtime_observed.json`.

Candidate capture compares native identity before/after measurements. P1's
serialized launcher records matching before/after identities in separate import
processes using the same environment, explicitly labeled as that scope, and
archives them. Canonical P1 identity is bound to its exact result JSON. The
report refuses cross-reference/native-runtime comparisons and checks observed
embedding, terminal norm, shared scalar, and cache metadata as well as policy
assertions. Historical v2 baseline remains evidence but cannot qualify for
selection without the new identity fields; current-source independent repeats
were already required. CPU validation and shell syntax pass. No production
native or model TT source has changed.

### 2026-11-16 08:53 UTC — dense control and scope of remaining work

The K32/no-SR diagnostic completed:2559 passes;68 fails minimum PCC;2528
passes PCC but fails top1 at31/32. Summary of20 completed diagnostic cases:
`resume_variants/completed_diagnostics_summary.json`. No proposed production
patch is applied.

The configured BF16 dense control is active and already fails68 with mean
PCC0.992920894/min0.833366257/top1=0.984435797. The paired EP40 weight
control and a small same-input singleton/chunk recurrence probe remain before
returning to the main datatype matrix. These baseline investigations do not
replace the selected policy's required default resume check. No threshold has
been changed, and failed diagnostics are not candidate performance evidence.

Native activation audit: `native_precision_contract.json` records mixed
BF16/BFP8 grouped internals, public BF16 scope, and the absence of a legal
intermediate-dtype switch. Changing weights, math fidelity, or SR does not
remove those internal casts. No native implementation change was attempted or
claimed; this is a fixed kernel constraint in the current supported policies.

## 08:58 UTC — dense control and same-input recurrence probe

The explicit normal `accuracy_bf16_dense` policy completed all three resume
splits in 591 s. Start68 fails minimum PCC (0.833366257; mean0.992920894,
top1=0.984435797); start2528 fails mean/minimum PCC (0.988953334/0.977610482,
top1=1); start2559 passes (0.992480746, top1=1). It is not a general repair.
Artifact: `resume_variants/accuracy_bf16_dense_v1/results.json`.

The same-input layer0 singleton/chunk-n1 probe completed normally in15s:
`linear_singleton_localize/L0_s2559_v1/result.json`. The first launch stopped
before TTNN import because the command used the wrong trace-tracking variable;
its assertion log is retained. The retry set `TT_METAL_TRACE_ALLOC_TRACKING=1`
and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; no reset or production edit.
Exact command is in the result and retry log. CPU interpretation is pending.

Paired `accuracy_bf16_dense_ep40` now runs all three splits with unchanged
execution policy and thresholds. It differs only in grouped expert storage,
BFP8 to BF16 across all40 layers. Full-capacity fit is not yet established.
After this control, resume the current-source baseline and coarse matrix;
no baseline diagnostic failure is silently waived for the eventual selection.

## Same-input recurrence interpretation

CPU analysis (`linear_singleton_localize/L0_s2559_v1/analysis.json` and
`analysis.md`) finds K512→K1024 leaves raw recurrence, new recurrent state,
gated values and final attention bit-identical under both boundary policies.
Tiny projection/conv-history differences remain; later effects are untested.
At fixedK1024, chunk-n1 and singleton have exactly equal q/k/v/beta/initial
state and first differ inside recurrence/decay. BF16-boundary final attention
RMS difference is1.65e-7; with FP32 carry final output is bit-identical.
Boundary carry itself changes final attention RMS1.42e-5 (~86x larger), but
neither is established as ground truth and the full-model carry variants
already failed. No isolated K-grouping/engine production repair is supported.

## 09:08 UTC — resume controls concluded; ranking resumes

The normal `accuracy_bf16_dense_ep40` diagnostic completed in603s, without
reset. It fails minimum PCC at68 (mean0.990693280/min0.569636479), minimum at2528
(mean0.990324362/min0.976647091), and mean at2559 (0.985254239). Top1 is0.992217898,
1,1 respectively. Raising grouped expert storage alone is not a repair.
`resume_variants/accuracy_bf16_dense_ep40_v1/results.json` retains the exact
policy, commands, all40-layer evidence and failed tensors. Production math is
unchanged; the eventual selected policy must still undergo the original gate.

The pre-matrix audit found stale-native P1-cache relabeling and source mutation
risks. Fixed before any P1 scores existed: `datatype_p1_runner.py` preserves
installed scoring but binds each newly created score to actual-process native
identity, source/input bytes, text/settings and exact reference-score identity.
Missing/mismatched bindings fail closed. `run_p1.sh` archives the process and
per-score evidence. Candidate measurements now compare Python/source/input
identities before and after, including installed readiness and TTNN Python;
native coverage includes `ttnn/api`. Report noise controls require successful
completion, normal closure and valid reference/native/Python identities; entry
coverage retains row indices. CPU rejection/syntax checks pass.

STOP was removed between jobs. Independent current-source BFP8 processes v3
and v4 now run serially via `run_candidate.sh`, each under3600s timeout and
installed wait_for3720s. Their source snapshot includes the finished P1 wrapper
and identity helper; these files stay frozen through the sweep.

## 09:14 UTC — independent baseline v3 complete

Current-source v3 passes every AIME/natural prefill/teacher gate with the same
scores as v2: AIME top1.98, natural prefill.979166667, natural teacher.981770833;
minimum top5.998697917 and top1001. Traced resident-logit teacher forcing is
47.698321357 and47.826379437 t/s/u; within-process spread0.128058080. This is not
yet the final cross-process noise floor; v4 is running. Native/Python/source
and reference identity checks pass and device closure is normal.
The report collector was exercised with v3: all identity/trace/finalization
checks pass; only P1 is missing, so it correctly does not select a policy.
Current sweep_results/plots are explicitly interim with no selected point.

Final-acceptance harnesses now retain actual-process native/source/input
identity before/after. Resume success and failure both capture the constructed
policy and selected artifact hash. Nonaligned/context fixtures are bound to
evidence. Exact commands are in `selected_acceptance_command_plan.json`;
no final acceptance has yet run. Verbose local diagnostics over the repository
500KB per-file limit are listed in `local_diagnostic_artifacts.json`; retained
on disk, with compact findings and reproduction scripts tracked. Nothing was
deleted or compressed.

## 09:20 UTC — two independent current-source baselines

Both v3 and v4 complete normally (339s/342s), with identical accuracy and
before/after source/native/input identity. Four traced teacher repeats are
47.698321357,47.826379437,47.723026384,47.827400916 t/s/u; the conservative
within/between-process spread is0.129079559 t/s/u. Report collection recognizes
two independent controls and rejects selection only because P1 remains absent.
No selected policy or final headline exists yet. A final targeted audit of
activation/residual policy consumption is underway before P1/cache creation.

## 09:23 UTC — actual boundary policy consumption

A targeted audit found activation/residual policy values were validated but
not connected to observed public tensors. Applied the reviewed
`precision_boundary_consumption_proposal.patch`, then repository Black23.10.1.
The four BF16-only interface fields now resolve to actual existing dtype
arguments (prefill residual buffers, terminal activation cast, both LM-head
outputs, sampler buffers) and metadata-only tensor checks at embedding,
decoder residual, public logits and sampler inputs. No new TTNN operations,
casts or readbacks were added. Internal native mixed precision remains fixed
and separately documented. This is policy enforcement/evidence, not a resume
numerical repair or support for new activation datatypes.

`precision_summary.observed_boundaries` records actual dispatched/captured
sites; it is not a trace replay counter. The report requires these observations
for all40 decode layers and the public prefill/logit/sampler sites. Post-selection
performance now saves the summary after measurement as well. AST/compile,
repository formatter and diff checks pass. Because source changed, v3/v4 stay
historical and new v5/v6 controls are required for ranking. No P1 cache has yet
been created, so none is silently relabeled. No further production changes are
planned before the matrix unless a runtime failure requires one.

Local checkpoint `c873e26136c8140fe419e523525265ee7d3939dc` records the boundary
plumbing, provenance guards, harnesses and compact diagnostic evidence. Normal
hooks initially fixed import order and one final newline; the second commit
attempt passes all hooks. No hooks were skipped and nothing was pushed.
The now-frozen source starts serial v5/v6 baselines; earlier checkpoint is
`a8fdbbc11b9bf58e597e5d4003dec559e3a50a99` and completed stage7 is`e03236d2`.

## 09:30 UTC — boundary-wired baseline v5

v5 completes in343s with the unchanged full-model accuracy: AIME prefill and
teacher top1.98; natural prefill.979166667 and teacher.981770833; minimum
top5.998697917/top1001. Traced repeats47.686910083 and47.826742645 t/s/u.
The report validates all94 actual boundary observations and every required
runtime identity field. Prepared allocator census is exactly equal to v3,
confirming no additional device allocation from this metadata-only enforcement.
v6 is running; P1 follows the final two-process noise calculation.

## 09:36 UTC — final-source baseline pair and P1 launch

v6 completes in346s with identical per-reference accuracy and all94 boundary
observations. Its repeats are47.702209688 and47.827742397 t/s/u. Final control
spread across v5/v6 is0.140832313 t/s/u, mean47.760901203;
mean resident-logit TTFT3847.110845ms. Both raw results pass the
report's complete runtime/native/Python/reference checks and normal closure.

Started serialized P1 with exact launcher:
`bash models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/run_p1.sh models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/p1_candidates.json accuracy_bf16_experts20_ep baseline_bfp8 baseline_bfp8_v1 only`.
Installed scoring/gates remain unchanged:2 external pinned-HF answers,
12,288 reasoning positions,256 traced decode tokens per answer, and default
2,048 HF positions. The exact expanded command and actual-process identities
are retained in `logs/p1_baseline_bfp8_v1.log` and
`p1/baseline_bfp8_v1.process.json`. Device timeout21600s,wait_for21720s.

## 09:51 UTC — user-directed stage cap and hand-off

The user capped stage time/tokens and directed wrap-up: keep the best passing
version, run only remaining required gates once, and carry untried ideas as
hand-off notes. No new precision search or numerical repair experiments will
run. The active baseline P1 run is allowed to finish normally. Production
runtime stays at checkpoint c873e26136c8140fe419e523525265ee7d3939dc.

Untried full-model policies (prepared configs, not measured rejections):
accuracy_bf16_dense; accuracy_bf16_experts20_ep as a ranked accuracy/performance
candidate; inner_experts_bfp4_lofi/hifi2; inner_shared_bfp4_lofi/hifi2;
inner_expert_gu_bfp4_lofi/hifi2; experts_bfp8_hifi2; attention_bfp8_lofi;
head_bfp8_lofi; shared_bfp8_lofi; projections_bfp8_lofi; kv_bfp8; ccl_bfp8.
Potential follow-up combinations and selective expert-down/shared-down
restorations are also deferred. Configs and serialized launchers are retained.
No BFP4 policy is selected without P1. The BFP8 KV component probe is not a
full-model candidate result. With only BFP8 full-model ranking measurements,
there is no evidence for a global precision optimum or a lower-precision win.

Resume numerical repair remains unresolved after AutoDebug/AutoFix experiments
recorded in AUTOFIX_resume.md, resume_variants/, hf_resume_control/, and
linear_singleton_localize/. The unapplied numerical proposals are hand-off
ideas, not accepted fixes. No more repair experiments are authorized by this
wrap-up. The default selected-path resume gate will run once and its actual
verdict will be retained; blocking failures cannot be called passing or
silently waived. Remaining acceptance, documentation, independent stage review,
and local commits will be completed within this wrap-up. No vLLM work or push.

## 10:10 UTC — retain BFP8 after P1 pass; final acceptance once

P1 completes normally, all required checks pass. Baseline versus fitted
accuracy_bf16_experts20_ep: 12,288 positions, dNLL +0.001005596,
SE0.000377739, allowed max(2*SE,0.004)=0.004; KLx1000=0.903216 below5.
Top1 agreement0.994059; traced decode512positions top1baseline0.90625 versus
reference0.908203, within noise. Two texts use128-position block SE rather
than text-level uncertainty. Advisory HF slice skipped by installed memory
preflight:67GiB weights vs49GiB free host RAM; no HF agreement result claimed.
Canonical doc/eval_readiness/precision_reasoning.json names baseline_bfp8.
The actual-process before/after identities and immutable score bindings pass.

Copied configs/baseline_bfp8.json byte-for-byte to
selected_precision_config.json. This preserves the completed optimized model
numerics while making the full policy the default normal construction path.
BFP8 is the sole fully ranked evaluated policy and passes both full-model
references and P1; no lower-precision optimality claim. Launching
`bash models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/run_final_acceptance.sh`
for the once-only remaining token-out, nonaligned, resume, qualitative, and
unchanged-KV long smoke. Exact per-job commands and statuses are saved under
final_acceptance/*.attempt.json and logs/final_*.log. No source edits during
these measurements. The full stage verdict remains pending acceptance/review.

The report initially found a pre-existing limitations schema mismatch: the
installed gate expects `items`, while historical stage5 readers use
`limitations`. Added an exact compatibility copy as `items`, keeping legacy
records unchanged and broadening no owner decision. The report then validates
both trials and P1 provenance and writes the selected red Pareto points,
JSON/CSV. No gate numeric criteria or runtime code changed. The post-selection
acceptance queue is active, session42775; script guards against repeated jobs.

## 10:15 UTC — post-selection token-out PASS

Normal default construction consumes selected_precision_config.json and all
runtime policy fields match. Warm128-in/128-out queued no-readback token-out:
46.690388244tokens/s/user,21.417684ms/step,127timed model+sampler replays,
zero timed token refreshes/readbacks/captures. TTFT2799.036538ms includes first
sample readback, excludes loading/compile/warmup/reset/tokenization. Caller-
observed delivery with127token readbacks is46.500566426tokens/s/user, separately.
Teacher forcing in this benchmark is47.771527372tokens/s/user, consistent with
ranking controls. Use46.690388244 for later serving comparisons; the raw
harness's historical caller-observed headline flag is explicitly distinguished
in post_selection_performance.json. Artifact post_selection_token_out.json,
normal device closure, wrapper exit0 after297s. Nonaligned acceptance now runs.

## 10:21 UTC — nonaligned acceptance PASS

Normal default selected policy passes logical prompt lengths1,33,2049 in4096
context. Next positions3,35,2051; each has exactly two model+sampler trace
replays and zero host refresh/readback/capture during decode. Program cache
remains9590. Actual native/Python/input identities and selected artifact stay
unchanged before/after; runtime policy equals selected JSON. Normal device
closure; wrapper exit0 after356s. Artifact:
final_acceptance/nonaligned_1_33_2049.json. This is an operational capability
check, not an HF accuracy or generated-text quality result.

## 10:36 UTC — final default resume gate FAIL, unchanged baseline behavior

Once-only default run completes normally in925s, wrapper exit1. Start32PASS;
start68 minimumPCC0.968224176FAIL; start2528 mean0.989604529/min0.973850864FAIL;
start2559 mean0.986428019FAIL. All four top1 and final-argmax criteria pass;
all four applicable negative controls fail as expected. No thresholds changed.
Artifact final_acceptance/resume_default.failed.json; both native and complete
Python/input identities remain stable, actual runtime policy equals selection.

All fresh/resumed logit tensors for all three failed cases are bitwise equal
to baseline_resume_diagnostics tensors. CPU torch.equal comparison is saved
in final_acceptance/resume_baseline_equivalence.json. Therefore the prior exact
HF per-position comparison applies directly; HF passes2559 and fails different
positions in other cases. This is unchanged baseline numerical sensitivity,
not a new selected-policy regression, but the required PCC gate still fails.
No blanket HF waiver or previous stage6 owner acceptance is extended. The
installed limitations.items records it as unresolved-after-autofix. No further
repair experiment under the user cap. Qualitative and context gates continue
once; the independent review will receive the raw failed gate and this evidence.

## 10:41 UTC — qualitative suite PASS within bounded scope

All six canonical chat-template prompts complete128greedy tokens with normal
selected construction and stable runtime/source/input identities. All six TT
token sequences exactly match stage7 outputs. Read all12 paired completions:
relevant reasoning, no new looping/language/leakage regression. q03 shared
“thinking thinking” and the short TT list-label oddity are unchanged. Both
HF and TT frequently exhaust the128-token budget in thinking; no EOS/final-
answer criterion was invented. Saved verdict.json distinguishes bounded smoke
from broad quality or completed-task correctness. No10-seed comparison claimed.

CPU-only installed checker command:
`python -m readiness_check.check_degenerate_output --root models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/final_acceptance/qualitative --scope autoregressive --json models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/final_acceptance/qualitative/degenerate_check.json`.
Exit0,12HF/TT records, no findings. Generation wrapper exit0 after288s.
The final32,769-token smoke now runs once in65,536 allocation.

The installed stage08 P1 gate ran once and exited0:
`MODEL_DIR=models/autoports/qwen_qwen3_6_35b_a3b bash "$TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/08-datatype-sweep.check.sh"`.
Log logs/stage08_final_gate.log, summary stage08_final_gate.json. This scoped
P1 pass does not override the separate default resume failure.

Repository hooks normalized terminal newlines in qualitative text/metadata
exports and the checker JSON; canonical generated token/completion records
were unchanged. Second hook run passes. Report plots now aggregate the two
baseline processes into one mean policy point and show the full repeat range
as an error bar; the single-policy frontier and red selection are explicit.
No runtime source or measured numerical result changed.


## 2026-11-16 owner acceptance and final context acceptance

Owner decision: accept the resumed-prefill PCC miss (start 68 minimum PCC 0.968 vs 0.98; the matched HF BF16 control also fails minimum PCC on the same case, and top-1, final argmax and negative controls pass) as a known limitation, the same as the stage 6 decision. Keep the selected precision configuration, record this in limitations, finish the stage review and complete stage 8. No further resumed-prefill investigation.

Applied to closing stage 8 with the existing selected configuration and recorded resumed-prefill PCC failures. The owner specifically cites start 68; acceptance does not assert that HF reproduces every TT-only failure at starts 68, 2528 or 2559. Raw thresholds, failed results and HF comparisons remain unchanged. No further resumed-prefill investigation is authorized in this stage. See `owner_decision.md` and `owner_acceptance.json`; `doc/eval_readiness/limitations.json` now records the explicit acceptance while retaining raw FAIL and the HF distinctions. Earlier unresolved-status entries below/above are historical. No production or selected precision policy change was made.

The final context artifact passes: logical32769 in65536 allocation, next position32771, two model/sampler replays, zero input refresh/readback/recapture, stable native/Python/input/selected hashes and allocator totals. Free DRAM12,954,371,584bytes/device; advertised262144 context is preserved. The interrupted wrapper did not persist its exit status; the artifact and log show success and normal device closure at10:57:23UTC. The recovered attempt explicitly uses exit_code=null. No hardware job remains or was rerun.

Final pre-review checks: `git diff --check` and repository `pre-commit run --files` over all stage-owned changed/untracked files passed (log `logs/wrapup_final_precommit.log`). Devices have no open holders. Both plots exist; top1 plot visually inspected after per-policy aggregation. Required CSV will be force-added because the repository globally ignores CSV. Independent fresh xhigh review round1 started after all device work completed.

## Stage 8 final review and checkpoint

Fresh independent xhigh reviewer `/root/stage8_review_round1` returned **clean-pass** in round1; report `stage_review_round1.md`. No Required Work remains under the controlling owner instructions. The reviewer independently rederived full-model metrics and P1 from raw tensors, verified current source/input hashes and runtime consumption, inspected all qualitative completions and stage7 equality, and checked final context/trace/nonaligned evidence. The raw resumed-prefill gate remains FAIL with explicit owner acceptance; no numerical repair or broad precision optimum is claimed.

Carried to next stage: the untried precision matrix listed above, limited two-text P1 coverage and skipped advisory HF slice, bounded thinking-only qualitative evidence, and preservation of the default selected policy and46.690388244tokens/s/user no-readback metric during later integration. Resumed-prefill is an owner-accepted known limitation; further investigation is not required or authorized in this stage.

Only tt-metal was touched, branch `main`; no serving-plugin/vLLM repository changes and no push. Prior stage8 checkpoints: `a8fdbbc11b9bf58e597e5d4003dec559e3a50a99` and `c873e26136c8140fe419e523525265ee7d3939dc`. Final acceptance checkpoint SHA will be appended after committing.

Final acceptance checkpoint: tt-metal `main` **`6281d5e4bf12a67cd6f3165c815837de59b9f982`**, committed locally with all repository hooks passing; required CSV included. No push. The CSV default CRLF caused `git diff --check` whitespace diagnostics despite hooks passing; normalized the CSV export to LF and set the report writer to emit LF. Parsed CSV rows are exactly unchanged. This formatting-only adjustment does not change runtime code, evidence, metrics or review conclusions. Final log/formatting checkpoint follows this record.
