# AutoDebug: optimized decoder PCC failures

Investigation date: 2026-11-13. Inspection only: no device opened, no reproduction or reset run, no source/test edits, and no AutoDebug relaunch. The only file written by this investigation is this report. Three independent source/evidence audits were used and their findings were checked against the code.

Paths abbreviated below are relative to `models/autoports/qwen_qwen3_8_flash_next/`, unless otherwise specified. Git HEAD is `3cd9937dfdbb2917990443e2e937e4dd3b9e60c7`; the optimized implementation and its evidence are untracked, so HEAD alone does not identify the investigated code.

**Revision boundary:** the reported failures and the controls discussed below used optimized source SHA `fb77bd2c...`. Unless explicitly called a subsequent edit, source line numbers and “selected default” refer to that failure revision. After the first report draft, the owning session changed the default to per-kind attention precision and GDN HiFi4 (observed new SHA `032d0b66...`). This investigator did not make that edit. The new default needs its own final validation; old failures or passing diagnostic controls must not be relabeled as its final gate results.

## Findings

**There is no demonstrated optimized GDN recurrence, request-state, or custom-kernel contract bug in the inspected code. There are concrete evidence/test discrepancies, and several distinct numerical failures that must not be treated as one synthetic exception.**

1. **The continuation failure is mischaracterized by its filename.** `final_continuation0.json` fails on an independent fresh 97-token prefill, row 5, while another request's decode trace is live. Resumed prefill, traced decode, repeated state/output equality, and request isolation pass. Whole-row PCC is **0.9895511866**; contribution PCC is **0.9863465428**. Both repetitions agree. The aggregate PCC near 0.99997 hides this row.
2. **The optimized wrapper can misreport the effective precision.** `final_continuation0.json` says BF16 experts, but `final_continuation0_selection.json` records actual BFP8 experts. The outer wrapper overwrites the inner probe's argument. This is a confirmed reporting/control-construction defect, not an explanation for corrupted outputs.
3. **Two batch outliers have established predecessor sensitivities.** GDN request 16's fresh-prefill contribution and PLE request 2's fresh-prefill contribution already failed literal HF in historical matched fixtures. Those predecessor passes used explicit, narrow backend/packed-lookup controls. The new plain probes omit those controls. They must be re-evaluated on the current implementation; an old classified pass cannot be copied forward.
4. **The new GDN outliers localize to Z/A/B numerical policy, not expert LoFi alone.** Request3 decode and request23 prefill fail the selected default. Attention BF16 alone repairs continuation but leaves request3 and introduces another batch outlier. Subsequent **BF16 Z/A/B + GDN-only HiFi4** and **BF16 Z/A/B + inherited native projection** controls repair the disputed batch outputs; their only remaining literal failure is the explicitly classified request16 contribution. BFP8/LoFi experts stay selected. This is a narrow intervention supported by completed controls, not proof of a defective kernel. Final default validation and latency still remain separate.
5. **OPT012 applies to the random-weight stress failure, but does not turn the entire evidence set into a pass.** All inspected “real HF” probes use real checkpoint weights with generated Gaussian hidden states. No recorded target-model activation corpus was found in the inspected optimized evidence. Same-contract target-activation coverage and adjudication of the real-weight output failures are still needed before claiming representative precision acceptance.

The current policy manifest correctly says `pending_final_gates`; that is consistent with the evidence.

## Direct observations: complete reported matrix

Thresholds below describe the scripts actually run, not a newly selected acceptance rule. Batch/continuation enforce PCC >= 0.995 for aggregate and every row, both whole output and residual-subtracted contribution (`tests/probe_qsa_decoder.py:32–70`).

| Artifact / case | Direct result | What it establishes |
|---|---|---|
| `doc/optimized_decoder/final_synthetic.xml`, GDN sequence 1 | Prefill 0.9999472499; decode step 0 0.9999150634; step 1 **0.9947856069** | One random-weight/random-activation numerical failure. Step numbers are zero-based. |
| Same XML, other 13 cases | Pass: GDN lengths 2,31,33,34,65,66; QSA lengths 1,2,31,33,34,65,66 | Those exact fixtures pass. No PLE case is in this 14-case matrix. |
| `final_continuation0.json` | `live_other_prefill[1]` and `[4]`, length 97 row 5: whole **0.9895511866**, contribution **0.9863465428** | Fresh-preﬁll outlier; not a resumed/decode error. Seed 101029. Length97 is processed as 32+32+32+1, and row5 is in its first full chunk, not the tail. |
| Same continuation artifact | Length97 aggregate whole 0.9999657273 / contribution 0.9999521375; all resumed/decode and equality/isolation checks pass | Aggregate success does not prove all-row success. Deterministic output alone does not prove correctness. |
| `final_batch0_32.json`, request 3 decode | Whole **0.9927616715**, contribution **0.9834337234** | A model-visible real-weight single-token output failure, with repeat determinism true. Its prefill length is 1. |
| Same batch, request 16 prefill | Aggregate contribution **0.9939829111**; row0 contribution **0.9905362725** | Fresh contribution-only failure, prefill length 2. |
| Same batch, request 23 prefill | Row2 contribution **0.9923265576**; whole row 0.9952174425 | Fresh contribution-only failure, prefill length 3. |
| `final_batch1_5.json`, request 2 prefill | Sole failure: row0 contribution **0.9949988127**; whole row 0.9971483350 | PLE miss is only 0.0000011873 below the scripted threshold. Every whole/decode gate passes; repeats are deterministic. |
| `final_hf0_len1.json` | Prefill 0.9999702573; minimum decode 0.9999430776 | Real GDN weights at length1 pass the probe's whole-output metrics. |
| `final_hf0.json`, length65 | Prefill 0.9999767542; minimum decode 0.9999662042 | This GDN probe measures aggregate whole PCC, not all-row contribution PCC. It is weaker than the failing continuation gate. |
| `final_hf1.json`, PLE33 | Pass; minimum prefill contribution 0.9960937707 | Real weights/packed table, random hidden states; FP64 all-row metrics. |
| `final_hf3.json` | QSA 1,2,31,32,33,65 pass; minimum listed prefill contribution 0.9972711802 | These exact QSA fixtures pass. |
| `final_batch3_32.json` | Strict pass; minimum contribution 0.9984466434 | Tuple batch32 works for this QSA fixture. |

`final_continuation1.json` is a **classified** pass (`literal_hf_strict_passed=false`, `classification_subject_to_stage_review=true`), not an unconditional literal-HF pass. `final_continuation3.json` is strict.

The synthetic test delegates to `tests/test_synthetic_decoder.py:90–165`. Its immediate assertion at line154 aborts sequence1 at step1, before step2, the second repetition, and fresh-state reset verification. Those later checks are **unmeasured for the failing fixture**. They must not be inferred from the other 13 passing cases.

### Weight and activation provenance

* Synthetic weights: `tests/synthetic_weights.py:39–70`, generated from pinned per-tensor statistics; synthetic hidden states: `tests/test_synthetic_decoder.py:92–93`.
* Real GDN checkpoint probe: `tests/probe_decoder.py:27–36`, Gaussian hidden states with standard deviation 0.02.
* Continuation: `tests/probe_public_continuation.py:75–81,116–120`, same generated distribution with explicit seeds.
* Batch: `tests/probe_batch.py:235–269`, global seed `1981 + layer`, prompt length `1 + request % 3`, generated prompt/decode/IDs.
* QSA: `tests/probe_qsa_decoder.py:115–118`; PLE: `tests/probe_ple_decoder.py:189–193`, likewise generated hidden states.

Consequently “real weights” is accurate; “real recorded activations” is not established. Passing GDN1 does not reproduce the synthetic sequence1 input or the batch request3 input. Running the batch script with `--batch 1` selects request0, not the failed request3 RNG position.

## Current-source controls observed during inspection

These artifacts were produced by the owning session while this investigation was reading files. **This investigator did not run them.** Completed control provenance/selection records were inspected, including model-source hashes matching the failure revision. A `control_*.json` policy file alone is not execution evidence.

| Actual intervention | 97-token row5 whole / contribution | Batch request3 decode whole / contribution | Interpretation |
|---|---|---|---|
| Selected default | .98955119 / .98634654 | .99276167 / .98343372 | Fails as above. |
| `control_hifi4`: expert fidelity only | .98928636 / .98603499 | .99278921 / .98349321 | Expert LoFi alone is insufficient to explain either failure. |
| `control_attn16`: GDN Z/A/B weights BF16, same DRAM backend | **.99979562 / .99974608**, entire continuation probe passes | .99279869 / .98351520 | Fixes continuation, not request3. Batch request23 improves, but request29 row2 now fails whole .99461651 / contribution .99168545. Not a safe universal replacement. |
| `control_shared16`: shared weights BF16 | .98955095 / .98634595 | .99274606 / .98339766 | Shared-weight precision alone does not repair these outputs. |
| `control_residual_dram` | Not assessed here | .99276167 / .98343372 | Request3 metrics match default exactly. L1 residual storage alone is not the cause of that failure. |
| `control_fused`: current FusedDecoder, **BFP8 experts** | **.99996883 / .99995911**, continuation passes | **.99998319 / .99996239** | Current fused BFP8 is a stronger contrast than an old BF16 baseline. Batch as a whole still fails the inherited request16 contribution gate. |
| `control_attn16_hifi4`: BF16 Z/A/B, **GDN-only HiFi4**, same DRAM geometry | Not assessed here | **.99987632 / .99975544** | Classified batch pass: all other literal metrics pass; only request16 fresh contribution is carried by the explicit GR control. |
| `control_attn16_interleaved`: BF16 Z/A/B, inherited native HiFi4 projection | Not assessed here | **.99987632 / .99975544** | Same classified batch outcome and matching reported metrics as the GDN-HiFi4 control. |
| `control_expert_l1_off`: expert input/output in DRAM | Not assessed here | .99276167 / .98343372 | Same failed-row metrics as default; expert L1 placement alone does not explain them. |
| `control_dense_interleaved`: disable all dense DRAM replacements, retain BFP8 weights | Not assessed here | .99273765 / .98337251 | Fails, with additional prefill outliers; removing the DRAM family alone is insufficient. This is a multi-role control. |

Artifacts are `final_{continuation0,batch0_32}_control_{hifi4,attn16,shared16,fused}.json` and `final_batch0_32_control_{residual_dram,attn16_hifi4,attn16_interleaved,expert_l1_off,dense_interleaved}.json`, with accompanying `_selection` and `_provenance` records. Fused batch completed at 16:23:45 UTC; fused continuation at 16:23:54; residual-DRAM at 16:25:29. BF16/GDN-HiFi4 completed at16:25:51 and BF16/native at16:26:01, both exit0 with `literal_hf_strict_passed=false`. Their request16 TT/GR-control row0 contribution is .9997075796.

These contrasts **do not prove an invalid DRAM kernel**. The same DRAM geometry passes the batch control with BF16/GDN-HiFi4. They demonstrate a sufficient numerical-policy intervention for this fixture while preserving BFP8/LoFi experts. They do not prove both settings are individually necessary: no completed BFP8-Z/A/B plus GDN-only-HiFi4 contrast is included here. In particular, do not confuse the failed **expert** HiFi4 control with a GDN-only HiFi4 control.

## Historical controls: separate inherited sensitivity from regression

The historical files below use matching probe source/fixture construction, but historical fused implementation hash `06df...` differs from current `d638...`. They are historical contrasts, not one-variable proof of current correctness.

| Historical artifact | Exact corresponding location | Result |
|---|---|---|
| `doc/fused_decoder/final_v5_continuation0.json` | `live_other_prefill[1].per_row[5]` | Whole .9999862909 / contribution .9999813437. BF16 experts. |
| `doc/fused_decoder/final_v5_batch0_32.json` | Request3 decode | Whole .9999909997 / contribution .9999795556. BF16 experts. |
| Same | Request23 prefill row2 | Whole .9999890924 / contribution .9999831319. |
| Same | Request16 prefill row0 | Contribution .9906031489; independent GR FP64 control reproduces literal disagreement .9906105995. Classified fresh-contribution pass, not strict literal pass. |
| `doc/fused_decoder/final_v5_batch1_5.json` | Request2 prefill row0 | Contribution .9949792027; independent actual-packed-lookup control sensitivity .9949601293; TT/control contribution .9999611378. BFP8 experts already used. |

The last PLE number is slightly worse than today's .9949988127. This strongly supports testing the inherited packed-lookup explanation first; it does not prove that the current candidate passes the matched control.

The control gate is deliberately narrow (`tests/probe_batch.py:29–75,385–414`): fresh prefill only, literal whole-output gates still >=.995, every TT/control metric >=.995, and a contribution discrepancy reproduced at the same row or aggregate coordinate by the independent control. PLE additionally checks every phase against the control. **Neither control can excuse request3 decode or continuation0's whole-row failure.**

The earlier request3 failure in fused stage-v2 had very similar PCC and was repaired by preserving the QKV reduction tree (`doc/fused_decoder/work_log.md:264–278`). Current code retains that repaired tree; numerical similarity is not evidence that the old bug returned.

## Source audit and causal boundaries

### What actually changed

`tt/optimized_decoder.py:504–671` constructs a fused decoder, changes selected weight representations/matmul programs, substitutes `OptimizedSparseMoe`, and optionally changes residual storage. Important effective settings:

| Operation | Actual selected path |
|---|---|
| GDN QKV and output projection | Inherited exact/precise implementations, including `FusedExactQKV`; not converted by `attention_dtype` for linear attention (`fused_decoder.py:2723–2740`, `optimized_decoder.py:578–579`). |
| GDN Z/A/B | Original checkpoint matrices concatenated into K=2560, N=6240; `DramLinear`, BFP8 weights, BF16 input/output, HiFi2, FP32 destination accumulation (`optimized_decoder.py:254–320,606–635`). |
| Shared packed gate/up/route | K=2560, logical N=1281; same DRAM wrapper, BFP8/HiFi2. Shared down remains the separately converted native projection by default. |
| Expert gate/up and down | BFP8, LoFi, FP32 destination, explicit BF16 outputs; gate K-block20, down K-block10, 40-core configurations (`optimized_decoder.py:359–425,711–725`). |
| Router and GR mixing down/up | Inherited precise projections. They still consume upstream values, so precision in attention can change later routing even if the router projection itself is exact. |
| Packed SiLU and residual writer | Same kernel source strings and compute arithmetic as fused; optimized wrappers add L1-compatible storage (`optimized_decoder.py:14–21,31–203`). |

`decoder_components.linear` recognizes the `PreciseGDNProjection` callable protocol (`tt/decoder_components.py:21–24`); `DramLinear` inherits that protocol. The replaced projection is called, rather than silently bypassed by a generic weight path.

GDN packed Z/A/B is split at `tt/fused_decoder.py:2081–2084`: A controls decay and B controls beta before the state update (`2131–2157`); Z gates the normalized output (`2160–2164`). Thus this projection can affect both current output and future state, even though QKV, output projection and the recurrence are unchanged. The attention result is injected before MLP input/routing (`tt/functional_decoder.py:134–142`). Expert fidelity is applied only after routing (`optimized_decoder.py:328–365`), so it cannot repair an already changed attention state or current top-10 selection in an isolated decoder layer.

**Interpretation, not proof:** small changes before routing can cross a discrete top-10 boundary and produce isolated row outliers amid high aggregate PCC. This predicts non-monotonic behavior such as attention-BF16 removing request23's outlier while creating request29's. The artifacts contain no paired intermediate/router captures for those exact rows, so routing amplification remains a hypothesis. Compare expert ID/weight pairs and selected sets, not just output ordering: a harmless permutation of paired experts is not a routing bug.

### State, logical dimensions and synchronization

* Each linear request gets independent conv `[1,1,3,10240]` BF16 and recurrent `[1,48,128,128]` FP32 storage (`functional_decoder.py:87–94`). PLE adds nine history rows and request-owned token history (`101–120`). Tuple APIs pass `state[i]` with input `i` (`144–151,180–190`); the fixture allocates states separately (`tests/probe_batch.py:438–443`). Batch32 is 32 independent one-request executions, not one M=32 decode matmul.
* Public prefill processes at most32 logical rows per chunk (`functional_decoder.py:157–177`). Synthetic sequence1 and every tuple decode use logical T=1, physically one 32-row tile. Batch prefill cycles T=1,2,3. The failed length97 row5 uses T=32. All these select the same one-tile-row DRAM matmul family; tail-only or multi-tile-M explanations do not cover the whole failure matrix.
* Default compact experts apply at T<=2: T=1 selects10 IDs; T=2 uses the paired union representation; T=3/32 use the full expert-axis sparse path (`optimized_decoder.py:331–358`). Therefore a compact-only indexing bug would not explain the failed length97 row5 in its first T=32 chunk.
* Compact GDN reader uses key head `h//3`, value head `h`, initial state tile slab `h*16`; writer uses corresponding output head and state slab (`fused_decoder.py:1623–1684`). With48 value heads,16 key heads and128-dimensional state, these match the model. Kernel loops use logical token count.
* The recurrence performs decay, k·state, beta*(v−read), rank-one update, then q·updated-state (`tt/kernels/gdn_compute_sfpu.cpp:166–194`), matching installed HF `modeling_qwen4_exp.py:433–444`. For T=1 it consumes initial-state16 tiles and creates the updated16 tiles. For T=32, each iteration consumes the prior16-tile state and alternates s2/s3. On the final iteration `exact_rank1` additionally packs16 tiles into separate `cb_state` for the final-state writer (`gdn_compute_sfpu.cpp:68–102`). q/k/v each consume4 tiles per token, decay/beta1; no inconsistent loop bound was identified.
* Three TT GDN conv-history rows versus four stored by HF is not an off-by-one: after append, HF keeps only the last sequence-length convolution outputs, which require the previous three rows. PLE's nine-row dilation3 history is likewise consistent with four taps.
* Warm decode explicitly synchronizes before disposal; restores precede replay; measured traces use `blocking=True` followed by `to_torch` (`test_synthetic_decoder.py:136–153`, `probe_batch.py:489–509`, `probe_public_continuation.py:321–354`). Continuation fresh-prefill output is read synchronously at line392. These PCC values represent completed outputs, not a host assertion before pending execution.

This audit lowers the priority of state cross-talk and queue/lifetime hypotheses; it does not prove every kernel race impossible. The failed synthetic fixture lacks its later replay/reset evidence because of its early assertion.

### Lowered numeric paths inspected

The explicit `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` selects the DRAM-sharded factory (`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2765–2766`), with width-sharded L1 activation, width-sharded DRAM weights, and width-sharded L1 output converted back to DRAM. Operand order is activation then weight, no bias or fused activation (`optimized_decoder.py:302–320`).

For the observed eight-bank Blackhole configuration (`doc/optimized_decoder/dram_gdn_v1.json`, `dram_banks=8`), GDN has Kt=80, weight-bank width ceil(195/8)=25 tiles; two workers per bank produce CN=2, CK=1, KB=2,40 K blocks, and two column passes with FP32 destination storage. These values are identical for logical T=1 and T=32 because both have physical Mt=1. The shared packed projection has41 logical N tiles,6 tiles per bank and one column pass. `cores=4` specifies activation/output storage shards, not compute-worker count: these projections use16 and14 active workers respectively. The factory's whole-K destination path and worker-group-rotated K traversal are explicit in `.../factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1257–1276,1313–1365,1525–1537`; consumption/packing is in `.../kernels/compute/bmm_dram_sharded_multicore.cpp:163–238`. This is a concrete reason not to expect bit-identical accumulation to the fused native projection. No buffer-count contradiction was found at this boundary.

The selected sparse path preserves FP32 partial buffers and marks FP32 reloads `UnpackToDest` (`.../sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:230–245,814–835`). It is not justified to attribute these failures to the historical TF32 partial-reload problem without further evidence. No complete hardware/LLK correctness proof is claimed by this inspection.

For concreteness, packed expert gate/up uses K=2560,N=1280, Kt=80,Nt=40,40 workers, one N tile per core, and four K blocks of20 tiles. Expert down uses K=640,N=2560, Kt=20,Nt=80,40 workers, two N tiles per core, and two K blocks of10 tiles; subblock width2 divides the output block. All logical T=1,2,3,32 cases have `per_core_M=1` (`models/demos/gpt_oss/tt/experts/config.py:134–204`). T=2 has20 compact slots containing the union and zero-weight padding, not necessarily20 distinct experts. T=1/2 compact outputs use L1; T=3/32 retain the512-expert output axis with sparse union execution and DRAM outputs. These contracts match the caller's selected geometry; no nominal-versus-runtime batch-size contradiction was found.

## Confirmed test/report issues and precise treatment

### Effective policy is not reliably reflected in the probe JSON

`tests/run_optimized_probe.py:22,31–35` sets outer `--expert-dtype` default BFP8 and unconditionally overwrites `kwargs['expert_dtype']`. Arguments after `--` belong to the inner probe. Consequently an inner BF16 default/report does not describe the model constructed. The selection record captures actual gate weight dtype (`39–49`), explaining the mismatch in continuation0/3.

Minimal test/docs correction: record requested inner settings separately from effective factory/model settings; include both gate/down dtypes, each dense projection class/dtype, actual fidelity and memory layout. An outer BF16 control must pass `--expert-dtype bfloat16` **before** `--`. A fused control with omitted outer dtype remains BFP8. `expert_compute` currently serializes an opaque compute-object string, so explicit compute fields and profiler rows are preferable. Per OPT013, construction records alone are not runtime-row proof.

`selected_decoder_policy.json` also overstates two sensitive exceptions: GR injection still uses default BF16 weights/native linear (`decoder_components.py:75`, `optimized_decoder.py:236`), and GDN's learned norm scale is BF16 (`decoder_components.py:224–226`). Narrow “all GR projections FP32/exact” to mixing down/up and distinguish norm scale dtypes. These are documentation inaccuracies, not newly introduced runtime defects.

### Preserve runtime coverage when making synthetic PCC diagnostic

OPT012 explicitly distinguishes random-weight/synthetic-activation stress from selection on real target weights and recorded/real activations. Its prerequisites and rejection criteria must be applied together:

* Do not reject a faster otherwise validated policy solely for synthetic sequence1 PCC .9947856.
* Do not blanket-`xfail`, skip, reseed, or immediately return from this test. In an optimized-specific test adapter, collect all numeric scores as diagnostic evidence while still executing hard finite-output, shape, changed-input, trace replay, reset determinism and state checks. Preserve the original failure artifact.
* Keep the optimized default exercised. Do not choose a slower fallback and then mark its lower-precision alternative an expected failure to manufacture a green suite; OPT012 explicitly forbids that policy-selection shortcut.
* Add same-contract target-weight/recorded-activation cases covering the disputed conditions. Preserve literal and contribution metrics with clearly stated acceptance roles. Structural/runtime failures and exact op-contract blockers remain hard failures.
* Do not erase real-weight model-visible failures just by calling their generated inputs “synthetic.” Current request3 and continuation0 observations require either a demonstrated fix, a justified same-contract target-activation adjudication under the owner's acceptance contract, or explicit unresolved status. OPT012 permits rejection based on target-weight model-visible output failure; it is not a tolerance waiver.

The exact reference read was `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/optimize/references/opt-advisories.md:115–121` (OPT012), with OPT013 at123–129.

### Metric and acceptance-contract discrepancies

Batch/continuation use FP32 `torch.corrcoef`; PLE's short probe uses FP64 streaming metrics. Save the same expected/device/residual tensors and recompute in FP64 before interpreting the marginal PLE miss. **It is not established that FP64 changes that result's classification**; raw batch tensors were not present in the JSON. Residual subtraction should remain explicit, and each row must retain its request/phase coordinate.

Historical `doc/fused_decoder/README.md:45` reports an owner acceptance rule of aggregate whole PCC>=.995 and a diagnostic row floor .99. Current batch/continuation scripts instead enforce .995 on every whole/contribution row, and optimized docs do not explicitly reconcile the two. This report does not import that historical reported instruction as authorization to change today's gate. State the active contract explicitly before publishing acceptance. Continuation0 row5 is below even .99; batch-wide pooled PCC cannot be reconstructed by averaging per-request correlations, and is not stored.

The plan runner changed during inspection: current `tests/run_optimized_plan.py:42–47` reports `completed_with_failures` and exits nonzero when continued jobs failed. An earlier scheduling-only `completed` label is not a current source defect. Always inspect required row exits and actual gate results, not just a plan label or selection `completed` flag.

## Smallest discriminating checks, in order

These are proposed owner-run checks, confined to optimized tests/docs and existing constructor controls; check2 was partially completed by the owner while the first draft was being reviewed, as recorded above. None was executed by this investigator.

1. **Resolve inherited controls and metric ambiguity without changing thresholds.** Re-run current PLE batch5 with the existing explicit packed-lookup control and current GDN batch32 with the existing explicit GR backend control; retain all strict results. These can classify only the matching fresh contribution anomalies. Save raw failing-row tensors and FP64 PCC, especially PLE request2. Do not apply the control to request3 decode or continuation0 whole output.
2. **Split Z/A/B weight precision from backend/fidelity.** The batch BF16/GDN-HiFi4 and BF16/native controls are now completed classified passes. Apply the resulting candidate to the complete disputed matrix, including synthetic changing decode and live-prefill continuation; use BFP8/GDN-only-HiFi4 if determining whether BF16 is necessary would affect the policy or latency decision. Supplying a `dense_programs` dictionary replaces the default dictionary; removing only `gdn` retains all other roles, while `{}` disables every dense role and is not a one-role control. Restore the same full prefix state in each comparison. These controls establish fixture-level correction, not final representative or performance acceptance.
3. **Localize the first divergence if finer attribution remains necessary.** For synthetic GDN1 step1, batch request3 decode, batch request23 row2 and length97 row5, record attention input, packed Z/A/B, recurrence inputs/state, attention result, MLP input, router logits/paired top10, routed and shared outputs. Compare current fused BFP8 and the failure revision with exact input/state hashes. If router inputs already differ, an exact router implementation does not exonerate upstream projection. Compact versus expanded recurrence with the same operands tests I/O/layout; both share the SFPU compute kernel, so equality does not independently validate that arithmetic. Compare both raw output and final state; do not feed HF intermediates into an acceptance run.
4. **Discriminate tuple/trace dependence cheaply.** Replay the exact saved request3 fixture alone and in original/reversed tuple order, with independently recreated complete state. Run the same length97 input fresh without a live trace and with the live trace. Existing isolation checks weigh against interference but do not replace this exact contrast. Do not substitute a newly sampled batch1 input.
5. **Close OPT012's representative-evidence gap.** Use checkpoint weights plus actual target-model activations, preserving non-aligned lengths, three changing decodes after length1, heterogeneous request histories, and independent prefill while trace remains live. Record raw outputs, per-row and aggregate metrics, replay/state results and source/policy hashes. Keep the random fixture's numerical result visible as stress evidence. Compare final candidate and baseline latency with the same harness, and verify effective dtype/fidelity in profiler rows before selection.

The clean historical ranking (`rank_*` artifacts and work log) shows promising performance, but a component time or a different candidate's timing is not a final default acceptance measurement. No latency conclusion is made for unmeasured diagnostic controls.

## Candidate ledger and other potential issues

| Candidate | Supporting evidence | Missing proof / contradiction | Status |
|---|---|---|---|
| Z/A/B numeric perturbation, potentially amplified by routing | Attention-BF16 repairs continuation; BF16/GDN-HiFi4 and BF16/native repair disputed batch outputs with expert policy unchanged | Exact intermediate/route correspondence absent; individual necessity of BF16 versus HiFi4 not established | Supported numerical-policy localization; routing mechanism and op-defect claim remain unproven |
| Expert LoFi alone | Selected policy lowers expert fidelity | Expert HiFi4 still fails; fidelity is downstream of current router/state | Insufficient explanation |
| BFP8 experts alone | Default differs from old BF16 GDN baseline | Current fused BFP8 passes request3 and continuation | Insufficient explanation |
| L1 residual implementation | Optimized wrappers differ in storage | Same kernel arithmetic; DRAM residual control retains request3's exact metrics | Demoted for request3 |
| Inherited GR/PLE numerical controls | Exact historical request/row matches; same harness seeds | Current TT/control comparison still required | Strong explanation to test for request16 / PLE only |
| State aliasing, off-by-one conv history, compact head mismatch | Stateful workload warrants audit | Independent allocations, matching mappings/recurrence, passing isolation, T32 failure outside compact-MoE branch | No concrete defect found |
| Hardware-specific race or low-level math defect | Cannot be ruled out without runtime controls | No timing/nondeterminism signature or direct contract contradiction found; deterministic controls change only selected numeric paths | Unproven; do not headline |
| HF chunk recurrence vs TT scan; divide vs reciprocal query scaling; inherited PLE rounding differences | Real source-level arithmetic differences | Shared with predecessor/current fused passing paths; no new failing-row causal trace | Other numerical differences, not demonstrated regressions |

`_validate_batch` checks count consistency but not distinct state-object identity; caller-supplied aliasing could violate the intended ownership contract. The reported fixtures create independent states, so this is outside the causal headline and outside the requested repair scope.

## Evidence binding and claim review

Failure-revision SHA256 values checked:

* `tt/optimized_decoder.py`: `fb77bd2ca4ea336a18ebed57958babf678760472f4a791ae38c07ac735411057`
* `tt/fused_decoder.py`: `d638f10da2929e6898f63bac36151590c64eb5feb1ad184c3942053f59ae1b5b`
* `tt/functional_decoder.py`: `e775e5fd6a37ebaca8faadd9d4f5414b82e44e4b57e8265fae331edec94fb2d1`
* `tt/decoder_components.py`: `7f2b95f6998237e9c5208542bc2d2147bf0899b2438d175b1aa3ff2bbaf4ad15`
* `tests/probe_public_continuation.py`: `3b7bfda318aad3e16b71a9b78e14d1773b2924969a9942958a2bb4ff609162d0`
* `tests/test_optimized_decoder.py`: `2184bc1754e6d280154820903975939a8576a1c54fb1d7bec2d37d80658bad78`

These model sources match the named original final-run provenance and the completed controls above. `run_optimized_probe.py` changed after those runs (including an explicit fused diagnostic factory); the original selection records independently establish actual construction/dtype. During final review, the owner also changed optimized source to SHA `032d0b666e87ed94972b4c6a9d8d844995519c1a44407000d0c93f862290c6e9`: `attention_dtype=None` resolves to BF16 for GDN and BFP8 for QSA, and the default GDN dense role uses HiFi4. This is a subsequent owner edit, not the revision that produced the reported failures. The sufficient-control result does not prove the new comment's claim that both BF16 and HiFi4 are individually required. Final validation of that new default is outside this evidence cutoff. Newly created policy files without completed results remain proposals, not evidence.

After drafting, the headline claims were checked against precise JSON paths, inherited state flow, effective constructor overrides, projection dispatch and the lowered recurrence/matmul paths. Claims that would require missing intermediate correspondences were left as hypotheses. Historical implementation changes, pre-existing arithmetic differences and generic aliasing risks were kept out of the causal headline. This report does not establish final hardware correctness, representative precision acceptance, or a source fix.
