# Datatype sweep work log

Stage 8, `ornith-ai/Ornith-1.5-9B`, started from optimized full-model commit `f8630a5a52013134e41b38a97352a7fed9d691c2` on local branch `main`. The contract is to select the fastest evaluated precision policy satisfying full-model accuracy and P1, preserve context and non-aligned prompts, ship the consumed default, and measure original token-out performance. No vLLM integration or remote push is part of this stage.

## Baseline and evaluation contract

The refreshed main AIME24 chat-template readiness reference has 161 input tokens and exactly 100 generated tokens. Baseline prefill top-1 is 99%; teacher-forced decode is 98%; top-5/top-100 are 100%. Initial readiness runs measure 101.13/100.75 traced tokens/s/user and setup-inclusive TTFT 6,620.12/6,527.13 ms. These are distinct from warmed selection measurements. See `baseline_provenance.json` and the three `baseline_*.log` files.

Acceptance requires top-1 >=90%, top-5 >=98%, and top-100 =100% in both phases on each reference. The second reference scores 512 actual contiguous Tale of Two Cities continuation tokens after a 128-token prefix at tokenizer offset 4,096, using the pinned original HF BF16 model. It is a numerical stress reference, not a chat-quality test. Exact checkpoint, tokenizer and input provenance are in `natural_text_provenance.json`.

All hardware jobs ran serially on chips 0–3, four Blackhole P300c devices, mesh 1x4/TP4, FABRIC_1D. Commands, timestamps, exit codes and bounded timeouts are in `commands.jsonl` and queue manifests. Initial health and device ownership checks found no conflicting users. No reset was required. Trace-sensitive runs use allocation tracking with program-cache allocations included; replay counters and program-count checks are in the individual artifacts.

## Candidate matrix and repairs

All 19 policies in `candidate_matrix.json` have complete full-model results on both references. `sweep_results.json` and `sweep_results.csv` retain exact policies, layer exceptions, compute fidelities, all phase scores, TTFT, traced teacher-forcing speed, command, hardware, mesh, regime and status. `README.md` contains the compact comparison table and group-level decisions.

Precision starts at BFP8. BFP4+LoFi was evaluated for every material group: attention input, attention output, combined attention, gate/up, down, combined MLP and LM head. Gate/up and combined MLP also have HiFi2 controls. Inner-layer BFP4 trials retain BFP8 in layers 0 and 31. Every BFP4 trial passes the short AIME reference but fails natural-text top-1, so none is eligible for selection. The fastest rejected BFP4 MLP/LoFi policy reaches 122.310 tokens/s/user but only 80.27% natural prefill and 81.05% natural decode top-1. Split attention and head trials also fail; their individual records prevent attributing a combined-group failure to an untested sibling.

BF16 KV, BF16 CCL, BFP8 decode activations, BF16 prefill activations, all-BF16/HiFi4 and canonical BFP8 projection fidelities were measured. All passing controls outside the P1 shortlist are slower than the selected policy. Their P1 status remains explicitly not run.

The BF16 controls exposed real runtime problems, repaired before their final measurements:

- A BF16 CCL same-dtype reshard incorrectly used a cast API; `to_memory_config` fixes the operation. Initial failure and passing focused/full-model controls remain on disk.
- AutoDebug source inspection identified retained BF16 inputs colliding with matmul circular buffers. AutoFix places affected large prefill tensors in DRAM from 4,096 physical rows; compressed policies keep the existing 8,192 boundary. An independent source audit caught non-aligned 4,065–4,095 inputs padding to 4,096; the decision now rounds physical rows.
- All-BF16 weights separately exceeded the special 512–1,024-row matmul CB budget with K16. The source ledger proves static end 1,488,832 versus allocation frontier 1,428,608. K8 is legal for all affected projection shapes and restores the budget; compressed weights keep K16.

`AUTODEBUG.md` and `AUTOFIX.md` record hypotheses, exact failures, source evidence and verified repairs. Both repaired BF16 profiles pass tracked, frozen-program-cache smoke runs at 511, 512, 513, 1,023, 1,024, 1,025, 4,095, 4,096, 4,097, 8,191, 8,192, 8,193, 16,385 and 32,769 tokens. Initial L1/K16 failures are preserved. The large launcher log is losslessly archived with hashes in `large_artifacts.json`.

The first short-prefix harness run omitted explicit sampling initialization and correctly hit the model guard. The repaired harness initializes sampling/RNG from authoritative histories on the first decode. The initial error is preserved; no model fix or weakened guard was needed.

## P1 and selection

The fitting BF16/HiFi4 reference generated four chat-template reasoning answers, seeds 1000–1003, temperature 1, top-p 0.95, top-k 20, presence penalty 1.5, unchanged cap 6,144. Cropping at first EOS gives 2,429, 1,699, 5,383 and 1,007 tokens, totaling 10,518. P1 uses the unchanged noise/KL/HF thresholds, minimum 1,000 positions, and four 256-position traced decode checks. The HF comparison requests 2,048 positions and scores the complete first 2,429-token answer.

All four BFP8 fidelity candidates pass P1. The selected `bfp8_all_lofi` has delta NLL +0.00174 nats/token, KL×1000 2.64908, and traced-decode top-1 difference -0.00098 over 1,024 positions. Reference-vs-HF KL×1000 is 0.54. `p1_final.json` and `../eval_readiness/precision_reasoning.json` contain the selected runner result; all four checks pass. There is no BFP4 required-to-fit limitation and no precision waiver. Rebuildable score tensors stay local; `p1_score_cache_manifest.json` records their hashes.

Three warmed baseline serial measurements are 113.574848, 113.622205 and 113.146600 tokens/s/user: median 113.574848, spread 0.475605. Differences within that spread are ties. Selected candidate teacher-forcing speed is 114.539849, above that spread; normal default reproduction is 114.414255. The selected candidate's warmed AIME TTFT is 106.419 ms. `selection.json` records the rule and comparisons.

Selected AIME prefill/decode top-1 is 97%, top-5/top-100 100%. Natural prefill top-1 is 462/512 (90.2344%); decode top-1 is 466/512 (91.0156%), top-5 511/512 (99.8047%), top-100 100%. Finite-sample 95% binomial half-widths are 3.34 points for AIME and 2.57/2.48 for natural prefill/decode. The near-threshold prefill result is a limitation, not a waiver or evidence of broad benchmark equivalence.

`selected_precision_config.json` is the exact winning policy and is loaded by the normal model builder with no override. It selects BFP8 weights/CCL/KV, LoFi for all five projection groups, BF16 residual/decode inputs/logits/sampling, phase-dependent prefill inputs, fixed norm/SDPA assumptions and no layer exceptions. Unsupported fixed-kernel values are rejected. `precision_propagation.json` ties every field to actual tensors, compute objects, consumers or validated fixed contracts. Both `selected_default.json` and `token_out_selected.json` use empty build kwargs.

The pyplot charts `top1_perf_pareto.png` and `top5_perf_pareto.png` plot all full-model configurations, their evaluated Pareto frontier, the selected red point and dotted minimum-accuracy lines. They use minimum teacher-forced accuracy across references and AIME traced throughput. Faster BFP4 points cannot win because they fail the simultaneous top-1 gate.

## Post-selection validation

The original optimized-full-model warmed token-out benchmark, through normal default construction, measures **116.141 tokens/s/user** resident no-readback and **88.885 ms TTFT**. The matched baseline is 114.810 tokens/s/user and 94.009 ms TTFT, a 1.16% decode improvement. Per-token-readback results are separately 115.756 and 114.491 tokens/s/user. See `token_out_selected.json` and `token_out_baseline.json`. Later serving comparisons should use the resident post-selection number, not teacher-forcing ranking speed.

`../context_contract.json` is recomputed for both KV dtypes and the BF16 reference. Four complete 32-layer capacity runs (baseline, BF16 KV, BF16 reference and selected normal default) pass 262,144-token prefill, non-aligned 262,143-token prefill and terminal decode position 262,143 with live traces and frozen program count. Advertised context remains 262,144. Baseline and selected prepared free DRAM are identical, 18.826613 GiB/device. Batch 32 was checked at context 512, not at maximum context.

Short-prefix controls use a 32-token prefix plus 256 real continuation tokens. BFP8 KV scores 237/256 top-1 at B1 and 236/256 for every B32 user. BF16 KV scores 238/256 at both batches. All top-5/top-100 scores are 100%, with same-input agreement across all 32 users. The selected resumed-prefill check passes the original 513-token workload at unchanged thresholds, including both negative controls (`prefill_resume_check.json`).

The six shared qualitative prompts use the pinned chat template and identical HF control settings. The selected French translation completes correctly; other outputs are coherent but mostly capped during reasoning. A thermodynamics overstatement triggered a fresh blind judge on ten matched seeds per backend. Scores were saved before unblinding: HF 0/10 and TT 1/10 material-error flags, below the fixed >20-point excess rule. All twenty hit the token cap; ten HF and nine TT are substantively incomplete. This small set does not prove equivalence. Historical full-model controls already document broad scientific overstatements in HF; the fresh judge's counts are not compared numerically with the old judge's counts. See `qualitative_verdict.json`, `qualitative_entropy_control/{rubric,blind_scores,unblinded_results}.json` and `blind_review.md`. Mechanical degeneration checks pass all 32 reviewed outputs.

The first readiness gate exposed a preexisting limitations-file schema mismatch (`limitations` versus required `items`). The original failure is preserved in `stage_gate_initial_schema_failure.log`. Migrated metadata retains historical evidence and adds the current capped-output comparison without waiving P1. Final gate output is `stage_gate.log`.

## Final checks and review

Host planner/cache tests: 22 pass (`host_tests.log`). Repository hooks pass (`final_hooks.log`); the source/hash/selection/context audit passes (`artifact_audit.json`); the actual `08-datatype-sweep.check.sh` passes (`stage_gate.log`). All 68 raw logs have lossless tracked gzip archives with SHA-256 hashes in `log_archives.json`. Independent fresh xhigh review returned **clean-pass** in round 1, with no required work (`stage_review_round1.md`). It independently checked all candidate policies, current-source P1 caches, traced counters, outputs, capacity, default reproduction and the original token-out benchmark. Residual risks remain the explicitly documented finite samples, capped outputs and future serving validation. Local checkpoint SHA is recorded below after commit. No hardware process remains active at handoff; no vLLM work or remote push was performed.

## Local checkpoint

Stage implementation, measurements, selected default, plots and clean-pass review are committed as `163c063da2d8f266fb3200c722829438e0997a91` (`Select validated Ornith BFP8 LoFi precision policy`). Commit hooks passed. This follow-up changes only the checkpoint log; its SHA is reported in the final handoff. No push was performed.
