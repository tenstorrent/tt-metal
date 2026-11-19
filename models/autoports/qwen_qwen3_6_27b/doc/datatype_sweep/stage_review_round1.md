# Stage Review

Verdict: clean-pass

Stage 8, datatype-sweep, `Qwen/Qwen3.6-27B`, reviewed independently on the live `main` worktree based on `5fee59eda8d6f213bd3e9a65323a69bed8b8369b`. The selected policy is `gdn_attn8_head8_mlp4_lofi`. No correctness, accuracy, selected-policy consumption, or required measurement gap remains. This verdict covers the implementation and completed evidence; updating the final status and recording local commit receipts are the stage owner's subsequent bookkeeping. The run plan prohibits pushing.

The user's stage-cap instruction closes optional search. Untried policies and slower candidates without P1 are handoff items, not requests for new experiments. No vLLM integration is authorized or claimed.

## Required Work

- None. All required evidence is present and consistent with the selected runtime source. No device rerun is requested.

## Other Concerns

- Documentation corrections were completed during review: `stage_artifacts_check.json` has **149 required checks passing and seven optional entries not evaluated**, for 156 inventory entries total. README/final work-log wording now uses that distinction, and the policy-consumption paragraph cites the completed normal-default benchmark and propagation evidence. I verified these corrections; they changed no runtime or measurement and need no hardware rerun.
- Preserve `work_log.md`'s `Carried to next stage` notes: isolated GDN-output variants, head HiFi4, broad-attention/head and additional layer/activation/CCL combinations, the combined down-only HiFi2 variant, and qualification of unselected BFP4 KV remain untried. The slower isolated `down4_lofi` P1 measurement remains unscored after its original runtime failure. None can conceal a faster measured policy: the selected policy has the highest measured throughput of all 25 rows.
- The P1 cleanup bridge is stage-local. Future callers that repeatedly construct generators must honor `close()` and synchronization; the unmodified upstream runner's optional `teardown()` hook alone does not do so for this generator. The successful repair does not establish recovery from an allocation failure in the middle of trace capture. The adapter aborts rather than continuing with residual allocations.

## Hard-Check Gaps

- The artifact inventory checks evidence structure, not prose counts, plot appearance, or text meaning. I separately inspected both PNGs and all 24 shared-suite TT/HF output texts, and independently recomputed the important numerical claims.
- Runtime receipts observe actual tensor metadata and configured fidelities; they are not new per-op profiler traces. Code inspection connects those observations to weight loading, per-group matmul configurations, paired SwiGLU, head matmul, cache allocation, residual boundaries, and sampler inputs. No contradiction between the measured policy and executed source was found.
- A vLLM adapter does not exist at this stage. `Model` and `build_generator` demonstrably load the required selected artifact by default. Adapter consumption and serving behavior remain stage 9 work, consistent with the explicit prohibition on beginning integration here.

## Evidence Re-derived

- All **25** completed full-model rows have both AIME24 P161/G100 and natural-text P128/G512 references, all 64 layers, and a 262144-token allocation. I checked reported accuracies against integer hit counts and denominators, per-reference prefill/TF classification, zero new captures during each TF window, exact decode replay counts, and stable trace IDs. JSON and CSV numerical rows agree.
- Summing the winner's 1220 timed decode tokens and dividing by the summed four TF elapsed times gives **37.645631771009946 tokens/s/user**. The four serial baseline repeat rates independently give median **30.88776440538331** and spread **0.03898973165338404**. The next fastest evaluated point is **36.421227901194776**, a gap of **1.224403869815170**, well outside that spread. Both fastest points pass final P1. The winner is faster than every evaluated point, including those without P1.
- Selected AIME TF/prefill accuracy is **98/100/100%**. Natural TF is **470/512, 506/512, 512/512** (**91.796875/98.828125/100%**); natural prefill is **471/512, 504/512, 512/512**. All meet the unchanged **90/98/100%** point-estimate thresholds on each reference. The JSON includes Wilson intervals and does not multiply accuracy sample size by timing repeats.
- Both selected BFP4 MLP groups have LoFi and HiFi2 measurements in the exact combined policy. Broad-MLP LoFi is **37.6456317710**, versus HiFi2 **35.9215634936**; combined gate/up LoFi is **36.4212279012**, versus HiFi2 **35.2031845989**. BFP8 attention/head/MLP fidelity comparisons and activation, CCL, KV and stronger-precision controls are represented in the matrix. No synthetic PCC rejection drives the selected policy.
- Independently loading the CPU score tensors and recomputing paired P1 statistics gives selected **dNLL 0.0024806536492728046**, block-cluster **SE 0.000561857302538242**, **KL x1000 3.851379902541775**, and **0.9879236297146745** argmax agreement over **23517** positions. The bound is `max(2*SE, 0.004) = 0.004`, with KL limit 5. Selected traced-decode top-1 is **0.935546875** over 1024 positions. The runner's HF control passes over 6144 positions, and all four final checks pass without a precision waiver.
- All three relevant P1 score-cache code/config hashes match current `tt/**/*.py` and the exact referenced policies: strongest reference `ad0bfd54d23e61e4`, winner `af5cdc9befae6cfb`, gate/up contender `8a10fcc62d870f53`. Each shares text hash `00aad6a2d8bd6fd5` and lengths `[6144, 6144, 6144, 5085]`. The final archived P1 JSON equals `doc/eval_readiness/precision_reasoning.json` exactly and names the shipped selection.
- The winner's explicit measurement and normal-default token-out artifact contain exactly the selected policy. `MODEL_PRECISION` records from resumed-prefill and qualitative default construction also match exactly across all 64 layers. The ten source hashes in every final selected gate and P1 receipt match the reviewed source. Only the two explicitly historical BF16 throughput controls have older source maps; neither is current-source selection evidence.
- Independent medians of the three 127-step resident windows reproduce **37.81770435692664 token-out tokens/s/user**, **26.442641535348546 ms/token**, and **75.38076722994447 ms TTFT**. Each window has 127 decode and sample replays, no captures/readbacks/input refreshes, and one end-of-window synchronization. Before/after trace IDs, capture counts, program cache and persistent addresses agree. This is the same B1/P128/G128 benchmark harness as stage 7; its result is properly separated from selection TF throughput.
- The selected cache remains BFP8 with the existing layout and full **262144** context. Capacity arithmetic agrees with actual cache shapes and allocator snapshots: 8192 physical pages per full-attention layer/rank, **19440537088** prepared free bytes/chip, a **4433600512-byte** increase over the refreshed baseline. The selected **32769-token** execution smoke and non-aligned lengths 33,127,129,511,513,1025 pass. A new max-context run is not triggered by this unchanged KV policy. Unselected changed-KV policies are not claimed to have passed their additional qualification.
- Resumed-prefill passes all four default-threshold cases, including start 68 and the last-token case. Both required negative controls fail as expected on the long windows. Trace-allocation tracking is enabled in `selected_trace_tracking.command.json`, program-cache allocations are included, and the guarded runtime audit passes all four trace kinds and 18 resident calls.
- Shared-suite metadata, rendered prompts, prompt token IDs, seeds, sampling settings and 128-token limits match the pinned HF control. All six prompts have greedy and sampled output on each side. The texts are relevant thinking continuations, including the French prompt's translation planning and the Fibonacci prompt's algorithm planning. No unexplained corruption or cross-request contamination is visible. This is an observational smoke, not final-answer accuracy or statistical equivalence.
- The final stage runner exits zero. A fresh read-only call to the inventory's `inspect()` function confirms **149 required passes**, with only seven expressly optional entries not evaluated. Final gate receipts are bounded, serial and exit zero. No device operation was performed during this review.

## Anomaly Ledger

- Observed anomaly: Primary P1 exhausted TRACE allocation on the fifth candidate, leaving `down4_lofi` unscored.
  Evidence: `p1_primary.log`, `p1_primary_precision_reasoning.json`, `AUTODEBUG.md`, `AUTOFIX.md`, `p1_runner.py`, `p1_lifecycle.json`, `p1_final_contenders.log`.
  Affected path: Multiple generator builds on one P1 mesh.
  Control or comparison: The isolated down4 full-model run captured successfully; repaired same-mesh lifecycle executes five builds, each returning TRACE usage from 1130496 bytes to zero. Both final full-model contenders release 22937600 bytes to zero with empty ID inventories.
  Likely subsystem: Cleanup hook mismatch, not measured numerical loss.
  Investigation performed: Inspected diagnosis, cleanup implementation and raw lifecycle/final logs. The repair synchronizes, closes, validates empty allocation and aborts on failure. Driver rejects candidate runtime errors.
  Resolution: Fixed lifecycle; the slower candidate's missing numerical score is explicitly controlled as an unmeasured handoff under the search cap.

- Observed anomaly: Original broad-MLP BFP4 policies miss one natural top-100 target; the combined attention/head/MLP policy passes it.
  Evidence: `full_mlp4_lofi.json`, `full_mlp4_hifi2.json`, `full_gdn_attn8_head8_mlp4_lofi.json`, final P1 score tensors.
  Affected path: Full-model numerical precision.
  Control or comparison: The original policies each score 511/512 top-100; the exact combined winner scores 512/512 and independently passes P1.
  Likely subsystem: Different complete-model arithmetic changes logits near ranking boundaries.
  Investigation performed: Recomputed fractions and selected P1 statistics; confirmed evidence is attached to each exact policy and current source.
  Resolution: Controlled; original policies remain accuracy rejects, with no transferred pass claim.

- Observed anomaly: Generic active-trace allocation warnings appear in ordinary benchmark/P1/qualitative logs.
  Evidence: `post_selection_token_out.log:109`, `selected_trace_tracking.command.json`, `selected_trace_tracking.log:113`, unchanged trace preparation code.
  Affected path: Eager helper work with resident traces.
  Control or comparison: Dedicated selected-policy run enables allocation tracking without excluding program cache, exercises repeated replay, eager prefill siblings and changed pages/tokens, and finishes with stable trace/cache/address inventories.
  Likely subsystem: Ordinary tracker-disabled warning, rather than an evidenced surviving unsafe allocation.
  Investigation performed: Inspected tracking environment, actual harness guards, replay/capture counters and command completion.
  Resolution: Controlled by the dedicated passing audit; no new broad allocation exemption was added.

- Observed anomaly: Story outputs open with “thinking thinking”; all suite outputs reach the common cap during thinking.
  Evidence: `qualitative_tt.json`, `../full_model/qualitative/hf_suite.json`, `qualitative_verdict.json`, `../eval_readiness/limitations.json`.
  Affected path: Prompt-correct qualitative generation.
  Control or comparison: Exact story opening appears in both greedy and sampled HF controls; all 24 texts share the 128-token limit.
  Likely subsystem: Checkpoint behavior and evaluation token budget.
  Investigation performed: Read all texts and checked exact matched metadata. No population claim was inferred from the single sampled seed.
  Resolution: Controlled; `original_model: "same"` is recorded. Task completion beyond the cap remains unmeasured.

- Observed anomaly: Historical BF16 rows and the first runner-gate failure could be mistaken for current acceptance failures.
  Evidence: Historical command source maps, current P1 cache identity, `stage_gate_initial.*`, `stage_gate.*`, limitations schema.
  Affected path: Evidence provenance and runner metadata.
  Control or comparison: Current selected receipts match all ten reviewed source hashes; current strongest-reference P1 scores match the full code/config hash. The gate passes after fixing the inherited limitations list schema.
  Likely subsystem: Earlier source revision and invalid metadata shape.
  Investigation performed: Recomputed source/cache identities and inspected both gate outcomes; no waiver or numerical threshold change was introduced.
  Resolution: Controlled historical rows; fixed gate metadata.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/qwen36-27b/multigoal_logs/08-08-datatype-sweep.prompt.txt`; installed version 0.1.24 `stage-review`, `datatype-sweep` and `references/precision-reasoning.md`, `tt-device-usage`, `qualitative-check`, and model-bringup startup instructions; user's stage-cap followup supplied in the review brief.
- Artifact paths: Stage README/work log, JSON/CSV results, all 25 raw numerical receipts and source maps, baseline repeats, selected policy/propagation, final P1 and CPU score tensors, lifecycle/debug/fix records, selected token-out/tracking/resume/qualitative artifacts and commands/logs, context contract, limitations, raw readiness reference shapes/provenance, shared HF suite, both Pareto PNGs, stage gate, artifact inventory and hook log.
- Code paths: `tt/precision.py`, `model.py`, `generator.py`, `multichip_decoder.py`, `optimized_decoder.py`, `flat_prefill.py`; stage diffs in context/evidence/benchmark tests; `optimized_runtime_audit.py`; sweep evaluator, report, propagation/capacity/inventory scripts, P1 adapter/driver and relevant installed P1 scoring/cache code.
- Commands run: Read-only `git status`, `git diff HEAD`, `rg`, `sed`, JSON/CSV analysis with Python standard library, local image viewing, CPU-only `torch.load(..., map_location="cpu", weights_only=True)` and paired-statistic/cache-hash calculations, and inventory `inspect()` with bytecode writes disabled. No TTNN import, device query/open/reset, server, long test, implementation edit, or push was performed. This report is the only reviewer-authored file.

## Residual Risk

- The selected policy's natural-text top-1 is lower than baseline and only about 1.8 percentage points above the required point threshold. P1 passes with four reasoning texts; it does not guarantee later benchmark or long-answer evaluation scores.
- The 32769-token smoke is the current selected-policy execution evidence; full advertised allocation is demonstrated separately. The policy preserves the existing maximum-context cache contract rather than claiming a new full-length selected execution.
- The qualitative suite establishes coherent matched thinking prefixes. Longer final answers, more seeds, served batching and serving capability gates are not stage 8 evidence.
- Prepared free DRAM is an allocator snapshot, not a transient peak or a multi-user capacity measurement. The selected policy is the fastest evaluated eligible policy, not a proof of a global optimum.
