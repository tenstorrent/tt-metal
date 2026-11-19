# Optimized full model work log

Started 2026-11-14T06:10:47.369960+00:00 from clean commit 7dde27883c7cf72ed1268a9ef5845b19134549c3.

Target: all64 real layers, Blackhole TP4 1x4, unchanged accepted decoder precision/layout and context262144. No vLLM integration or broad datatype search. Installed skills resolved from TT_MODEL_BRINGUP_ROOT; tt-autodebug enabled in Codex inventory. Device holders: none; bounded tt-smi lists all4 p300c chips. Stage6 baseline and prior rejection evidence remain immutable.

Initial recorded baseline: TTFT86.924ms, token-out32.292ms including per-step readback; layer bound31.055ms. Fresh same-harness baseline follows before source changes.

## Candidates before device validation

Read `tech_reports/LLMs/llms.md` sections3.3/4.5 and TraceCorrectness.md. CPU audit identified ignored `read_from_device=False`, per-output sync, full cache reset cost and eager prefill dispatch. Candidate1 honors borrowed persistent token/logit output; caller must consume before another replay. Candidate2 adds one prepared B1/P128/start0 prefill trace, using existing input/state/logit allocations and unchanged layer policy. Eager general lengths, resume and batches retain their contracts. Full reset remains unchanged pending poisoning/masking evidence; prior same-policy LM-head reader candidates remain rejected on measured total-terminal latency.

Baseline started before runtime edits; its imported modules and source identity describe stage6 code. Only the baseline job owns chips0..3 during CPU implementation. All device jobs are serial and bounded.

## Initial full-path measurements

Fresh stage6 baseline (`baseline.json`): P128/G128/C262144, TTFT87.031ms, readback-inclusive decode30.976t/s/u. First candidate (`performance_candidate.json`) passes all64 greedy/sampled eager-versus-trace and no-readback endpoint equivalence, stable4 trace IDs/program24456, changed pages/tokens, nonaligned33/127/129/511/513/1025 and32769 smoke. Candidate no-readback32.2575ms,31.0005t/s/u; TTFT86.9058ms. Both gains are small; reduced-layer TTFT improvement does not establish a material all64 gain. Final same-process eager/traced TTFT and sampling-parameter A/B measurements will decide the retained default.

New reduced profile (`profile.json`, `perf/`) is complete with no readbacks inside decode. Exact kernel work extrapolates to31.183–31.490ms including terminal work. Generic cross-drain op-to-op gap annotations are invalid and excluded, as in prior stages. L1 semaphore/active-trace allocation warnings are generic setup warnings; strict tracking acceptance remains required, not waived.

Final serialized acceptance started with `run_acceptance.py`; each bounded command has its own `.command.json`, source identities and log. No device reset has been needed.

## Selected path and optimization stop

Final resident `performance_final.json` passes: medianTTFT86.850814ms, decode32.258325ms/token (30.999750t/s/u), all64/P128/G128/C262144. Same-process P128 helper A/B is86.855779ms eager versus86.650346ms traced (0.23652% lower); sampling-param A/B is86.730679ms uncached versus86.746810ms cached (0.01860% higher, no measurable gain). The cache is retained for eliminating identical uploads, not a latency claim. The last two retained changes are below3%; Stop Rule2 ends optional tuning. This is not a claim of significant speedup or exhaustive search, and does not waive any correctness gate.

Fresh correct split greedy0.512781ms beats force-argmax2.725168ms,200iterations each. Both choose valid maxima, including ties; split's unchanged equal-value tie ordering need not be lowest-index HF ordering (`sampling_decision.md`). Full-stack strict tracking, reduced complete-shape trace contract, positive-control rejection and separate Watcher pass. No host transfer/sync is observed inside the resident traced path by the executed API sentinels. Fresh mixed-slot sampling and reduced B32 regressions pass; all64 B32 and final quality gates remain in the serial queue.

Nonaligned32769-token prefill plus resident decode passes at allocated262144 context; positions reach32770. Four traces and24456 programs remain unchanged. Sampled memory is19,071,786,496allocated and15,006,936,576free bytes/device, not a peak measurement. `doc/context_contract.json` retains262144 and the stage6 maximum test because cache dtype/layout is unchanged.

No AutoFix invocation was needed: no device gate has failed. A CPU audit found a diagnostic-only test preparing33 without the new128 bucket; the test now prepares both and a separate strict-tracking host-compatibility run will verify it.

## Not tried (stop rule)

Expected whole-model gains below are unmeasured unless stated. These are handoffs,
not failures or empirical rejections; preserve the existing safe defaults.

- Terminal distributed norm followed by ordinary AG, and distributed norm plus
  fused AG-matmul: adapt actual BF16/H5120/N62080 head packing and persistent
  buffers before comparing. Final norm currently costs about31–37us/token;
  its standalone removal alone has a small full-model ceiling.
- Hidden-sharded terminal head with vocab reduce-scatter, or full-vocab partial
  all-reduce: modeled communication is much larger than the selected hidden AG
  plus candidate exchange; neither family was measured or rejected here.
- Further unchanged-policy LM-head geometry, larger legal K blocks, core/output
  subblocks and one/two/three-reader adaptations beyond the stage6 tested
  candidates. Stage6's specific slower/admission-failing candidates remain
  rejected; the entire geometry family does not.
- Reducing full-cache reset work, avoiding duplicate fresh GDN clearing, and
  direct state import/publication changes: preserve explicit reset semantics,
  stale-page safety and rounded attention windows before benchmarking.
- Additional traced prefill buckets, trace reuse across chunk starts, multi-row
  tracing, larger internal chunks and alternate long-prefill kernel geometries.
  Logical lengths remain unrestricted on the current eager fallback.
- An all-active B32 trace that omits inactive GDN state preservation, and native
  masked-state updates for mixed slots. Retain the tested inactive-row behavior.
- Additional terminal/sample buffer reuse to remove clone/layout staging while
  preserving common sampler ownership; no custom sampler or full-vocab gather.
- Earlier decoder handoffs remain untried: ephemeral/DiT fused-norm scratch for
  64 owners; phase-specific GDN HiFi4; lower-movement prefill norm; prediction,
  column and live-row recurrence combinations; BF16-DRAM joint combine/conv;
  long-prefill input-L1, row-cap, concat/window and geometry crosses under selected
  fidelity; QKV batched/fragment movement; SwiGLU bulk/tail configs; attention-gate
  whole-layer combinations; broader persistent-buffer signature capacity. Exact
  predecessor evidence/manifests are in
  `../optimized_multichip_decoder/final29_handoff.md` and `final29_candidate_decisions.md`.
- Broad precision/fidelity/KV/CCL Pareto exploration and BFP4/P1 selection belong
  to the explicitly deferred datatype-sweep stage, not a failed stage7 search.

Complete-model gate work is **not** a stop-rule handoff: finish refreshed AIME24,
qualitative/autoregressive, all64 mixed/B32/resume, readback metric and summary,
then independent review and local checkpoint commits.

## Refreshed all64 fixed slots

`slots32.json` passes after complete B1 and B32 lengths1..512 preparation. Identical33-token prompts produce equal finite logits in all32 rows;128 resident decode steps preserve one inactive row, then remap/release pass.31 active rows deliver16.237918t/s/user (503.375470 aggregate synthetic-regression tokens/s), zero per-token reads. This is a fixed-slot functional/performance regression at context1024, not a serving or natural-answer benchmark and not a replacement for the stage6 full-context bounded-pool capacity proof.

## Standalone-loader import repair

The first AIME24 prefill command exited1 before evaluation: readiness loads generator.py through a standalone module spec, and the newly added relative helper import lacked a parent package. Preserved failure log/command/source are in `import_failure/`. Changed exactly one import statement to the existing absolute autoport convention. `import_resolution_proof.json` verifies the exact text delta, standalone loader module and identical helper class. Hooks pass. This is a direct, understood import repair; no device graph, parameter, policy or math changes. No reset or AutoFix escalation was needed. The serial queue resumed from prefill (`acceptance_resume.log`) and now includes the host-compatibility, all64 mixed and stage-gate jobs after quality runs.

Already completed package-import probes retain their original hashes and remain valid through the explicit import-only equivalence proof; the collector re-verifies the exact source replacement and permits only the enumerated package-mode receipts. It does not rewrite logs, silently update their identities or exempt readiness results. All readiness jobs must run on the repaired exact source.

The repaired standalone prefill run passes top1/top5/top100=100/100 each (`prefill.log`, exact-current-source `prefill.command.json`). Thus the import failure is closed by full-model hardware execution, not only the CPU import proof. Teacher forcing and remaining quality/contract checks continue.

## Refreshed AIME24 numerical gates

Final absolute-import source passes chat-template100-reference-position prefill top1/top5/top100=100/100 each and traced teacher forcing99/100,100/100,100/100. These equal stage6 numerical results and satisfy top5>=98%, top100=100%. Teacher-forcing decode31.04t/s/u is separate from the31.000 resident autoregressive measurement; its cold TTFT93.504s includes first preparation and is not a warmed latency. Raw logs/commands: `prefill.{log,command.json}`, `teacher_forcing.{log,command.json}`.

## Refreshed resumed prefill

`prefill_resume.json` passes starts32/68/2528/2559 on natural2560 tokens. Minimum case-mean PCC0.999431966, minimum individual PCC0.985754633 against runner floor0.98, minimum top1 agreement0.992217898, and every last token matches. No-history and fresh-suffix controls fail at both eligible257-position windows. These values reproduce the accepted stage6 receipt exactly; existing recurrent/KV continuation is preserved. KV-only automatic prefix caching remains excluded for recurrent_state; this is not a vLLM prefix-cache certification.

## Refreshed free-running readiness

The installed autoregressive runner passes128 tokens. Fresh TT is token-exact with matched cached HF and stage6 TT. Both actual completions were read; they coherently explain supervised/unsupervised learning and begin the same fruit analogy. Leading closing-think token is identical on both sides. The CLI applies its existing trailing-whitespace strip to both24-token inputs; the shared suite separately tests the unstripped default chat template. `autoregressive_verdict.json` and `autoregressive/` preserve text, tokens and scope.

## Refreshed shared qualitative suite

Read all12 fresh TT completions and all12 matched HF controls for the six canonical prompts, greedy and sampled, seed0,128 tokens. Every TT token history exactly equals stage6 TT, and metadata/rendered prompts/prompt IDs/settings match HF. Outputs are coherent and on topic; they mostly remain in default thinking at this cap. Story prompt2 has the same initial thinking thinking phrase on both HF and TT, without sustained collapse. `qualitative_verdict.json` records per-prompt observations/snippets, control reuse and limits. This is observational regression evidence, not a10-seed statistical quality claim or final-answer-success score.

## Hard gates complete

All20 bounded jobs and `collect_evidence.py` pass; see `hard_gates.json`, `accuracy.json`, `perf_summary.json`, `collect_evidence.log` and per-job exact commands. Final all64 mixed33/128/513 orderings and preemption rebuild pass on exact repaired source; host compatibility returns to device sampling with stable traces under tracking. Installed stage07 check exits0, no degeneracy. Final tt-smi health exits0 and fuser reports no holders; no reset/recovery was used. Only independent stage review and local checkpoint commits remain. The historical pending statements above describe progress at their recorded point, not unresolved gates.

## Independent review closure

Fresh xhigh stage-review round1 returned **clean-pass**; see `stage_review_round1.md`. No Required Work remains. Registered the narrowly observed matched HF/TT story wording in `../eval_readiness/limitations.json` with `original_model: "same"` and exact evidence paths. Compact required CSVs are explicitly staged despite repository ignore rules; bulky raw Tracy capture remains local at the documented path. These are documentation/checkpoint housekeeping, with no runtime change or additional hardware run. Local checkpoint SHAs are recorded in `commit_receipt.json`. No push or vLLM integration was performed.

Checkpoint hooks normalized generated report whitespace and CSV line endings only. Compact CSV values and runtime sources are unchanged; raw Tracy captures are excluded locally from staging.

## Local checkpoint

Implementation/evidence commit: `e84009a83ff67e3345e43666d52e6395611df573` on `main`, based on `7dde27883c7cf72ed1268a9ef5845b19134549c3`. All repository hooks pass; runtime hashes match accepted evidence. All12 compact CSVs are included with unchanged parsed rows. The worktree was clean after this checkpoint. This receipt and log entry are committed separately; no push was performed. Stage7 optimized-full-model is complete with independent clean-pass.
