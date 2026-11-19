# Independent review input — stage3 optimized-decoder, round1

Target: Qwen/Qwen3.6-27B at pinned revision8f33b98937c6031bb49e713dee3e8e185aafd26f.
Repository: /home/ttuser/qwen36-27b/tt-metal, branch main,
starting HEAD4c79322a4c142187308864977f00cbe5012c0352. Live shared worktree;
implementation and measured harnesses frozen. Source SHA is in delivery194_manifest.json.
Review is before local checkpoint commit; user says never push.

Original owner contract:

- Deliver models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py and tests/docs only;
  start from completed fused decoder, no multichip/full-model/vLLM work.
- Tests must select optimized runtime, preserve fused prefill/decode semantics,
  determinism, paged KV and representative layer kinds; PCC≥.995 unchanged.
- Preserve262144 context; update context_contract for capacity changes, no reduction
  except proven hard physical limit. Arbitrary non-aligned logical lengths valid.
- Report warmed prefill and traced warmed decode before/after; delivered code must
  beat best correct traced baselines/candidates, not merely reduce op count.
- Default BFP8 weights. Measure actual-weight BFP4/LoFi for dominant MLP matmuls;
  final full-model precision choice belongs to stage8.
- Start with topology audit, address applicable canonical precision/configs,
  sharded layouts, packed versus well-tuned legal separate projections, DRAM
  decode matmuls, large prefill configs, SDPA/composites, runtime movement and
  MoE active execution if applicable. Keep packing only if faster or separate invalid.
- Advice-enabled tt-perf-report prefill/decode tables/CSV; material rejected
  advice must have before/after evidence or exact contract blocker. First API
  errors do not suffice; repair/adapt and retry or use AutoFix.
- No unnecessary runtime torch/from_torch/to_torch, layout/reshard or host fallback.
- Stress/repeated replay and watcher-clean correctness. Trace allocation tracking
  includes program-cache allocations, positive controls and safe lifetime evidence.
- README/work_log record commands, PCC/perf, configs, conclusions, rejected
  options, limitations and exact artifacts.
- Assess optimize checklist until owner's stop rule:≥70% roofline AND last two
  kept gains<3%, OR8h actual device work. All-items-assessed shortcut excluded.
  Untried work goes in hand-off notes; quality gates stay unchanged.
- Fresh xhigh stage-review clean-pass (or pass-with-carried after3 rounds),
  findings fixed/rereviewed, local stage-owned commit, no push, log SHAs.
- Unmet requirements/findings are work. Stop only after AutoFix fails.

Skills:
/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/{optimize,tt-device-usage,tt-enable-tracing,stage-review}/SKILL.md
/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8/skills/{autodebug,autofix}/SKILL.md

Inspect first: README.md, work_log.md, checklist.md, delivery194_manifest.json,
final_evidence_check.json, final_measurements.md, final_profile_conclusions.md,
stop_rule_selection.json, capacity_delivery194.json, ../context_contract.json,
kernel_contracts.md, kernel_contracts_attention.md, runtime_audit.json.
Then raw final194_* result JSON/logs/exit ledgers and perf/final194_* reports,
implementation and delivered tests. Candidate/sequence summary CSVs index
experiments; candidates/source_archives preserve measured source.

Treat owner-written docs as claims. Re-derive important numbers and inspect
actual runner assertions, default wiring, contracts, source and evidence.
This is decoder scope: text activations are precision probes, no TT generation.
Return required Stage Review format with findings first, verdict, Required Work,
Other Concerns, Hard-Check Gaps, Anomaly Ledger, Scope Inspected and Residual Risk.
Do not spawn another reviewer, run TT devices, reset devices, start servers, or
modify implementation. Read-only commands/small artifact analysis are allowed.
Write the review report to stage_review_round1.md only.
