# Independent stage review contract

Stage 4: multichip decoder, Qwen/Qwen3.6-35B-A3B. Review the live worktree
on branch `main`, starting commit `2901d8a506198dc69a9e99714db3a9c590b6ff95`.
Implementation scope is only `tt/multichip_decoder.py` plus tests and docs under
`models/autoports/qwen_qwen3_6_35b_a3b`. No full-model or vLLM work is authorized
in this stage. No push; passing stage requires local stage-owned commits.

The user's required contract:

- Start from the completed optimized decoder; make best practical use of the
  available mesh. Supporting smaller meshes is unnecessary.
- Choose mesh/hardware strategy before the final path. Document every tensor,
  activation, KV-cache, collective and expert strategy, calculated per-device
  shapes/shards/padding, and rejected alternatives.
- Recompute context capability and memory. Preserve advertised context unless
  a proven hard physical limit requires a reduction. Run32K–64K smoke; maximum
  only if KV dtype/layout changes.
- Validate whole-output prefill/decode PCC against the single-chip optimized
  TTNN baseline for every meaningful layer kind. Per-row PCC is diagnostic;
  an HF-reproduced miss is a controlled limitation, not an unexplained defect.
- Validate paging, page tables, current positions, local KV heads/cache layout,
  and stacked-decoder input/output contracts. Preserve nonaligned logical
  lengths through internal sharding, padding and chunking.
- Warmed decode trace replay must work on the target mesh. Report warmed
  single-chip versus mesh latency, speedup and efficiency.
- Produce tt-perf-report human tables, CSV and provenance; inspect communication,
  DRAM, compute, CCL and data movement before accepting the path.
- Runtime fallback audit, appropriate determinism/stress and watcher-clean
  evidence are required.
- Follow multichip's MoE EP/TP/hybrid and batched-decode rule. Execute gate-selected
  active experts; dense all-expert execution requires proof it wins.
- README and work log record exact commands, PCC, performance, plan, limitations
  and artifacts. The real implementation must fit full-model weight/KV
  expectations and serve as the next stage's layer-stack baseline.
- Independent stage-review must return clean-pass, or pass-with-carried only
  after three rounds. Fix findings and re-review. Local commits and SHAs required.
- Unmet requirements and failed gates are work; stop only after autofix fails.

Skill root:
`/home/ttuser/qwen36-35b-a3b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills`
Read `stage-review/SKILL.md`, `multichip/SKILL.md`, `tt-device-usage/SKILL.md`,
and applicable `optimize/SKILL.md` requirements.

Start with README, work_log, gate_summary, artifact_manifest, mesh_plan, capacity,
latency_comparison, profile findings/summaries, implementation and test runners.
Treat prose as claims; independently inspect actual JSON/CSV/log/source evidence.
No generated-text evaluation is claimed in this decoder-only stage. Reviewer
may run read-only CPU analysis and write its review report, but must not open
devices, reset hardware, run servers or modify implementation/tests.
