# Independent functional-decoder stage review request

Use `$stage-review` as a fresh independent reviewer, round1 of at most3. This is
functional-decoder only for `Qwen/Qwen3.8-Flash-Next`, not an optimization, multichip,
full-model or serving stage. Do not spawn a reviewer. Read-only implementation access;
you may write the requested review report. Do not import/open TT devices, run hardware,
reset hardware, start servers, or run long experiments.

## Original user contract

Implement only `models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py` and
its model-local functional-decoder evidence. Document prefill/decode contracts for all
meaningful target layer kinds, with real target config shapes. Decode must run fully
with TTNN traced execution. Tests must cover paged prefill/decode, page tables/current
positions, longest feasible context and nonaligned chunk/page/tile boundaries, real
weights and HF-versus-TTNN PCC>=0.995 unless model-specific evidence justifies another
bar. Do not reduce advertised capability merely for convenience; any reduction requires
hard physical device evidence and the largest feasible value. Record context contract,
warmed prefill and traced warmed decode performance with human tt-perf-report tables,
CSV and provenance. Runtime passes must contain no Torch/from_torch/to_torch/host
fallback. Determinism and Watcher-clean evidence must exist. README/work_log must name
commands, PCC, performance, limitations and exact artifacts. Satisfy functional-decoder
skill final objectives. Required verdict is clean-pass, or pass-with-carried only after
round3; findings must be fixed and rereviewed. Local stage-owned commits after passing
review, never push. Unmet requirements are work; stop only after AutoFix fails.

## Active skill paths

- `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/stage-review/SKILL.md`
- `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/functional-decoder/SKILL.md`
- `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/tt-device-usage/SKILL.md`
- `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/tt-enable-tracing/SKILL.md`

## Workspace and inputs

Live worktree `/home/ttuser/qwen38-flash/tt-metal`, branch main, base commit
`d7e788cd88199501a8546ddb7bc381a601af47a9`. All stage changes are under the model autoport;
local commit is intentionally pending the independent verdict. No push is authorized.
This is the actual shared tree, not a copied snapshot. Stage measurements and required
pre-review gates will be closed before this request is dispatched.

Model root `/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next`.
Inspect first:
- `doc/functional_decoder/README.md`, `work_log.md`, `final_gate_bindings.json`, and the
  latest `pre_review_gate_inventory_v*.json` and `final_semantic_inspection_v*.json`.
- `doc/context_contract.json`, `doc/eval_readiness/limitations.json`.
- All26 runtime files under `tt/`, with the public FunctionalDecoder boundary.
- Real short/public continuation/batch records for representative layers0/1/3 and
  their `tests/probe_*` harnesses. Read actual numerical fields and failure records.
- `gdn_final_full_context.json`, `ple_final_full_context.json`,
  `qsa_ordered_device_full_boundary_v1.json`, `qsa_ordered_full_decoder_v1.json`.
- `ple_public_hf_4768_private_v1.json`, `ple_public_hf_4767_private_v1.json`,
  `final_capacity_qsa_closure_v4.json`, `final_batch_capacity_qsa_closure_v4.json`,
  `tests/capacity_contract_final_assessment.md`.
- `final_watcher_summary_format_v2.json`,
  `synthetic_final_decoders_v4_format_summary_v2.json`,
  `performance_evidence_offline_v3.json`, and six final
  `tracy/layer{0,1,3}_{prefill,decode}_device_trace_v3/` records.
- `chunk_input_final_v6.json`, `chunk_output_final_v5.json`, related logs/Watcher maps.
- `plans/format19_transition_v1.json` and exact source-equivalence validators/snapshots.
- `evidence_archives.json` and multipart manifests, packaging/view reports, original
  logs/provenance. Raw files are present here; archive restoration is documented.

Pinned HF snapshot:
`/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa`.
The installed Transformers qwen4_exp source and large external fixture paths are named
in the actual records. Do not infer that a raw runner exit0 is sufficient by itself.

## Review stance and output

Treat README/work-log and owner inspection as claims. Re-derive important claims from
source, runner artifacts, logs, JSON and actual values. Checks passing are a floor;
look for contradictions, stale evidence, incorrect cache/trace/position handling,
precision anomalies, capability reductions or unsupported performance numbers.
Apply the skill's anomaly ledger and review standards; do not invent a stronger
artifact format solely for its own sake. If hardware follow-up is necessary, identify
it as required work and do not run it. Write the required verdict/findings/anomaly
ledger/scope/residual-risk format to
`doc/functional_decoder/stage_review_round1.md`, then return the verdict and findings.
Only the fresh independent reviewer supplies the verdict. This request supplies no
owner theory of what should be accepted.
