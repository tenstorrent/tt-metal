# Stage 3 independent review brief

Target: ornith-ai/Ornith-1.5-9B, revision
489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
Autoport: models/autoports/ornith_ai_ornith_1_5_9b.
Live worktree on main, starting HEAD 814ba2ac. Stage files are not committed until
review closes. Review the current files; do not use the starting commit as the
implementation under review.

## Original stage contract

Use optimize and tt-device-usage on the completed fused decoder in this checkout.
Work only on optimized_decoder.py and its tests/docs (including context_contract).
Do not begin multichip-decoder, full-model or vLLM work.

- optimized_decoder.py exists; delivered tests execute it, not a functional fallback.
- Preserve fused prefill/decode semantics, paged KV cache, determinism and every
  meaningful layer kind. PCC stays at the functional acceptance bar (0.995).
- Preserve context 262144; update the context contract for layout/sharding/cache
  changes. Reductions require a hard physical limit and largest-feasible evidence.
- Support valid non-aligned logical lengths; no seq_len % chunk_size public restriction.
- Report before/after warmed prefill and traced warmed decode latency. The final
  default must beat the best correct baseline/candidates, not merely use fewer ops.
- BFP8 weights remain the default. Measure real-weight BFP4/LoFi MLP candidates with
  profiler rows; stage 8 makes the final precision choice.
- Prefill/decode tt-perf-report tables and CSV exist. Try actionable advice and
  substantiate rejected advice; do not reject a material optimization on its first
  API/config error.
- Begin with an operation-topology audit: repeated same-input matmuls, layout/reshard
  costs, packed/fused projections, optimized composites and lower-movement replacements.
- Address model-applicable precision, sharding, projection packing, DRAM decode
  matmuls, large prefill configs, SDPA, memory/program/compute configs, movement and
  active-expert opportunities. Keep packing only if it beats a tuned legal separate
  candidate, or that candidate is invalid.
- No unnecessary torch/from_torch/to_torch, tilize/untilize, reshard or host fallback
  in measured runtime. Stress/repetition and watcher-clean correctness are required.
- README/work_log contain commands, PCC/perf, configs, advice, rejected options,
  limitations and exact artifacts.
- Assess optimize checklist until its stop rule: >=70% roofline and last two kept
  gains <3%, or 8 hours of device work. Untried items are hand-off notes; quality
  gates never weaken.
- Independent stage-review clean-pass, or pass-with-carried after three rounds.
  Fix/rereview findings. Commit stage-owned changes locally, never push, log SHAs.
- Unmet requirements/findings/failed gates remain work; stopping requires failed
  autofix or an unrecoverable external dependency under the selected skills.

## Sources and inspection scope

Installed skill root:
/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23.
Read skills/stage-review/SKILL.md and relevant optimize, tt-device-usage and
trace-lifetime requirements. The parent has applied those skills.

Primary evidence is this directory's README.md, work_log.md, final_audit.json,
final_accuracy.csv, final_performance.csv, perf_summary.json, *_final.json,
*_watcher.json, *_long.json, *_trajectory128.json, *_paired.json and
tracy_final/{linear,full}/{prefill,decode}_perf_report.{csv,txt}.
Compare implementation/source hashes, commands and raw/archived logs. Experiments
are inventoried in candidate_measurements.csv, matmul_geometry.csv, queue_results.jsonl
and the queue manifests. BFP4 profiles are under bfp4_profile. pre_advice contains
explicitly superseded validation, not the final signoff. context_contract.json is
in the parent doc directory. Source/tests are under ../../tt and ../../tests.

Treat owner prose as claims. Re-derive important facts from code and measured data.
The reviewer is read-only except for its requested report. No TTNN imports/device
open, device jobs, tests, servers, resets, or source edits. Small host-only artifact
analyses are allowed. Report precise required follow-up, not broad suggestions.
