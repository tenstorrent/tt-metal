# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None that blocks this stage. The final Tracy report still marks the five projection matmuls `SLOW` relative to its ideal model, but the stage attacked them with BFP4/LoFi, DRAM-sharded weights, advisor 1-D alternatives, packed-versus-split MLP, role-specific core grids, and legal block-width sweeps. The selected final default is the fastest correct traced-decode candidate reproduced under final source.

## Hard-Check Gaps

- The preserved Watcher JSON records the clean real-weight check and its then-current Watcher-log hash; the live `generated/watcher/watcher.log` was subsequently updated by the remainder of the same five-test Watcher suite, so its current hash differs. The current log also has no `error|assert|hang|stuck|timeout` match, and the work log records the complete command and five-pass result. This is an artifact-retention detail, not evidence of a failed or stale optimized path.
- The 8192 context value is a tested supported floor rather than an exact adjacent maximum. This is explicitly disclosed; 16384 has a final-source hard DRAM OOM, and the HF-advertised 40960 workload is independently impossible on one device from the byte calculation. No capability was reduced by this stage: the prior 4096 floor increased to 8192.

## Anomaly Ledger

- Observed anomaly: all-BFP4 failed the original seeded-random precision stress while passing real target-model boundaries.
  Evidence: historical synthetic PCC 0.975395/0.972469 versus final real PCC 0.999996978 prefill and 0.998990068 decode, plus final non-aligned length-31 decode PCC 0.998826-0.999354 and cache PCC at least 0.994387.
  Affected path: precision-policy selection.
  Control or comparison: conservative BFP8-attention/BFP4-MLP seeded-random diagnostic passes; two prompt-derived HF layer-32 boundary artifacts exercise final all-BFP4.
  Likely subsystem: the artificial random distribution is more precision-sensitive than the target-model activation distribution.
  Investigation performed: real length-17 prefill/decode, advancing traced positions 17-20, independently captured length-35 HF boundary used for non-aligned length 31 and positions 31-34, cache-consuming comparisons, and final timing.
  Resolution: controlled. OPT-012 is satisfied; synthetic evidence does not veto the faster real-weight winner.

- Observed anomaly: the first 16/20-core gate/up and down maxima exceeded L1.
  Evidence: gate16/block10 requests 2,093,312 bytes and gate20/block8 requests 1,739,520 bytes; down16/block50 requests 2,087,168 bytes and down20/block40 requests 1,700,608 bytes, all versus 1,572,864 available.
  Affected path: dominant BFP4/LoFi MLP geometry search.
  Control or comparison: phase-specific gate/up input shards make the 16/20/32/40/80-core K widths genuine; adapted gate16/block5, gate20/block4, gate32/block5, gate40/block4, gate80/block2 and down blocks 25/10/5 and 20/10/8/5 all execute.
  Likely subsystem: exact circular-buffer L1 capacity at the largest legal block widths.
  Investigation performed: full final-policy sweep in `all_bfp4_role_geometry_v2.json`, followed by a 500-replay tie-break in `all_bfp4_role_geometry_v2_precise.json`.
  Resolution: controlled. The exact blockers are recorded, legal adaptations were measured, and down-32 wins at 1.217996 ms.

- Observed anomaly: the complete shard-advisor layout family failed decode PCC while its matmul-only isolate passed but regressed latency.
  Evidence: full legal family decode PCC 0.985269 at 1.743 ms; advisor matmuls with production head layouts decode PCC 0.999272 at 1.737 ms; selected DRAM-sharded final is 0.998990 at 1.217 ms.
  Affected path: OPT-015 advisor seed.
  Control or comparison: advisor layouts were implemented as an executable family and then isolated from the production head layouts.
  Likely subsystem: advised head/norm/RoPE layout family changes numerical behavior; 1-D interleaved-weight matmuls lose to DRAM-sharded weights.
  Investigation performed: full legal family plus matmul-only isolate, with PCC and same-harness timing.
  Resolution: controlled. Advisor direction was applied where beneficial and the slower/failing recommendations were rejected with evidence.

## Scope Inspected

- Goal/skill paths: user optimized-decoder contract; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/shard-advise/SKILL.md`; `.agents/skills/shard-advise/SETUP.md`; `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/optimized_decoder/README.md`, `work_log.md`, `perf_report.md`, `results/final/*.json`, `results/final/capacity/*.json`, `results/candidates/all_bfp4_role_geometry_v2*.json`, `results/candidates/advisor_full_vs_matmuls.json`, final Tracy CSV, activation artifacts, `doc/context_contract.json`, and `shard_advise/report.json` plus `final_ir.mlir`.
- Code paths: `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`, and `tests/test_context_capacity.py`.
- Commands run: `git status --short`; `git branch --show-current`; `git rev-parse HEAD`; `git diff --stat`; `sha256sum` over final source, contract, capacity test, advisor, and activation artifacts; `jq` queries over candidate/final/advisor JSON; read-only CSV analysis over the final Tracy signpost windows; source and documentation inspection with `sed`, `grep`, `find`, and `git diff`.
- Recomputed final source identities: `optimized_decoder.py` `b136bfe65d8453a991b8870c591c8601d39ccad391111629408d2e73b25197b4`; `test_optimized_decoder.py` `e249c009d8a8addc70da4c5bd7ff70cae14b55e7299f1b70a013bfff67cbe936`; `test_context_capacity.py` `3246ee7a1ccd4b4c4706354404ffa83beddb522625ceecf020f6817deffdbdbe`; `context_contract.json` `c003fad34c835d9a2ac33b96749d45fc7c62614296af6e9d9fae44e636c56b7c`. Final correctness, timing, profile, capacity, and v2 geometry artifacts match these identities.
- Advisor identities: `report.json` `af538b90b1c0da0fe0eb05851892208e5c3c888cb5109725521fd646398453f2`; `final_ir.mlir` `f01e034478e12a86960528aa8d2a956097e1bd9c9c640fb3495614704a37334a`. The report contains 26 total ops, 23 final choices, and a spill pass with one spill; the IR contains the dense attention plus separate gate/up/down matmuls and their authoritative layouts/program configs.
- Git/scope: branch `mvasiljevic/model/qwen-qwen3-32b`, starting HEAD `f42cc399282678717d9240251210e464a8fbefea`. Stage changes are confined to the optimized decoder, model-local optimized/capacity tests, context contract, and optimized-decoder docs/artifacts. The unrelated untracked `.agents/prompts/forge_goals/03-multichip-provenance-from-ir.txt` is visibly isolated and must remain excluded from the stage commit.

## Residual Risk

- Evidence is one real representative dense layer (layer 32) under the compiler-derived batch-32 single-device contract. All 64 Qwen3-32B decoder layers share that architecture, but full-stack accumulation, generation quality, multichip behavior, and serving are intentionally outside this stage.
- The selected down-32 advantage over down-40 is small (about 1 microsecond per replay), though it is directionally consistent in the final 500-replay tie-break and final default reproduction. Future runtime or firmware changes may warrant retuning without invalidating current correctness.
- The final signposted decode contains necessary composite/layout boundary operations. Their device cost is small and the final path wins whole-layer traced latency, but later TTNN composite support could remove more boundaries.
