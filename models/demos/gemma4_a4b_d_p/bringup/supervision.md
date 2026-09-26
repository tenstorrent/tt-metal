# Gemma-4 26B-A4B bring-up: supervision log

Every intervention by the supervisor (the /bringup session) in run1: trigger, classification, action, result.
Framework fixes are the `[bringup][F<n>]` commits; the owner's decisions are marked "owner".

| When (2026-09-25/26) | Task | Trigger | Class | Action |
|---|---|---|---|---|
| 21:40 | R.2 | path check charged the supervisor's own commit and a dashboard re-export to the agent | framework | stopped before attempt 2, fixed (F11) |
| 21:45 | R.3 | chunked != one-shot (PCC 0.99983) | model code | none; the reference agent fixed it (MKL M-dependent sgemm) |
| 21:50 | R.3 | a CPU-reference failure would have escalated to ttnn-expert-debugger | framework (owner caught it) | per-role policy (F12) |
| 22:00 | G.s4096 | text top-1 16-19% on raw book text (ERNIE 69-87%) | input | stopped before the 55k golden; owner chose chat wrapping (F13) |
| 22:10 | R.2 | user-turn wrapping still 16%; model-turn 67% | input | owner: one deterministic input; canonical prompt, model_turn (F14) |
| 22:50 | G.s16384 | owner asked whether 43% is a red flag | input | checked HF = golden 100% past the window; parity-length rule (F15) |
| 22:55 | B.1 | box test used a nonexistent MeshDevice API; device policy too strict | framework | fixed (F16) |
| 23:05 | PL.0 | waited for a plan approval before any plan existed | framework | approval scope (F17) |
| 23:11 | PL.1 | plan ready | approval point | owner approved; experts: hack unified_routed_expert_moe for gelu_tanh (owner) |
| 23:15 | C.sliding.attn_norm | file edit mentioning ttnn flagged as device access | framework | stopped the needless attempt 2, parse Python (F18) |
| 23:40 | C.sliding.attn_residual | CPU-only golden analysis flagged; then the supervisor edited during the agent step | framework + supervisor error | stopped attempt 3 within seconds; only device-opening code is flagged (F19) |
| 01:00 | C.sliding.experts | device open fails on all chips (active ethernet core timeout) twice | box | stopped the run; owner reset the board and set "always 2D fabric" (F20) |
| 01:50 | C.global.attn_norm | skill, dashboard styles, intake sanity gate, pause, infra stop | planned framework work | stopped the just-started test agent; F21 |
| 02:10 | S.global.04 | supervising session at 80% context | hand-off | paused before C.global.ffn_norm; hand-off below |
| ~02:35 | C.global.ffn_norm | new session took over from the hand-off | resume | resumed run1; ffn_norm, S.global.05, mlp, S.global.06, post_mlp_norm passed |
| 02:50 | C.global.post_mlp_norm | owner: teletext should alternate Index / Model graph every minute | framework (planned) | paused after post_mlp_norm; F24 carousel, selftest 115 passed |
| 02:57 | S.global.07 | owner: carousel every 20 s, not 1 min | framework (planned) | paused after S.global.07; F24 interval 20 s, selftest 115 passed |
| 04:04 | L.s56320 | gate FAIL: pytest-timeout at 300 s after all checks passed; fix agent started | framework | killed the orchestrator and fix agent (no edits made); F25 test timeout from spec + timeout stops for a person; selftest 117 passed; resume |
| 04:20 | K.1 | L.s56320 passed (566b22d99d2); the contract agent started on shared adapter.py without the owner's OK | framework | killed the orchestrator and agent (no edits); F26 shared-code approval point; K.1 waits for `approve shared:K.1` |
| 04:29 | K.1 | owner: yes to K.1 on adapter.py, yes to the no-host-work rule and gate | approval + framework | `approve shared:K.1`; F27 agent rule 5 + host_transfers_per_layer (reported by X.1 for Gemma); selftest 120 passed; resume |
| 04:48 | K.1 | gate FAIL: shared producer sends Gemma to the MLA KV reader; attempt 2 could not touch it | framework | stopped the run; owner: drop the approval, let the agent change the engine; F28 full prefill in X.1, F29; resume |
| 04:55 | K.1 | gate PASS 4b0a1482594 (checks 0 failed, acks_early 0, KV PCC k 0.9945 v 0.9943) | review: accepted | producer change is a generic adapter hook (+docs); reader compares device KV to the golden files; runtime builds RoPE once at load; only upload is the engine's token ids. F30 skill: the overseer decides |
| 05:23 | X.1 | profile: no device durations; the ladder/profile model is the hybrid harness (~20 s/chunk of host glue) | framework + owner picks | killed the fix agent; F32 fast loop (one chunk on the golden prefix, no precompile), F33 run_block sections + standing assemble step; owner delegated picks: P.1 assemble all-device model, then X.1 baseline, then P.2 SDPA config A |
| 05:50 | P.1 | gate PASS de51b4c945b (0.743 s/chunk vs ~20 s hybrid, host transfers 0, accuracy unchanged); false device alarm started attempt 2 | review: accepted; framework | stopped attempt 2; F34 parse-based device check, no retry after a pass, profile timings on the dashboard |

## Hand-off (2026-09-26 02:10)

- **State:** 36 of 56 implement tasks passed (all 28 sliding-layer tasks; global layer 5 through S.global.04). Every
  task before implementation (R, G, B, PL) passed. Run `run1` is PAUSED before `C.global.ffn_norm`.
- **Resume:** `PYTHONPATH=$PWD python -u -m models.demos.common.bringup.orchestrator resume --spec models/demos/gemma4_a4b_d_p/bringup/spec.yaml >> /localdev/dnijemcevic/bringup/gemma4_a4b_d_p/runs/run1/orchestrator.log 2>&1`
  (background), then watch that log (filter: gate PASS|FAIL|HANG, attempts, STOPPED, WAITING, paused, problems, exit).
- **Next:** C/S.global.ffn_norm .. ffn_residual (10 tasks), then L.s4096, L.s16384, L.last, L.s56320 (ladder),
  K.1 (contract: registering the adapter touches shared `models/demos/common/prefill/adapter.py`, ask the owner
  first), X.1 (profile), X.2 (opportunity list, stops for the owner's picks).
- **Owner rules in force:** always FABRIC_2D; experts use unified_routed_expert_moe with the GeluTanh hack (committed
  ce75b1f0d7f); ask before shared-code changes; never tt-smi -r; short device checks; pause before any edit.
- **Open items:** the 28 sliding tasks passed on FABRIC_1D_RING before the 2D rule (the ladder re-exercises them on 2D);
  hf.parity_seq is 512 < window 1024 (covered by the hand check in BREADCRUMBS); dashboards: standard
  https://claude.ai/artifact/6PuG6EV3pyhchG1GTXBUuo and teletext https://claude.ai/artifact/C1yGG9j68BTTrcacwFc6nc
  (republish after every gate: `python -m models.demos.common.bringup.dashboard.export --spec <spec>`).
