# Supervision: mimo_v2_6_d_p_2x2 run1

| time | task | trigger | classification | action | commit |
|---|---|---|---|---|---|
| 2026-09-28 04:27 | intake | owner approved spec (rule 5 added: bring-up ops awareness) | approval | `approve intake`, early ledger, init-run run1, launch | 3c14efebf9c |
| 2026-09-28 04:30 | - | owner: "you are working autonomously, I go to sleep ... Make 2x2 work at similar perf" | delegation | plan and perf approvals delegated to the overseer; board resets not delegated | - |
| 2026-09-28 04:30 | - | dashboards | - | standard https://claude.ai/artifact/3rDqBJAPGGyKxyuLThTm5d, teletext https://claude.ai/artifact/7NSv41J2ABUBuj8gW9fjRV | - |
| 2026-09-28 04:30-04:50 | R.1-R.3, G.*, B.1, PL.0 | gates | - | all passed on their first check, no agent; G.* golden_reused 1 (goldens shared with the 1x4 prior) | fe698820d1b .. 74cd83d8774 |
| 2026-09-28 04:40 | - | dashboards lacked a comparison with the prior | framework | agent in a separate worktree added the "vs prior" view (both styles) + selftest; cherry-picked while the run waited | see git log `[bringup] dashboard:` |
| 2026-09-28 04:55 | PL.1 | plan ready | approval point | owner reviewed (MoE: row halves, dispatch within column, all_reduce axis 1, all_gather axis 0; TP=4 attention with a 2-stage all_reduce; no SP) and approved: `approve plan` | - |

Note: the dashboard's prior chunk time is 225.5 ms (the prior's X.3 profile); the 1x4 run's final number after the perf work is 204.9 ms (ad-hoc profile /localdev/dnijemcevic/bringup/mimo_v2_6_d_p/profiles/adhoc_20260928T024054.json). Compare against 204.9 ms at the end.
| 2026-09-28 07:40-07:48 | C.* S.* M.1 L.* K.1 X.1 X.2 | gates | - | all passed on first attempt; no existing TTNN op touched, no fork changed; K.1 added one registry row to common/prefill/adapter.py (accepted) | d6248c46320 .. 5247a2e0c87 |
| 2026-09-28 07:55 | X.2 | perf picks | approval point | ad-hoc op profile: vs 1x4 final (204.9 ms) the 2x2 chunk (278.5 ms) loses +47 ms in fabric dispatch/combine and +28.5 ms in full SDPA (HiFi4 under rule 7 vs 1x4's HiFi2). Owner: "just do hifi2 and that's it" -> P.1 (preset A for full layers), gate reruns the full-layer attention component tests and full_dense/full_moe swap tests (monitoring point 6), last rung, profile; X.3 now depends on P.1 | - |
| 2026-09-28 14:00-14:40 | P.1 X.3 O.1 | gates | - | P.1 (SDPA preset A): device 278.5 -> 249.9 ms, frozen full-layer attention component + swap tests rerun and pass; X.3 full 56k: layer PCC 0.9984-0.9988, state 0.9963, TTFT 2524 ms; O.1: 8 fork calls, 0 uncovered, 0 failed across all models' cases (fork test harnesses generalized to dispatch groups > 1; no fork code changed) | 3187fc945e0, bc86acc0474, a0757f27235 |

## Run1 complete (2026-09-28 14:40): 60/60 PASS, no retries, no debugger, no owner resets

vs the 1x4 prior (final state, 204.9 ms chunk after its perf work):
- accuracy (X.3 full 56k): identical to 4 digits (layer PCC 0.9984-0.9988, state min 0.9963)
- 50k->55k chunk device: 249.9 ms (1x4 204.9 ms). The gap is the MoE fabric dispatch + combine within each column (41.7 + 26.3 ms vs 8.7 + 12.6 local on 1x4); owner declined local dispatch / dispatch tuning.
- 0->55k warm TTFT: 2524 ms (1x4 X.3: 2209 ms, measured before its last perf work)
- bring-up ops: reused rms_norm_ttnn (incl. return_residual_sum), sdpa (V 128), unified_routed_expert_ffn (high_precision), dispatch / combine / offset_cumsum (now on a 2-device axis, fabric on). No fork extended, none created; no existing TTNN op touched.
