# Supervision: DeepSeek-V4.1-Flash, galaxy_8x4 full (seeded from the quietbox_2x2 run)

## 2026-10-26 10:41:59 UTC start (overseer: Claude session on the owner Mac, driving bh-glx-120-b06u08 over ssh)
- Host: bh-glx-120-b06u08 (Exabox 120-B67, Prefill slot, Slurm job 153825, 32 chips, TDP_LIMIT_MAX 0x82).
- Spec: seed spec unchanged except paths (owner-approved intake): art /scratch/$USER/bringup (node-local), hf hf_full
  (HF rev 5254448 text-only + the 4 vision/MTP shards, re-downloaded here), golden on /data (shared NFS).
- Goldens imported from bh-qb-12 via a private HF relay (minimal sets: s10240 52.9 GB, 32 non-seed chunk_00 layers left
  out; s56320 37 GB, 400 files left out); verify_copy intact on /data. Dry checks before intake: B.1 PASS, G.s10240 PASS,
  G.s56320 PASS, CK.1 all checkpoint/prompt metrics ok (intake_approved was 0 before approval).
- Note to W.1/M.1 (overseer note): chunk 5120 -> 160 query rows per chip, never gated (plan checked 8x4 at chunk 16384);
  first run of 4-column TP (loudbox_2x4 never ran). Plan deploy chunk 8192 vs seed spec deploy.chunk 4096: inconsistent,
  not blocking.
- Known infra: real-time profiler D2H socket pin failed on devices 7 and 10 during B.1 (F155 perf measurement may be
  affected at X.1); a 2x2 submesh of a galaxy fails its fabric handshake (no TT_VISIBLE_DEVICES 2x2 subset exists).
- 2026-10-26 10:42:47 CK.1 gate passed, commit refused (no git identity on the node): infra. Set repo-local user.name/email, gate CK.1 --commit-only, resume.
- 2026-10-26 10:50:58 W.1 pre-check FAIL (187 s): TT_FATAL flat_routed_expert: no gate/up split for 24 tile columns (flat_routed_expert_plan.cpp:433), first expert layer on 8x4. Classified: model/fork code (expected for a port). No action; port attempt 1 started. Review: split derived from shape, 2x2 path kept.

## 2026-10-26 11:30:52 overseer session state (outside this tree; for a hand-off)
- Overseer: a Claude Code session on the owner Mac, driving this node over ssh (Host bh-glx-120-b06u08 in the Mac ~/.ssh/config;
  login node Host exabox = slurm-login-01). Node reserved via the ai-workspace exabox skill: Slurm job 153825, Prefill
  all-day slot 120-B67 (schedule tab 10/26-11/01), no time limit. Release: ai-workspace skills/exabox/scripts/release.sh.
- Orchestrator: tmux ds41-8x4, launcher /scratch/dnijemcevic/run_ds41_8x4.sh (now runs "orchestrator resume"); log
  /scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/orchestrator.log.
- Session-only, re-create in a new session: the watch (bringup.watch --spec <this spec>, re-armed after every event) and
  the 30-min heartbeat cron (:07/:37). No Stop hook (heartbeat_stop.py) installed on the Mac yet.
- Dashboard (standard) published: https://claude.ai/artifact/TCBJdnmec628NHLQDXJM37 (dashboard.links records it); refresh at
  milestones (W.1 verdict, M.1).
- Git: repo-local identity set (Djordje Nijemcevic <dnijemcevic@tenstorrent.com>); every commit is mirrored by
  .git/hooks/post-commit to /data/dnijemcevic/tt-metal-ds41.git (log /scratch/dnijemcevic/mirror.log). Nothing pushed to
  GitHub (no credentials on the node; pushes are the owner call).
- Data: checkpoint /scratch/.../hf_full (HF rev 5254448, all 48 shards); goldens /data/dnijemcevic/bringup/
  deepseek_v41_flash_d_p/golden_full/galaxy_8x4 (minimal sets, verify_copy intact) and golden_scaled/quietbox_2x2/s4096_c2048
  (+ galaxy_2x2 symlink). /scratch/.../hf_scaled/galaxy_2x2 = scaled cut (unused after the 2x2 attempt).
- Relay cleanup pending: private HF dataset djordjenTT/ds41-goldens-relay (holds s4096_c2048 only); temporary write token in
  bh-qb-12:/localdev/dnijemcevic/.hf_write_token and this node ~/.hf_relay_token (owner revokes the token when done).
- Open: framework gap: gates do not apply testing/cards.py TT_VISIBLE_DEVICES, and no 2x2 subset exists on a galaxy (a
  2x2 run here needs a mesh-graph descriptor or a submesh of the 8x4); real-time profiler D2H pin failure on devices 7/10
  (may affect X.1); plan.md checked 8x4 at chunk 16384, the run uses 5120 (OV-1 on W.1/M.1).
- 2026-10-26 11:31:07 owner: hand off at 90% context (not the skill default 75%); dashboard republish only at milestones (each costs ~100k tokens of overseer context).
- 2026-10-26 13:24:03 (restored; attempt 2 removed it) KNOWLEDGE CONTRADICTED DS41-76: 768 = 24 tile columns hits flat_routed_expert_plan.cpp:433 on 8x4; agent pads the shared expert to 1280 at TP 4 (DS41-115). Review: 576 -> 1280 is a 2.2x pad; prefer extending the fork split unless 1280 is a clean derived rule.
- 2026-10-26 13:44:57 W.1 PASS 9582c9d (attempt 1; gate unit reused). Attempt 2 started only because supervision.md was flagged outside the allowed paths: framework bug (IGNORED / stamps.SKIP did not cover bringup/ledgers/<run>/, F183). Classified framework. Paused, aborted attempt 2 (OV-2), restored state.json and results/W.1.json from 9582c9d, kept attempt 2 BREADCRUMBS notes (honest, no code change). Fix F189: orchestrator.IGNORED and stamps.SKIP cover ledgers/<run>/; selftest test_seeded_ledger_overseer_files.py (2 passed), test_orchestrator + test_stamps 45 passed. Resumed.
- 2026-10-26 13:46:15 W.1 diff review (9582c9da53d): accept new fork ttnn.bringup.update_padded_kv_cache (80-row stripes per chip on 8x4; stock op refuses non-tile stripes; README/CHANGELOG/unit test); accept derived q_chunk. Shared expert pad max(ipart, 33 tiles): TP4 576 -> 1280 (2.2x): derived from the op one-subgrid plan, stands; PERF CANDIDATE: extend flat_routed_expert gate/up split below 33 tile columns. Fabric setup nconn = len(dirs) (no repeated unused connections) touches the 2x2 path: SEED-BOX RE-VERIFY on a quietbox_2x2 (F-S8 not automated).
- 2026-10-26 14:06:52 OWNER (strong): no padding / inflated work to get around a kernel limit, ever. My W.1 review wrongly let the shared-expert pad (TP4 576 -> 1280) stand as a perf candidate; corrected: FIX.1 (extend the flat_routed_expert fork, remove all padding incl. TP2 1152 -> 1280, gate W.1), note OV-3 on every task. Pausing during the M.1 pre-check so FIX.1 runs before any M.1 agent attempt.
- 2026-10-26 14:15:10 Killed the orchestrator mid M.1 pre-check (owner: no waits; FIX.1 invalidates M.1), cleared PAUSE, resumed: FIX.1 started at once, device free, resume clean. Lesson: to reorder, kill + resume (F95) instead of pause-and-wait.
- 2026-10-26 14:32:04 HF relay repo djordjenTT/ds41-goldens-relay deleted (account has no datasets left). Token kept (owner): bh-qb-12:/localdev/dnijemcevic/.hf_write_token and ~/.hf_relay_token here, for a re-relay if L.s56320 needs golden files left out of the minimal copy; owner revokes it after L.s56320.
- 2026-10-26 14:36:11 FIX.1 attempt 1 aborted at 21 min (no code change; spent on before-timing, about to use device-profiler per-op mode, which the skill forbids for picks; the RT profiler fails on devices 7/10). My fix-job text asked for before/after timing: my brief caused the detour. OV-4: kernel first, one plain-profile section time after, no per-op device profiler. Attempt 2 started at once.
- 2026-10-26 15:17:45 OWNER: work autonomously; no lazy shortcuts; fix the framework cause of every bug (not a local patch): the goal is DeepSeek-V4.1 AND a robust framework. Framework causes queued for the next task boundary: (1) all units passed + only a path-scope problem starts a new attempt (F189 not-done); (2) no launch preflight for git identity (CK.1 commit refused); (3) gates do not apply cards.py TT_VISIBLE_DEVICES for a run mesh smaller than the box, no report when no subset exists; (4) RT profiler D2H pin failure on devices 7/10 not surfaced before X.1 (cause unknown: investigate).

## 2026-10-26 15:23:05 HAND-OFF: overseer moves from the owner Mac to this node (tmux session overseer)
Where the run is: CK.1, G.s10240, G.s56320, B.1, W.1 PASS (W.1 9582c9da53d). FIX.1 running (owner order: no padding; it
extends ttnn.bringup flat_routed_expert so the shared expert runs unpadded at 18 / 36 tile columns; plan change is
conditional: only widths with no subgrid split go to the whole-grid one-subgrid plan; the owner accepted that). Queued:
FIX.2 (root-cause the 11 first-call hangs of the fork tests on a Galaxy chip, fix the 11x10 grid asserts in the fork test
builder), FIX.3 (root-cause the real-time profiler D2H pin failure on devices 7 / 10; leads: iommu=pt, 2 MB + 4 KB
buffer, HugePages_Free 0 with a mesh open). Then M.1, the ladder, K.1, X.*, O.1, Z.1. Notes OV-1..OV-6 in overseer.yaml.
Owner rules (binding, beyond the skill):
- Work autonomously. Time matters: never wait when a kill + resume (F95) or an abort does the job; no long blind waits:
  short timeouts on checks, report what runs and its ETA.
- No lazy shortcuts, ever: no padding, duplicated or inflated work to get around a kernel / fork limit (OV-3). Review every
  gate commit for that; a shortcut gets a fix job at once and the run is reordered so it lands before dependent steps.
- Fix the framework cause of every bug (with a selftest and a dev/BREADCRUMBS F-entry), never a local patch: the goal is
  DeepSeek-V4.1 AND a robust framework. Framework edits only at a task boundary (stop the orchestrator, edit, selftest,
  commit, resume), never while an agent step runs.
- Perf numbers: device per-op times come from the real-time profiler with BRINGUP_PROFILE_OPS=1 (as on the 2x2); the device
  profiler for a single op. FIX.1 must report its op before/after per-call time (OV-6); if its commit lacks it, measure it.
- Hand off at 90% context (not 75%); republish the dashboard only at milestones (FIX.1, M.1, the ladder).
- Never push to GitHub without asking; commits are mirrored to /data/dnijemcevic/tt-metal-ds41.git by a post-commit hook.
Framework fixes due at the next task boundary (owner-approved): (1) a gate whose units all passed with only a path-scope
problem must commit and stop for the overseer, not start a new attempt (F189 not-done); (2) launch preflight: git identity
(and any other commit precondition) checked before the first gate; (3) gates apply testing/cards.py TT_VISIBLE_DEVICES when
the run mesh is smaller than the box, and report when no subset exists (no 2x2 subset on a Galaxy); (4) via FIX.3.
Open for the owner: FIX.1 op before/after numbers; 2x2 QuietBox re-verify after W.1 fabric-setup change (nconn = len(dirs));
HF write token (qb-12 /localdev/dnijemcevic/.hf_write_token, ~/.hf_relay_token here) revoked after L.s56320; the
/scratch hf_scaled/galaxy_2x2 (272 GB) is unused. Dashboard: https://claude.ai/artifact/TCBJdnmec628NHLQDXJM37 (publishing
needs a session with the Artifact tool; a node session may not have it: if so, skip and say so).
Slurm: job 153825 on this node (Prefill all-day slot), no time limit; release only on the owner word.

## 2026-10-26 15:29:53 overseer session on this node (tmux overseer) takes over
- Watch + heartbeat cron (:07/:37) armed; Stop hook heartbeat_stop.py installed in ~/.claude/settings.json on this node.
  Node gotcha: plain `python` does not exist; every bringup command needs `source python_env/bin/activate`.
- 15:25:38 FIX.1 PASS c9217573868 (attempt 2). Diff review: shared-expert padding removed (TP4 576, TP2 1152 at real
  width); fork plan falls back to one subgrid only where the 2/3-subgrid plan had no gate/up split (old plans unchanged);
  se6_dcompute static_assert kt_d % 8 dropped (hk is matmul_block kt_dim with rt_dim 1, unused by the LLK: agent cites
  llk_unpack_AB_matmul.h). New unit test test_flat_widths.py (I 576/1152, glm/full, vs float64 within noise limit).
  W.1 gate re-ran on the new code (agent check 15:24:42) and passed. Accepted: honest fix, no padding (OV-3 met).
  KNOWLEDGE CONTRADICTED DS41-76 / DS41-115: the agent's own corrections of the padding rationale; expected.
  Review items: 2 fork env reads (settings lint) + stale padded fork case -> note for O.1 / Z.1.
  OV-6 before number missing (agent: after = 0.153 ms/layer shared expert on the model path, MoE block 4.662 ms): I
  measure the before (I 1280 padded call on the same build; It 40 plans unchanged by FIX.1) at the next boundary.
  SEED-BOX RE-VERIFY list += FIX.1 (2x2 shared expert now runs I 1152 unpadded).
- Framework fixes (1)-(3) written in dev worktree /scratch/dnijemcevic/fwdev: F190 (pass + out-of-scope -> stop for
  overseer), F191 (launch preflight: git identity, merge/rebase, index.lock), F192 (run mesh smaller than the box:
  cards.py subset at launch, PREFLIGHT stop when none). Land at the FIX.2 boundary.
- 15:41 heartbeat: orchestrator alive, FIX.2 attempt 1 active (transcript 15:40). INCIDENT (mine): the dev-worktree
  selftest run (test_resume_after_a_fix -> main resume -> F192 run_mesh_cards) probed this box's REAL cards via
  run_safe_pytest card_probe.py from ~15:33 to 15:42, next to FIX.2. Killed (no probe left; card cache untouched,
  dated 11:34). Possible effect on FIX.2: per-card lock waits or a probe-side reset between its runs; read FIX.2's
  hang findings with that in mind. Framework cause fixed in the dev tree: selftest conftest autouse fixture hides
  /dev/tenstorrent and the card cache from every selftest (test_no_selftest_sees_the_real_cards).
- 15:55 dev-tree selftests (F190-F192 + isolation): 594 passed, 9 skipped, 2 xfailed; 1 failure from before these fixes
  (test_e2e_runtime::test_no_runner_contract_test_takes_a_fixed_threshold flags xing40_a4b_d_p's frozen
  server_rules.py:393 fixed PCC, a different model's frozen test; not for this run). test_fork_cases::test_a_volatile_cache_key
  crashed (SIGBUS importing ttnn while FIX.2 rebuilt it); rerun alone after FIX.2. Fixes ready to land at the FIX.2 boundary.
- 16:39 FIX.2 PASS 677bc394faf. Review: real root cause, accepted. se_dyn leader core compared my_x/my_y (translated NoC
  coordinate on Blackhole) with reader 0's physical NoC0 coordinate: on Galaxy chips (physical column 1 harvested) no
  core matched, every core polled the semaphore forever (SE_DYN_PULL plans only). Now logical coordinates
  (get_absolute_logical_x/y). flat_expert.py test builder generalised from the 11x10 assert to E-shifted layouts (12x10).
  W.1 re-ran on the final code (agent check) and passed. KNOWLEDGE CONTRADICTED DS41-119 (mesh runs hung too, on the
  dyn-pull path) and DS41-63 (fork perf bands fail on a Galaxy chip: QuietBox baselines): agent's corrections, expected.
  SEED-BOX RE-VERIFY list += FIX.2 (dyn-pull leader change runs on the 2x2 p150s too).
- 16:39:31 killed the orchestrator at FIX.3 attempt 1 start (<1 min of agent work) to land F190-F192 (kill + resume).
  Moved F191/F192 code out of core/gate.py and testing/cards.py (both measured: stamps.py would have voided every
  passed unit) into orchestrator.py / __main__.py (stamp-skipped). Selftests 595 passed, 1 failure from before these
  fixes (xing40 frozen contract test). Commit ff690328e21.
- 16:58 FIX.1 before-number (OV-6): perf_shared_expert.py test_shared_flat L00 time, with the pre-FIX.1 padded
  shared_expert.py (9582c9d) loaded by a scratch pytest plugin, on the current build (It 40 plans unchanged by FIX.1).
  Log /scratch/dnijemcevic/fix1_before.log. FIX.3 resumes after it.
- (correction: the before-measurement ran 16:45-16:51, not 16:58.) FIX.1 before/after (OV-6), 8x4, L00, chunk 5120,
  640 rows a chip, EXPERTS_PRECISION=glm, synchronized wall time of 5 calls (perf_shared_expert.py test_shared_flat time):
  shared expert on the model's path (rows, minus the test's rows clone): padded 1280 = 0.702 / 0.206 ms (two runs;
  the first is an outlier) -> unpadded 576 = 0.153 ms (FIX.1 agent, one run). Own-x path 0.553 / 0.511 -> 0.513 ms
  (dominated by the x conversion). MoE block 4.517 / 4.609 -> 4.662 ms: within run-to-run noise. Plans: padded
  gate/up 40 down 30; unpadded gate/up 18 down 30. Accuracy unchanged (rel 0.0213 both). Verdict: ~0.05 ms a layer
  saved on the shared expert (-26%), no visible MoE-block change at this noise level; X.1's RT-profiler per-op rows
  will give the precise number. Log /scratch/dnijemcevic/fix1_before.log.
- 16:51:24 resumed (launcher in tmux ds41-8x4): preflight passed, FIX.3 attempt starts.
- 16:52 FRAMEWORK BUG: on the 16:51 resume FIX.3 "passed" with no agent (gate B.1 + W.1 reused; commit 5b1a3ec2106 holds
  only records, no code: VOID). Cause: F87 gates every resumed task first, also a fix job cut off mid-agent whose gate
  passed before the job. Killed the run in M.1's precheck (seconds in). Fix F193 (7fd0e51f4ee): a precheck-False task
  gates first on resume only after a STOP; cut off -> its agent runs. Selftest fails without it. Suite 595 passed + F193,
  only the xing40 failure from before. `rerun --from FIX.3 --no-run`, resumed 16:58:46: FIX.3 overseer attempt 2
  started (attempt 1 = the 16:39 cut-off).
- 17:50 OWNER: no sudo on this node; the owner asks the admins for `sudo systemctl restart tt-iommu-domains` (chips 7 /
  10 identity IOMMU). Until then (owner, "want"): profile on the 30 working chips. DUE at the FIX.2 boundary (framework,
  orchestrator.py): F194 stop -> warn + record the missing chips on the task (review), no stop; plus ONE device-profiler
  run of one layer before X.1's numbers are used, to check chips 7 / 10 are not slower than the rest.
- 17:41 FIX.2 REOPENED (owner): its first pass left 14 fork unit tests failing on a Galaxy chip (12 TP2/TP4 TT_FATAL
  p.nsg == 1, 2 time bands); I passed it without checking its own done-condition (its gate had only W.1). Gate now W.1 +
  tests/unit/ on a Galaxy chip; brief requires all 14 to pass, no exceptions. FIX.4 removed (owner: no new fix numbers).
- 18:10 OWNER: device time of the 8 seed layers before M.1. Plan at the FIX.2 pause: (1) FIX.3 classified framework
  (no agent): I fix its knowledge text, commit on reused units; (2) seed profile: test_profile.py on rung seed (layers
  0 1 2 3 14 20 21 24, chunk 5120 at 5120), BRINGUP_PROFILE_OPS=1, RT profiler (30 of 32 chips: 7 / 10 IOMMU), via a
  scratch pytest plugin (scratchpad/seedprof/run.sh), results outside the ledger; ~18 min load + profile; (3) resume -> M.1.
- 18:31 FIX.2 PASS 9b8bf7c7d40 (attempt 2). Fork tests/unit on a Galaxy chip: 38 passed, 0 failed, 3 skipped (2x2-mesh
  tests, one card). Cause of the 12: flat_schedule_defines asserted nsg == 1 for every plan (only the overlap consumer
  needs it); assert moved to flat_combine_overlap. Time bands now per chip class (arch, grid): p150 11x10 unchanged,
  Galaxy 12x10 measured here. W.1 passed on the new code. Accepted. QB re-verify (owner): FIX.1, FIX.2, W.1 fabric.
- 18:33 FIX.3 knowledge text repointed at F194 by the overseer (owner: no agent for text): 3589c089004. FIX.3 is
  classified framework: resume commits it on reused units, no agent.
- 18:34 seed profile started (run still paused): scratchpad/seedprof/run.sh, log /scratch/dnijemcevic/seed_profile.log.
- 18:37 INFRA: something removed this user's /tmp entries (/tmp/claude-4338 incl. the session scratchpad, /tmp/tmux-4338,
  /tmp/tt-device-*.lock / .dirty); other users' /tmp untouched; cause unknown (no run of ours deletes /tmp). Effects: the
  seed profile (shell in the scratchpad) killed at 4 min (SIGTERM, mid weight load); the tmux socket gone (restored:
  mkdir -m 700 /tmp/tmux-4338 + SIGUSR1 to the tmux server); the next try segfaulted on an OpenMPI session dir created
  without x (mode 0640, /tmp/ompi.<host>.4338): chmod 700, umask 022 added to the run scripts. Overseer files now live
  in /scratch/dnijemcevic (seedprof/), not the /tmp scratchpad. 18:39 seed profile restarted (tmux seedprof).
2026-10-26 19:36:47 node drained for scheduled maintenance (Retimer FW + BusBar test, vincentdu) at 19:05, rebooted 19:07; reservation 153825 lost; run paused before FIX.3, resumable; uncommitted ledger files backed up to /data/dnijemcevic/bringup/deepseek_v41_flash_d_p/ledger_backup_20261026T1936

## 2026-10-26 21:47 new overseer session after the maintenance reboot (slurm job 155336, node bh-glx-120-b06u08)
- 21:44:54 autoresume: launch preflight passed with NO IOMMU warning. Box check (infra): 32 chips in /dev/tenstorrent; every
  card's IOMMU group now DMA-FQ (chips 7 / 10 no longer identity: the reboot fixed DS41-120's cause, so the admins'
  tt-iommu-domains restart is no longer needed); kernel cmdline iommu=pt. B.1 test_box.py re-ran fresh in FIX.3's gate
  (box rebooted -> not reused): PASS 19 s. RT profiler: M.1's precheck opened the full 8x4 mesh (4x8_Mesh_flat, all 32
  chips) at 21:45 with no "Real-time profiler" error (before the reboot every 8x4 open logged it for 7 / 10). X.1 profile
  will confirm device time from all 32 chips; the F194 "one device-profiler layer on 7 / 10" check becomes moot if X.1 has
  all 32.
- 21:45 FIX.3 PASS 89e622744f9 (B.1 fresh + W.1 reused; records only, as classified: framework, no agent). FIX.3 done.
- 21:45 M.1 started: precheck runs the s10240 ladder (40 layers). Watch + heartbeat cron (:07/:37) armed; Stop hook present.
- 21:47 stopped the run (kill + resume, F95) 4 min into M.1's precheck: the owner's 18:10 order (seed-layer device time before M.1) never ran (killed 18:40 by the /tmp wipe; results empty). Seed profile restarted in tmux seedprof (log /scratch/dnijemcevic/seed_profile.log), ~18 min; then resume -> M.1.
- 22:06 seed profile DONE (owner 18:10 order; results /scratch/dnijemcevic/seedprof/results/W.1_profile.json, ops
  W.1_ops.json.gz; outside the ledger). Rung seed, layers 0 1 2 3 14 20 21 24, chunk 5120 at 5120 (the second chunk
  ends at 10240), profiler rt, BRINGUP_PROFILE_OPS=1. All 32 chips report device time, 7 / 10 included and not slower
  (experts 21.8 / 27.2 ms vs a 18.2-30.3 range; attention 15.2 / 16.7 vs 14.5-17.9): the F194 single-layer
  device-profiler check on 7 / 10 is MOOT. Device kernel time, max over chips per section, summed: ~70.6 ms
  (experts 33.2, attention 17.9, ffn_norm 2.1, q_a 2.1, indexer 1.9, compressor 1.9, moe_add 1.7, attn_hc 1.6,
  router 1.4, ffn_hc 1.3, shared_expert 1.18, ffn_residual 1.0, engram_add 0.9, attn_norm 0.9, attn_residual 0.85,
  engram_kv 0.59, kv 0.18). Critical chip 29: kernel 55.7 ms against a 5726 ms device timeline; host dispatch 5544 ms
  over 697 calls (gaps 5639 ms). Wall time is all host dispatch (no trace yet; for X.*). Expert time varies by chip
  (18-30 ms): load imbalance across expert chips, for X.*.

## 2026-10-27 05:39 new overseer session (slurm job 155944, node bh-glx-120-b06u08) after an INFRA interruption
- INFRA: slurm job 155336 (previous overseer) ended 2026-10-26 22:17:55 UTC; no reboot (uptime 10:32 at 05:39); every
  process of this user got SIGHUP ~11 min into M.1's precheck. Most likely Slurm preemption (partition
  PreemptMode=CANCEL; another user held the node until 04:24 UTC); Slurm accounting is off, so unconfirmed. Lost: ~7.3 h
  of run time (22:17 -> 05:38). Classification: infrastructure; no model or framework cause.
- Just before the SIGHUP (22:17:50) the previous session queued FIX.4 (gate W.1): routed-expert weight load is ~113 s
  per layer (M.1 precheck log 22:07-22:15), ~75 min of host work before every M.1 / ladder / K.1 / X.* run; use the
  flat-expert fork's cache_prefix + callable weights (GLM-5.3 experts_ag.py does) and cut the miss path honestly
  (bit-identical tensors). Endorsed: it pays for itself on the first M.1 re-run. Not in the 22:06 entry, logged here.
- 05:38:56 autoresume (job 155944): FIX.4 placed before M.1, overseer attempt 1 started.
- Box health (SAFE_PYTEST had marked all 32 cards dirty on the SIGHUP; the markers lived in /tmp, which the new job
  wiped, so no automatic reset would follow): 05:40 test_box.py on the full 8x4 mesh PASS (17 s, log
  /scratch/dnijemcevic/boxcheck_0540.log), run next to FIX.4's first minute (agent still reading code). Box trusted.
  Framework gap noted: a card's dirty state kept only in /tmp does not survive a node's /tmp cleanup (the 18:37 wipe and
  this job start); candidate F-entry at the next boundary (keep the markers under /scratch or check the box at launch).
- Watch + heartbeat cron (:07/:37) armed; Stop hook present in ~/.claude/settings.json.
- 06:41 FIX.4 PASS 655d1d2716a (attempt 1; gate W.1 kept from the agent's final check, which ran on the final code with
  every layer a cache hit). Diff review: tt/experts.py passes FlatRoutedExpert cache_prefix + a weights callable (miss
  only), keyed by revision, mesh, scaled/real, layer, EXPERTS_IMPL, stored-tensor descriptors; completion marker with
  file sizes + bfp4_stats (a killed dump repacks, not loads a truncated file). No fork change. Miss path vectorised,
  exactness checks kept (one per stack). Evidence: perf_experts_load.py::test_host_weights_identical (all 384 experts,
  layer 0: every bf16 tensor torch.equal to the old loop), W.1 metrics identical to the last digit, device tensors of
  devices 0 / 31 equal hit vs miss. W.1 model load 1057 s -> 65.6 s (all hits); a miss is still ~117 s / layer (75 %
  the fork's bfp4 packing; follow-up, not now: one-time cost per layer). Disk 7.1 GiB / layer bfp4 (~285 GiB for 40;
  /scratch 2.1 TB free). New Setting EXPERTS_WEIGHT_CACHE (on). Accepted: honest, no shortcut, values unchanged.
  M.1's first run pays 32 misses (~62 min of load), later runs ~5 min. Seed box: the cache path is mesh-keyed; no
  change to the 2x2's numerics (QB re-verify list unchanged).
- 06:42 M.1 started (precheck: the s10240 ladder over 40 layers).
- 07:22:28 INFRA (overseer, not the run): the overseer's Claude process exited while idle (last turn ended 07:15:57,
  Stop hook passed, no command of mine running; the watch bz0eyrc49 was killed with it). The run was untouched (M.1
  precheck went on). The owner restarted the session at 07:23:46 (run_overseer_continue.sh, --resume). Evidence: the
  pane (tmux %1) ended at 07:22:28, the same second as the owner's ssh client attached to it (pts/2 from 10.32.64.6,
  07:14:19-07:22:28) disconnected; /tmp/claude-4338 and the tmux server intact (not a /tmp cleanup), no OOM (411 GB
  free), Stop hook ran in 40 ms with no error. Likely: keys or EOF from the dropping client reached Claude's TUI
  (an exit on Ctrl-C/Ctrl-D leaves no transcript marker); not provable with user-level logs. Fix at the launcher
  (/scratch/dnijemcevic, outside the repo), whatever the cause: overseer_guard.sh sets remain-on-exit + a pane-died
  hook on tmux "overseer" that runs overseer_respawn.sh, which resumes the SAME session (id in overseer_session) with a
  "you were restarted" prompt; held by overseer.nostart or after >3 exits in 15 min (crash loop); kill-session ends it
  for good. run_overseer.sh now starts with --session-id and records it; autoresume.sh sets the guard on every start.
  Applied to the live pane now; mechanism tested on a throwaway tmux session. Owner tip: detach with C-b d before
  closing the ssh window. Watch + heartbeat cron re-armed.
- 07:42 M.1 precheck STOPPED on pytest-timeout (3600 s) while still building the expert weight cache (cold: 32
  layers x ~117 s; 35 of 40 layers cached when killed). The framework retried the unit once by itself (F170); the retry
  loads 35 layers from the cache (~1 s each) and misses 5 (~10 min). Classification: framework (a 3600 s precheck
  timeout cannot cover a first cold-cache load of 40 layers); no waste beyond the one-time cache build, which the
  retry keeps. Candidate fix at the next boundary: the precheck/gate timeout scales with the uncached layer count, or
  the cache is warmed as its own step before a full-model gate.
- 07:44 owner: push the branch (the QB does not see /data). Push FAILED: no GitHub credentials on this node (https
  asks for a username; no ssh key; gh not logged in). Asked the owner.
- 07:48 owner logged gh in on this node (account djordjenTT); pushed dnijemcevic/model_bringup 34d20d4ee70..655d1d2716a (fast-forward, 16 commits up to FIX.4) for the QB re-verify of FIX.1, FIX.2, W.1 fabric.
- 08:01 M.1 PASS 03a4bcb9dab on the precheck (no agent; retry after the cold-cache timeout, 1144 s with the cache
  warm). Rung s10240, all 40 layers on 8x4: 105 metrics, 0 failures; top5_overlap 1.0 (>= 0.994); err_final_hidden
  0.189 (limit 0.256, 74 %); tightest err_state_window_kv_L27 78 % of its limit, err_layer_L30 72 %; errors grow with
  depth as expected (L00 0.025 -> L30 0.147); host_transfers_per_layer 0. Commit holds records only (no code change).
  L.s10240 started. Dashboard republished (milestone).
- 08:10 L.s10240 PASS 9960b0b018b (523 s). L.last started.
- 08:32 L.last PASS af70ec04593 (483 s; top5 1.0, err_final_hidden 0.185 / 0.242). L.s56320 started. DASHBOARD STALE wake: not republished (owner: milestones only; next at the ladder's end).
- 08:35 L.s56320 PASS ac0f241e2ca (full target 56320, 11 chunks). L.smoke started.
- 08:42 L.smoke FAIL (529 s): device greedy 8 tokens 'We need answer one word. Capital of' (coherent start of the
  model's thinking; no 'Paris'). Classified FRAMEWORK: testing/smoke.py run_smoke decodes a fixed max_new=8, while the
  CPU smoke (intake/check_hf_sanity.py smoke, n=24) needed 24 tokens ('...Capital of France = Paris...</think>Paris').
  Paused + aborted the L.smoke fix attempt 1 at once (note OV-10; not counted). Owner asked to see the cause first;
  proposed fix (shared budget 24 / intake.smoke.max_new, CPU-greedy prefix match recorded, selftest, gate --unit
  L.smoke) awaiting the owner's word. Owner (08:41): pause after L.smoke; then a profile chart of one 5k chunk at
  0, 50k, 100k, 200k, 500k (500k: check KV memory on 8x4 first); then the owner will rebase onto latest tt-metal
  (a month-old tt-d-gen is of little use before that), so K.1 waits.
- 09:0x F195 (0384fc770c1): device smoke budget = the CPU smoke's (24 / intake.smoke.max_new); selftest; suite 601
  passed, only the old xing40 failure. L.smoke classified framework, `gate --unit L.smoke`: device 'We need answer one
  word. Capital of France = Paris' (token 11) PASS; `--continue` committed 9fe2e3a915f. Run stays PAUSED before K.1
  (owner: pause after L.smoke; rebase onto latest tt-metal next; month-old tt-d-gen not useful before that).
- Owner order: profile one 5120 chunk of the whole model at context 0, 50k, 100k, 200k, 500k (positions 0, 51200,
  102400, 204800, 512000), device + host time, section breakdown each. Overseer measurement outside the ledger:
  /scratch/dnijemcevic/posprof (plugin: synthetic rung, zero KV prefix, random tokens, as testing/positions.py;
  run_profile unchanged: warm, plain wall, one RT-profiler run). 500k fits: deploy.seq 1044480. One process per
  position. Framework gap: run_profile only profiles a golden rung's last chunk; a position profile with sections is
  what X.3's test_positions should record (wall only today) -> F-entry at the next boundary.
- 08:44-09:36 owner: profile one 5120 chunk (all 40 layers) at context 0 / 50k / 100k / 200k / 500k: device total 329.9
  / 350.6 / 360.0 / 372.7 / 403.7 ms; only the indexer grows (3.6 -> 63.9 ms, ~0.12 ms per 1k); attention 82 -> ~93 flat
  from 50k; experts + moe_add ~181-186 flat. Results /scratch/dnijemcevic/posprof/results/POS*.json (FABRIC_2D).
- 09:40 F196 (14b46baf60f): clean wall timed only after a run adds no program-cache entry, misses refused in the timed
  runs. First use, position 0: NOT converging (249 -> 831 entries over ~30 runs, +20..40 a run; walls 15-43 s). Killed
  10:03 (would run to 100). Census (overseer plugin posprof/census_plugin.py, 6 runs): every new entry is
  ttnn.bringup.flat_routed_expert_with_program (+40/+40/+20/+40/+38/+20, i.e. 20-40 of 40 layers a run); its key bakes
  every buffer address (flat_with_program_device_operation.hpp:33). ~0.5 s host per miss. => no host number so far is
  warm. Owner: ignore host times until the op is fixed; profiling must always check cache entries before calling a run
  warm -> F197 (record_warm: <run>_new_programs on the plain, profiled and position runs; NOT WARM printed).
  FIX.5 queued (gate W.1): addresses as runtime args patched on a hit, out of the hash; bit-identical; F196 converges.
- 10:08 owner: switch 8x4 to FABRIC_2D_TORUS_XY; spec edited, intake re-approved (owner's word). Then device-time
  profile at the 5 positions on torus. Indexer ring hard-codes ttnn.Topology.Linear (tt/indexer.py:121): note for an
  agent after torus is verified. Framework gap: the selftest suite took CHIP_IN_USE_0_PCIe (a box test waited on it).
- 10:18 torus box test PASS (4x8_Mesh_flat_torus_xy, collectives exact; it first timed out at 300 s behind a chip lock
  held by my selftest suite, PID 3690127: killed; cards reset by the next run). 10:19 torus position profile pos 0:
  HANG at 10:25 ('device timeout in fetch queue wait'); triage: flat_routed_expert_with_program cores waiting
  (se3/se6 compute, semaphore_read), DRISC 'ready, done 0'. Series stopped (STOP + kill). Classified MODEL / FORK (the
  routed-expert path is not torus-safe), not infra: FIX.6 queued (gate B.1 + W.1 on torus; also the indexer's
  Topology.Linear). FIX.5 set held until FIX.6 (its W.1 gate would hang on torus). Both need the orchestrator resumed:
  owner's call (the run is paused for the rebase).
- 10:3x owner: drop torus for now (a new flat expert op in the works fixes it; FIX.6 removed, spec back to FABRIC_2D,
  intake re-approved); hack the cache misses, then pause for the rebase. Key log (posprof/keylog_plugin.py, 5 runs):
  of the op's key only a(xg) moves (56 DRAM addresses; arena, words, relay_words, counts, token_index fixed). Hack
  (owner-approved, overseer-written, default off): Setting EXPERTS_XG_PINNED; on = gather_x writes one persistent DRAM
  buffer (ccl.all_gather out=, ccl.dealloc skips pinned addresses: the shared expert frees take_rows' rows).
- 10:51 F196 warm wall, position 0, XG_PINNED=on: warm after 1 run (249 -> 249 entries), 10 timed runs with misses
  refused all completed: median 860.5 ms (830-974). The 15-43 s before was program rebuilds. First verified warm host
  number. Next: 500k device profile with the hack (vs 403.7 ms), accuracy of the hacked path, F197 suite, commit, push.
- 10:57 500k device profile, XG_PINNED=on: 383.5 ms (plain + profiled runs built 0 programs: warm), was 403.7 ms with
  misses (experts + moe_add 182.2 -> 166.9: collective waits from staggered program launches); warm plain wall 930 ms.
- 11:0x accuracy of the hacked path: s10240 ladder with XG_PINNED=on: every metric identical to L.s10240 (9960b0b018b)
  to the last digit (err_final_hidden 0.18905364, top5 1.0, L39 0.68742542). Selftests 607 passed (only the old xing40
  failure). Committed F197 + the hack (default off) + records; pushed (owner: include the hack). PAUSED for the
  owner's rebase onto latest tt-metal. Pending after the rebase: FIX.5 (proper op fix, or moot with the new op),
  torus with the new op, K.1 (needs a current tt-d-gen), X.*.

## 2026-10-27 11:20 HAND-OFF: node bh-glx-120-b06u08 may be released (slurm 155944); the next overseer may be on another Galaxy
Where the run is: CK.1, G.s10240, G.s56320, B.1, W.1, FIX.1-4, M.1 (all 40 layers), L.s10240, L.last, L.s56320 (full
56320 target: top5 1.0, err_final_hidden 0.189 / 0.259), L.smoke ('...Capital of France = Paris', F195) PASS.
Orchestrator STOPPED (not running; no PAUSE file needed): `orchestrator resume` continues with FIX.5 (queued) then K.1.
Branch dnijemcevic/model_bringup on GitHub = f9e7c977c3a (everything up to the pause); every later local commit is
mirrored to /data/dnijemcevic/tt-metal-ds41.git (post-commit hook). The OWNER IS REBASING the branch onto latest
tt-metal (2026-10-27): work only from the rebased branch the owner pushes; never push onto the old one meanwhile.
After the rebase every passed gate re-runs once on resume (ttnn / tt_metal are in the stamps): expected, ~2-3 h.

What lives where (a new Galaxy has none of the node-local parts):
- Repo + ~/.claude (memory, transcripts) and /scratch are NODE-LOCAL. Shared NFS /data keeps: goldens
  (/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4, spec paths.golden), the git mirror, the
  2026-10-26 ledger backup, and node_handoff_20261027/ (this hand-off's copies: scratch_bringup/runs (orchestrator log,
  agents' transcripts, gate logs, state), profiles, input; overseer/ = the launcher scripts (run_ds41_8x4.sh,
  run_overseer.sh, overseer_guard.sh + overseer_respawn.sh (tmux pane-died -> resume the same session),
  autoresume.sh, ensure_overseer.sh), overseer prompts, hf_dl.py, posprof/ + seedprof/ (measurement plugins + results)).
- To rebuild on a new Galaxy: clone/fetch the rebased branch, build; checkpoint: hf_dl.py (HF deepseek-ai/
  DeepSeek-V4.1-Flash rev 5254448103b6254fc646dacbca1cbbf20e9216c1, ~501 GB, ~7 min here; the 4 ignored shards were
  added separately: all 48 shards needed, see 2026-10-26 entries) into /scratch/$USER/bringup/deepseek_v41_flash_d_p/
  hf_full; copy node_handoff_20261027/scratch_bringup/runs back to /scratch/$USER/bringup/deepseek_v41_flash_d_p/runs
  (the orchestrator's state, logs and attempt numbers); the expert weight cache (tt_cache, 285 GB) rebuilds itself
  on first load (~117 s a layer, ~75 min once: the first M.1-type run needs a timeout above 3600 s or the framework's
  one retry, see 07:42 entry). Node gotchas: `source python_env/bin/activate` (no plain python), umask 022, the
  overseer's tmux guard (overseer_guard.sh).
- If the new box is not a Galaxy 8x4 (32 x p150b, 12x10 grid): the spec's box and the goldens are 8x4-specific.

Open items, in order:
1. After the rebase: fetch, build, box test (test_box.py, seconds), resume; watch the gates re-run.
2. FIX.5 (queued): flat_routed_expert_with_program bakes buffer addresses into its program key -> misses the cache on
   every run unless EXPERTS_XG_PINNED=on (the overseer hack, default off, bit-identical). The owner says a NEW flat
   expert op is in the works (also fixes torus): when it lands, drop FIX.5 and check the new op converges under F196
   (warm = a run adds no program-cache entry) and works on FABRIC_2D_TORUS_XY.
3. Torus (owner: wanted): FABRIC_2D_TORUS_XY hangs in flat_routed_expert_with_program (10:25, triage in
   node_handoff/overseer/posprof/pos0.log... copied as posprof/); retry with the new op; then the indexer's
   Topology.Linear (tt/indexer.py:121). Then re-measure the 5-position profile on torus.
4. K.1 needs a CURRENT tt-d-gen (owner: a month-old one is of no use): dgen-build agent on the latest tt-d-gen.
5. X.*: perf picks are the owner's. Candidates from today: trace capture (warm wall 860 ms vs ~250-330 ms device per
   40-layer 5120 chunk), sparse expert dispatch (full column gather today: 28 % of the routed MoE), torus.
6. 2x2 QuietBox re-verify of FIX.1, FIX.2, W.1 fabric change (owner, on the QB).
7. Framework: the selftest suite takes a UMD chip lock (find which test); dirty-card markers live in /tmp (lost on a
   /tmp wipe); a 3600 s gate timeout cannot cover a cold expert-cache build; all at the next boundary.
Measurements of 2026-10-27 (outside the ledger, copies in node_handoff/overseer/posprof/results): device time per
5120 chunk (FABRIC_2D, before the warm fix: MoE slightly inflated) at context 0 / 50k / 100k / 200k / 500k: 329.9 /
350.6 / 360.0 / 372.7 / 403.7 ms; only the indexer grows (3.6 -> 63.9 ms). Verified warm (XG_PINNED=on): 500k device
383.5 ms; host wall 860 ms (0k, 10 runs, misses refused) and 930 ms (500k).
Owner rules: the 2026-10-26 15:23 HAND-OFF list still binds (autonomy, no shortcuts, framework fixes with a selftest,
push only on the owner's word, dashboard at milestones). Plus today: never call a run warm without the program-cache
check (F196 / F197); host times stay hidden until verified warm.

## 2026-10-27 16:44 overseer session on bh-glx-120-b06u08 (slurm 157973) after the owner's rebase
- Repo on the rebased dnijemcevic/model_bringup at 7c06953a55c (= rebase_switched); old head kept as ds41-prerebase.
  No python_env change in the new base (requirements untouched; setup.py drops two header globs only): env kept.
  Build: /scratch/dnijemcevic/build_rebase.sh (Release, Tracy, clang-20 as before; stage lines in build_rebase.log).
- Owner order (2026-10-27): complete the bring-up end to end on this Galaxy, K.1 with the full 40-layer runner smoke.
  Spec serving: require_engine true, server_repo /scratch/$USER/tt-d-gen (/localdev does not exist on this node),
  answer 7 rewritten (the smoke runs here). Intake re-approved on the owner's word (16:45). dgen-build agent started
  in the background on tt-d-gen's latest commit (nice 19, -j24 beside the tt-metal build).
- FIX.5 stands: no new flat expert op in the rebased branch (7c06953 is Marko's combine-trailer hang fix only).
- Framework gap (F200, selftest test_engine_wait.py): with require_engine and no build, K.1's gate would fail and start
  an agent attempt that cannot succeed. Now a gate that runs a runner test (its file names dgen_engine / use_engine)
  waits with "tt-d-gen build missing" (no gate, no agent); independent tasks go on (F109); it runs once the build is found.
- Hand-off item 6: the owner's bab52a1 records the 2x2 passing after the rebase (L.*, K.1 contract, O.1 238 fork
  cases). Whether that closes the QB re-verify of FIX.1, FIX.2 and the W.1 fabric change stays the owner's call.
- Guard (overseer_guard.sh: remain-on-exit + pane-died respawn) re-applied; heartbeat cron :07/:37; Stop hook present.
- F.1 (F199, run-closing strict fork suite) added with ledger_gen --extend; runs after Z.1 (cost on the 8x4 stated before it starts).
- 16:48 build_ok = 5308dbb; box test PASS (24.7 s; box_rebase.log); 16:49 orchestrator resumed in tmux ds41-8x4.
  It placed FIX.5 first (queued fix, gate = W.1's frozen tests: these re-run on the rebased code). A resume does not
  re-gate the other PASS tasks by itself, so at the FIX.5 boundary (pause) the overseer runs `core.verify --tasks M.1
  'L.*' --batch` (the stamps reuse nothing across the rebase: ~2-3 h, owner-expected), then resumes with K.1.
- 16:59 tt-d-gen built by dgen-build: /scratch/dnijemcevic/tt-d-gen @ a9701ce6675 (main, 2026-10-27), tt-blaze 7393302a49a, 576 s cold; found via spec serving.server_repo (K.1 will not wait).
- 16:59 found: dgen_engine.server_repo does not expand $USER in serving.server_repo (the template default has it too): K.1 would wait forever. Pause requested (stops after FIX.5); fix + selftest at that boundary, with the M.1 / L.* verify.
- 17:05 OWNER: perf tasks (X.*) are off for now. Fix what is needed, prove the rebase broke nothing (M.1 + L.* verify), run K.1 (resume --until K.1). K.1's decode-via-prefill stand-in (test-side tail re-prefill on the migrated KV) is the accepted contract proof; no runner token output.
- 18:13 FIX.5 PASS be9736f2900 (agent's W.1 run on the final code reused). Reviewed: patch_addresses keyed program (no buffer address in the key; moved addresses re-applied on a hit), shipped default EXPERTS_PROGRAM_KEY=patch; bit-identical 24/24; F196 warm after 1 run (237 entries), 10 timed runs no miss, warm wall 999.8 ms (pre-rebase 860 ms with XG_PINNED: other base, not comparable); fork unit tests 38 pass / 4 skip (2x2-only). Accepted: honest fix, no padding. SEED-BOX RE-VERIFY list += FIX.5 (weighted_case column-barrier change). server_repo $USER fix committed with selftest.
- 18:50 REBASE VERIFIED: verify M.1 L.s10240 L.last L.s56320 L.smoke 5/5 PASS on 5161e706d64 (7-8 min each); err_final_hidden / top5 / err_layer_L39 identical to the pre-rebase values; smoke -> '...Capital of France = Paris'. Launcher now resumes with --until K.1 (owner: no perf tasks for now). Resumed.
- 18:57 K.1 gate unit test_contract FAIL (357 s): FileNotFoundError golden chunk_00/layer_04 (full-run goldens dump per-chunk only the 8 step layers; kv_cache has the final state of all 40). Classified: model code (tt/runners/kv_contract.golden_rows assumed per-chunk dumps for every served layer, true on the 2x2). Aborted the attempt that had started without the diagnosis (OV-11), wrote the note for K.1 (full comparison on step layers, last-window vs the final ring state on the others), resumed.
- 18:57 attempt 2 started with only OV-11 (my abort reason, misclassified as 'overseer fixes it') before OV-12 landed: paused, aborted attempt 2 (~1 min), closed OV-11 / OV-13; resuming so attempt 3 has OV-12. Lesson: write the note before the abort.
- 19:20 K.1 attempt 3 fixed kv_contract (test_contract + test_kv_layout PASS, kept); test_adapter_acks FAIL: 'no ladder rung s4096' - server_rules.py carries the 2x2 seed's geometry literals (2048 / 4096 / s4096). Paused, aborted attempt 4 (OV-14, then withdrawn), note OV-15: derive the geometry from the spec (seed must still get s4096/2048/4096), and audit the other contract tests for the same per-chunk golden gap. Framework lesson for F185 (seeded runs): the seed's contract tests are carried unchecked; box literals there break the new box - recorded in dev/BREADCRUMBS F200 follow-up. SEED-BOX RE-VERIFY list += server_rules geometry + kv_contract golden_rows.
- 19:47 K.1 attempt 5 done (the gate is re-running every unit on its changes). Its own runs: test_runner_smoke FAIL - tt-d-gen ENGINE_FATAL 'chunk_size=5120 is not divisible by sp_factor=1800000': latest tt-d-gen (#1600, layer-completion ring) dropped device_prefill_pipeline's layers_per_chunk; testing/dgen_prefill_driver.py passes positionally -> args shifted. FRAMEWORK bug. test_adapter_acks: window_kv layer 20 pcc 0.981 below the contract limit (agent: DS41-125). Path problem: the agent's contract-test edits (asked by OV-15) are outside K.1's paths. Pause requested (stops before attempt 6); at the pause: driver fix + selftest, K.1 paths, classify the acks PCC.
- 20:01 paused before K.1 attempt 6. Driver fix committed b813d15e7fe (selftest; checked against the real tt-d-gen binding). OWNER: Paris end to end first, A vs B later. Running only K.1#test_runner_smoke (gate --unit) in tmux ds41-smoke. Owner asked for a MoE-branch investigation (flat vs block modes) -> background agent, report /scratch/dnijemcevic/moe_port_report.md; after Paris + B: reconcile the branch's MoE updates with ours and re-measure the 5-position profile (device + host).
- 20:08 MoE port report: /scratch/dnijemcevic/moe_port_report.md (origin f81cb90cf5a, 10 owner commits: flat sync with Marko's layout EXPERTS_LAYOUT, block mode DS41_MOE_IMPL=block via ttnn.bringup.prepared_program, FABRIC_2D only; FIX.5 complementary (origin flat still keys addresses); 9 textual conflicts; both sides use F200 -> renumber ours at the port; new weight cache ~75 min cold outside a gate; full rebuild).
- 20:12 OWNER (MoE port, after Paris + B): non-torus first (FABRIC_2D), torus only once it works (block OP supports torus; only tt/moe_block_adapter.py:159 refuses it, D10). Block mode needs the weight re-layout: it loads routed weights via TtExperts in EXPERTS_LAYOUT (marko, D6) -> one new expert cache (~285 GB, ~75 min cold, build outside a gate), shared with flat+marko; keep the legacy cache. Order: reconcile + rebuild -> marko cache -> flat FABRIC_2D W.1/M.1/L.* -> block FABRIC_2D same -> 5-position profile (device + warm host) both -> torus.
- 20:15 runner smoke run 1 (20:01) hung: engine waited on /tt_prefill_layer_acks_<id> (pre-ring name) while the runner published /tt_prefill_layer_completions_<id>; runner idle waiting for the first chunk. Killed it. Framework fix ca3041407a8 (dgen_engine.ack_shm_name from the runner's scheduler_shm_name, driver default, selftest); one-line overseer edit in the model's test_runner_contract.py plan (uses D.ack_shm_name; uncommitted with the agent's contract edits, K.1 commits it). Re-running K.1#test_runner_smoke.
- 20:31 K.1#test_runner_smoke run 2: tt-d-gen drove the prefill end to end (attached 12.3 s, ADMITTED slot 0, PREFILL_DONE 160 tokens, no errors), KV read-back of all 40 layers OK; then the stand-in decode HUNG on its first tail (32 tokens at start 128) in flat_routed_expert (se3_compute / se6_dcompute / se5_recv, triage generated/tt-triage/triage.txt). Box test after: PASS. Diagnostic (outside the gate, own results dir): same smoke with DS41_EXPERTS_PROGRAM_KEY=addresses (pre-FIX.5 key) to see whether FIX.5's patch path is involved.
- 20:47 Diagnostic run 3 (EXPERTS_PROGRAM_KEY=addresses): prefill via tt-d-gen PASS again, decode HUNG again at 20:46:15 -> FIX.5 not the cause; the hang is flat_routed_expert on the decode tail input. HAND-OFF: HANDOFF_20261027_galaxy.md (this directory); node released; branch dnijemcevic/model_bringup_galaxy.

## 2026-10-27 20:58 overseer session on bh-glx-110-c10u02 (slurm 158554), branch dnijemcevic/model_bringup_galaxy a681324c835
- Node setup: tt-metal build (owner's build_node.sh) done 21:01:45, ttnn imports. Run dir restored: the 11:20 hand-off's
  run1 (stamps, run.json, timeline) as base + node_handoff_20261027b's orchestrator.log, logs, briefs, K.1/FIX.5
  transcripts (next K.1 attempt = 6); PAUSE dropped. Stamps are the 11:20 ones: K.1's kept units may re-run once.
- Agent definitions were not linked on this node: ~/.claude/agents -> models/demos/common/bringup/agents/*.md.
- 21:02 box test BLOCKED: /tmp/tt-device.lock (the pool's legacy interop lock) is a stale 0660 file of a previous user
  (dlukic, 19:00), unopenable for us. Infra: this slurm job holds the node alone (squeue), so no legacy runner can
  compete; /scratch/dnijemcevic/node_env.sh sets TTPOOL_LEGACY_LOCK/DIRTY to our own files (sourced by ~/.bashrc and
  the launchers). 21:03 box test PASS (30.8 s, box_node.log).
- Expert weight cache: copied by the owner from b06u08 (same commit, same mesh): DONE 21:02, 285 GB, 120 files.
  (My own parallel rsync of it was stopped at once when the owner said so; its 24 KB partial removed so the owner's
  move into tt_cache could not land inside it.)
- hf_full: hf_dl.py with all 48 shards (no ignore patterns), tmux hfdl, started 20:59.
- tt-d-gen: the dgen-build agent could not clone (no GitHub credentials on this node: no gh login, no registered key).
  b06u08's checkout is a9701ce6675 = origin main (ls-remote from b06u08 21:01), so its checkout + build tree are
  copied to the same paths (copy_dgen.sh); a later tt-d-gen update needs `gh auth login` here.
- 21:12 F201 committed 3c5ad4d385c (framework): a HANG unit's generated/tt-triage + generated/inspector are copied into
  logs/<stem>.hang-<unit>-<time>/ before the next unit runs; overseer abort, agent timeouts and orchestrator stops wait for
  a running tt-triage (selftest test_hangdump.py; core/overseer selftests 67 pass). Diagnostic runs outside gates go
  through /scratch/dnijemcevic/diag/run.sh (flock-serialized, copies the same dirs to /data/dnijemcevic/diag/ on exit).
- 21:14 hf_full complete (48 shards). Fast hang loop built (owner: no 15-min iterations): diag/test_repro_tail.py = the
  K.1 smoke's stand-in decode without the runner (REPRO_LAYERS subset, prefix prefilled on the device or loaded from a
  runner run's dump), diag/moe_dump_plugin.py (pytest -p: per-call routing dump + sync to localise the hanging layer,
  load_served/tail inputs saved). Always --no-precompile (the precompile pass costs 3 min).
- 21:19 NEW BLOCKER (box portability, model + fork): the device model does not build on this Galaxy:
  weighted_combine.py:245 '40 y writers > the L1 hand-off's 32'. flat_expert._layout's DRAM readers follow harvesting
  (get_optimal_dram_bank_to_logical_worker_assignment): reader columns 0,1,7,8 here (E=7) vs E=6 on b06u08 -> plan
  down 40 (was 32), same cap/gu/mt. Consequences: the copied expert cache misses on this box (new layout hash; ~75 min
  cold rebuild at the first full load, after the fix); the cache marker dropped 2 stale L00 laydbf29... files not in
  its marker. bringup-engineer (fix role, outside the orchestrator) started on it: hand-off > 32 writers if contained,
  else a fork option capping y writers; E=6 plan unchanged.
- 21:20 owner did gh auth login on this node; dgen-build agent started (tt-d-gen main a9701ce6675).
- 21:40 marker: no change. _CacheMarker.write lists every *.flat.* file present (all layouts), so layouts of several
  boxes coexist; it deletes only files no completed write listed (an interrupted dump). The 2 dropped L00 files were
  such leftovers from b06u08. (My earlier claim to the owner that it keeps one layout was wrong; corrected.)
- 21:40 owner decision (cache portability): the E=7 fix caps the plan's down cores at 32 so the weight layout key
  (WEIGHT_LAYOUT_KEYS, count-based) equals the E=6 one and the copied cache hits; told the fix agent.
- 21:42 run-2 triage (K.1.units.log) parsed offline: se3_compute / se6_dcompute trisc1 inside
  matmul_block_math_dynamic_throttle on all 32 chips, readers / se5_recv / se6_drecv / combine kernels in cb waits,
  routers waiting for credits: compute fed more work than its data (count / loop bound on near-empty experts?) rather
  than a lost semaphore. Data from the fast loop decides.
- OWNER SIDE NOTE (21:45): matmuls on more than 110 cores at a time may hang a card (power-supply current limits,
  rare). If nothing else explains the hang: count the matmul cores per program (Inspector kernels.yaml /
  programs_log core ranges) and flag any program over 110. Flat expert today: 48 gate/up + 32 down = 80 (E=6).
- 21:50 OWNER ORDER (binding sequence): (1) fix the decode hang, Paris end to end (tt-d-gen prefill + stand-in decode).
  ONLY AFTER Paris is confirmed: (2) reconcile with origin dnijemcevic/model_bringup (owner's new MoE, f81cb90+;
  moe_port_report.md: 9 textual conflicts, renumber our F200/F201 against theirs, full rebuild); (3) profile one new
  5120-token chunk at context 0 / 50k / 100k / 200k / 500k (device + verified-warm host, F196/F197) in BOTH MoE modes:
  flat and block (DS41_MOE_IMPL=block; may need the expert weight re-layout = a new cache, build it outside a gate);
  (4) once block works on FABRIC_2D: try torus (FABRIC_2D_TORUS_XY: wrap-around links, ring instead of line;
  untested, expected further gain; adapter D10 assert at tt/moe_block_adapter.py:159 dropped with the owner's word).
  B (test_adapter_acks) and the K.1 commit stay after Paris as in the hand-off; perf tasks X.* stay off.
- 21:55 portability fix accepted + committed (bringup-engineer, fix role outside the orchestrator): fork option
  flat_routed_expert max_y_writers (default 0 = off; plan, program and op keys; README/CHANGELOG/INDEX; unit test
  test_max_y_writers.py 2/2 on this chip); the model passes the combine's L1_MAX_WRITERS = 32 (a receive-side layout
  limit of sparse_combine_rf: per-writer landed counters at state words 128..191 and wx/wy[32]; words 352..511 free if
  more writers are ever wanted). E=7 chips: nd 32, per 5, the same weight layout key as E=6 -> copied cache HITS
  (layer 0: 7.3 s, model up 34 s). E=6: plan unchanged by code reading + unit assertion (not on an E=6 device).
  SEED-BOX RE-VERIFY += max_y_writers (programs recompile once: new key field).
- 21:38 the 1-layer repro (prefill [0,128) at start 0, layer 0 only) HANGS: Timeout, every device on physical 15-3 /
  15-2 = logical (11,1)/(11,0) - the same core that led run 2's triage on b06u08 (combine kernel in cb_wait_front).
  The K.1 hang now reproduces in ~2 min with ONE layer and without the runner.
- 23:05 K.1 DECODE HANG ROOT-CAUSED + FIXED (bringup-engineer, fix role, 4 h with the 4-min loop; overseer F202):
  not the op and not hot experts: test_runner_smoke.open_mesh() set the fabric without the spec's
  fabric_router_config (10880 B payload) -> the weighted combine's 10272 B packets over the router default 4352 B
  hung the first MoE call. The runner sets the payload itself (K-79): prefill passed, the stand-in decode hung.
  Evidence: the op alone on the captured routing passes (PCC 0.99989) on a correctly opened mesh and hangs with the
  model's exact triage signature on a default-payload mesh. Fixes: combine_rf refuses a too-small payload at build
  (no kernel change; tests/unit/test_fabric_payload.py 2/2), test_hot_experts.py regression (3/3 + the capture),
  F202 harness.open_mesh used by both smoke tests (DS41 and xing40) + lint selftest. 40-layer device model: prefill +
  4 tail steps pass. The smoke test edit is an overseer/framework edit of a contract test (its own mesh setup; no
  check, threshold or golden touched), like the 20:15 one. DS41-127 in known_issues.md (agent).
  Fork tests on 8x4 (flat tests/unit + combine_rf tests/unit + test_flat_routed_expert_ttnn.py): 57 pass, 17 fail,
  4 skip, 3 errors; none runs the changed code: test_acc_dtype 3 errors (2x2 mesh hard-coded, fabric init fails on a
  Galaxy), test_column_planner_mc 15 fail (planner alone forced to 8x4, overflows its L1), test_flat_routed_expert_op
  [tp4] AssertionError (16,1,6,2) (NEW on this node, uninvestigated), k2 perf band 4.7 % slow. -> owner report; F.1.
- 23:18 GOAL MET: K.1#test_runner_smoke PASS (gate --unit, 780 s): tt-d-gen a9701ce6675 drives the real prefill runner
  over all 40 layers (160-token padded smoke prompt), KV of every served cache of 40 layers read through the exported
  table (192 records, [0,128)), stand-in decode on the device (tail recompute at start 128, ~1.2 s a step):
  'We need answer one word. Capital of France = Paris'. Unit stamp kept; dump copy /data/dnijemcevic/k1_smoke_20261027_2305.
  Next per the owner's order: B (test_adapter_acks), K.1 commit, then the reconcile with origin model_bringup, the
  5-position profile (flat + block), torus.
- 00:45 K.1 attempt 6 (in progress), reviewed live: B implemented as decided (caches.GoldenKV(window="last"), no
  limit/golden touched; test_adapter_acks PASS in its check 00:09). test_runner_contract then found DS41-128: on TP 4
  tp_sum's row-block reduce-scatter adds the partials in a row-dependent ring order -> the pulled-back chunk rewrote
  layer 0 window KV with other bytes (704 rows, 1 bf16 ulp). Agent fix: setting TP_SUM_BLOCKS=cols (column blocks,
  order fixed per column; same volume; 'rows' keeps the old path). Accepted in principle (determinism of the same path
  is an owner rule). Consequences: runner smoke re-runs in K.1 (model code changed); after the K.1 commit the overseer
  re-verifies M.1 + L.* (accuracy, ~40 min) before the reconcile; SEED-BOX RE-VERIFY += TP_SUM_BLOCKS; perf of the
  column RS shows in the 5-position profile.
- 01:08 K.1 PASS 20e3d9ffa5c (attempt 6; all 5 units from the agent's check reused, F169): B + DS41-128 TP_SUM_BLOCKS=cols;
  Paris through tt-d-gen again on the new code. Orchestrator stopped at --until K.1.
- 01:46 re-verify on 20e3d9ffa5c (TP sums' new add order): M.1, L.s10240, L.last, L.s56320, L.smoke 5/5 PASS (7-8 min
  each); L.s56320 final hidden rel 0.189 (was 0.189), top5 1.0, top1 0.997; L.smoke '...Capital of France = Paris'.
  Dashboard exported (not republished: the 313 KB framework page must be read whole before a publish; deferred).
- 02:20 RECONCILE c9ba76f8618 (bringup-engineer, fix role): merge of origin/dnijemcevic/model_bringup d762b21db30 (14
  commits: Marko flat sync, EXPERTS_LAYOUT=marko, MOE_IMPL=block, fused MoE head/tail, torus allowed for block D14,
  their F200/F201). 16 conflicts resolved keeping both sides; ours renumbered F200->F203 (engine wait), F201->F204
  (hangdump); F202 kept. Build OK; selftests 637 pass / 1 fail (xing40 server_rules.py:393 fixed threshold, untouched
  by either side since the base) / 9 skip. Fork device tests: merge-related all pass (max_y_writers 4/4, with_program
  patch legacy+marko, hot_experts 6/6 + K.1 capture, fabric payload 2/2, flat_widths 6/6). NOT merge-caused, open
  (owner): column_planner_mc 15 (L1 overflow forced to 8x4), acc_dtype 3 (2x2-only), flat_routed_expert_op[tp4]
  (16,1,6,2), NEW from the sync: flat_routed_expert_op[mimo] arena_tiles C++ 604 vs mirror 570, indexed_perf k2 flat
  +21.7 % / tp4 +8.2 %. Marko layout on this E=7 box = the capped legacy key lay9c3da689dd35 -> cache hit, no rebuild.
  1 layer: flat (marko) PASS; block fused head FAILS at build (routing_stream.py:612 len(lst) <= 16 on FABRIC_2D lines,
  DS41-129); block with DS41_MOE_HEAD=ops PASS (same tokens as flat).
- 03:01 post-merge verify (flat, marko): M.1, L.s10240, L.last, L.s56320, L.smoke 5/5 PASS; L.s56320 final hidden rel
  0.1899, top5 1.0; smoke Paris. Starting the 5-position profile series (posprof/series.sh: flat device, flat wall,
  block(ops head) device, block wall; ~2 h).
- 05:10 5-POSITION PROFILE on c9ba76f8618 (merged; FABRIC_2D; one new 5120-token chunk; device = real-time profiler,
  0 new programs in every profiled run; wall = warm host wall, F196 + F205 settle, 10 timed runs, misses refused;
  results /scratch/dnijemcevic/posprof/results, logs posprof/{FLAT,FLAT2,BLOCK}*):
    ctx    flat dev  block(ops head) dev   flat wall  block wall   (10-27 flat dev, old layout)
    0      308.1     290.4                 671.8      1089.5       329.9
    50k    314.6     304.2                 695.8      1099.9       350.6
    100k   323.3     310.1                 803.5*     1136.6       360.0
    200k   338.0     320.7                 1005.3     1120.6       372.7
    500k   375.7     363.0                 746.9      1185.3       403.7
  * flat 100k hit the 60-run settle cap (not settled). Block's fused head (its default) does not build on 8x4
  FABRIC_2D (DS41-129) -> measured with DS41_MOE_HEAD=ops. Findings: block -3..-6 % device vs flat; flat -7..-10 %
  vs 10-27; block host overhead ~800 ms/chunk vs flat ~360-670; flat's host wall is position-dependent and not
  monotonic (200k ~1005 ms in both series; 500k 747): a host-side position effect, uninvestigated (perf off).
- 05:26 OWNER: push. Pushed dnijemcevic/model_bringup_galaxy a681324c835..2dc76a55232 (fast-forward, 25 commits: F204/F202/F205, max_y_writers, payload check, K.1 pass, reconcile merge c9ba76f8618, records).
- 06:20 TORUS (FABRIC_2D_TORUS_XY, wrap links present): block fused head (its default) + block ops head run; flat HANGS
  on the first MoE call (as on b06u08; kept /data/dnijemcevic/diag/torus_flat_1028_051322). Block fused on torus:
  L.s10240 PASS (top5 1.0, final hidden rel 0.192), L.smoke Paris. Profile: device 219.5 / 233.9 / 237.5 / 249.1 /
  284.4 ms (0/50k/100k/200k/500k; -24..-29 % vs flat 2D), warm wall 615 / 641 / 634 / 644 / 646 ms. At 0k: MoE 104.5 ms
  (flat 179.1, block-ops 159.8; 160 op calls vs 600/800), ffn_norm 3.6 (9.8), attention 71.0 (77.4), 2029 op calls.
- 07:00 HOST-TIME ROOT CAUSE (cProfile of 5 warm runs, posprof/hostprof_*.pstats): block-ops' extra ~435 ms/chunk =
  models/demos/moe_block/moe_block.py:903 _tp_rs_rows (ops head) calls fabric_reduce_scatter WITHOUT
  cache_descriptors=True -> the mesh program descriptor rebuilt in Python 40x/chunk (366 ms cum). Verified: with
  cache_descriptors=True the block-ops warm wall 1089.5 -> 841.0 ms. Patch kept at
  /scratch/dnijemcevic/posprof/moe_block_cache_descriptors.patch (reverted in the tree: owner's code, owner's call).
  Remaining block-ops vs flat (~200 ms): +840 ttnn calls/chunk (122 ms dispatch; inherent to the ops head),
  prepared_program.generic_op 21 ms (120 calls), str.replace 13 ms (1170 calls: kernel source templated per call),
  combine_plan.build_mc 12 ms (3 builds/chunk), begin_chunk +10 ms. Flat itself: ~443 ms/chunk ttnn dispatch
  (3575 calls, ~124 us each: host dispatch-bound, device 308 ms), engram host rows ~100 ms/chunk on the critical path.
- 07:19 OWNER: A. Fast-forwarded origin dnijemcevic/model_bringup d762b21db30..74c043c800e (= model_bringup_galaxy tip; galaxy pushed 2dc76a55232..74c043c800e). Both branches identical; next owner rebase: git rebase --rebase-merges or rerere trained on c9ba76f8618, else the 16-file fork conflicts recur.
- 07:26 OWNER: run the perf part; focus: fusions that cut HOST time; see whether agents find model-level improvements;
  work autonomously (branches pushed first: both at 8ad76bad6bb). Orchestrator resumed --until X.2 (X.1 warm profile
  on the shipped config: flat, FABRIC_2D; X.2 opportunity list). FRAMEWORK GAP (queued, F206 pending): X.2 ranks
  device time only and no agent proposes picks; host time (wall - device, ttnn calls per section, host hot spots) is
  not in the perf review. Interim: a bringup-engineer (perf review, CPU only, no code) writes perf_proposals.md from
  tonight's cProfile + ops data; the top proposals become perf tasks P.* (step perf, role perf), each behind a
  pre-named tt/settings.py switch (ab.change/ab.env = DS41_<NAME>), judged by the orchestrator's A/B report (frozen
  tests, ladder, profile with chunk_wall_ms), accept/reject per pick = owner.
- 09:48 FIX.6 PASS 142f265538e: MOE_IMPL default 'block' (fused head + tail) on the torus spec; all 11 gate units
  reused from the agent's check (W.1, M.1, L.s10240, L.last, L.s56320, L.smoke Paris, K.1 contract tests + runner
  smoke Paris through tt-d-gen on the torus, 664 s). The agent stalled after its check (no transcript write or child
  process for 15 min): its process was ended (SIGTERM, review item 'rc=143'); the gate reused the passed check.
  Paused before X.1 to land the engram hand-off.
- 09:50 ENGRAM HOST PATH (owner): bench tools/engram_bench.py (CPU/NVMe, byte-equal --check). Cold, today's code:
  ~4.6 s per engram layer per 5120-token chunk (read-ahead 256 KB x 122,880 random rows ~ 30 GB at the NVMe's
  ~6.9 GB/s); MADV_RANDOM: 0.79 s, 8 threads 0.48 s; pread+FADV_RANDOM the same. Hand-off for the 2x2 session:
  bringup/ENGRAM_HOST_PATH.md (problem, numbers, staged plan, 2x2 vs Galaxy split and pitfalls).
- 11:38 OWNER: accept P.1 (GENERIC_OP_PREPARED: warm wall 673.2 -> 594.3 ms, accuracy identical, gate PASS with the change on). SEED-BOX RE-VERIFY += fabric_reduce_scatter prepared (setting on).
- 12:10 P.2 TAIL_A2A_PREPARED A/B: on 544.4 ms warm wall; off side crashed (Fatal Python error: Bus error in load_tensor_flatbuffer while loading cached expert weights at model build; files re-read cleanly: transient). Off = the accepted P.1 tree: P.1 on 594.3 ms, all 103 accuracy metrics identical to P.2 on -> -49.9 ms. Brought to the owner without a re-run. Framework F207 (A/B side crash retried once, selftest) committed at this pause.
- 12:49 OWNER DELEGATION: "clean win -> accept yourself". The overseer accepts a perf pick only when ALL hold:
  gate PASS with the change on; every accuracy metric unchanged, or within its limit and no worse than the old path;
  warm wall improves and device time does not get worse; diff read (old path selectable, no precision change, no
  padding, nothing outside its paths). Anything else goes to the owner. One line per delegated accept, here.
- 12:49 DELEGATED ACCEPT P.2 TAIL_A2A_PREPARED: warm wall 594.3 -> 544.4 ms (-49.9), 103 accuracy metrics
  identical, gate PASS; diff: TailA2A prepared=False default + setting. SEED-BOX RE-VERIFY += TailA2A prepared.
- 13:39 PRECISION (owner: 'major fuckup', realistic = whatever the reference uses): KV caches bf16 vs reference fp8 window / fp4 compressed + index keys; GEMM activation inputs bf16 vs fp8; attention output before wo_a bf16 vs fp8. Cause: rule 7 spells out weights only, rule 17 only permits low-bit caches, SC.1 recorded 'decode-side cache bf16 lossless'; overseer did not flag it. Proposed: framework (rule 7 for every stored/GEMM-input tensor + automated precision audit vs the reference's rounding list, F208) + fix jobs (activations -> bfp8; caches need a served-format decision with tt-d-gen). Pause requested so P.4/P.5 wait for the precision fix (P.3's A/B finishes first).
- 13:45 OWNER: reject P.3 (FUSED_HEAD_PREPARED_PER_LAYER: 40 per-layer programs, warm wall 538.2 -> 552.6 ms, no gain;
  the agent's diagnostic: 7520 patch slots a call, kept as DS41-131). The reject left the fork changes in the tree
  (F209 pending): reverted by hand, patch at /scratch/dnijemcevic/posprof/P3_rejected_leftovers.patch; notes kept.
  Overseer error: the resume that applied the reject cleared PAUSE and started P.4 (should have been --until P.3);
  P.4 aborted within a minute (OV-17), orchestrator stopped. Overseer error: `pgrep -f "bringup.orchestrator resume"`
  matched this Claude session's own argv: some "orchestrator alive" heartbeat lines today may be false; exact checks
  from now on.
- 13:48 OWNER: specs fixed (both 2x2 and Galaxy): the stale "Precision during bring-up: HiFi4, fp32 accumulation, bf16
  KV cache" line replaced by "realistic = whatever the reference uses" (every weight, cache and GEMM input); intakes
  re-approved. F208 framework (rule 7 + spec check + selftest; suite 641 pass / 1 known failure). Next: the fix plan
  for the KV caches and GEMM activation inputs (owner discussion). P.4, P.5 held until then.

## 2026-10-28 14:35 HAND-OFF (context 84 %): node bh-glx-110-c10u02 (slurm 158554), branch dnijemcevic/model_bringup
Pushed: origin dnijemcevic/model_bringup = 673f9d93ddd (engram hand-off docs). LOCAL, NOT PUSHED (owner: no push without
agreement): F207 63a55fcfc67, P.1 01cd1c615ee, P.2 7078290a9f3, P.3 reject record 954a7829f3b, F208 + specs 2fff5b8b667.
Where the run is: FIX.7 (GEMM inputs -> bfp8, setting GEMM_INPUTS, owner F208) running in tmux ds41-8x4 via
/scratch/dnijemcevic/run_ds41_fix7.sh (--until FIX.7): W.1, M.1, L.s10240 PASS in its check; then L.last, L.s56320,
L.smoke, K.1. P.4 / P.5 HELD (pause/abort OV-17) until the precision work and a re-baseline; X.3 after P.5.
Perf so far (block MoE fused head+tail on FABRIC_2D_TORUS_XY, the default since FIX.6): warm wall 673 -> 544 ms per
chunk (P.1 prepared reduce-scatter -79, P.2 prepared fused tail -50); P.3 rejected. Owner delegation: clean wins
(gate PASS, accuracy unchanged/within limits and no worse, wall better, device not worse, diff clean) the overseer
accepts; anything else to the owner.
De-risk agent output dir: /scratch/dnijemcevic/ckv_derisk/ (outside the repo until FIX.7 commits). Precision plan (owner; F208): phase 0 precision audit hook + reference quantizer checks; 1 GEMM inputs bfp8 (FIX.7);
compressed KV FIRST (owner: hardest, biggest risk): de-risk agent running CPU-only L0 (CKV_4BIT_DERISK.md: option A
reference e2m1 + e4m3 scale per 16 vs B row-packed bfp4; RoPE is inside the quantized row; index_topk 512 rows /
candidate blocks of 8 -> row-major with in-row scales, gather+decode); device L1 (one card) / L2 (1-layer repro) only
after FIX.7 frees the mesh, L3 (ladder + K.1) once as the fix job's gate. Then window KV -> bfp8 tiles, index keys
-> bfp4 tiles (record geometry: owner), contract + K.1, re-baseline X.1, resume P.4/P.5.
Open owner decisions: window KV bfp8 tiles; index keys bfp4 + 16-row records (32-row vs half-tile); bfp4 tooling
(tt-d-gen vs ours); GLM / xing40 spec rules flagged by F208 (remove / owner exception); pad-row masking as a perf
item; engram: "never hold the tables in host RAM" spec rule vs the RAM path; F209 (reject must revert fork changes,
keep notes) to implement. Engram work: on the 2x2 session (ENGRAM_HOST_PATH.md, 4b order).
Overseer notes: use `pgrep -f "^python -u -m models.demos.common.bringup.orchestrator"` (the plain pattern matches this
session's argv); a resume clears PAUSE (use --until to stop); watch + cron :07/:37 + Stop hook armed.
- 14:45 COMPRESSED-KV DE-RISK L0 DONE (report /scratch/dnijemcevic/ckv_derisk/CKV_4BIT_DERISK.md; codec.py + 8 passing
  tests, study.py, study_results.json; intended repo paths in the report; moves into the repo after FIX.7 commits).
  Today: write tt/model.py:422 -> TtCompressor.write (tt/compressor.py:548; served cache via update_padded_kv_cache,
  attention table via slice_write, replicated); read: stock sparse_sdpa(kv_format=BF16) (tt/attention.py:589) as 512
  single rows per query; 38/40 layers read one of 4 sources (L2, L8, L14, L20). L0 (golden s10240 native rows):
  rows vs native bf16 0.0017 | A (reference e2m1 + e4m3/16, codec bit-exact) 0.095 | B (row-packed bfp4, host
  rounding) 0.111-0.120 | B truncated 0.21-0.22; attention output vs native: A 0.025-0.047, B 0.026-0.053 (1-11 %
  above A), B truncated 0.18-0.25. RECOMMENDATION B (288 B/row, 3.6x less than bf16; native bfp4 unpack after a
  byte shuffle into tiles, no tilize, no scale multiply), IF the device packer rounds to nearest (else SFPU rounding
  first): L1 test 1 decides. Work: write fork op (pack bfp4 -> 288-B rows, both stores); read: sparse_sdpa fork
  ROW_BFP4 (gather rows like GLM scaled-fp8, shuffle into 16 bfp4 tiles / 32 rows), window rows stay separate (two
  KV tensors); contract change (owner + server). L1 plan (one card, after FIX.7): packer bytes, device rounding,
  write bytes, gather/decode 512 rows + sentinels, sparse_sdpa ROW_BFP4 vs CPU, micro-benchmark. L2: diag repro
  behind CKV_FORMAT. DS41-133 drafted in the report.
  INCIDENT: the de-risk agent imported ttnn on the host (cluster open) and held the chips 14:35:58-14:37:43 between
  FIX.7's gate units (FIX.7 L.s56320 delayed ~2 min, unharmed). Lesson for briefs: "no device" includes `import ttnn`.
- SWITCH POINT (owner): continue in a fresh session (/bringup, "report on deepseek_v41_flash_d_p galaxy_8x4") from
  the 14:35 hand-off entry + this one. FIX.7 still in its gate check (L.s56320, L.smoke, K.1 to go).
- NEXT SESSION, first device step (one card, after FIX.7 commits): the device packer's bfp4 rounding rule (L1 test
  1). Crafted bf16 tile (exact half-steps, just above / below, groups whose max forces shifts) -> packed to
  bfloat4_b BY THE DEVICE (ttnn.typecast to bfloat4_b, or a matmul with a bfp4 output) -> read back -> compare with
  RNE and truncation predictions and with the host packer (from_torch dtype=bfloat4_b = the de-risk codec). Hints:
  PR #49811 and the K3 write-up say the Blackhole packer truncates fp8; unknown for bfp4. Truncation -> SFPU rounding
  before the pack in the write kernel (else option B's error doubles: 0.21 vs 0.11).
- PRECISION-INFO STATUS (owner asked, 14:55): FIXED: agents' rule 7 (caches + GEMM inputs at the reference's width;
  bf16 where the reference is narrower is wrong even if a contract / spec / test says so); spec check
  stale_precision_rules (F208, selftest); the stale line removed from both DS41 specs. STILL OPEN: (1) the frozen
  serving_contract.md still says "One format: bf16, ROW_MAJOR. The owner's rule is a bf16 KV cache" - rewritten with
  the cache work (remaining item 5); (2) glm53_flash_d_p(_lb) and xing40_a4b_d_p(_8x4) specs carry HiFi4 lines, now
  flagged by the check - owner decision (remove / owner exception); (3) no automated check of the device's actual
  formats yet (precision audit hook, rest of F208, remaining item 6); the model's settings.py texts and knowledge
  files were not audited for old precision wording - the audit hook catches it in practice.
  REMAINING bf16 WORK after FIX.7: 1 compressed KV -> bfp4 rows (de-risked, B; packer-rounding test first); 2 window
  KV -> bfp8 tiles; 3 indexer keys -> bfp4 tiles (record geometry: owner); 4 index query -> bfp4/bfp8; 5 serving
  contract + K.1 for the new formats (owner + server sign-off; bfp4 tooling); 6 precision audit hook. Then X.1
  re-baseline, P.4/P.5, F209, pad-row masking, 2x2 seed-box re-verify (incl. MOE_IMPL=block on its FABRIC_2D).
- 15:46 FIX.7 PASS 0b85058efbc (GEMM inputs -> bfp8, GEMM_INPUTS setting; all 11 units incl. Paris through tt-d-gen).
  A transient runner bus error (BUS_ADRERR "non-existent physical address", device-mapped memory; second today after
  P.2's off side) failed one smoke run; the box test passed (15:40) and the agent's re-run passed (15:45). Watch for
  a third: then a board reset (owner).
- 15:50 OWNER: read-ahead fix done here, not on the 2x2: reference/weights.py EngramTable madvise(MADV_RANDOM) on the
  two table mappings. Cold layer-1 lookup through EngramTable: 4,480 -> 610 ms, disk read ~23 GB -> ~0.95 GB, rows
  byte-equal; 2-layer repro (engram layer 1) passes. ENGRAM_HOST_PATH.md: step 2's MADV_RANDOM part is DONE.
- 15:55 PUSHED everything: origin dnijemcevic/model_bringup = 5b275cb7622 (F207, P.1, P.2, P.3 reject, F208 + specs,
  FIX.7, read-ahead fix, hand-off records). Orchestrator stopped after FIX.7 (--until FIX.7); P.4/P.5 held; device idle.
  NEXT (in order): (1) report FIX.7's per-matmul before/after numbers (accuracy margins, device time per chunk) from
  its agent report (runs/galaxy_8x4-full/run1/agents/FIX.7.overseer.1.jsonl final message, results/FIX.7.json) - owner
  asked, not yet done; (2) the bfp4 packer-rounding test on one card (recipe above); (3) move
  /scratch/dnijemcevic/ckv_derisk/ files into the repo (paths in its report) and run the compressed-KV L1 tests on one
  card; (4) owner decisions still open (see 14:55 and 14:35 entries).
- 16:05 DS-V4.1 DECODE KV FORMATS (tt-blaze llong/dsv41-reload-quad-engram4x1 @ 2d8abc70e8, read-only agent): NO 4-bit
  KV. SWA layers 0-1: window KV bfloat8_b TILE [1,1,128,512] (unpacker -> bf16; per-token RMW of one tile). CSA2
  layers 2-39: window + compressed KV in ONE bf16 ROW_MAJOR tensor per layer (rows [0,128) circular window, 128 +
  p//ratio compressed), 1024 B/row, replicated on 8 TP chips ("Released compressed KV is E2M1 with E4M3 scales; the
  device cache is BF16", tests/blaze/deepseekv4_1/harness.py:483). Indexer keys: uint8 rows of 192 B = 128 e4m3 +
  fp32 per-row scale (max-abs/448, not pow2) + pad, 32-row bank granules / 256-row chunks per chip, quantized on the
  device (truncating), dequantized in the scorer kernel. Decode attention HiFi4, bf16 activations, weights bfp8/bfp4.
  "reload" = ring of per-layer program+weight images (not KV import); "quad" = 4 Galaxies x 16 stages of 4x2;
  "engram4x1" = engram tables 4x1-sharded (static, 272 B rows: e4m3 + e8m0 repacked as BFP8). NO prefill->decode
  KV import exists yet (decode starts cold at position 0); the inter-stage page carries bf16 rows (kv 1024 B, index
  key 256 B). Fit with our plan: window KV bfp8 tiles matches only the SWA layers (CSA2 window is bf16 RM);
  compressed KV bfp4 rows and index keys bfp4 tiles are NOT directly consumable (decode: bf16 RM / e4m3+fp32 rows);
  a prefill->decode loader is needed either way. Bears on the cache decisions (owner).
- 16:10 OWNER: the decode finding may change what to work on next. The NEXT list of the 15:55 entry is SUSPENDED: the
  session after the compact absorbs the state only (read these entries, the run's files, the de-risk report) and takes
  NO action - no device runs, no agents, no orchestrator, no edits, no pushes - until the owner says what to do. Open
  questions for the owner's reconsideration: prefill KV at the reference's width (with a conversion at the
  prefill->decode hand-over, which needs a loader anyway) vs matching decode's formats (bf16 rows for CSA2 KV, e4m3 +
  fp32 rows for index keys); whether the compressed-KV 4-bit work continues; FIX.7's per-matmul numbers still to report.
- 18:05 ENGRAM 2x2 COMMITS TAKEN (rebased my 3 record commits on origin c3d41c620db, local). L.s56320 PASS (bfp8 rows,
  582 s). K.1 runner smoke FAIL twice, neither from the model: run 1 (resident tables in the runner, 17:00) slowed to
  ~2 min a step from 17:10 and died SIGBUS ~17:38, when user dlukic started a device job on this node
  (run_safe_pytest --profile deepseek_v3_d_p test_dispatch_sweep, 17:38:11); run 2 (DS41_ENGRAM_TABLE=random) waited on
  dlukic's CHIP_IN_USE lock until the 600 s no-progress stop. Cause class: infra (another user on our slurm node; our
  private TTPOOL_LEGACY_LOCK makes us invisible to their tools). Today's two earlier BUS_ADRERRs likely the same. Owner
  is coordinating with dlukic. The slowdown of run 1 before 17:38 is unexplained (resident in the runner?): the next
  Paris run with the default tables decides.
  OWNER: P.6 ENGRAM_ASYNC added (perf task before P.4; P.4 now deps P.6): measure lookup / pack / PCIe upload per
  engram layer first (upload never measured), then layer 14 off the critical path (second CQ + event), next-chunk
  prefetch only if the engine exposes later tokens. Owner assigned this engram task to the Galaxy session (tell the
  2x2). Chain ready, NOT launched (waits for dlukic off the node): /scratch/dnijemcevic/run_ds41_chain.sh = Paris
  (resident) -> if PASS the 5-position profile with resident tables (posprof/series_engram.sh, tag ENG) ->
  orchestrator --until P.6. P.6's wall bar (< 544) to be replaced by the ENG profile's position-0 wall before P.6 starts.
- 18:25 OWNER: P.6 has nothing to decide ("faster by pure logic, my input should not be needed"): overlap ALL engram
  host work (both layers' hash, lookup, pack and upload) with dispatch; begin_chunk returns at once, take_engram_rows
  waits per layer. The overseer accepts its A/B on the clean-win rule without asking; anything else (gate fail,
  accuracy worse, wall not better) is reported.
- 18:35 OWNER: after P.6 and the resident-engram profile -> 4-bit compressed KV. Queued FIX.8 HELD (CKV_FORMAT=bfp4_rows;
  gate as FIX.7: W.1, M.1, L.s10240, L.last, L.s56320, L.smoke, K.1; paths model + ttnn/ttnn/bringup). Scope: only the
  attention table goes bfp4 rows; served records stay bf16 holding the rounded values (contract + tt-d-gen unchanged);
  bf16 path kept behind the switch. Release after: the overseer's single-card bfp4/bfp8 packer rounding test (~10 device
  min) and the de-risk files (/scratch/dnijemcevic/ckv_derisk, report section 7 now cites the K3 artifact and PR #49811)
  moved into the repo; the packer result goes to FIX.8 as a note.

## 2026-10-28 18:55 HAND-OFF (owner: the current plan, in case of a switch to another machine)
Node bh-glx-110-c10u02 (slurm 158554), branch dnijemcevic/model_bringup. origin = c3d41c620db (the 2x2's engram commits
on top of 5b275cb7622). Local commits on top (records, P.6, FIX.8, the ckv_derisk files, overseer_tools): push on the
owner's word only; a move to another machine needs them pushed first.
NODE-LOCAL (/scratch is this node's disk): the run dir /scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/
galaxy_8x4-full/run1 (orchestrator.log, stamps, unit logs, verify/), the profile results and the launchers. Copies of the
launchers, the profile scripts and the TBLOCK baseline: ledgers/galaxy_8x4-full/overseer_tools/ (README there). On a new
node the stamps are gone: passed units re-run once.
State: orchestrator stopped (after FIX.7, --until FIX.7). The 2x2 engram commits taken (ENGRAM_TABLE=resident bfp8 tables
in host RAM ~209 GB built at process start outside pytest; under pytest ENGRAM_TABLE=random, F222; ENGRAM_ROWS=bfp8).
L.s56320 PASS on them (bfp8 rows, 582 s). K.1 runner smoke (Paris) NOT yet passed on them: two runs spoiled by another
user's device jobs (18:05 entry); the slowdown of run 1 from 17:10 (resident tables inside tt-d-gen's runner?) is open.
BLOCKER: user dlukic runs device jobs on this node from SSH (robo_test.py loop driven by `claude -n dsp-fb`, deepseek_v3
test_dispatch_sweep under tracy, every ~20 min). Slurm does not stop it (no pam_slurm_adopt, /dev/tenstorrent 0666) and
the locks do not meet (his stale /tmp/tt-device.lock vs our TTPOOL_LEGACY_LOCK). Owner is coordinating. Check before any
device run: `ps -u dlukic -o pid,lstart,cmd | grep -E "pytest|tracy|robo_test|run_safe" | grep -v grep` must be empty.
PLAN, in order (owner 2026-10-28; engrams BEFORE 4-bit):
 1. Paris (K.1#test_runner_smoke) with the default resident engram tables, ~13 min.
 2. Only if it passes: the 5-position profile of the current default (block MoE, fused head + tail, FABRIC_2D_TORUS_XY)
    WITH the engram tables resident (DS41_ENGRAM_TABLE=resident forced: pytest would pick random): one new 5120 chunk at
    0 / 51200 / 102400 / 204800 / 512000, device time (real-time profiler) + F196/F205 warm wall, ~60-70 min, tag ENG
    (posprof/series_engram.sh). Then publish a page like https://claude.ai/artifact/HRfMf31S4781YkU2M8NDUf (per-section
    tables at 0k and 500k, device and host time vs context), compared with the TBLOCK baseline.
    Steps 1-2-3 are one launcher: /scratch/dnijemcevic/run_ds41_chain.sh (tmux).
 3. P.6 ENGRAM_ASYNC (perf task, before P.4; P.4 deps P.6): all engram host work (both layers' hash, lookup, pack AND
    upload) off the main thread; begin_chunk returns at once, take_engram_rows(i) waits just before layer i's engram op;
    upload via a 2nd command queue + event if the mesh has one. Owner: nothing to decide - the overseer accepts its A/B
    as a clean win (gate PASS, accuracy unchanged, wall better, device not worse); only a failure goes to the owner.
    Before it starts: set its chunk_wall_ms bar (now < 544) to the ENG profile's position-0 wall. The PCIe upload time
    was never measured (the 2x2's 7.6 ms/layer is ids -> TILE bytes on the host); P.6 reports it.
 4. 4-bit compressed KV: (a) the overseer's single-card packer rounding test for bfloat4_b AND bfloat8_b (~10 device min;
    recipe in the 14:35-14:55 entries; PR #49811 says the Blackhole packer truncates fp8 - unknown for bfp formats; FIX.7's
    bfp8 GEMM inputs depend on it too); (b) release FIX.8 (queued HELD) with the result as a note. FIX.8: only the
    attention table (AttnTable, read by sparse_sdpa) as bfloat4_b rows of 288 B; the served records stay bf16 rows with
    the rounded values (serving contract + tt-d-gen unchanged; load_served repacks exactly); write = fork op, read =
    sparse_sdpa fork ROW_BFP4 with the window as a 2nd tensor; CKV_FORMAT=bf16|bfp4_rows, bf16 kept; gate as FIX.7.
    Design: bringup/ckv_derisk/CKV_4BIT_DERISK.md (now in the repo; header lists the owner's design change; section 7
    cites the K3 fp8-wire artifact and PR #49811). ~4-6 agent-hours + ~1.5 device-hours.
 5. Then: window KV to the reference's width on the same two-tensor fork (fp8 per 32 power-of-two: the K3 cast ops
    nearly as-is, or bfp8); index keys + index query to fp4 width; serving-contract text; X.1 re-baseline; P.4 / P.5.
Reference material from today: the precision table reference vs our prefill vs tt-blaze decode
https://claude.ai/artifact/QP9f8tn3xDrojyyk8fMLUs (wider than the reference: the three KV caches, the index query, the
shared expert's x/h; narrower: bfp8/bfp4 re-rounded weights, bf16 cross-chip partial sums, RoPE tables, sink, HiFi2
SDPA). Decode (tt-blaze llong/dsv41-reload-quad-engram4x1) has no prefill KV import; a loader converts our formats
losslessly (index-key scale must be a power of two); the compressor's partial group is not handed over (64-aligned
hand-over, contract :59-61, :123-124).
Open for the owner: dlukic / node access; push; the in-RAM engram slowdown in the runner if Paris shows it again.
- 19:05 OWNER: the 4-bit compressed KV is DEVELOPED ON THE 2x2 (packer rounding test bfp4 + bfp8, the write fork op, the
  sparse_sdpa ROW_BFP4 fork, single-card tests 0-5, the model change behind CKV_FORMAT, the 2x2 ladder) and pushed; the
  Galaxy's FIX.8 (still HELD) becomes the full-size integration gate (hold: the 2x2's push + P.6 done; its text says so).
  FILE OWNERSHIP while both run: the 2x2 owns tt/compressor.py, tt/attention.py, tt/sparse_attention.py, the new forks in
  ttnn/ttnn/bringup, the CKV_FORMAT block of tt/settings.py and bringup/ckv_derisk/; the Galaxy (P.6) owns tt/engram.py,
  tt/engram_kv.py, begin_chunk / take_engram_rows in tt/model.py and the ENGRAM_ASYNC block of tt/settings.py (the
  engram code moves back from the 2x2 to the Galaxy for P.6, owner). Anything else shared: coordinate through the owner;
  ids: 2x2 DS41-150..169 / F220..F229 (as ENGRAM_HOST_PATH.md 4a), Galaxy DS41-130..149 / F207..F219. Pushed.
- 19:20 OWNER: the 4-bit KV runs on the 2x2 (pushes to dnijemcevic/model_bringup_bfp4kv). Galaxy queue after P.6, all
  independent of the 4-bit work and of the 2x2's files: FIX.9 (HELD until P.6) DS41-132 shared expert x / h as bfp8 in
  the fused head (precision: above the reference today), then P.4 / P.5 (host time), then the index query to fp4 width
  (indexer.py only; the index-key cache waits for FIX.8). FIX.8 (integration of the 2x2's work) takes priority whenever
  the bfp4kv branch is ready. No-device side work: the precision audit hook (rest of F208), F209.
- 19:25 OWNER: work autonomously; PUSH after each major milestone (Paris pass, profile published, P.6 accepted, each fix
  job) together with a hand-off-quality entry here, in case something interrupts. Replaces "push only on my word" for
  this branch and this plan.

## 2026-10-28 19:35 MILESTONE + HAND-OFF: the 2x2 engram commits pass on the Galaxy (rows), the in-RAM tables do not
- L.s56320 PASS (16:50, 582 s) and K.1 runner smoke = Paris through tt-d-gen PASS (19:12, 968 s) with ENGRAM_ROWS=bfp8,
  tables read from disk (DS41_ENGRAM_TABLE=random). The bfp8 rows are correct end to end at full size.
- ENGRAM_TABLE=resident (the default outside pytest) breaks the runner on this host: the 18:39 run (dlukic NOT on the
  device) ran ~2 min a layer in the first forward (py-spy: torch.cat in se_engine3 upload, ~59 cores user; 496 GB read
  from disk; swap full; THP compact_stall 611k) and was stopped at 19:11; the 17:00 run (same pattern) died SIGBUS -
  dlukic's 17:38 job may not have been its cause after all. The Galaxy spec's owner rules still say "never hold the
  203 GB of tables in host RAM" / "Host RAM is shared (503 GB)": for the owner (wording vs the owner's RAM decision).
- QUEUED FIX.10 (gate L.s56320 + K.1): make resident coexist with the full model on this host (root cause by
  measurement, fix at the cause, rows bit-identical). Runs NOW (orchestrator --until FIX.10). Plan after it: the resident
  5-position profile (posprof/series_engram.sh) -> P.6 (bar = the profile's position-0 wall) -> FIX.9 -> P.4 / P.5 ->
  index query fp4; FIX.8 (2x2's 4-bit KV, branch dnijemcevic/model_bringup_bfp4kv) jumps in when ready.
- Launcher: /scratch/dnijemcevic/run_ds41_chain2.sh (copy in overseer_tools/).
- 19:45 OWNER: ignore the 4-bit compressed-KV work for now (the 2x2 proposed something very slow). FIX.8 stays HELD and is
  not integrated; do not watch or pull dnijemcevic/model_bringup_bfp4kv. Galaxy plan unchanged otherwise: FIX.10 ->
  resident profile -> P.6 -> FIX.9 -> P.4 / P.5 -> index query fp4.

## 2026-10-28 20:55 MILESTONE + HAND-OFF: FIX.10 PASS f17604c27c2 (resident engram tables fit the Galaxy host)
- Root causes (by measurement): DS41-133 numba's omp layer raised the shared OpenMP max-threads ICV to 64 at its first
  launch, torch's pool followed (32 -> 64) and every host torch op ran 5-20x slower (fused head weight layout 112 s a
  layer); fix: tt/engram_bfp8.py _run restores torch's thread count around each launch. DS41-134 MADV_HUGEPAGE on 200 GB
  compacted synchronously (THP defrag 'madvise') and direct reclaim evicted ~95 GB of the weight cache; fix:
  ENGRAM_HUGEPAGES=off default (build 74 s, gather 8.5 vs 6.3 ms a layer) + DONTNEED of the read table bytes. New
  ENGRAM_BUILD=background: the build overlaps the weight load on a thread (nogil numba). First forward 3.5 s a layer
  (112 resident before, ~6 random); runner load 3m21s. Gate: L.s56320 + all K.1 units (Paris) PASS with resident.
  Overseer review: diff honest (no test or threshold touched; rows bit-identical; tasks.yaml = the FIX.10 insertion).
- The 17:00 SIGBUS: FIX.7 also saw a tt-d-gen "Non-existent physical address" SIGBUS with random tables; three
  transient bus errors today, cause open (not clearly dlukic, not the tables). A fourth: ask the owner for a board reset.
- NOW: the resident 5-position profile (tmux ds41-chain, /scratch/dnijemcevic/run_ds41_chain2.sh -> posprof/series_engram.sh,
  tag ENG, ~60-70 min, results /scratch/dnijemcevic/posprof/results/ENG*). Then: publish the profile page; set P.6's
  chunk_wall_ms bar to the ENG position-0 wall; orchestrator --until P.6 (accept on a clean win); release FIX.9 -> P.4 /
  P.5 -> index query fp4. 4-bit KV (FIX.8) set aside (owner 19:45).
- 22:00 OWNER (order): finish the resident profile + page -> 4-bit KV integration (merge origin/dnijemcevic/model_bringup_bfp4kv
  @ 9697878dfb0 "READY FOR GALAXY": CKV_FORMAT default bfp4_rows, 320 B rows, ttnn.bringup.ckv_bfp4 + sparse_sdpa ROW_BFP4,
  F223; 2x2 rung `last` ladder + profile pass; ttnn rebuild needed; FIX.8 released as the full-size gate) -> P.6 (bar
  from the post-FIX.8 wall) -> FIX.9 -> P.4 / P.5 -> index query fp4. Do not touch the tree until the profile series
  ends (each position is a fresh process loading the code).

## 2026-10-28 22:20 MILESTONE + HAND-OFF: resident-engram 5-position profile done (f17604c27c2)
Page: https://claude.ai/artifact/EeioFZEY2jHgdzPRYXbaxc . Results: overseer_tools/profile_eng_20261028/ (+ /scratch/dnijemcevic/posprof/results/ENG*).
| context | device before (TBLOCK) | device now | wall before | wall now |
| 0       | 219.5 | 220.8 | 615.0 | 432.2 |
| 51.2k   | 233.9 | 229.1 | 640.8 | 463.7 |
| 102.4k  | 237.5 | 237.4 | 633.8 | 464.8 |
| 204.8k  | 249.1 | 248.2 | 644.1 | 469.7 |
| 512k    | 284.4 | 288.2 | 645.8 | 464.0 |
Wall -28..-30 % (P.1 + P.2 + the engram host path); device flat (FIX.7 casts +88 op calls, ~+1 ms). Still host-bound.
NEXT (owner order 22:00): merge origin/dnijemcevic/model_bringup_bfp4kv @ 9697878dfb0 -> ttnn rebuild -> release FIX.8
(orchestrator --until FIX.8) -> P.6 (set its chunk_wall_ms bar from FIX.8's wall) -> FIX.9 -> P.4 / P.5 -> index query fp4.

## 2026-10-29 01:35 MILESTONE + HAND-OFF: FIX.8 PASS c258011c8a5 (4-bit compressed KV at full size)
- The 2x2's CKV_FORMAT=bfp4_rows (merged bf6c8b497a6 from model_bringup_bfp4kv @ 9697878dfb0) runs on the 8x4 with NO
  code change; all 11 units PASS (W.1, M.1, L.s10240, L.last, L.s56320, L.smoke, K.1 incl. Paris through tt-d-gen).
  final hidden s56320 0.1889 -> 0.1961 (limit 0.2586), top5 1.0, compressed-KV state worst 0.1547 -> 0.1973 (0.3294),
  contract ckv PCC 0.9847 -> 0.9767 (>= 0.937); closest: W.1 layer 24 0.0318 / 0.0336.
- Attention tables at 1,044,480 tokens: 2570.5 -> 817.4 MiB a chip (-1.71 GiB). sparse_sdpa +2.5 % (0k) / +4.2 % (512k);
  device +0.2..0.6 %; host +~5 ms a chunk = ckv_bfp4 rebuilding its ProgramDescriptor every call (DS41-135) -> queued
  perf task P.7 CKV_BFP4_PREPARED (after P.6, before P.4). Warm walls bf16 / bfp4: 0k 419.7 / 441.6 and 412.5 / 426.3,
  512k 451.3 / 472.0 (noisy, spread 30-130 ms).
- Framework: F210 (gate commits leave dashboard/*.html out; 77c60cd4291) and F211 (commit-only ignores the framework's
  own records under a task's paths; 50720ab8ba6) - FIX.8's passed gate was refused twice; committed without a re-run.
  Selftests 647 pass (the 1 failure = the known xing40 fixed threshold).
- NEXT: P.6 ENGRAM_ASYNC (bar chunk_wall_ms < 434 = FIX.8's bfp4 walls at 0k), orchestrator --until P.6, accept on a
  clean win; then release FIX.9 (shared expert bfp8) -> P.7 -> P.4 -> P.5 -> index query fp4.

## 2026-10-29 02:55 MILESTONE + HAND-OFF: P.6 ENGRAM_ASYNC accepted + committed 6e20afb4b50 (clean win)
- A/B (quick, last rung): accuracy identical to the digit; warm wall 421.0 on / 462.6 off (-41.6 ms, -9 %), spread 26 vs
  62 ms; gate: chunk_wall_ms 402.1 (< 434), device_ms_total 219.78 (<= 230). begin_chunk 48.9 -> 0.6 ms; ~42 of 49 ms of
  engram host work hidden (layer 1 still waits ~6 ms); PCIe upload ~2.7 ms a layer (first measurement); the old
  upload held the GIL ~35 ms a layer (DS41-136, fixed with a reused staging buffer). No 2nd command queue (one CQ in
  the harness and tt-d-gen). Overseer accepted under the clean-win rule.
- Framework tonight: F210 (dashboards out of gate commits; second try 6801b494a86: explicit file list, the exclude
  pathspec skipped explicitly named new files), F211 (commit-only ignores framework records), F212 (device_ms_* with
  wall-only profile segments refused; P.4-P.7 commands fixed), F213 (a released fix is not placed before a perf pick
  whose A/B is measured; FIX.9 was, aborted within 30 s), F214 PENDING (P.6's recorded accept vanished from
  state.json; workaround: gate + commit by hand).
- NEXT: orchestrator --until P.5: FIX.9 (shared expert x/h bfp8, placed, attempt 3) -> P.7 CKV_BFP4_PREPARED -> P.4 ->
  P.5 (perf picks stop for the overseer's A/B decision; clean wins accepted) -> index query fp4 (to queue).
- 05:30 OWNER: a fresh 0k-512k profile on the current code after FIX.9 (tag CUR; posprof/series_cur.sh, ~65 min).
  Orchestrator PAUSE requested (stops after FIX.9); tmux ds41-chain3 = /scratch/dnijemcevic/run_ds41_chain3.sh waits
  for it, checks FIX.9 PASS, runs the profile, then resumes --until P.5 (P.7 -> P.4 -> P.5). Then publish the page
  (update https://claude.ai/artifact/EeioFZEY2jHgdzPRYXbaxc or a new one) comparing ENG (22:10) and CUR.

## 2026-10-29 06:50 MILESTONE + HAND-OFF: FIX.9 PASS d1571ee35ac (shared expert x / h bfp8, DS41-132 closed)
- se_engine3 x_dtype / h_dtype (TILE_BYTES from the dtype), x typecast once into a persistent bfp8 buffer (the router
  keeps bf16 x), h packed precise (SE_PACK_PRECISE=on). moe_block defaults unchanged (bf16, pack_precise False) for
  other users / the 2x2 ops head. Head 446 -> 451 us + 7 us typecast a layer; SE rel vs native 0.0116 -> 0.0165 (the
  reference's own fp8 rounding). All gate units PASS incl. test_adapter_acks (layer 20 nope PCC within its limit) and
  Paris through tt-d-gen. Overseer reviewed the diff.
- DS41-137 (new): a packer's bfp8 output without bfp8_pack_precise rounds twice (ties away both times): +bias. Likely
  every FIX.7 bfp8 activation written by a matmul packer; candidate follow-up (precise packing, same speed); the
  ComputeKernel config hash ignores the flag (a source define works around it; the 2x2 found the same).
- NOW: tmux ds41-chain3: fresh 0k-512k profile CUR on d1571ee35ac (~65 min, started 06:45), then resume --until P.5
  (P.7 CKV_BFP4_PREPARED -> P.4 -> P.5). Then publish the profile page (ENG vs CUR).
- 07:40 SP=32 STUDY (read-only agent) + OWNER: deployment box is the 8x4. Attention today is TP=4 x SP=8 in the
  projections only (sparse_sdpa / indexer already 160 rows x 64 heads a chip); the two 10.5 MB head<->seq all_to_all
  cost 22 ms a chunk (more than sparse_sdpa), plus q_a RS+AG and wo_b RS of bf16 partials and 4x redundant norm/RoPE;
  ~1525 of ~2030 op calls a chunk are attention's. DeepSeek-V3 sparse MLA keeps top-k local and reshards heads->seq only
  when heads/tp < 32. OWNER ORDER: P.7 -> P.8 C INDEXER_TOPK_LOCAL -> P.9 B ATTN_SP32_PROJ (wq_b/wo_a/wo_b full rows,
  +3.5 GiB a chip) -> P.10 A ATTN_SP32 (CONDITIONAL on P.9's host savings, else rejected unrun) -> P.4 -> P.5 (both
  re-checked against the new layout first). B and A gates add L.s56320 + K.1 (layout change). Estimates: attention
  73 -> ~45 ms device a chunk, ~270 op calls fewer, wall -30..-50 ms; risks: untuned M=160 matmuls, small all_to_all
  host cost, +3.8 GiB weights a chip (fits the 8x4 by estimate). The 0.75 ms/layer net is an estimate.
