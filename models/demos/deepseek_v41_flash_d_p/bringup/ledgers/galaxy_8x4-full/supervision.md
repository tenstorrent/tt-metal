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
