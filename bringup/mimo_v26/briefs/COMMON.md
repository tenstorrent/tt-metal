# Common brief (all MiMo-V2.6-Flash track agents) — read first

Model: MiMo-V2.6-Flash-RL on Blackhole Galaxy 4x8 (32 chips). Read, in order: `bringup/ONBOARDING.md`, `.claude/skills/tt-moe-bringup/SKILL.md`,
the skill your brief names, `bringup/GUIDE_dsv4_lessons.md`, `/mnt/tt-data/ssinghal/bringup_study/mimo_v2_6.md` and `bringup_study/mimo/` (config, HF modeling code
`modeling_mimo_v2.py`; NOTE `sgl.py` in the study dir is a 404 stub, no sglang source is on disk; the checkpoint is at /mnt/tt-data/MiMo-V2.6-Flash-RL). The study is UNVERIFIED notes: check every MiMo fact you rely on against config/modeling code and record the check.

Rules
- Work ONLY in your worktree (`/mnt/tt-data/ssinghal/mimo/wt/<track>`, branch `ssinghal/mimo-<track>`). `pwd` before any build/cp/rm. Never write to
  `/mnt/tt-data/ssinghal/tests/tt-metal` (DeepSeek repo, jobs run from it) or to the integration clone `/mnt/tt-data/ssinghal/mimo/tt-metal`.
  Read DSV4 code with `git show old/ssinghal/dsv4p1:<path>` (ref fetched into the clone) or by reading files in the old repo.
- Device jobs ONLY via `models/common/bringup_kit/launch.py` (pinned commit, clean tree, flock, hangwatch >= 45 min for builds, >= 20 for compile). One device job per host,
  only on the host(s) named in your brief. Check `host_health.py <host>` first. Never use .31/.42; never kill/reset anything that is not your own pid.
- Log every measurement, decision, failure, retraction with `log_tool.py add` (commit first; tree must be clean). Numbers need commit+host+log path+shape.
  Use `--status inferred` for anything not measured at that commit. Log dir: `/mnt/tt-data/ssinghal/mimo-logs/<track>_<what>.log`.
  Do not quote numbers from stale/superseded/inferred entries. Real text prompts only (no random tokens) for any accuracy or routing claim.
- Decisions already made (do not re-ask): routed experts bfp8, KV replication OK, MTP last, 64k first milestone, no full-model CPU run, bfp8 attention weights default.
- ASK the coordinator (write it in your final report / stop) before: lowering precision (bfp4, fp8 KV), skipping/weakening a gate, deleting data, C++ rebuild in a shared build
  (use a private build dir and ask), installing packages, pushing, scope changes.
- STOP and report when: a gate fails twice for the same cause, host faulty/hang with the same signature twice, blocked on another track, or 3 h without a measured result.
- Skills are unvalidated. If a skill/guide/kit tool misleads you or lacks something, fix it in the SAME commit as the discovery, log it (track `skill-defect`), and update that
  skill's "Last validated against commit" line only for what you exercised. Keep a list of defects in your final report.
- Commit with long timeouts (NFS hooks are slow); message ends with the Co-Authored-By line given by the harness. Do not push.
- Final report (handoff): worktree, branch, commit ids, files changed, evidence (LOG ids, log paths, exact repro commands), what did not work, what is unverified, composition/conflicts
  with other tracks, skill defects found. Also write a `handoff` LOG entry.
- Early cheap deliverable within 3 h (unit/4-layer level) before any long run.
