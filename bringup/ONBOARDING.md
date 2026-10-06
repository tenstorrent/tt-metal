# Onboarding: MiMo-V2.6-Flash bring-up on Blackhole Galaxy

Read this first, then the skills in `.claude/skills/` and `bringup/GUIDE_dsv4_lessons.md`.
Written 2026-10-06 by the DeepSeek-V4.1-Flash (DSV4) bring-up lead. Anything here that names a number
is tied to a commit; re-measure before relying on it.

## 1. Goal

Bring up **MiMo-V2.6-Flash-RL** (HF `XiaomiMiMo/MiMo-V2.6-Flash-RL`; the OpenRouter `xiaomi/mimo-v2.6-flash` is the same model)
on a Blackhole Galaxy (4x8 mesh, 32 chips), autonomously, end to end: prefill, decode, paged KV, MoE, 64k context milestone,
MTP speculative decode last. Second goal: make the process repeatable. The skills and the bring-up log are the product
as much as the model; fix them whenever they mislead you.

Owner: ssinghal@tenstorrent.com. Use they/them for people unless told otherwise.

## 2. Where things are

| What | Where |
|---|---|
| This area (new clone, branch `ssinghal/mimo`) | `/mnt/tt-data/ssinghal/mimo/tt-metal` |
| DSV4 reference implementation (read-only for you) | `/mnt/tt-data/ssinghal/tests/tt-metal`, branch `ssinghal/dsv4p1`, package `models/demos/blackhole/deepseek_v41_flash` |
| Spec decode work (not yet merged) | same repo, branch `ssinghal/dsv4p1-spec` |
| MiMo study (config, modeling code, DFlash drafter, notes) | `/mnt/tt-data/ssinghal/bringup_study/mimo_v2_6.md` and `bringup_study/mimo/` |
| MLA-style SDPA probe report (unequal head dims) | `/mnt/tt-data/ssinghal/bringup_study/mimo_sdpa_probe.md` (may still be in progress) |
| DSV4 prefill research notes, agent brief | `/mnt/tt-data/ssinghal/prefill_study/` |
| DSV4 logs | `/mnt/tt-data/ssinghal/dsv4-logs/` |
| Build slots (limit concurrent model loads) | `/mnt/tt-data/ssinghal/buildslots` |
| Skills, guide | `.claude/skills/*/SKILL.md`, `bringup/GUIDE_dsv4_lessons.md` (force-added; `.claude/` is gitignored) |

IMPORTANT: this clone's `main` does NOT contain the DSV4 package or the later one-copy ring MoE ops. To read DSV4 code, open
the old repo. To reuse a commit: `git fetch /mnt/tt-data/ssinghal/tests/tt-metal <branch>:refs/remotes/old/<branch>`
and cherry-pick or copy files. Never edit the old repo's checkout; other agents run device jobs from it.
This clone has `build_Release` (with `build` symlink) and `python_env`; check the binaries match the sources before running
(`git log -1`, build timestamp) and rebuild in place only if you change C++.

## 3. What the DSV4 bring-up produced (what you can reuse)

- DSV4: 40 layers, TP=8 on columns, EP=32, Engram on host, traced chunked prefill, paged KV, DSpark spec decode. All of it works
  on `ssinghal/dsv4p1`. Prefill about 4.3k to 6.4k tok/s total, decode 10 to 23 tok/s/user depending on batch (B=4 to 128) at commit `35b7afd`.
- **Unified prefill MoE** (`ttnn.experimental.deepseek_prefill.unified_routed_expert_moe`: dispatch, expert FFN, combine,
  `post_combine_reduce`) is 1.7x to 2x faster than `moe_compute` in prefill. "One-copy ring" mode reads the decode `moe_compute`
  ring weights in place on an 8-column grid, so there is a single copy of expert weights in DRAM. It needs 8 live DRAM banks
  and tile counts that suit the ring layout; DSV4 fits (9 gate/up, 20 down tiles per core). MiMo experts differ in size, so
  re-derive the tile counts and keep the fallback (separate copy) available. Policy: `tt/uni_policy.py` on dsv4p1.
- **Chunk rule** (`tt/chunk_rule.py`): 2048 tokens per mesh row for U<=16 users per row. 8192 ran out of DRAM at 64k.
- **Model.reconfigure**: one build serves many batch sizes (release traces, buffers, CCL semaphores, program cache, in that order).
- **Build slots + stagger**: each DSV4 build read about 640 GB over NFS (60 to 90 min). `tt/build_slots.py` is git-ignored in the old
  repo; copy it if you need it. MiMo is 172.9 GB, so loads are about 4x cheaper, still stagger them.
- **Decode spec**: verify rows per mesh row are U x (1+k); mHC-style fast paths only exist for T=4/16/32 in DSV4 (not relevant to MiMo
  directly, but expect row-count cliffs). Compile all runners before capturing any trace; one scenario per process.
- The lessons with evidence and status labels are in `bringup/GUIDE_dsv4_lessons.md`. Treat rows marked open or early as hypotheses.

## 4. MiMo-V2.6-Flash facts (from the study; re-verify against the files before building on them)

- 48 layers: 9 global attention, 39 sliding-window (128, with sinks). Only layer 0 is dense; the rest are MoE.
- 256 routed experts, top-8, **no shared expert**. Sigmoid router with a selection-only bias (bias picks experts, does not weight them).
- Head dim 192 for QK, 128 for V (unequal), V scaled by 0.707, only the first 64 of 192 dims get RoPE.
- Checkpoint: experts packed fp4 with e8m0 scales, qkv fp8 with 128x128 block scales, 172.92 GB total.
- 3 MTP layers in the checkpoint (no HF reference code for them). A separate DFlash drafter exists in `bringup_study/mimo/dflash`.
- KV in bf16 is 5,760 B per token per user per chip (with replication).
- Mesh mapping on 32 chips: EP=32 gives 8 experts per chip.

## 5. Decisions already made (do not re-litigate)

- Expert weights **bfp8** (bfp4 lost accuracy on DSV4 and is off by default). Reproduces fp4 checkpoint values exactly.
- **KV replication is fine** when kv heads are fewer than TP.
- **MTP** is in scope, after plain decode and prefill are correct.
- **64k context** is the first milestone.
- All Galaxy hosts have enough RAM. Reference policy: **do not run the whole model on host for reference**. Prefill-only and decode-only
  reference slices (per layer or a few layers) are enough.
- **Unequal head dims (192/128): validate with MLA-style SDPA.** That is the probe in `mimo_sdpa_probe.md`; the unevenness question
  is whether padding V to 192 wastes too much compute.
- Skills are written for Claude agents; many small skills, one per task type. Bring-up logs are per model, every entry carries a commit id.
- Parallel bring-up: multiple agents, multiple worktrees. Skills are validated by actually running MiMo with them.

## 6. Hardware and lab rules (hard-won)

- Device jobs: run through the flock wrapper `/tmp/dsv4_dev.lock` convention (see `tt-lab-ops`). One job per host at a time.
- Good Galaxy hosts as of today: .30, .32 to .35, .40, .41, .43 to .48. **.31 is faulty** (wrong first tokens in mesh rows 0 and 1).
  **.42 hangs on its first command.** .36 to .39 unreachable. Re-check with the host health tool before trusting this list.
  DSV4 device tests were pinned to .45; ask the user which host MiMo uses.
- Hangwatch: the default 8 minutes killed healthy model builds. Use at least 45 minutes for builds.
- After a killed job, the device can be dirty (hang at startup). Reset, then re-run on a clean host before concluding anything.
- Do not use `pkill -f` with patterns that match your own shell; kill by pid.
- Do not state a number without commit, host, and log path. Do not reuse numbers across commits or hosts.
- Ask before: installing packages into shared envs, force-pushing, deleting data, touching other agents' worktrees.

## 7. Plan

Follow `.claude/skills/tt-moe-bringup/SKILL.md` (tracks, merge gates). Short version:

1. Track 0: read config and modeling code, write the log's first entry; verify every fact in section 4 against files.
2. Single-layer correctness on device vs a host reference slice: attention (incl. 192/128 dims, sinks, sliding window, partial RoPE),
   router (sigmoid + selection-only bias), dense layer 0, MoE layer. Use `tt-moe-correctness`.
3. Chained layers, then full 48-layer decode and traced chunked prefill, paged KV, 4k then 64k.
4. Performance: per-op profile, then the DSV4 playbook (unified prefill MoE, chunk rule, reconfigure).
5. MTP spec decode last.
Use `tt-multiagent` to split independent tracks across worktrees under `/mnt/tt-data/ssinghal/mimo/` or a sibling `wt/` dir.

## 7b. Open risks

- Unequal head dims may not map onto existing SDPA ops; the probe decides pad-to-192 vs a new path.
- 8 experts per chip with bfp8 and the ring layout: tile counts unverified for the one-copy mode.
- Expert skew in real data can be extreme (DSV4: one expert about 63x the mean); random-token tests hide it. Test with real text.
- MTP has no HF reference code: needs a reference built from the config and checkpoint tensor names.
- Skills are unvalidated against a real bring-up (all say "Last validated against commit: <none yet>"). Update that line as you validate.

## 8. How to work

- Every action on the bring-up gets a log entry with the commit id (`tt-bringup-log`). Stale entries: re-check commit before trusting.
- Commit small, on branch `ssinghal/mimo` or sub-branches per track. Commit messages end with the Claude co-author line.
- If a skill is wrong or missing, fix it in the same commit as the discovery and note it in the log.
- Report to the user in plain terms: what ran, on which commit and host, what the number is, what is unverified.
