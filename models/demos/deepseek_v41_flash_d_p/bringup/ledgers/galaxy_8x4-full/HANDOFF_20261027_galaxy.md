# HAND-OFF 2026-10-27 ~20:50 UTC: DeepSeek-V4.1 galaxy_8x4-full, node bh-glx-120-b06u08 released

Branch: **dnijemcevic/model_bringup_galaxy** (forked from the local dnijemcevic/model_bringup run branch; pushed at this
hand-off). It is the owner's rebased tip 7c06953a55c + this session's commits + a WIP commit of the K.1 agent's
uncommitted contract edits. **origin/dnijemcevic/model_bringup has moved on to f81cb90cf5a (the owner's MoE work, 10
commits) and does NOT contain this session's work**: reconcile before continuing (see "MoE port" below).

Owner rules: the 2026-10-26 15:23 and 2026-10-27 11:20 HAND-OFF lists in supervision.md still bind (autonomy, no
shortcuts / no padding, framework fixes at the cause with a selftest + dev/BREADCRUMBS entry, push only on the owner's
word, never call a run warm without the program-cache check). Owner today: **perf tasks (X.*) are OFF until K.1's
Paris works end to end**; the launcher resumes with `--until K.1`.

## Where the run is

PASS on the rebased code (verified 18:14-18:50 with core.verify, numbers identical to pre-rebase):
CK.1, G.s10240, G.s56320, B.1, W.1, FIX.1-4, **FIX.5** (be9736f2900), M.1 (40 layers), L.s10240, L.last, L.s56320
(top5 1.0, err_final_hidden 0.189), L.smoke (device: "...Capital of France = Paris").

**K.1: in progress, orchestrator PAUSED before contract attempt 6** (PAUSE file in runs/galaxy_8x4-full/run1).
Units: test_contract PASS, test_kv_layout PASS (kept stamps); test_adapter_acks FAIL (limit question, below);
test_runner_contract not run in the gate; test_runner_smoke FAIL (decode hang, below).
Then (not started): X.1-X.3 (off), O.1, Z.1, F.1 (strict fork suite, added today).

## What works end to end now

tt-d-gen (latest main a9701ce6675, built here at /scratch/dnijemcevic/tt-d-gen, node-local) drives the real prefill
runner on the Paris prompt over all 40 layers: attached 12.3 s, ADMITTED slot 0, PREFILL_DONE 160 tokens, no errors;
the KV of all 40 layers reads back through the exported table (192 records). This is the first e2e serving prefill.

## Current bug: the stand-in decode hangs (no Paris yet through the runner path)

test_runner_smoke step 3 (the test plays decode: fresh mesh, device model with all 40 layers, load_served(migrated
KV), tail [128, 160) recomputed as one 5120-token chunk at start=128, greedy) hangs on its FIRST tail call:
"Timeout detected ... device timeout in fetch queue wait" (run 2, 20:28). tt-triage: the stuck kernels are
flat_routed_expert's (se3_compute, se6_dcompute, se5_recv, se_yrm) + fabric erisc routers. Box test right after: PASS.
L.smoke (same prompt, start 0, no migrated state) passes with the same code; difference = tail at start 128 with 32
real tokens (most experts get 0 tokens) and state from load_served.
Diagnostic run 3 (DS41_EXPERTS_PROGRAM_KEY=addresses, i.e. FIX.5's patch path off; outside the gate, results in
/scratch/.../smoke_diag_results, log smoke_diag.log): **RESULT: see the last line of this file.**
Next: if it hangs with addresses too -> the fork on near-empty experts at a non-zero start: K.1 agent with
ttnn-expert-debugger (note: reproduce standalone with flat_routed_expert_with_program on 32 tokens / start 128 /
counts mostly 0, see tests/unit and FIX.2's first-call hang history); if it passes -> FIX.5's patch path is at fault:
fix job on FIX.5 (re-run W.1 + the smoke).

## Open decision: test_adapter_acks limit (owner said: Paris first, then B)

window_kv PCC at deep step layers over all positions (L20 0.981 / limit 0.990; L21, L24 similar) misses a limit
derived from the reference drift on the LAST 128 positions only (drift-s10240.json contract_state_window_kv_L* is
the final ring). Accuracy is fine (ladder err per layer passes over all positions). Owner-recommended **B**: compare
window KV for every layer on the drift's positions (last 128 = the final ring); not yet implemented (K.1 agent, note).

## This session's changes (all on the galaxy branch)

- 5308dbb660d spec serving: require_engine true, server_repo /scratch/$USER/tt-d-gen, answer 7 (smoke runs here);
  intake re-approved; F.1 appended; F200 orchestrator: a runner-test gate waits for a missing tt-d-gen build
  (no gate, no agent; independent tasks go on). Selftest test_engine_wait.py.
- be9736f2900 FIX.5 (agent): flat_routed_expert_with_program patch_addresses (address-free program-cache key),
  EXPERTS_PROGRAM_KEY=patch shipped; bit-identical 24/24; warm after 1 run; warm wall 999.8 ms (pre-rebase 860 ms
  with XG_PINNED, other base: not comparable, uninvestigated). SEED-BOX RE-VERIFY += FIX.5.
- 5161e706d64 dgen_engine.server_repo expands $USER; b813d15e7fe the driver names device_prefill_pipeline's args
  (tt-d-gen 1cf343887 dropped layers_per_chunk); ca3041407a8 tt-d-gen attaches to the runner's completion ring
  /tt_prefill_layer_completions_<id> (dgen_engine.ack_shm_name), not /tt_prefill_layer_acks_<id>.
  NOTE: both sides use breadcrumb id **F200** (origin's f81cb90 has another F200): renumber ours at the reconcile.
- WIP commit (uncommitted at hand-off, K.1 attempts 3/5 + one overseer line): tt/runners/kv_contract.golden_rows
  (non-step layers: final-state ring; this golden dumps per-chunk only the 8 step layers), tests/bringup/contract/*
  (server_rules geometry from the spec: s4096/2048/4096 on the 2x2 seed, s10240/5120/10240 on 8x4; the other contract
  tests for the same golden gap), test_runner_contract plan uses D.ack_shm_name (overseer). K.1's paths do not yet
  allow the contract tests (the gate flagged them): widen K.1's paths in tasks.yaml before resuming.
  SEED-BOX RE-VERIFY += server_rules geometry, kv_contract golden_rows.

## MoE port (owner: after Paris + B)

Report: moe_port_report.md (this directory; correction at its top: the block OP supports torus, only
tt/moe_block_adapter.py:159 refuses it). Plan (owner): reconcile origin f81cb90 with this branch (9 textual conflicts,
mostly keep both kwargs; renumber F200), full rebuild; build the Marko-layout expert cache once outside a gate
(~285 GB, ~75 min cold; block mode loads routed weights through the same cache, D6), keep the legacy cache; flat on
FABRIC_2D: W.1 / M.1 / L.*; block (DS41_MOE_IMPL=block) on FABRIC_2D: same; then the 5-position profile (context
0 / 50k / 100k / 200k / 500k; device + warm host, F196/F197) for both; torus only after FABRIC_2D works (drop the
adapter's D10 assert with the owner's word).

## What does not work / pending

1. Runner smoke decode hang (above). 2. test_adapter_acks limit (B). 3. test_runner_contract not yet run in a gate.
4. Torus: FABRIC_2D_TORUS_XY hangs in flat_routed_expert_with_program (unchanged). 5. Perf X.* off. 6. 2x2 QuietBox
re-verify of FIX.1, FIX.2, W.1 fabric change, FIX.5, the contract geometry: owner's call (bab52a1 records the 2x2
passing after the rebase). 7. Framework: seeded runs carry the seed's contract tests unchecked (box literals);
runner tests load the model twice (~4.5 min each; only routed experts are cached); selftest UMD chip lock; /tmp
dirty-card markers. 8. xing40's serve/server.py still names the old ack channel.

## What lives where

- Repo + /scratch are node-local. On /data: goldens (golden_full/galaxy_8x4), the git mirror
  /data/dnijemcevic/tt-metal-ds41.git, **/data/dnijemcevic/node_handoff_20261027b/** (this hand-off: orchestrator.log,
  logs/ incl. K.1.units.log with the hang + triage lines, briefs, verify/, K.1 / FIX.5 agent transcripts, runner smoke
  run 2 dir, overseer/ scripts + logs, tt-d-gen build README), and node_handoff_20261027/ (the previous one).
- New Galaxy: checkout dnijemcevic/model_bringup_galaxy, build (./build_metal.sh, Release; python_env as is), hf_full
  checkpoint via hf_dl.py (~501 GB, all 48 shards), copy node_handoff_20261027b/{orchestrator.log,logs,briefs,state*}
  back under /scratch/$USER/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1 (state, attempt numbers; the
  PAUSE file is not needed), dgen-build agent REPO=/scratch/$USER/tt-d-gen (latest), expert cache rebuilds itself on
  first load (~75 min). Then box test, then `orchestrator resume --until K.1`.

## Diagnostic result
Run 3 (DS41_EXPERTS_PROGRAM_KEY=addresses, FIX.5's patch path OFF): the tt-d-gen prefill again PASSED (prefill done
in 495 s incl. the runner's 40-layer load, KV read 192 records), and the stand-in decode HUNG again on its first tail
call (Timeout detected 20:46:15). **FIX.5 is not the cause.** The hang is flat_routed_expert on the decode tail input
(32 real tokens at start 128, mostly-empty experts, migrated state). Run 3's tt-triage dump was KILLED by the overseer before it was written (overseer error); the Inspector data Metal
saved at the hang survives: node_handoff_20261027b/inspector_run3_hang_2046/programs_log.yaml (programs enqueued up to
the hang). The hang reproduced 2/2: a re-run (~15 min, gate --unit K.1#test_runner_smoke) regenerates the dump;
run 2's stuck-kernel list is above and in node_handoff_20261027b/logs/K.1.units.log. Log: node_handoff_20261027b/
overseer/smoke_diag.log.

**Next on the new machine (in order):** (1) reproduce the hang without the runner: the device model alone, a 5120-token
chunk with 32 real tokens at start=128 after load_served / or after a real prefill of 128 tokens (L.smoke's path with
start 128), to see whether migrated state matters; (2) standalone flat_routed_expert_with_program with that call's
counts (most experts 0); (3) fix in the fork (K.1 agent + ttnn-expert-debugger, or a fix job), then the runner smoke
(`gate K.1 --unit 'K.1#test_runner_smoke'`, ~15 min) -> Paris; (4) B for test_adapter_acks; (5) K.1 commit; (6) the
MoE port; (7) the 5-position profile.
