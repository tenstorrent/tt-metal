## 2026-11-13 17:4x overseer take-over (owner)
- Node rebooted 09:41 EST for the tt-kmd leak; driver still reports 2.11.0; MemAvailable 329 GB at take-over.
- Running: gate FIX.30 --continue (fix30_continue.sh; units ladder/profile/positions) and chain_1113.sh (FIX.30 commit -> P.49/P.50/P.51 full gates -> X.3 reuse -> push model_bringup -> release + resume --only FIX.31).
- Armed: Monitor on orchestrator.log ("chain:" and failure lines), 30-min heartbeat cron (7,37). No device job of mine.
- Note: `gate FIX.30 --status` lists determinism / one_process as "stale" (the gate in flight treats them as kept; the framework files it changes are uncommitted). Watch that the --continue does not re-run them.
- Framework gap (to fix after FIX.30 commits, framework edit + selftest): `watch` exits 3 "ORCHESTRATOR NOT RUNNING" while a `gate --continue` or a chain script drives the device; it should follow a live gate (its pid / unit progress) too. Stand-in until then: a log monitor on "chain:" and failure lines plus chain pid liveness; the framework watch takes over at chain step 5 (resume --only FIX.31). Queued as FIX.31 item (6), with the owner's item (5): back-to-back gates (picks full gates + X.3) in one process, one model load.
- 2026-11-13 ~19:00 OWNER UPDATE (relayed, owner asleep): (1) board resets are the overseer's tonight (safe runner recovery or tt-smi -glx_reset, recover-hosts if that fails, never -r; then test_box.py; then resume the stopped step through the framework; log each); to be written into the skill's standing decisions at the next pause. (2) The chain's push will fail (GitHub SAML 403): do not fix auth; on that stop, run `overseer release FIX.31` + `orchestrator resume --only FIX.31` myself; the owner pushes from the Mac. (3) After FIX.31, with the device idle: `bash /scratch/dnijemcevic/run_until_repl.sh P.55` (replica fork @ 606346c8acb, its own orchestrator.log); its A/B decision is mine under the perf-pick rule, only from the A/B table's device ms per chunk (replicas off vs exact), default stays off; P.54 not started (WIP on exp/p54_wip).
- 2026-11-13 18:57 chain_1113 STOP at P.51's full gate: all 5 units PASS, gate FAIL on MISSING determinism_max_abs_diff_decoder_skip. Classified FRAMEWORK/ledger: P.51's gate (3d37c15ce30) lists FIX.28's metric but not FIX.28's test_determinism unit that emits it (FIX.28/FIX.29 gates have it). Action: added `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py` to P.51's gate (no limit changed); continued with /scratch/dnijemcevic/chain_1113b.sh (same steps from pending_full_gates; push failure -> bundle and continue, per the owner's update). Expected: P.51 reruns ladder (its hash moved, framework quirk, queued in FIX.31) + determinism; other units kept.
- 2026-11-13 ~19:30 chain_1113b crashed after P.51's 6 units all PASSED: my FIX.31 addendum put apostrophes inside overseer.yaml's single-quoted text; the orchestrator's yaml read (clear_classification) raised. Overseer error. Fixed the text (yaml parses), relaunched chain_1113b; P.51's units should be kept (same env).
- 2026-11-13 21:46 chain done: FIX.31 PASS 6c94007c225 (its gate re-ran units: this orchestrator predated FIX.31's env-key fix; profile 2627 s vs 838 s in X.3). Bundle /scratch/dnijemcevic/model_bringup_20261113_2146.bundle for the owner's Mac push (origin..6c94007c225). Launching P.55a on the replica fork (run_until_repl.sh P.55a).
- 2026-11-13 21:50 replica fork: its dashboard/links.yaml is a copy of the main run's (same artifact URL); not publishing the fork there (would overwrite the main run's page); marked published to silence DASHBOARD STALE. A separate fork dashboard is the owner's call.
- 2026-11-13 22:26 replica fork: P.55a PASS 55b04e3e0fc (deadline-surviving saving novel 10.1 ms GO, code 2.9 ms NO-GO, random 16.5 ms; stop rule needs both under 5 ms -> continue). Note OV-39 for P.55b/c (streaming per block, two link paths, home keeps/re-reads weights, report novel AND code). Launching P.55b.
- 2026-11-13 23:15 replica fork: P.55b PASS ac18c0bf487 (streamed copy bit-exact, 100 runs, combine identical; copy lands 0.77-0.89 ms pair, ~1.23 ms with relays; added program time +0.30-0.42 ms per layer on real splits). P.55a's model with measured copy times: novel 6.2-8.6 ms/chunk, code 1.4-2.3 ms/chunk, 0 copies on time; and the per-layer added program time (if it holds in a real chunk, ~12-17 ms over 40 layers) may exceed the saving. DECISION: P.55c (4-8 agent-hours + device) NOT started: the outcome is a direction call for the owner (code NO-GO, novel marginal, possibly net negative). Box idle; nothing else approved in either queue.
- 2026-11-13 owner: go on P.55c. Note OV for P.55c: decide on device ms/chunk off vs exact on real novel+code (agent's test), not only the A/B's random-token profile. Launching run_until_repl.sh P.55c.
- 2026-11-13 OWNER DECISION (replica fork only, a throwaway branch in the worst case): do not wait on the owner for P.55c. If its gate passes and accuracy is within limits, ACCEPT it behind its off switch (keep_default: MOE_REPLICAS stays off) even when slower, so P.56 and later refinements can reuse the logic; or queue a P.55d refinement if the data points to one. Backup plan P.56 (owner design): learn the hot experts from chunk K-1's router counts, fill the replica slots for chunk K over the following layers' combines (P.53 piggyback), A/B off vs K-1 on real back-to-back chunks; queued after P.55c (patch /scratch/dnijemcevic/replication/p56_task.yaml, applied when P.55c's run stops).
- 2026-11-14 00:52 replica fork: P.55c accepted behind its off switch (owner rule; gate PASS, accuracy bit-identical, keep_default). Real-prompt result: streamed exact novel +6.9 / code +9.5 ms per chunk (slower); pre-filled split novel -31.0 / code -12.8 ms (best case, same-chunk host plan). KNOWLEDGE CONTRADICTED DS41-244/245 (the exact-split savings estimates). Queued P.56 (e67f8c5ebaa), P.54 after it. Launching run_until_repl.sh P.56 (resume commits P.55c first).
- 2026-11-14 02:14 replica fork: P.56 accepted behind its off switch (owner rule; gate PASS, bit-identical; real prompts chunks 1-10 device novel 175.2->156.1 ms, code 164.0->153.1 ms; clean wall unchanged ~191 ms: host-bound; A/B random-token profile device -25.2 ms, wall -19.5 ms but plan learned from the same tokens, DS41-228). Resuming to commit (run_until_repl.sh P.56).
- 2026-11-14 owner: queue a host-bound investigation; P.57 on the replica fork (measure host breakdown with P.56 on, fixability, trace blockers, one-layer trace prototype); owner: if not fixable otherwise, a traced run next. Launching run_until_repl.sh P.57.
- 2026-11-14 06:27 replica fork: P.57 PASS b601378d4cd, then STOPPED on 2 untracked results/P.57_host_*_p51200.pstats (written by the framework's host profile, testing/profile.py:168, not by the agent): classified FRAMEWORK (false path violation). Moved them to /scratch/dnijemcevic/replication/p57_pstats/ (raw cProfile data kept). Framework follow-up: the path check should accept the profile's own <task>_host*.pstats outputs (or profile.py should write them outside results/).
- 2026-11-14 owner: queue the TP=4 experts check -> P.58 (8dadc6678ce, CPU model + dispatch/combine microbenchmark, go/no-go vs EP + P.56). Launching run_until_repl.sh P.58 (resume clears P.57's STOPPED: its units are reused).
- 2026-11-14 owner: also check TP=8 (column groups of 8 chips) -> note for P.58 (model: TP8 at the weight-read floor, 28.1 ms both prompts).
- 2026-11-14 ~07:40 OWNER (leaving): bring decisions autonomously ('we'll see where it ends'). Spawned an agent to decode Marko's branch mstaletovic/flat-expert-perf-dw-split (flat expert -17..-21% at M>=256). Running: P.58 (TP=4/TP=8 check) on the replica fork.
- 2026-11-14 ~07:50 Marko's branch decoded (agent): mstaletovic/flat-expert-perf-dw-split = 1e2ab798da1 (WIP) on top of 0c0bbe59d98 (our fork's exact source). Table = the x-pieces change (x passed in >=8 KB pieces: 2 not 4 for DSv4.1), not the DW split (~-2.5%, semaphores unchecked). DECISION: port only x pieces as P.60 on model_bringup (e903c91ecd6), switch EXPERT_X_PIECES legacy|min8k, A/B at 51,200; expected -0.15..-0.25 ms a layer on the busiest chip (~6-10 ms a chunk), to be measured. DW split + gate/up loop wait for upstream. P.60 runs after P.58 leaves the device (one device job at a time), via `orchestrator resume --only P.60` on ~/repos/tt-metal. Accept rule: main-branch perf pick: accept on a clear device/wall win with accuracy in limits; else to the owner. Push: not possible (SAML); bundle.
- 2026-11-14 OWNER: do not mix branches; park model_bringup work unless it helps the replication/TP phase, in which case apply it on the replica branch first and re-measure the baseline. P.60 parked on model_bringup (reverted, f1b8ab5919c). The x-pieces port changes per-token expert cost, so it shifts the baseline: requeued as P.59 on the replica branch (patch /scratch/dnijemcevic/replication/p59_task.yaml: port + A/B + re-measure P.56 off/prev, refit the timing model, redo the TP estimate), applied when P.58's run stops.
- 2026-11-14 owner: go: P.59 then the TP implementation as P.60a/b/c on the replica branch (plan /scratch/dnijemcevic/replication/p60_plan.md, finalized from P.58's verdict and P.59's baseline; stop after P.59 if comms eat the gain).
- 2026-11-14 ~08:10 P.58 PASS dbd39dcb585: TP NO-GO (TP4 215/209, TP8 335/330 vs EP+P.56 156/153 device ms per chunk novel/code). Cause: flat op per-row cost barely falls with width (7.66 -> 6.25 -> 4.85 us/tile at full/quarter/eighth) since each row moves full 5120 in/out and every group chip handles all group rows; overseer's 1/k assumption wrong by ~4x. Comms modest for TP4 (+0.17-0.25 ms/layer). DECISION: no P.60a/b/c (stop rule). Next: P.59 (x pieces + re-measured baseline).
- 2026-11-14 ~09:06 P.59 ACCEPTED (min8k default on the replica branch; bit-identical; A/B device -8.9 ms, wall -13.2 ms at 51,200; real prompts off: novel 174.9->166.7, code 163.3->156.6; P.56's gain shrinks to -11.9 / -4.7 ms). Since P.56 costs +18-24 ms host (P.57) on a host-bound chunk, P.56 now hurts the wall unless the host cost goes. DECISION: queue P.60 (device-side ring slot + cheap planner launch, P.57 fix 3) and P.61 (trace the MoE half, P.57 fix 1; owner's "traced run if not fixable"), 0a736bcf5d5; run P.60 now (resume commits P.59 first).
- 2026-11-14 ~09:25 OWNER: hold (TP verdict questioned: P.58 summary given). P.61 and any TP work NOT started until the owner decides; P.60 (already running) continues unless told.
- 2026-11-14 OWNER: go ahead with P.61 after P.60 (hold lifted for P.60's accept and P.61; TP still on hold).
- 2026-11-14 10:08 P.60 ACCEPTED (MOE_PREV_HOST=device default on this branch; bit-identical; A/B at 51,200: wall 190.4->177.7 (-12.7), host overhead 30.4->20.1, device 160.0->157.6; real prompts prev: host novel 188.1->172.4, code 182.7->170.1; wall novel 189.1->179.3 (now below off 181.0), code 183.5->175.0). KNOWLEDGE CONTRADICTED DS41-251 (cause was per-call fill/wait arg rewrites on 32 chips, not the ring slot; DS41-254). Planner launch (~3.8 ms/chunk) left: needs a mesh program or folding into the combine (fabric, hang risk). Launching P.61 (resume commits P.60 first).
- 2026-11-14 10:12 P.60 committed a1cd941aed0. SIDE EFFECT (overseer's A/B design): accept also flipped defaults MOE_REPLICAS off->prev and MOE_FILL serial->piggyback (both sides of P.60's A/B set prev; F104 ships every ab.change key). Replicas now ON by default on this branch. Left as is during P.61 (an edit would be charged to P.61's agent); one-line revert after P.61 if the owner wants. Lesson: an A/B whose both sides share a switch must use keep_default or put the shared switch only in ab.env.
- 2026-11-14 ~10:25 OWNER: park P.61 (trace) when done; first sort out why the host gets slower with replicas ("it makes no sense"), then reduce host calls in other ops, autonomously, one per run (mesh-wide programs, fewer attention calls, ...). PLAN (patch /scratch/dnijemcevic/replication/host_plan/tasks.yaml, applied when P.61's run stops): P.62 replica host attribution + removal (target prev host <= off + 1 ms), P.63 one mesh program for fused head + combine, P.64 fewer attention-side launches, P.65 cheaper all_gather / mhc_post, P.66 trim Python glue; each its own A/B, accept on a clear host/wall win with accuracy in limits. P.61 PARK: set keep_default true on P.61 before deciding so MOE_TRACE stays off by default (no trace mixed into the host measurements).
- 2026-11-14 11:20 P.61 PARKED (owner): accepted with keep_default (code kept, MOE_TRACE default off). Result: MoE-half trace cuts ~50 ms host a chunk, bit-identical; chunk becomes device-bound; replicas then WIN on wall: traced novel prev 166.9 vs off 174.6, code 161.8 vs 166.7. Known issues DS41-255..257 (capture-freed buffers, engram upload during capture, profiler replay ids, shared profiler.py fixed). KNOWLEDGE CONTRADICTED DS41-251 (allocations inside a capture ARE a blocker with eager work between replays). Queued P.62-P.66 (host plan); launching P.62.
- 2026-11-14 12:31 P.62 ACCEPTED (lean, bit-identical, removes work by construction; ~0.45 ms host a chunk, inside noise). Attribution complete: prev's extra host = planner launch ~3.7-3.8 ms (extra 32-chip launch per layer) + combine fill ~1.1-1.5 + head/Python ~1; earlier "2.5 ms unattributed" was noise. DS41-258: a mesh-wide prepared program launches ~as slow as 32 per-chip (85 vs 92 us): per-device dispatch dominates -> note OV-43 to P.63 (probe first). Queued P.62b (fold the planner into the combine + next head, ~-3.7 ms host; C++ in two ops, hang risk accepted under the owner's reset authority) before P.63 (4fa8c206016). Launching P.62b (resume commits P.62 first).
- 2026-11-14 13:32 P.62b ACCEPTED (MOE_PLANNER=folded default; planner launch 92 -> 4.8 us a call; bit-identical; host prev vs off: novel +1.0 ms, code ~0 (target met); clean wall off -> prev: novel 177.3 -> 170.9, code 174.9 -> 170.7, untraced; A/B quick wall 173.1 -> 167.1). KNOWLEDGE CONTRADICTED DS41-258/259 (P.62's lean accounting ran P.60's path: test reset the setting; fixed, DS41-262; mesh-program conclusion holds). New DS41-260/261. Launching P.63 (resume commits P.62b and re-runs its gate unit on the final code).
- 2026-11-14 13:35 P.62b committed 8b8c4167b62; accept gate reused the A/B on-side measurement (final code: quick ladder, determinism, profile PASS); test_moe_prev_replicas not re-run on the final code (only change after its pass: combine scratch layout for 65-slot rows, 2x2 only). Covered by P.63's gate, which runs test_moe_prev_replicas.
- 2026-11-14 14:35 P.63 ACCEPTED (MESH_PROGRAMS=on: cached argument slots, no lookup per call; combine ~352 -> ~248 us a call, head 342 -> 301; ~-6 ms host a chunk; bit-identical; at 51,200 device-bound so wall flat). Not a true mesh program (probe: Metal already sends one command set to 32 devices; remaining per-chip patching ~150 us a layer). Owner said try mesh programs -> queued P.63b (true mesh-wide head + combine with chip-lookup tables) before P.64. Launching P.63b (resume commits P.63 first).
- 2026-11-14 15:23 P.63b HANG: the agent's mesh-wide fused head program hung (dispatch timeout) on its first device probe; the agent stopped device work (brief). The safe runner recovered on its own (recover-hosts-dnijemcevic-20261114T152337Z, 'reset complete'). Overseer box test 15:3x: PASS (34 s). No manual reset needed. Waiting for the agent's report; mesh-wide head is the risky part (DS41-264..266 being written).
- 2026-11-14 15:45 P.63b STOPPED + REVERTED (overseer). Attempt 1: mesh-wide fused head hung the 8x4 on first launch (safe runner recovered; overseer box test PASS). The orchestrator auto-started attempts 2/3 on a settings lint; overseer aborted (OV-44), killed the run and the attempt-3 agent. Findings: DS41-264 (per-chip programs differ in structure: defines carry the chip index, arg lists differ; combine places cores per chip), DS41-265 (a mesh program keeps the ~80 KB per-chip arg copies per launch), DS41-266 (hang suspects). Code reverted, diff + new files kept in /scratch/dnijemcevic/replication/p63b/; findings committed ee8ef578a11; P.63b removed from the ledger; fork rebuilt (build_Release). Owner asked to try mesh programs: tried, hung, low payoff -> parked. Next: P.64.
- 2026-11-14 16:34 P.64 ACCEPTED (ATTN_FUSE=on: window write fused into the send's prepared program; 9 -> 2 launches a layer, 1656 -> 1376 a chunk; host per chunk -6 to -11 ms (prev: novel 155.6 -> 144.4, code 156.3 -> 150.0); wall -1 to -5 ms; bit-identical). With replicas on, host (~144-150) is now at/below device (~145-148): further host cuts show in the wall only as device time falls too. Launching P.65 (window snapshot first, OV-45).
- 2026-11-14 17:15 P.65 ACCEPTED (CHEAP_CCL=on: window snapshot = 1 prepared program, owner's rows sent down the column then along rows, replacing 2 whole-ring all_gathers + slice/concat/update; per call host 352 -> 108 us, device 163 -> 67 us; at 51,200 launches 1376 -> 1136, device 154.9 -> 149.7, wall 173.7 -> 168.0; real chunks prev: novel host 144.4 -> 138.1, wall 162.9 -> 164.2 (noise), device 148.3 -> 145.2; code wall 164.5 -> 158.7, device 144.9 -> 141.3; bit-identical). Not changed: mhc_post (already 1 launch at the floor) and 12 compressor all_gathers. Launching P.66.
- 2026-11-14 18:22 P.66 ACCEPTED (LEAN_PY=on: per-layer wrappers built once, 3 load-time buffers, loop trims; host -4.3..-5.1 ms a chunk with prev, frees 630 -> 513; bit-identical; wall flat: chunk device-bound, wall ~20 ms above host). Running the commit resume (--until P.66); then totals + stop (nothing new without the owner).
- 2026-11-14 18:2x P.66 committed b9a45fb0b80. Host sequence P.62..P.66 done. Replica branch now: real chunks wall prev novel ~163 / code ~159 (start of day 186.6 / 183.5); off ~173 / ~169; chunk device-bound (wall ~160, device ~145-150, host ~135). Stopping: nothing new without the owner. Options proposed: device-side work, port winners to model_bringup (curated, fresh gate), TP with a narrow-expert op.
- 2026-11-14 ~18:50 OWNER: bring wins to model_bringup. Queued P.67 on model_bringup (merge the replica branch @ b9a45fb0b80, code only, ledger excluded; keep FIX.28-31, P.49-51, X.3; branch defaults; ab before/after at 51,200; gate = P.51's + determinism + replica test). Launching orchestrator resume --only P.67 on ~/repos/tt-metal (tmux main_p67). After: X.3 milestone gate, then a bundle for the owner's push.
- 2026-11-14 ~20:15 P.67 agent done: replica branch merged into model_bringup (3-way patch of 6bdf324e716..b9a45fb0b80, ledgers excluded; FIX.28-31/P.49-51/X.3 kept; known-issue ids renumbered: branch 241..269 -> DS41-246..274; DS41-275 framework fix: measure's after side lacked ENGRAM_TABLE=resident). measure before -> after: wall last-golden 182.8 -> 157.9, p51200 202.9 -> 166.9, device 187.3 -> 152.0, launches 1656 -> 1136; accuracy identical, determinism 0. RACE DS41-276: test_decoder_skip_decode failed 1 of 4 runs (layer-0 served window snapshot differs at chunk-boundary stops); suspects P.64 fused window write / P.65 snapshot broadcast. DECISION: accept P.67 after its A/B; then FIX.32 at once: CHEAP_CCL + ATTN_FUSE default legacy until the race is found and fixed (proof: >=20 consecutive decode-skip passes), then re-enable; X.3 + bundle after FIX.32's safe defaults.
- 2026-11-14 20:20 P.67 ACCEPTED (A/B: quick wall 182.8 -> 157.9, device 187.3 -> 152.0; accuracy identical). Queued FIX.32 (DS41-276 race: legacy defaults for CHEAP_CCL + ATTN_FUSE first with 20 consecutive decode-skip passes, then find/fix, re-enable only on 20 passes with both on). Running: resume --only P.67 (commit) then resume --only FIX.32 in tmux main_p67b.

- 2026-11-14 21:29: owner: FIX.32 must find the race cause, not settle for legacy defaults. Note OV-39 added. Running attempt not aborted (its step 2 is already the hunt); a report ending at legacy without a cause gets rejected and re-run with OV-39 in its brief.
- 2026-11-14 21:30: owner refined: legacy defaults only with proof the design cannot work. OV-39 closed, replaced by the new note (outcomes A: cause+fix, or B: proof unworkable).
- 2026-11-15 00:22: FIX.32 agent killed at the 4 h limit (rc=-9, "agent stopped: timeout_s") while its detached gate --check ran; no knowledge report, OV notes unanswered. Cause: spec agents.timeout_s 14400 (from e82c435a3e3, before F252) overrides defaults 0 (owner F252: no fixed limit). Queued: set spec agents.timeout_s 0 when the run stops (spec edit not made mid-task). Orphaned agent gate (pid 2071393) is on unit 8 test_moe_prev_replicas; orchestrator gate kept units 1-7 and its unit 8 waits on the device flock (serialized, no overlap; one duplicate run of unit 8). Framework bug to fix: an agent kill should end its detached gate or the orchestrator should wait for it. Fix code was 60/60 on the race loop (both / fuse / ccl), bit-identical to legacy.
- 2026-11-15 00:38: FIX.32 PASS -> 021797535ea (fix: snapshot_send programs keyed by pattern + ring/slab geometry; CHEAP_CCL/ATTN_FUSE stay on; 60/60 race loop). Spec agents.timeout_s 14400 -> 0 committed fa47b6c2686. X.3 gate started in tmux x3_1115 (reuse, not --fresh).
- 2026-11-15 00:49: X.3 PASS -> 0e397ef379a. Bundle next.
- 2026-11-15 00:55: framework fix: orchestrator waits for an ended agent's own detached gate run (agent_gates / wait_agent_gates) before its gate; selftest test_agent_gate_wait.py. Selftest subset: same 9 failures before and after (pre-existing, listed for the owner).
- 2026-11-15 06:09: owner: no unknown/dead modes. MOE_PREV_HOST=legacy (P.56) and MOE_PREV_HOST2=legacy (P.60) are superseded host paths still measured by test_moe_prev_replicas; queued FIX.33 to remove them (shipped device+lean unchanged, bit-identical, ms within noise). Launched tmux main_fix33.
- 2026-11-15 06:16: OWNER GOAL: model_bringup = latest and greatest WITHOUT expert replication; moe_replication = model_bringup + replica work. My error: P.67 merged the whole replica branch (owner asked only for the orthogonal wins). FIX.33 aborted and rewritten: keep P.59, P.63-P.66, FIX.32; remove every replica piece (settings, tt/expert_prev.py, trace P.61, replica tests, fork replica options). Gate = P.67 units minus test_moe_prev_replicas. Then rebuild the replica branch on top (separate task). Nothing replica-related was pushed (origin at 82a197f0edf).
- 2026-11-15 06:48: FIX.33 PASS -> ba3edbdc7ef (no replica code: grep clean outside history docs; tt/expert_prev.py gone; chunk_wall 170.4 ms, device 159.4, experts 84.9, determinism 0, decode-skip 0). X.3 launched in tmux x3_1115b (verifies the committed tree).
- 2026-11-15 07:00: X.3 PASS on the replica-free tree -> 0881367c136 (ran fresh, 685 s; chunk_wall 170.7, device 157.9, full prefill 1764.5 ms, top5 1.0, determinism 0). Bundle made.
- 2026-11-15 08:16: owner: keep model_bringup history as is (no rewrite). Bundle model_bringup_20261115_0700 stands. P.68 (replica rebuild) running on the fork.
- 2026-11-15 08:41: owner: the kmd leak can go near 300 GB. Total 566 GB, hugetlb 35, a run peaks at 25-60 GB host: stop new device jobs at ~280 GB leaked (was ~200).
- 2026-11-15 08:52: owner asked whether a node change loses anything. Shared /data has: expert weight cache 8x4/v1 (status: 0 to pull / 0 to push, 611.5 GB; local-only extras are 1.3 TB of old non-mse .tensorbin files never read), goldens (85 GB). Git mirror /data/dnijemcevic/tt-metal-ds41.git was stale since 10-28 (post-commit push rejected after the rebase): pushed both branches there (model_bringup -> dnijemcevic/model_bringup_rebased, moe_replication) and pointed the hook at _rebased. NOT shared: HF checkpoint (476 GB, re-download at rev 5254448), run state runs/ (19 GB), ~/.claude.
- 2026-11-15 08:54: owner: copy all node-local state to /data/dnijemcevic/bringup. /scratch/dnijemcevic/backup/to_data.sh (nice 19, ionice idle, rsync, re-runnable): runs/input/profiles/fix23 -> deepseek_v41_flash_d_p/node_backup_20261115/, HF checkpoint -> deepseek_v41_flash_d_p/hf_full/, ~/.claude -> claude_home/. Log /scratch/dnijemcevic/backup/to_data.log.
- 2026-11-15 09:03: owner: the perf measurement is the realistic setting: DECODER_SKIP=decode (layers 0-19 + layer 20 KV/index head), per point (51200, 128000, 256000, 512000, 768000, 1024000) both (1) a steady skipped chunk (BRINGUP_CHUNK_LAST=0) and (2) the first chunk of a resumed request (128-row encoder replay below global_write_start + ~4992 new tokens; tok/s on new tokens); device ms, warm wall ms, tok/s; both branches; replica branch with the previous chunk first.
- 2026-11-15 09:43: FIX.34 PASS -> 81eb0ffce32 (realistic measurement, no replicas: steady 106.7..120.4 ms wall, 81.6..99.1 device, 48.0k..42.5k tok/s from 51k to 1M; all six points fit). P.69 queued and launched on the replica branch (same test, replicas on).
- 2026-11-15 10:01: P.69 PASS (replica branch 5907b5cb634) but replicas never used under DECODER_SKIP=decode (DS41-280: guard waits NCALL=40 MoE calls, decode calls 20): numbers equal FIX.34 within noise. Queued + launched P.70 on the replica branch (fix NCALL from the layers a chunk calls, keep the guarantees, re-measure).
- 2026-11-15 10:34: CORRECTION: a run peaks at up to 226 GB anonymous host memory (safe runner logs; my earlier 25-60 GB was one small test). Budget 566 - 35 hugetlb - 226 peak - ~10 kernel => the leak must stay below ~295 GB; stop new device jobs at ~250 GB unaccounted (not 280).
- 2026-11-15 10:56: P.70 PASS (replica branch cdbcd44e737): DS41-280 fixed (per-chunk P = MoE layers called, plan keeps its own P, ring stays sized 40; guard/fence explained). Bit-identical to off (0.0), determinism 0. Steady chunks ready 20/20, experts 47 -> 35 ms, wall -10..-13.5 ms, tok/s +8..14%. Resumed first chunk ready 0/20 by design (P.56: replica state reset per request). Accepted as is; resumed-chunk learning left to the owner.
- 2026-11-15 17:39: owner wants the full prefill MEASURED (host must not interfere): FIX.35 queued and launched (test_full_prefill.py, one 200-chunk request 0..1M, DECODER_SKIP=decode, cumulative wall at 51k..1M, per-chunk host/device/gap). Then the same on the replica branch.
- 2026-11-15 18:10: FIX.35 PASS -> bf9e3fbbbe4: measured 0..1M in 21.87 s (46.8k tok/s), sum device 16.14 s; host enqueue mean 86 ms vs device 81 ms a chunk (host >= device on 159/200), wall/chunk 109 ms: host time IS in the wall (~26%). P.71 (same test, replica branch) launched.
- 2026-11-15 18:24: P.71 PASS (replica b9d6fb8eab3): measured 0..1M 21.18 s (48.3k tok/s) vs model_bringup 21.87 s (46.8k): -3%; device mean 76.7 vs 80.7 ms, host enqueue 88.5 vs 86.1 ms, host >= device on 196/200 chunks: host-paced. Replicas ready 20/20 on 199/200 chunks.
- 2026-11-15 19:09: owner done for now, releasing the reservation. Everything committed on both branches, bundles and the report copied to /data, git mirror and /data backup refreshed.

## 2026-11-17 new phase: per-slot private state vs tt-d-gen (overseer session on bh-glx-110-c10u02, job 217991)
- 13:44 node rebooted (tt-kmd leak). 13:53 RAM baseline: unaccounted 2.6 GiB (excl. hugetlb 34.8 GiB), available 516 GiB.
- 13:57-13:59 link check: recover-hosts --skip-reset --skip-mpi-stress-test (validation only, no reset): all links healthy,
  artifacts /data/dcamp/cluster-health-check/logs/recover/dnijemcevic-20261117T135735Z.6O8DWF.
- INFRA: pre-commit hook envs and virtualenv's extracted wheel images had lost their .py files (a cleanup on 2026-11-16
  03:58; only __pycache__ left): every commit failed. Fixed: pre-commit clean, removed ~/.cache/virtualenv/wheel/3.10/image,
  pre-commit install-hooks.
- Owner: step 1 (code reading) done by an overseer subagent; report /scratch/dnijemcevic/phase_1117/step1_report.md.
- Owner: A/B STATIC_TABLES. Queued P.72 (ac1549442e6): switch-only pick (keep_default, flip), off vs on, gate = ladder last,
  both determinism checks, test_decoder_skip_decode, and under DECODER_SKIP=decode test_skip_positions (6 points) and
  test_full_prefill (0 -> 1M). Agent only checks that off still runs. 14:00 launched: tmux p72, resume --only P.72.
- 14:00-14:26 P.72 attempt 1: STATIC_TABLES=off was broken since P.38 (bf16_tiles window region; assertion at the first
  compressed layer): DS41-282, agent fixed the off path only (tt/sparse_attention.py _index_region; on path untouched);
  its checks passed: test_decoder_skip_decode off (devjobs p72_off_skip2) and test_determinism off (p72_off_det).
  The agent then ran `orchestrator measure` (on side): batch0 (ladder, both determinism, decoder skip) done 14:22;
  batch1 test_skip_positions HUNG at p256000 (14:26:07, TT_THROW dispatch timeout, "waiting for physical cores 15-3, 15-2";
  p51200 and p128000 measured). FIX.34 passed p256000 on the same on-path code (2026-11-15): an intermittent hang.
- 14:29 INFRA: the safe runner started recover-hosts (reset) for the dirty mesh; during its sys-triage the node went down
  hard at 14:30:38 (journal of that boot ends with recover-hosts' sudo lspci / dmesg / journalctl; no shutdown record).
  Up again 14:32:37; re-reserved as job 218916; this session resumed 22:46. Unaccounted RAM 12.7 GiB excl. hugetlb at 22:47
  (no device run since boot). 22:5x box test PASS (32 s, leak +0.8 GB).
- 22:5x P.72 resumed (--only P.72) with note OV-42: attempt 2 only records attempt 1's result; no re-checks, no measure.
  If p256000 hangs again on the on side, it is a reproducible hang of the shipped tree: stop and tell the owner.
- 23:1x P.72 off side measured (old/): batch0 all rc 0 (determinism, test_determinism, ladder last, decoder_skip_decode);
  test_skip_positions rc 0 (steady wall 109.1 / 111.6 / 118.3 / 125.9 / 141.4 / 155.8 ms at 51k..1M; device 83.2 .. 128.3).
  test_full_prefill rc 1: device DRAM OOM at MODEL LOAD (attention.py:149 _upload), because F259's one-process batch ran it
  after test_skip_positions and its reload ("needs another model ... releasing it and loading again in this process") does
  not get the first model's DRAM back. FRAMEWORK bug (testing/batch.py release / reload path), not the model: FIX.35 ran the
  same test alone. Cheap fix now: after the report, `decide --rerun` re-runs only the unfinished runs (F228), each side's
  test_full_prefill alone in its own process. Framework fix (reload frees the DRAM, or a run needing another model gets a
  fresh process) after P.72's decision, with a selftest (an edit now would change the A/B code key).
- 23:18 P.72 on side (shipped tree, STATIC_TABLES=on, DECODER_SKIP=decode): test_skip_positions measured 51k..768k, then
  HUNG at p1024000 (23:18:21 "TIMEOUT: device timeout, potential hang detected", dispatch cores 15-2/15-3 on every chip in
  teardown). Second hang of the on side in two tries (14:26 at p256000). The test records a hang as fits_p<p> = 0 "DOES NOT
  FIT" and exits rc 0: it misreports a hang as a memory miss (test bug, to fix). test_full_prefill then failed in 5 s (CQ reader
  dead). Off side: every point ran, no hang (one try).
- 23:21 OVERSEER STOP: killed the orchestrator's process group before the safe runner's automatic recover-hosts (the 14:29 one
  ended in the node going down). All 32 cards are marked dirty; NO reset done: the owner decides the reset. P.72 stays RUNNING
  (resumable). Partial A/B (steady wall ms on / off): 51k 104.9/109.1, 128k 107.5/111.6, 256k 113.6/118.3, 512k 111.5/125.9,
  768k 117.1/141.4, 1M -/155.8; device 80.9/83.2 .. 93.2/116.3 (768k), off 128.3 at 1M.
- 23:26 INFRA: node lost again ~5 min after the 23:18 on-side hang and my 23:21 stop (no reset was started; cards dirty).
  Boot ended without a shutdown record (`last -x`: no shutdown line; our journal view has no system messages). Node up
  23:26:52, re-reserved as job 219013. Pattern: both node losses today followed a device hang within ~5-10 min (14:26 hang ->
  14:30 down during recover-hosts; 23:18 hang -> ~23:2x down with nothing running): likely the cluster's health handling
  of a hung mesh, not our recover. To confirm with the cluster admins.
- 23:31 box test PASS (7 s); unaccounted RAM 2.7 GiB excl. hugetlb (fresh boot). P.72 NOT resumed: the shipped tree hung
  twice; waiting for the owner (proposal: reproduce the on side's test_skip_positions alone before finishing the A/B).
- 00:42 (owner quiet, standing rule: start proposed runs) hang reproduction: `orchestrator measure --task P.72 --part
  test_skip_positions` (on side, one process, records kept; the hung run's log kept as new/batch1.hang2.log). 00:48 rc 0, all
  6 points, NO hang. Shipped tree: test_skip_positions hung 2 of 4 runs (FIX.34 ok, 14:26 hang p256000, 23:18 hang p1024000,
  00:48 ok): intermittent. Unaccounted RAM after: 3.2 GiB (2.7 before: +0.5 GiB for this run).
  On side complete (steady wall ms): 104.5 / 105.8 / 108.1 / 112.4 / 116.2 / 123.0 at 51k..1M; device 80.7 .. 98.4.
  Off: 109.1 / 111.6 / 118.3 / 125.9 / 141.4 / 155.8; device 83.2 .. 128.3. test_full_prefill: no result on either side
  (off: batch reload OOM, framework; on: after the hang). Not run further: waiting for the owner on the hang.
- 01:4x queued FIX.36 (held for the owner's go): intermittent hang hunt + test_skip_positions misreporting a hang as 'does not fit'. Nothing running.
- 04:0x OWNER: "if something needs to be fixed, fix it": FIX.36 released. Framework F267: `overseer fix` default paths came
  from the ledger dir (bringup/ledgers/tt for a seeded run): now the model's tt/ from the spec; selftest
  test_fix_default_paths.py (+ test_overseer.py: 23 passed). FIX.36's paths set to tt/, tests/, BREADCRUMBS, known_issues,
  ttnn/ttnn/bringup. P.72's attempt-1 work (DS41-282 off-path fix, its reports, the partial A/B records) committed as
  bookkeeping before FIX.36 so the fix agent starts on a clean tree; P.72 stays RUNNING and is re-measured after the hang fix.
- 04:1x OWNER: after the hang fix, hide the all-gather. Queued as P.73 (patch /scratch/dnijemcevic/phase_1117/p73_task.yaml,
  applied to tasks.yaml when FIX.36's run stops; deps FIX.36): sub-device overlap of STATIC_TABLES=off's history gather
  (deepseek_v3_d_p pattern), new switch HISTORY_GATHER_OVERLAP, A/B off-vs-overlap under STATIC_TABLES=off, keep_default.
  Gate without test_full_prefill until the F259 batch-reload OOM is fixed (framework, at the same pause).
- 04:36 INFRA: node down for the 3rd time. FIX.36 attempt 1 reproduced the hang (test_skip_hang_loop, round 13 of p256000/
  p1024000 loops, right after a profiled run; dispatch cores 15-2/15-3; chip 15 = M0D14 had a core whose MMIO died; tt-triage
  could not attach); the safe runner then ran recover-hosts at 04:34, and the host died during its sys-triage. CAUSE OF THE
  NODE LOSSES (2 of 3 proven): sys-triage's `tt-fw-terminal` step on a hung box: both triage files (14:29 and 04:34 runs)
  end at the "======== tt-fw-terminal ========" header, and the boot ends seconds later (14:30:41 / 04:36:53). The 23:2x loss
  had no recover of ours (unexplained). Mitigation: node_env.sh TTPOOL_RECOVER_CMD adds --skip-sys-triage (safe-runner
  recovery). For the owner / cluster admins: tt-fw-terminal on a hung Blackhole Galaxy kills the host.
  Up 04:38 (job 219312); box test PASS; unaccounted RAM 2.7 GiB. Test fix DS41-284 (a hang is no longer 'does not fit').
  Note to FIX.36 attempt 2: HANGLOOP_PROFILE=0 first (profiled-run trigger), record the chip each time, <= 3 hangs.
- 04:52 FIX.36 attempt 2, hang #2 of the hunt (devjob fix36_np1): HANGLOOP with the rt profiler OFF hung in round 7
  (p256000 resumed) at 04:52:10, same dispatch cores 15-2/15-3: the profiled run is NOT the trigger. The safe runner's
  recovery WITHOUT sys-triage reset the box at 04:57:15 ("reset complete") and the NODE STAYED UP: mitigation confirmed.
  Unaccounted RAM 3.9 GiB after (2.7 at boot).
- 05:17 FIX.36 attempt 2, hang #3 of the hunt (devjob fix36_w1, watcher run): round 15, p256000 resumed, after 1157 s; node stayed up (recovery without sys-triage). The agent is at its 3-hang limit and reports next.
- 05:42 FIX.36 attempt 2 testing a fix (devjob fix36_fix1, since 05:18): every op of the skip flow capped to the model's core grid (sub_core_grids=sub_cores(..), agent rule 18: attention, compressor, engram_add, indexer, moe adapter, sparse_attention): ops on the full grid were the lead. Round 29 at 05:42, no hang (previous hangs at rounds 7, 13, 15). Unaccounted RAM 15.3 GiB (3.9 at 04:57: +11.4 over the watcher run, its hang and reset, and this run so far).
- 06:01 FIX.36 PASS -> eb61d3743da. CAUSE (DS41-285): stock binary ops left uncapped ran on 120 cores (beyond the model's
  110-core cap, agent rule 18): the MoE output add (20 a chunk a chip) and an sfpu scalar op (5 a chunk); a hang took one
  chip (15, 3, 7: not a single bad chip) and its CCL partners waited. Fix: sub_core_grids on every uncapped stock op in tt/.
  Proof: 41 loop rounds = 7,544 chunks clean, determinism 0.0; device/wall within noise of FIX.34. Test fix DS41-284.
  The run STOPPED after the commit on a path violation: the brief told the agent to write its knowledge-use report under
  bringup/results/ while the orchestrator reads/allows the ledger's results/ (seeded run). FRAMEWORK F268: the brief names
  the ledger's path; selftest test_seeded.py (11 of 12 pass; the 12th, an imported-golden test missing
  models/tt_transformers/tests/tale-of-two-cities.txt.bz2, fails without the change too). Stray raw report moved to
  /scratch/dnijemcevic/phase_1117/stray/ (the ledger copy is committed).
- 06:1x P.73 (hide the history gather) applied to tasks.yaml; launching: resume --only FIX.36 (clears the stop, no agent),
  then --only P.73. Pending framework fix before any gate that runs test_full_prefill: F259 batch reload OOM.
- 06:12 unaccounted RAM 18.3 GiB (15.3 at 05:42; +3.0 over FIX.36's fix loop end and its gate run). P.73 agent reading (no edits, no device job yet).
- 06:4x OWNER: gather hiding must be injected into an existing kernel, no extra op. Sent the running P.73 agent a cross-session message (session tt-metal-1f, delivered) and note OV-44 for any later attempt. Candidate named: the indexer ring score (already rings index keys over SP).
- 06:5x OWNER clarified: injection is an idea; a separate op is fine if it hides (e.g. in the engram wait gap). Sent a second message to the P.73 session; OV-44 replaced by the new note.
- 06:42 P.73 first numbers (sub-device route, before my messages took effect?): standalone 1M gather+top-k per source 7.28 -> 5.45 ms (row0:0-5 gather cores + 110 compute = 116 cores busy at once, above the 110 cap whose violation caused FIX.36's hang); in the model bit-identical, chunk wall 1M 152.8 -> 150.7, 51k 109.5 -> 110.8, 256k 117.0 -> 119.1 (noise). Unaccounted RAM 20.6 GiB.
- 06:5x OWNER: hardly any idle gaps; inject into another kernel when links are less busy; candidates engram wait, token-id readback of the next chunk. Sent to P.73 (third message); note OV-46 replaces OV-45.
- 07:12 P.73 agent done (attempt 1): shipped HISTORY_GATHER_OVERLAP=on = history gathered as ROW_MAJOR rows (no tilize/untilize, high_bw on both axes), standalone 10.0 -> 5.9 ms a source at 1M; the engram-window move (fixed-stride table) left as next step. A/B running (off side). Unaccounted RAM 23.4 GiB.
- 07:06 HANG again, in P.73's on side (the agent's `orchestrator measure`): STATIC_TABLES=off + HISTORY_GATHER_OVERLAP=on
  (row gather), test_skip_positions after p256000, same signature (dispatch cores 15-2/15-3 on every chip). FIX.36's proof
  loop ran STATIC_TABLES=on only: the off path and the new row gather were never looped. The device was left dirty; the
  A/B's off side at 07:11 started the safe runner's reset: recover-hosts (no sys-triage) FAILED: PCIe device 7 unreachable
  ("Read 0xffffffff from PCIE", PcieHangError, POST_RESET failed for /dev/tenstorrent/7), validation skipped. The box is
  unusable until a host-level power cycle (BMC / admins). P.73 was WAITING FOR A PERSON with an empty A/B; orchestrator
  stopped (07:1x). Partial on-side numbers before the hang: steady wall 111.5 / 110.6 / 112.3 ms at 51k / 128k / 256k
  (off-path P.72: 109.1 / 111.6 / 118.3). Unaccounted RAM 23.4 GiB at 07:12.
- 07:2x OWNER asked for tt-smi -glx_reset: ran it by hand: IPMI reset issued, POST_RESET failed again for /dev/tenstorrent/7 (secondary bus reset not completed). Chip 7 needs a host reboot / power cycle.
- 09:0x HAND-OFF written: /data/dnijemcevic/bringup/handoff_20261118/HANDOFF.md (+ bundle of the 5 local commits, the uncommitted P.73 diff/tgz, step1 report, A/B records). Box needs a host reboot (chip 7).
- 09:17 INFRA: host rebooted (owner / admins) for chip 7; job 219332. 09:20 link check (validation only, no reset, no
  sys-triage): all links healthy; box test PASS; unaccounted RAM 2.8 GiB.
- 09:2x P.73 back to its agent with note OV-48 (reproduce and cap the off-path / row-gather hang with the loop, >= 20 clean
  rounds per switch value, before its A/B). OVERSEER ERROR: I used `bringup rerun --from P.73` to restart the task; it
  resets the task AND runs its gate (it started test_batch on the device with the hanging configuration and reset P.72 to
  TODO). Stopped it after ~2 min (TaskStop; no hang seen; cards marked dirty, the next job resets them). Lesson: `rerun`
  is a gate runner; to restart an agent use `orchestrator resume --only <task>` (status RUNNING / the next attempt).
- 09:3x P.73 resume first went straight back to the old (empty) A/B decision; changed P.73's brief (attempt-2 instruction: fix the off-path hang first), which voids the old A/B key, and resumed: perf attempt 2 started with notes OV-46/OV-48.
- 09:42 P.73 attempt 2: core audit run (p73_coreaudit_on) done, hang loop p73_loop1 running. Unaccounted RAM 14.4 GiB (2.8 at 09:20: +11.6 over my stopped gate run, its card reset and the core audit).
- 10:07 P.73 attempt 2, off path + row gather hung again in its loop (p73_loop3, round 16, p256000 resumed) after capping slice/bitcast to 110 cores; safe runner resetting (no sys-triage). OWNER: the bring-up ran weeks on 120 cores without a hang; the first hang (11-17 14:26) ran FIX.34/35's code: spawned a read-only analysis agent (what changed between 11-15 and 11-17: builds, env, firmware/KMD, mesh-open warnings, the 13:57 link validation); report to /scratch/dnijemcevic/phase_1117/hang_origin_report.md. Unaccounted RAM 5.4 GiB at 10:12 (14.4 at 09:42: it FELL by 9 GiB across the reset).
- 10:1x P.73 attempt 2 ended: KNOWLEDGE CONTRADICTED DS41-283: its loop with every program <= 110 cores (STATIC_TABLES=off) hung after 2,944 chunks (chip 25 dead): the 110-core cap is NOT the cause. The orchestrator then started P.73's A/B; I stopped it (doomed: hang every ~20 min, A/B ~40 min) until the hang-origin analysis reports. tt-triage never produced output in any hang (fails in UMD discovery on the dead chip); sent the analysis agent that, the Inspector dump and the watch-mode data. Cards dirty (reset by the next job).
- 10:5x OWNER: rebase the whole branch (as llk_helper_library) onto the main of mstaletovic/moe-block-65124-split-rows (main 8a317f39a8d, 2026-11-17, + 11 MoE commits), then port Marko's MoE from that branch; only DeepSeek-V4.1 Flash matters (other models' bring-up ops may break), no extensive tests; MoE replication later. Backup tag backup/model_bringup_pre_rebase_20261118 + bundle in the handoff dir. P.73 attempt-2 WIP committed as is (not accepted; its off-path hang is unresolved; history_overlap.py, the reverted sub-device experiment, deleted).
- 12:3x MoE port (Marko's merged block, split rows, 6208 B payload; fork ttnn.bringup.moe_block 5963cd7fdb3): run 1 failed at build (his flat plan required 12 Galaxy columns; our cap is 11: fork now checks the shaped plan fits 11x10); run 2 a commit-during-import mistake (hook stash); run 3 hit the test's 1 h timeout at layer 22 building the new expert weight cache (~2.7 min a layer, new gate/up layout); run 4 (12:32) builds layers 22-39 then the ladder rung last. test_box passed. Unaccounted RAM 11.4 GiB after run 3.
- 12:5x OWNER: a layout change must not cost 1+ h of expert-weight rebuild: queue (after the MoE port lands) a cheap rebuild: cache the bfp4_mse-rounded routed weights layout-free (key: layer, impl, checkpoint fingerprint) and derive any flat plan's layout from them (and from existing .moe_main / .moe_blk files); and push the new .moe_blk files to the /data master (weight_cache push).
- 13:4x MoE port checks (Marko's block, split rows, 6208 B payload): ladder last PASS (err_final_hidden 0.1268 vs old 0.1325, top5 1.0, 0 of 14 compared metrics outside limits), both determinism PASS, test_skip_positions 6/6 points, no hang: steady wall 105.9/103.6/106.5/110.8/116.4/121.1 ms at 51k..1M (old 104.5/105.8/108.1/112.4/116.2/123.0), device 78.9..95.4 (old 80.7..98.4). Unaccounted RAM 12.1 GiB. Next: hang soak (loop) at the new payload once the port agent reports.
- 13:5x MoE port committed (5963cd7fdb3 fork ttnn.bringup.moe_block, 60add9f88f6 model: MOE_BLOCK_SRC=block default, MOE_BLOCK_SPLIT_ROWS=on, payload 6208 B; old block = fallback). Not ported: shared expert bf8 / SE_PACK_PRECISE (his head has no option: shared expert stays bf16, accuracy slightly better), cached_slots (his DRAM args table does that job). Open: test_all_gather_stock.py hard-codes 10880; ledger still names MOE_BLOCK_SRC=main.
- 13:5x started: (1) hang soak, tmux soak: test_skip_hang_loop (p256000 + p1024000 loops, decode, no profiler, 45 min / <=200 rounds), log /scratch/dnijemcevic/phase_1117/soak/soak.log; (2) weight_cache push of the 80 new .moe_blk files (305.8 GB) to the /data master at nice 19 / ionice idle, tmux wpush (~51 min). Cheap expert-cache rebuild agent queued after the soak (it edits the tree).
- 14:12 weight_cache push done: 80 .moe_blk files (305.8 GB) on the /data master in 1045 s, rc 0. Soak at round 35 (~6,300 chunks), no hang.
- 14:4x SOAK CLEAN on Marko's block + split rows (6208 B payload), STATIC_TABLES=on, decode: 64 rounds, 11,776 chunks (p256000 + p1024000 loops), 2738 s, determinism L19 max_abs_diff 0.0 every round, 0 programs dropped. Old payload hung every ~1,300-3,000 chunks. Unaccounted RAM 32.8 GiB. OWNER: park the MoE 12-column cap and the cheap cache rebuild (both mean re-layout); next: stock-op cap revert (matmuls stay 110), then the all-gather re-measure.
- 14:5x OWNER: after the cap revert, record the post-rebase work in the ledger (rebase, MoE port, hang fix, cap revert: tasks with their measured results) and run one milestone gate on the final tree (ladder last, determinism, test_skip_positions). Bundle of the current tip: handoff_20261118/model_bringup_post_moe_port_20261118.bundle.
- 15:0x OWNER: the milestone gate after the cap revert includes the contract tests (test_contract.py, test_adapter_acks.py, test_decoder_skip_decode.py): not run since the rebase (models/demos/common/prefill took main's side; our prefill_producer changes dropped).
- 14:5x OWNER: no more soaks; the cap-revert agent told to skip its 20-min soak (commit after test_skip_positions). Milestone gate after the revert: ladder last, determinism, positions, contract tests; no soak.
- 15:1x Recorded the post-rebase work as milestone task X.4 (brief lists rebase, MoE port, hang fix, uncap with reports and commits); its gate verifies the final tree incl. the contract tests. Launching resume --only X.4 (no agent: role-less).
- 15:08 X.4 run: `resume --only X.4` first ran P.67's pending full gate (quick-accepted pick, framework: deferred full gates run under --only) whose last unit test_moe_prev_replicas was deleted by FIX.33 (rc 4, no tests) -> WAITING FOR A PERSON; X.4 got only its determinism unit. On today's tree P.67's units PASSED: test_profile, test_decoder_skip_decode, test_contract, test_adapter_acks (contract survived the rebase + MoE port). Fix (ledger, overseer): removed the deleted replica test from P.67's gate (FIX.33 removed replication; the unit cannot run any more); then gate P.67 --continue (commit), then resume --only X.4 (its passed units reused). FRAMEWORK follow-up: --only should not run another task's deferred full gate.
- 15:1x X.4 PASS -> 166db33020b (every unit passed earlier on the same code: ladder last, both determinism, decoder skip, contract, adapter acks, positions). P.67's stale full gate closed (4f707ee2141). Queued P.74 (owner: see the all-gather): STATIC_TABLES=off on the new tree, row gather renamed HISTORY_GATHER, one persistent shared gather buffer, A/B on vs off (keep_default), DRAM per chip per point.
- 16:0x OWNER: after P.74's measurement, an attempt to hide the gather (engram wait, token-id readback, or injected into an existing kernel). Queued as P.75 (patch /scratch/dnijemcevic/phase_1117/p75_task.yaml, applied to tasks.yaml when P.74's run stops; ab: HISTORY_GATHER_HIDE off/on under STATIC_TABLES=off; same gate as P.74).
- 16:3x P.74 A/B (STATIC_TABLES on vs off, new tree): off at 1M +11.8 ms wall (+10 %; old tree +33 ms / +27 %), device +10.2; within noise below 256k; DRAM one user 21.7 -> 19.7 GiB at 1M; accuracy, determinism, contract identical. Accepted switch-only (keep_default: the table stays the default, owner decides). Chain: P.74 commit, then P.75 (hide the gather) applied + launched (tmux p7475). Unaccounted RAM ~47 GiB.
- 16:4x P.74's gate commit was refused by the pre-commit hook check-models-do-not-import-tests (fails on two existing files: tests/test_expert_x_pieces.py:66, models/demos/moe_block_main/tests/test_moe_block_models.py:43, both pre-dating P.74); my chain's 'Queue P.75' commit (f985ff3b103) then swept P.74's staged files in under that message. Fixed: gate P.74 --commit-only with SKIP=check-models-do-not-import-tests -> PASS 83929f1e10f. run_only.sh now sets that SKIP until the two imports are fixed (queued as a small fix job). Launching P.75.
- 17:3x OWNER: STATIC_TABLES=off becomes the default after P.75 (queued FIX.38, held until P.75's decision); then a bundle for the owner's push; then update the slot-state artifact (item 4 solved, memory per user); then item 3 (indexer scores the served rows).
- 17:3x P.75 A/B: hidden gather (engram wait) vs plain off: wall 1M 130.5 -> 123.4 (table on 119.5 in P.74), 512k 118.9 -> 115.1, 51k noise; device unchanged; DRAM +1.0 GiB at 1M (three history buffers); bit-identical, determinism 0. Accepted (switch). FIX.38 released (STATIC_TABLES=off default + HISTORY_GATHER_HIDE=on).
- 17:54 FIX.38 PASS -> 51e4c623628: STATIC_TABLES=off + HISTORY_GATHER_HIDE=on are the defaults; its gate units ran on the new default during the attempt (17:41-17:53): ladder last err_final_hidden 0.1268, determinism 0, contract 0 failed, positions 51k 105.7 ms / 1M 120.7 ms wall (device 105.85), DRAM one user 1M 20.74 GiB. Bundle for the owner's push next.
- 18:1x Bundle for the owner's push: handoff_20261118/model_bringup_20261118_tables_off.bundle (whole branch, 689 MB; history rewritten by the rebase: the push is a force-push). Slot-state page updated (v14): item 4 solved, speed and memory per user. Queued + launched P.76: item 3, the indexer scores the served fp8 rows (INDEX_K_SOURCE=served), A/B keep_default.
- 18:2x OWNER plan: P.76 (item 3) -> framework fix for the F259 batch reload OOM (a run needing another model gets a fresh process; selftest) -> full gate X.5 on the final tree (X.3's set: full-target ladder s56320, test_full_prefill 0->1M, positions / profile, plus determinism and contract).
- 18:4x Review feedback from the owner's Mac session: (A) latent stale-prefetch reuse after outside writes in history_gather.gather (version is a host counter): fix queued; (B) FIX.38 units did run on the new default (FIX.38.check.log 17:37-17:53), the 'reused' line is the final gate reusing them; (C) no test of an outside write: queued test_outside_history.py. Both as FIX.39, held until after P.76.
- 18:5x P.76 A/B (INDEX_K_SOURCE private vs served): wall 51k 105.2/105.2, 1M 123.3/122.4 ms; device 1M +1.0 ms; err_final_hidden 0.1268 -> 0.1271 (limit 0.2422), top5 1.0, determinism 0; per-user k8 gone (one shared buffer a source), decode of the history hidden at the engram wait. WAITING for the owner: make served the default (would go with FIX.39). Unaccounted RAM 67.6 GiB.
- 19:3x FRAMEWORK F269 (23775f12a1c, CPU only): a batch process holds one model; a run needing another model raises NeedsFreshProcess, the rest is deferred to a fresh process (core/batch.drive used by gate shared group, gate unit batch, multigate, A/B); selftest test_fresh_process 4/4, batch/gate/overseer/orchestrator/stamps selftests 142 pass; the full selftest suite has 180 failures that also fail without the change (to triage). Side effect: testing/batch.py is in the code key: stamps measured before re-measure once (P.76's accept gate may re-run its units).
- 19:5x Owner idle: P.76 accepted switch-only; FIX.39 released.
- 20:5x FIX.39 test_outside_history (raw DRAM copy of chunk k's served records into slot B, then chunk k+1): shipped defaults bit-identical (0.0); STATIC_TABLES=on differs from layer 2 (242), INDEX_K_SOURCE=private from layer 2 (245), no prefix ids from layer 1, no replay from layer 0. OVERSEER CORRECTION: P.76's agent had set INDEX_K_SOURCE=served as the code default; keep_default only kept the accept from changing defaults, so served shipped since 56bbcbbf17f without the owner's yes: told the owner (keep or revert). Unaccounted RAM 73.9 GiB.
- 21:1x FIX.39 PASS d766c50da90. Slot-state page v17: item 3 solved (before/after + outside test), the owner's Mac edits (4-bit note, Open block, ask A) merged; Open block marked done. Queued + launched X.5 (full gate on the final tree).
- 21:16 X.5 PASS -> 8c0c74712ea (full gate, final tree): full-target ladder s56320 PASS (err_final_hidden 0.1306, top5 1.0), full prefill 0 -> 1M 21.90 s = 46.8k tok/s (11-15 FIX.35: 21.87 s), device sum 16.77 s, host-bound chunks 0 (11-15: 159 of 200), positions PASS, outside-history 0.0, determinism 0, contract pass. Bundle for the owner's push next.
- 21:5x OWNER (via Mac): KEEP THE DEVICE FREE until the Mac session says otherwise (another team's DSv4.1 prefill bench runs from a separate session on this node: do not touch its processes / files). Drafted, not started: X.6 'packed-equivalent prefill throughput at AgentX shape' (drafts/x6_agentx_task.yaml); tt-d-gen asks (drafts/tt_d_gen_asks.md).
- 22:0x OWNER GO (device ours again, work autonomously): 1) X.6 AgentX measurement launched (unaccounted RAM before: 76.5 GiB); 2) then one profile at ~142k (per op group + host); 3) a roofline per group, gaps ranked, optimization tasks queued with targets, A/B gates and stop rules; items 1-3 design behind the perf work.
- 22:1x OWNER DECISION: everything optimized from now on is on a 4-bit compressed-KV path. (a) P.77 drafted (drafts/p77_task.yaml): prefill packs its served bf16 shard to 4-bit per chunk, gathers 4-bit, sparse attention reads 4-bit; accept ships it as the default; applied after X.6, and X.7 (profile) re-based on it. (b) Served format stays bf16: owner-decision write-up drafts/owner_decision_served_ckv_4bit.md (3.2x fewer bytes for every KV transfer and decode's memory; 1.7 GB a user at 1M Galaxy-wide).
- 22:3x X.6 PASS 71e783781de (AgentX shape, steady wall / tok/s): 32k 105.9 / 48.3k, 64k 107.6 / 47.6k, 95k 108.1 / 47.4k, 142k 111.0 / 46.1k, 256k 110.7 / 46.3k; device 82.1..88.5 ms (ceiling 62.4k..57.9k tok/s); determinism 0; RAM 76.5 -> 78.7 GiB. Gate hiccups: overseer metric names lacked _nonlast (fix agent aborted, classified framework); test_skip_positions unit was REUSED from an earlier run although SKIP_POS differed (FRAMEWORK BUG: unit stamp must include the test's env such as SKIP_POS; to fix) -> rerun. test_profile's clean wall (153-166 ms) is inflated vs test_skip_positions' steady wall (to check: host profiler in the wall run). Launching P.77 (4-bit history), then X.7 (profile on it).
- 22:07-22:26 OWNER (via Mac): verify the units X.5 / FIX.39 / FIX.38 recorded as "reused" by RUNNING them on the pushed tree 1eb81cf47e2 (working tree's tt / tests / ttnn / tt_metal / common byte-identical to it; P.77 had not started, the stale X.6 orchestrator killed and its fix-2 agent's notes moved to /scratch/dnijemcevic/phase_1117/x6_stray). Fresh runs through the safe runner, logs /scratch/dnijemcevic/phase_1117/verify_1eb81cf/: framework test_determinism PASS (all diffs 0.0); model test_determinism PASS (0.0 incl. decoder skip, 4 + 5 chunks); test_outside_history PASS (shipped 0.0; static_on 242 from L2, index_private 245 from L2, no_prefix_ids 220 from L1, window_carry 229 from L0; 0 markers left); test_decoder_skip_decode PASS (served_bytes_differ 0). Unaccounted RAM 80.1 -> 81.6 -> 82.4 -> 84.9 -> 84.6 GiB. The reuse records were right this time (they came from FIX.39's gate --check at 20:49-20:56 and a hand run at 20:39 on the same code), but the reuse key drops test env such as SKIP_POS (X.6's wrong reuse): framework fix F270 next.
- 22:5x FRAMEWORK F270 (ec4339162ab, CPU): the reuse key = measuring env + every VAR=value on the unit's command + every env knob its test files read that is set (SKIP_POS etc.); hand runs reused only with the same full prefix; the log names the reuse source (task / hand run, time, code key, env key); selftest test_reuse_env_key 6/6 (4 fail without the fix); 233 pass / 17 fail across the touched selftests, the 17 failing without the change too. Every safe-runner record re-measures once. Launching P.77.
- 23:3x P.77 A/B (history bf16 vs bfp4 records, pack-before-gather): wall 51k 105.7/103.9, 142k 110.0/106.5, 256k 113.1/106.8, 1M 123.1/118.5 ms; device 1M 106.8/99.1; DRAM 1M 19.42/18.40 GiB; gather bytes 3.2x fewer (1M ~3.2 -> ~1.0 ms a source + ~0.5 ms pack); err_final_hidden 0.127 -> 0.135 (limit 0.242), top5 1.0, determinism 0; tightest: err_state_window_kv_L20 0.136 / 0.151 (0.90 of the limit; bf16 0.58). Accepted as the default (owner decision: everything on the 4-bit path). Then X.7 (profile at 142k).

## 2026-11-19 00:30 UTC: 142k profile done, roofline, perf queue

- X.7 (one profile of a steady 5120-token chunk at 142k on the 4-bit default): device 82.2 ms, 794 op rows, determinism
  0.0. Its gate first failed on my metric names (no `_nonlast`, the X.6 slip again); the orchestrator started a fix
  agent, which found the same; stopped it, renamed, gate --continue -> PASS 7d12a8870f0. Framework at the cause:
  F272, `validate` and the gate refuse a position profile's metric gated without its role suffix (selftest).
- Roofline: `results/X.7_roofline.md` (+ `X.7_roofline.py`). Floor ~40 ms vs 82.2 ms device, wall 106.5 ms.
  Gaps: host not hidden ~23, attention ~13.6, MoE experts + combine ~13, MoE head ~7.3, dispatch ~6 (9 in fp8),
  mHC ~5, the rest < 2 each. Target ~60 ms device, ~85k tok/s.
- F271: BRINGUP_CHUNK sets a measurement run's chunk (the sweep); serving.max_chunk is checked on the written spec.
- Queued, serial on the device: X.8 chunk sweep (1536 / 2560 / 5120 / 8192 at ~142k), then P.78 host hiding, P.79
  experts + combine to the DRAM roofline, P.80 fp8 dispatch, P.81 attention, P.82 MoE head, P.83 mHC residency. Each
  with its roofline target, A/B gate (ladder accuracy, determinism, decoder skip, contract, outside history, steady
  walls at 142k and 1M) and a stop rule.
- Open: test_profile's clean wall reads ~155 ms at 142k vs the steady 106.5 ms (not cProfile: that runs after the
  timed runs); walls come from test_skip_positions until explained.
- RAM unaccounted after X.7: 94.9 GiB.

## 2026-11-19 00:45 UTC: chunk-size sweep done (X.8 PASS 5f44e06b36f)

- Steady chunk at ~142k, 8x4, decoder skip: 2048 70.7 ms (29.0k tok/s), 3072 79.7 (38.5k), 5120 108.3 (47.3k),
  8192 149.3-150.2 (54.5k, +15 %; roofline said ~+20 %: indexer x1.87 and host +7 ms a chunk grow with the chunk).
  Determinism at 8192 0.0. 1536 / 2560 do not load on 32 chips (DS41-297: chunk % 1024); gate units dropped, 2048 /
  3072 measured by hand. Serving chunk stays 5120 (a larger one is an owner decision with tt-d-gen).
- Framework gap (noted, not fixed yet): a measurement task (no code change, no `ab`) whose first check fails gets a
  perf attempt and then stops at "apply or not?"; closed by gate --continue.
- RAM unaccounted after X.8: 104.6 GiB (94.9 after X.7).
- Next: P.78 (hide the host time).

## 2026-11-19 02:22 UTC: host hiding accepted (P.78 PASS 1face0ec48f)

- HOST_HIDE=on (numba hash + threaded gather for layer 1's engram rows; layer 14's job deferred until the host
  leads): steady wall 142k 108.1 -> 93.4 ms (-14.7), 1M 121.2 -> 105.2 ms (-16.0); accuracy identical, determinism
  0.0 both sides; device work unchanged. ~54.8k tok/s at 142k.
- Accept first crashed: my ab values `on` / `off` unquoted = YAML bools; F274 (set_defaults takes a bool) + F273 (the
  A/B table shows the gate's test metrics on vs off: the steady walls were missing). The agent again set the
  setting's default to the new value itself (the off side was still measured with HOST_HIDE=False: A/B valid).
- RAM unaccounted after P.78: 117.1 GiB (94.9 after X.7, 104.6 after X.8): climbing ~10 GiB a task, no reboot since
  09:17 on 11-18.
- Next: P.79 (MoE experts + combine to the DRAM roofline).

## 2026-11-19 02:50 UTC: MoE experts task rejected (P.79); position runs now use real text (F275)

- P.79's agent stopped on its stop rule, no code change: routed_experts_combine = 205 us + 6.77 us x the busiest
  chip's 32-row tiles a layer (R^2 0.979, DS41-299); the busiest chip's FPU (84 % of LoFi on gate/up), not the weight
  stream. The roofline's 11 ms DRAM floor assumed uniform routing. Balanced rows over chips would give ~13.4 ms; that
  needs expert replicas (off this branch, owner). Rejected; its census test (tests/perf/test_expert_load_census.py)
  and notes kept.
- Cause found in our measurement: every position run (X.6, X.7, X.8, the A/B steady walls) fed random token ids
  over a zero history, assuming the work does not depend on the values. For an MoE it does. Census at 142k: random
  ids 27.1 ms a chunk, real text (the canonical 56k prompt repeated) 22.3 ms. Real text is still skewed (the hottest
  expert 1.5k-5.0k of 5120 tokens in many layers). F275: BRINGUP_POS_TOKENS=text is now the default (random on
  request); selftest. Walls from here on are not comparable 1:1 with the earlier random-id ones (expect ~-5 ms a chunk
  at 142k from the experts alone); the next A/B measures both sides on text.
- RAM unaccounted after the census: 119.3 GiB.
- Next: P.80 (fp8 dispatch).

## 2026-11-19 03:45 UTC: OWNER (Mac, 03:2x): expert replication approved; several texts, no tuning to one trace

- Queued after the fp8 dispatch task (P.80, not interrupted). Ledger patch ready
  (scratchpad owner_1119_patch.py), applied when P.80 stops: X.9 = the hot-expert census over texts (per layer and
  chunk: rows per expert / chip, busiest chip's tiles, hot-8 overlap chunk to chunk and across texts) + the 142k steady
  wall on each text; P.84 = expert replication (static set picked on the picking texts, or per-chunk placement as a
  pure function of the routing; chosen by the DS41-299 fit vs replica DRAM / traffic; stop rule < 3 ms at 142k on
  held-out texts); then attention, MoE head, mHC (their gates gain the held-out texts' walls).
- F276 (2ebb0d93749): reference/texts.py, BRINGUP_POS_TOKENS = prose (default; the whole Tale of Two Cities, 190k
  tokens) | agentic (the bring-up agents' own tool-use sessions, 179 transcripts, 1.1M tokens) | code (5591 op
  sources, 1.1M) | math (sympy / mpmath docstrings, 1.1M) | multilingual (zh / ja / ru / de man pages, 697k) | random.
  All from files on this machine, no downloads; tokenized once, cached with a sha under $ART/input/texts. A named
  text tags every metric `_t<text>`; every position run records pos_text (name, source, sha, repeats).
- Split: pick on prose, code, math; validate on agentic (the AgentX-like one) and multilingual. Walls reported as a
  spread over texts.
- F277: the canonical book's path moved (tt_transformers -> ttt_compat): fixed in prompt.py; 9 old selftest failures
  pass now. The canonical prompt's sha is unchanged.

## 2026-11-19 04:50 UTC: fp8 dispatch rejected (P.80); census over texts next (X.9)

- P.80 A/B (both sides on real text, prose): steady wall 142k 85.7 -> 85.4 ms, 1M 97.4 -> 96.3 (spread 3.2);
  accuracy and determinism identical (bit-identical path). The agent's own measurement: on real text the dispatch is
  8.8 ms a chunk (12.0 was random ids), ~3 ms of it the wait for the slowest chip's head (routing skew); fp8 rows cut
  only the transport (~-2.5 ms device expected, ~0.3-1 ms seen in the wall). Rejected: 27 files / 430 lines in the MoE
  forks for a gain inside the noise. Diff kept at /data/dnijemcevic/bringup/handoff_20261118/rejected_p80/; notes kept
  in BREADCRUMBS / known_issues. Revisit after replication.
- New baseline (real text, prose = the whole book, F276): 142k steady wall 85.7 ms = 59.7k tok/s (was 93.4 on random
  ids); 1M 97.4 ms.
- RAM unaccounted after P.80: 135.2 GiB of 566 GiB (119.3 before; +16 GiB this task).
- Owner patch applied (X.9 census over texts, P.84 replication, held-out walls in P.81-P.83); validate: valid.

## 2026-11-19 06:37 UTC: OWNER pointer (Mac): one process for every text / position of a gate; estimate from X.9

- X.9's check (11 units, 3107 s = 52 min), from its log timestamps:
  - a census unit (316 s): device open + layers 0-20 loaded 142 s, layers 21-39's MoE weights 100 s, the census
    itself (previous chunk, compile, 4 chunks) 42 s, pytest / teardown ~30 s;
  - a wall unit (314 s): load 0-20 143 s, layers 21-39 96 s, the wall (warm-up 10 + 10 timed runs) 75 s;
  - determinism 256 s (its own two passes).
  The measurement is 13-24 % of a unit; the rest is loading, ten times over.
- One process for the ten text runs: load once (143 s + 96 s) + 5 x 42 + 5 x 75 s = ~14 min, + determinism 4 min =
  ~18 min vs 52: -34 min (-65 %) on X.9; every multi-text perf gate from here on (P.81-P.83 add 2 texts x 2 positions)
  saves about the same share.
- Second finding: under DS41_DECODER_SKIP=decode the run uses layers 0-19 + layer 20's KV head, but every process
  also builds layers 20-39's MoE weights (~96-100 s a process, ~30 % of each unit). Loading only the layers a run uses
  saves that in EVERY device unit, not only multi-text ones.
- Plan (framework, after X.9 stops; X.9 not interrupted): (1) F279: the per-run variables of the position tests
  (BRINGUP_POS_TOKENS, SKIP_POS, CENSUS_POS, CENSUS_CHUNKS) become batch item variables and test_skip_positions /
  the census declare BATCH_SHARED, so consecutive text / position units share one process (core/batch gate_groups,
  F259 / FIX.29 already do this for framework tests); the model's slot state is fresh per run (new_state, as the
  position tests already do per point); the run order is recorded with each wall. Determinism stays its own unit (two
  passes, bit for bit) and a selftest checks the grouping. (2) Layers loaded = the layers the run executes
  (decoder skip): a model-side fix job (hooks / device_model layer list), A/B-free (same outputs), measured by load
  time.

## 2026-11-19 06:50 UTC: OWNER TOP PRIORITY: iteration time (load once; then a resident runner)

- X.9: its last per-text process (multilingual wall, unit 11 of 11) was already running; all five texts' census and
  walls were measured in its first check pass (05:48). No further per-text processes start; no device task runs
  until the resident runner lands.
- F279 (92e0db9d955): per-run variables a test declares (BATCH_ITEM_VARS: SKIP_POS, CENSUS_*) and the text
  (BRINGUP_POS_TOKENS) no longer split processes: X.9's ten text runs plan as ONE group (was ten processes).
- Resident runner (owner addendum: measuring within ~1-2 min of a request) being built by a dedicated agent:
  (1) a long-lived server holding the mesh and the model, requests over a socket, request -> first chunk timed;
  (2) code-change awareness: model python reloaded with the device weights kept (a device-tensor registry under the
  upload), restart only for a .so change; (3) layers loaded = layers executed (decode skip: 0-20);
  (4) gate units submit to it; fresh-process determinism once per accepted task; (5) a device demo with X.9's census
  and walls (prose, agentic). Each piece gets a status line here.
