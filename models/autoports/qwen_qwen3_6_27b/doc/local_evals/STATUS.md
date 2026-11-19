2026-11-17T13:14:22Z evals session created (brief CLAUDE.md)
2026-11-17T13:17:42Z GPQA Diamond: NEEDS OWNER: hf login on M4 (no ~/.cache/huggingface/token; Idavidrein/gpqa gated, only README cached). Continuing with other deliverables.
2026-11-17T13:19:11Z eval server launch (pool 1048576 tokens, 32 slots, stage-10 config)
2026-11-17T13:22:12Z first server launch failed (HF_HOME unset in stage-10 env.sh); relaunched (server/run2). KV pool override committed tt-metal b2d6be51
2026-11-17T13:22:12Z math data downloaded: AIME26 30, HMMT Feb25 30, Nov25 30, Feb26 33 (MathArena HF); client+scorer written; SWE-bench/TB prep delegated
2026-11-17T13:52:52Z eval server healthy (startup ~33 min); thinking smoke OK; benchmark sweep started
2026-11-17T13:55:30Z owner did hf login; GPQA Diamond unblocked: 198 items x 3 runs built (HF_TOKEN_PATH=~/.cache/huggingface/token since HF_HOME is ~/qwen36-27b/hf)
2026-11-17T14:20:30Z SWE-bench (5 inst, gold 5/5) and Terminal-Bench 2.0 (5 tasks, oracle 5/5) harnesses ready; sweep at 4096/512
2026-11-17T15:57:29Z benchmark sweep complete (4096/512 c32 rerun with seed base 9100)
2026-11-17T15:57:29Z eval phase started: SWE-bench 5 + TB 5 agents, math run0 (c12), GPQA run0 (c10)
2026-11-17T16:00:02Z sweep table written to RESULTS.md (16/16 rows)
2026-11-17T16:32:12Z eval phase running: first SWE instance submitted (sklearn-10297), GPQA 1/198; server prefill-bound by agents (~850 prompt tok/s, ~110 gen tok/s), 0 preemptions
2026-11-17T17:00:21Z GPQA run1 started (c8) to fill agent-freed slots
2026-11-17T18:02:44Z progress: math 5/123, GPQA r0 11/198, r1 5/198, SWE 4/5 submitted; gen ~137 tok/s aggregate at ~30 long-context users (~210 ms/step); projection ~15 h for math + GPQA x3
2026-11-17T18:31:10Z Terminal-Bench: cobol-modernization, overfull-hbox, prove-plus-comm PASS; 2 medium tasks running
2026-11-17T18:33:31Z SWE-bench partial scoring (official docker harness): django-10880, sklearn-10297, sphinx-10323, sympy-11618 RESOLVED; pytest-10051 agent still running
2026-11-17T18:58:29Z TB run1 (card 3h timeout): 3/5 PASS; adaptive-rejection-sampler & chess-best-move AgentTimeout at 3h (~43K out tokens, shared load). Started run2 for those 2 with timeout x36 (~9h), labelled relaxed
2026-11-17T19:01:52Z SWE-bench Verified 5/5 RESOLVED (official docker harness, run id qwen36-run1-final)
2026-11-17T20:27:15Z progress: math 18/123, GPQA r0 24/198, r1 15/198; TB run2 (2 tasks relaxed) running; 0 preemptions
2026-11-17T20:58:28Z throughput w/o agent prefill ~209 gen tok/s; answers long (math ~39K, GPQA ~19K tok avg) -> ETA ~15 h for math + GPQA r0,r1; GPQA r2 only if time allows
2026-11-17T22:01:23Z stopped TB run2 (relaxed) after ~3h: its 2 agents re-prefilled 150-300K-token contexts every turn (~400 prompt tok/s), halving reasoning throughput; will rerun the 2 tasks on an idle server after reasoning evals
2026-11-17T23:00:00Z GPQA run2 started as filler (c12, KV gate 0.75 via logs/kv_gate.txt) - long math answers left only 16 running under the 0.6 gate
2026-11-17T23:58:21Z progress: math 34/123, GPQA r0 42, r1 28, r2 10; 32 running, 0 preemptions
2026-11-17T23:58:55Z AIME26 complete: 26/30 = 86.7% (card 94.1); all 4 misses truncated at 81,920 (soft enumeration loops). Suspect BFP4/LoFi MLP precision; planned bfp8 A/B after deliverables
2026-11-18T00:29:35Z progress: math 34/123, GPQA r0 43, r1 28, r2 11; ~207 gen tok/s; remaining ~18M tokens (GPQA avg ~27K tok incl. truncations) -> ETA ~24 h
2026-11-18T00:45:48Z KV 92% with 32 running; lowered run2 gate to 0.55; preemption possible (recompute path, see PIPELINE-NOTES #1 follow-up)
2026-11-18T01:01:05Z first vLLM preemption (KV ~100%); recompute re-prefill ~52K tokens observed in slot 30; continuing
2026-11-18T01:02:18Z 2 preemptions at ~100% KV; stopped GPQA run2 client (newest in-flight requests aborted; 15 finished kept, resumable) -> KV 76%
2026-11-18T02:58:01Z only 11 running at 62% KV (long tails); resumed GPQA run2 now (c8, gate file 0.75), cancelled the wait-for-math chain
2026-11-18T03:27:16Z progress: math 46/123, GPQA r0 56, r1 42, r2 16; 32 running, 2 preemptions
2026-11-18T04:55:58Z decision: stopped GPQA run3 (run2; 19/198 kept, reported as partial/not used). Long traces (avg ~31K tok, ~11% truncated) + KV limit -> ~150-200 gen tok/s; finishing math + GPQA r0,r1 (~17 h) then precision A/B + TB rerun is more informative than a 3rd GPQA run
2026-11-18T05:54:10Z progress: math 58/123, GPQA r0 68, r1 49; 4 preemptions total
2026-11-18T07:24:05Z paused GPQA run1 (54/198 kept) so math + GPQA run0 finish first; auto-resume when math client exits. Throughput is attention-bound: step ~45 ms + ~0.1 ms per 1K in-flight context tokens (~100-200 gen tok/s at 50K-token traces)
2026-11-18T08:22:19Z progress: math 72/123, GPQA r0 80/198
2026-11-18T08:22:23Z HMMT Feb 25 complete: 23/30 = 76.7% (card 93.8); all 7 misses truncated at 81,920
2026-11-18T09:49:38Z progress: math 81/123, GPQA r0 88/198
2026-11-18T10:48:16Z client now skips items finished by another client; post-math chain: restart GPQA run0 at c22, then run1 as filler
2026-11-18T11:46:28Z progress: math 90/123, GPQA r0 98/198
2026-11-18T12:15:42Z only 13 running at 62% KV: resumed GPQA run1 now as filler (c10, gate file 0.75); post-math chain now only restarts run0 at c22
2026-11-18T13:14:04Z progress: math 98/123, GPQA r0 113, r1 58; 32 running
2026-11-18T14:12:19Z progress: math 103/123, GPQA r0 121, r1 63
2026-11-18T15:11:33Z progress: math 109/123, GPQA r0 123, r1 69
2026-11-18T15:40:47Z priority fix: run1 filler gate 0.5 (was starving run0); post-math chain restarts run0 at c22 with fixed gate 0.75
2026-11-18T16:39:07Z progress: math 115/123, GPQA r0 132, r1 77
2026-11-18T16:39:11Z HMMT Nov 25 complete: 24/30 = 80.0% (card 90.7); 6 misses all truncated at 81,920
2026-11-18T17:37:22Z progress: math 116/123 (7 in flight), GPQA r0 137, r1 80
2026-11-18T18:35:37Z progress: math 119/123, GPQA r0 140, r1 84
2026-11-18T18:47:48Z math run0 client finished; restarting GPQA run0 client at c22 (fixed gate 0.75)
2026-11-18T19:05:00Z MATH COMPLETE: AIME26 26/30=86.7 (card 94.1), HMMT Feb25 23/30=76.7 (93.8), Nov25 24/30=80.0 (90.7), Feb26 22/33=66.7 (84.3). 28/123 truncated at 81,920; every non-truncated answer (95/95) is correct -> gap is entirely truncation. GPQA run0 restarted at c22 (145/198)
2026-11-18T20:03:13Z progress: GPQA r0 155/198, r1 90
2026-11-18T20:24:24Z 6 preemptions at ~99% KV: stopped GPQA run1 permanently (90/198, reported as partial) so run0 completes and the device frees for the precision A/B
2026-11-18T21:22:50Z progress: GPQA r0 174/198
2026-11-18T21:54:38Z GPQA run0 tail (<12 running): starting TB rerun (2 timed-out tasks, timeout x36) job qwen36-run2b-relaxed
2026-11-18T22:49:11Z OWNER RULE (Vladan): one workload at a time on the server, never two clients at once. Stopped TB relaxed rerun job qwen36-run2b-relaxed (it overlapped GPQA run0 tail; discarded, not reported). New order: GPQA run0 finishes alone -> score -> TB 2 tasks alone, one task at a time -> precision A/B
2026-11-18T23:29:23Z GPQA run0 client exited; server idle. TB relaxed rerun task 1/2 adaptive-rejection-sampler alone (timeout x36)
2026-11-18T23:29:36Z GPQA COMPLETE run0: 159/198 = 80.3% (card 87.8); 28 truncated at 81,920 (14%); accuracy on non-truncated 156/170 = 91.8%; 7 answers parsed from reasoning (3 correct). Partial run1 90/198: 81.1% (run0 on same 90: 78.9%); partial run2 19/198: 84.2%. 6 vLLM preemptions total over the campaign
2026-11-19T00:35:53Z TB task 1/2 done: PASSED 0/1. Starting task 2/2 chess-best-move alone
2026-11-19T01:33:32Z TB task 2/2 done: PASSED 0/1
2026-11-19T01:33:45Z TB relaxed rerun (alone, one task at a time): adaptive-rejection-sampler FAIL (7/9 tests, 3904 s), chess-best-move FAIL (wrong move, 3433 s). TB final: 3/5 pass. Next: stop eval server, start bfp8_baseline A/B server
2026-11-19T01:36:16Z eval server stopped cleanly (no device holders); launched precision A/B server (bfp8_baseline policy via QWEN_VLLM_PRECISION_CONFIG, otherwise identical) -> server/ab_bfp8
2026-11-19T01:42:31Z A/B server healthy (bfp8_baseline confirmed in MODEL_PRECISION, KV 1,049,600); A/B run started alone: AIME26+HMMT Feb26 (63), same prompts/seeds/sampling, c32 gate 0.7
2026-11-19T03:10:55Z A/B progress: 11/63
2026-11-19T04:09:06Z A/B progress: 25/63, KV 92%
2026-11-19T05:07:17Z A/B progress: 39/63, 1 preemption
2026-11-19T05:07:33Z A/B interim: AIME26 under bfp8_baseline = 26/30, 4 truncated (same as selected). Same problems truncate under both (15, 28, 30; HMMT Feb26-2); #10 and #29 swap. Mean tokens similar (47K vs 45K). Precision hypothesis NOT supported so far on AIME26
2026-11-19T07:37:40Z A/B COMPLETE: bfp8_baseline 50/63 (AIME26 26/30, HMMT Feb26 24/33), 13 truncated vs selected 48/63, 15 truncated; 12 truncations shared; mean tokens equal. Precision hypothesis rejected (difference within sampling noise). Next: stop A/B server, restart selected server, continue the 56 truncated traces (+49,152 tokens, total 131,072)
2026-11-19T07:39:53Z A/B server stopped cleanly; relaunched selected-config eval server (server/run3) for budget-continuation test
2026-11-19T07:45:46Z budget-continuation test started alone: 28 truncated math traces continued +49,152 tokens (total 131,072), c8
2026-11-19T10:39:46Z BUDGET CONTINUATION (math, selected config): continuing the 28 truncated traces to a 131,072 total budget -> AIME26 93.3 (card 94.1), HMMT Feb25 96.7 (93.8), Nov25 90.0 (90.7), Feb26 81.8 (84.3). 25/28 concluded, median +8K tokens. Gap to card was purely the output budget. Starting GPQA continuation (28 truncated) alone
2026-11-19T13:05:47Z BUDGET CONTINUATION (GPQA): 28 truncated traces continued to 131,072 -> 27 concluded, 17 correct; GPQA 80.3 -> 87.4 (card 87.8). All reasoning evals match the card at a 131K budget
2026-11-19T13:06:21Z ALL DELIVERABLES DONE. RESULTS.md final (sweep 16/16; GPQA 80.3 @81,920 / 87.4 @131,072 vs 87.8; AIME26 86.7/93.3 vs 94.1; HMMT Feb25 76.7/96.7 vs 93.8, Nov25 80.0/90.0 vs 90.7, Feb26 66.7/81.8 vs 84.3; SWE-bench Verified 5/5; Terminal-Bench 3/5). Eval server stopped cleanly; no device holders, no eval processes or containers left. Local tt-metal commits: b2d6be51 (KV pool env), 002b2466 (precision env); nothing pushed
