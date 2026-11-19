2026-11-14T10:59:04Z started evals session; launching vLLM server (32 slots, ctx 262144)
2026-11-14T11:02:23Z server ready (~3 min after launch)
2026-11-14T11:02:29Z benchmark sweep started (16 points)
2026-11-14T11:42:00Z benchmark sweep done: 16/16 points, 0 failed requests (bench/summary.md)
2026-11-14T13:09:12Z GPQA Diamond started (198 q, thinking on, temp 1.0/top_p .95/top_k 20/pp 1.5, max 32768, conc 32)
2026-11-14T13:11:39Z TB harness smoke (hello-world, not counted) passed; SWE-bench mini-swe-agent run started (5 instances)
2026-11-14T13:31:49Z Terminal-Bench run started (5 tasks: hello-world, fix-permissions, csv-to-parquet, fix-git, openssl-selfsigned-cert; Terminus-2)
2026-11-14T13:47:00Z SWE-bench agent done: 5/5 submitted patches; official docker eval started
2026-11-14T13:47:00Z GPQA Diamond generation done: 198/198 responses
2026-11-14T13:48:56Z SWE-bench official docker eval done: 5/5 resolved
2026-11-14T13:49:01Z Terminal-Bench first run done: 4 pass, fix-git harness error (image build failed)
2026-11-14T14:21:21Z TB replacement task sqlite-db-truncate: fail (agent timeout); TB final 4/5 (fix-git unbuildable: upstream repo deleted)
2026-11-14T14:23:36Z ALL DONE: RESULTS.md complete (sweep 16/16, GPQA 78.8%, TB 4/5, SWE-V 5/5); vLLM server stopped cleanly, chips free (no /dev/tenstorrent holders), no eval containers
2026-11-16T01:15:09Z GPQA gap investigation started (78.8 vs card 86.4); relaunching server
2026-11-16T01:17:50Z gap: server up (bfp8); started GPQA repeats 2+3 (full 198 each) and 128K-cap rerun of the 8 no-answer questions
2026-11-16T01:43:25Z gap step1: run1 extraction OK (all 190 non-blank answers parsed; 8 no-answer = 7 hit 32K cap + 1 EOS-in-reasoning); 128K-cap rerun in progress (2/8 back, both now correct)
2026-11-16T01:43:25Z gap step2: card has no GPQA recipe beyond general sampling (T1.0/top_p.95/top_k20/min_p0/pp1.5) = ours; device presence_penalty verified active; chat template = repo template
2026-11-16T01:43:25Z gap step4 prep: local commit 3c4ed3f5 adds opt-in ORNITH_PRECISION_CONFIG (for bf16_hifi4 A/B); default unchanged
2026-11-16T02:39:29Z gap: repeats done (78.8/80.3/82.3, mean 80.5 sd 1.8); 128K cap fixed 4/8 run1 no-answers; TT-vs-HF logits 63/66 top1 match; bfp8 server stopped, launching bf16_hifi4 server
2026-11-16T02:44:08Z gap step4: bf16_hifi4 server up; GPQA full run (32K cap, same sampling) started
2026-11-16T03:21:26Z gap step4: bf16_hifi4 GPQA = 81.82% (162/198), +1.35 vs bfp8 mean (SE 2.1, n.s.); bf16 logits closer to HF (KL 0.009 vs 0.022)
2026-11-16T03:21:26Z gap step5: TT vs HF fp32 logits on wrong-answer prefixes: top1 63/66 (bfp8), 65/66 (bf16), mismatches near-ties; EOS-in-think is model behavior (HF gives similar EOS prob)
2026-11-16T03:21:26Z GAP DONE: explained/bounded in RESULTS.md 2b - noise (run1 lowest; 3-run mean 80.5), 32K cap (~1.4), strict parse (~0.7), precision (~1.4 n.s.), residual ~2.5 within card-vs-ours sampling error + unknown card recipe; no TT defect. bf16 server stopped, chips free
2026-11-16T15:02:15Z CI dispatch: started (per CI-DISPATCH.md); checking gh auth, workflow inputs, past runs
2026-11-16T15:13:26Z CI dispatch: branches prepared locally (vkovacevic/ornith-1.5-9b-ci) - tt-metal 97042da5f66e (lean: upstream tag v0.83.0-dev20261108 + 1 runtime commit), vllm-tt-plugin 2a1504554aa7 (origin/main+3 local fixes), tt-inference-server d67d443f7429 (Ornith spec/benchmarks/evals on Qwen3.8 template). PUSH BLOCKED by permission check; waiting for owner to push or allow; nothing dispatched
2026-11-16T15:23:04Z CI dispatch: pushed vkovacevic/ornith-1.5-9b-ci (owner-approved) - tt-metal 97042da5f66ea14c97e1de4bb579900fcc69600a, vllm-tt-plugin 2a1504554aa7babb1d426dfbca8c7db8c0a795ef, tt-inference-server d67d443f7429bc7e97435bc50ab3aedd122bc6a2
2026-11-16T15:23:41Z CI dispatch: release run (full evals) https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/43895740668 -> startup_failure within seconds ('workflow file issue', same as every dispatch since Nov 10; tt-shield ref unchanged ba2f0331). Per CI-DISPATCH.md: STOPPED, no retry, agentic not dispatched. Branches stay pushed for a later dispatch
2026-11-16T15:38:01Z issue opened: https://github.com/tenstorrent/tt-agentic-bringup-qb2/issues/99 ; final-numbers comment: https://github.com/tenstorrent/tt-agentic-bringup-qb2/issues/99#issuecomment-6921031278 (no CI dispatch)
2026-11-16T15:52:19Z issue #99: edited final-numbers comment https://github.com/tenstorrent/tt-agentic-bringup-qb2/issues/99#issuecomment-6921031278 -> full 16-point sweep table + separate evals table (no new comment)
2026-11-18T00:56:07Z FULL-AGENTIC: started (per FULL-AGENTIC.md); server up (same bfp8 stage-10/11 config, logs/server_full_agentic.log); full Terminal-Bench core 0.1.1 run 'full' (80 tasks, Terminus-2, n-concurrent 8) started
2026-11-18T01:26:21Z TB full: first attempt aborted after ~30 min (harness crash on content=None when EOS inside <think>; fixed in ornith_agent.py, see PIPELINE-NOTES.md); full 80-task run restarted from scratch (n-concurrent 8)
2026-11-18T07:25:43Z TB full DONE: 25/77 passed (32.5%), 3 excluded (env cannot be built: fix-git, qemu-alpine-ssh, qemu-startup), 53/77 agent timeouts, wall 2h05m; results in RESULTS.md §5
2026-11-18T07:25:43Z SWE full: started mini-swe-agent on all 500 SWE-bench Verified (8 workers, swebench/full_out) + incremental official docker eval loop (swebench/eval_loop_full.sh, removes per-instance images after eval)
2026-11-18T07:26:25Z issue #99: posted full Terminal-Bench comment https://github.com/tenstorrent/tt-agentic-bringup-qb2/issues/99#issuecomment-6972022867
2026-11-18T08:12:45Z SWE full: KV saturated at 8 workers (99%, 4 preemptions per /metrics); restarted agent with 6 workers (3 finished instances kept, in-flight ones redone)
2026-11-18T09:13:39Z SWE full: still KV-saturated at 6 workers (preemptions 6 total); restarted with 4 workers (6 instances done/kept)
2026-11-18T10:44:16Z SWE full: 7/500 agent-done (3 resolved of first 6 evaluated) after ~3.3 h; pace ~2-4 instances/h at 4 workers (long-context decode ~110 tok/s total, contexts up to 80K, no prefix cache) -> ETA for all 500 is ~4-6 days; continuing unless owner says otherwise. Preemptions total 7 (1 since 4-worker restart)
2026-11-18T13:33:54Z SWE full: server crashed 12:17 (KV preemption race -> 'Drain decode submissions before resetting slots'); fixed in tt-metal local commit 84c2b2f95 (drain readbacks before slot release), verified with forced-preemption test; 24 crash-burned instances discarded for redo; server restarted; agent resumed (4 workers) with server watchdog
2026-11-18T14:43:46Z WATCHDOG: server down -> SWE agent stopped
2026-11-18T14:42:58Z EXTERNAL INTERRUPTION: GitHub Actions CI job (tt-d-gen kv-manager-heavy-checks) ran /opt/tt-forge-infra/scripts/reset.sh on this machine -> chips reset under the server (not a model bug); watchdog stopped SWE agent 14:43:46Z, 0 instances affected
2026-11-18T15:19:32Z SWE full: tt-smi -r, server restarted, agent resumed (4 workers, 9 instances done); supervisor.sh now auto-restarts on server death and logs each event here
2026-11-18T17:34:13Z SWE full: 14/500 done, 8 resolved; pace since restart ~2.2 instances/h (4 workers, 0 preemptions) -> ~9 days remaining at this rate; continuing (owner may cut to a subset)
2026-11-18T19:34:24Z SWE full: 23/500 done, 14/22 evaluated resolved, 1 empty patch (astropy-14369, step limit 150 = genuine agent result); last 2 h pace ~4.5/h -> ETA ~4-5 days
2026-11-18T21:34:31Z SWE full: 27/500 done, 16/25 evaluated resolved, 2 empty patches; 2 KV preemptions at 4 workers handled without crash (fix 84c2b2f95 holds in production)
2026-11-18T23:34:41Z SWE full: 33/500 done, 18/27 evaluated resolved, 5 empty patches; server stable (preemptions 2, no crash)
2026-11-19T01:34:55Z SWE full: 41/500 done, 25/36 evaluated resolved, 5 empty patches (all step-limit); stable, ~3.5-4/h
2026-11-19T01:59:30Z SUPERVISOR: vLLM server down (external reset or crash) -> agent stopped, 6 affected instances discarded for redo; tt-smi -r + server restart
2026-11-19T02:03:25Z SUPERVISOR: server back up -> agent resumed
2026-11-19T02:10:54Z note: 01:59Z server death = tt-metal realtime profiler SIGBUS in RealtimeProfilerManager::drain_device_pages (receiver thread), not CI/model; no clean env switch to disable it, so supervisor auto-restart kept (details in PIPELINE-NOTES.md)
2026-11-19T03:36:15Z SWE full: fixed supervisor discard bug (it had dropped 6 genuine step-limit results at the 01:59Z restart); originals restored, 2 redo attempts archived uncounted; agent resumed (42/500 done)
2026-11-19T05:36:28Z SWE full: 45/500 done, 28/39 evaluated resolved, 6 empty (step limit); preemptions 7 since 02:03 restart, no crash
2026-11-19T07:36:55Z SWE full: 45/500 (no change in 2 h): two agents at 172K/123K-token contexts monopolize KV + re-prefill each step; letting them run to step limit
- 2026-11-19T09:39:19Z: SWE stall root cause found: mini-swe-agent sandboxes die after 2 h (stock `sleep 2h`); all 7 step-limit failures were dead-sandbox runs. Fixed (container_timeout 48h); those 7 are being re-run; 39/500 valid preds; agent and supervisor restarted.
- 2026-11-19T10:39:50Z: SWE 40/500 valid (29 resolved). After the sandbox fix: 0 dead sandboxes, 4 workers, KV peak 95%, no new preemptions. Server averages ~1.9K prompt tok/s and ~81 gen tok/s; decode at 4 concurrent is the bottleneck. Currently re-running the 7 long instances; will re-estimate the ETA once regular instances resume.
- 2026-11-19T12:40:06Z: SWE 41/500 valid (29 resolved). Redos with working sandboxes now submit real patches (django-10097 resolved in 48 steps; astropy-14369 unresolved in 113 steps). 4 redos still running (3 for ~3 h). Preemptions 8->11, no crash.
