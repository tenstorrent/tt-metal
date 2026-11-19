2026-11-18T10:36:43Z evals session created (brief CLAUDE.md)
2026-11-18T10:40:00Z server launch 1 (final serving cmd, resident2/C262144/budget9, timeout raised to 7d)
2026-11-18T10:51:27Z server ready (~25 min load); smoke OK; top_k=20 sampled on device; bench block1 (128/128,1024/1024 x c1/8/16/32) started
2026-11-18T10:55:00Z SWE-Pro: Harbor venv provisioned, 731 leak-free tasks generated, 5 task images prebuilt; fix commit absent in all 5 (swepro/verify_images_prebuild.txt)
2026-11-18T11:11:54Z bench 128/128 c1/8/16/32 done (all requests OK)
2026-11-18T13:54:13Z bench 1024/1024 c1/8/16/32 done (116/116 OK, 0 preemptions)
2026-11-18T13:56:38Z GPQA run1 started (c1); SWE-Pro run started (5 tasks, 1 trial at a time, owner config); egress blocked except server (verified in live container)
2026-11-18T16:41:51Z SWE task 1/5 ansible: agent timeout at 9900 s after 8 model calls; verifier ran, reward 0 (unresolved). GPQA run1 8/8 correct so far. Element-web trial started.
2026-11-18T19:27:10Z SWE task 2/5 element-web: agent timeout after 9 model calls, reward 0 (unresolved); egress check clean. GPQA q8 in long reasoning (~30K tok, 3 h). vuls trial started. KV 13%, 0 preemptions.
2026-11-18T22:17:53Z SWE task 3/5 vuls: unresolved (see RESULTS). GPQA 10/10 correct (q8 took 46,852 tok / 4.3 h, stop). Found+fixed egress-watcher fix-hash bug (element/vuls checks used wrong hash); openlibrary trial started.
2026-11-18T22:22:29Z OWNER DECISION: run everything sequentially, one client at a time. Stopped GPQA (kept 10 scored answers, 10/10; aborted in-flight q10) and SWE-bench Pro (discarded all 3 finished trials, ansible/element-web/vuls, plus the running openlibrary trial; moved to swepro/discarded_shared_run/). Server idle, healthy.
2026-11-18T22:22:29Z PLAN: (1) bench alone: 4096/512 c1/8/16/32, then 32768/512 c1/c8 (c16/c32 skipped: only 2 slots, queueing only, ~26 h). (2) SWE-bench Pro alone: same 5 leak-free tasks, one at a time, owner config (2.75 h agent limit). (3) GPQA alone: resume from the 10 answered, in order, running score here every few hours.
2026-11-18T22:33:06Z bench 4096/512 c1 done alone: TTFT 239.5 s, TPOT 108.0 ms (9.26 tok/s/user)
2026-11-19T03:13:16Z bench 4096/512 c8/c16/c32 done alone (all ok, 0 fail): TPOT 319/333/341 ms (prefills of queued requests interleave with decode), output 1.72 tok/s total. Started 32768/512 c1.
2026-11-19T03:45:23Z bench 32768/512 c1 done: TTFT 1945 s (32K prefill ~16.8 tok/s), TPOT 114.7 ms (8.72 tok/s/user). Started 32768/512 c8 (last bench row).
2026-11-19T08:15:34Z BENCH SWEEP COMPLETE: 32768/512 c8 done (TTFT mean 8996 s, TPOT 326 ms, 0 fail, 0 preemptions); table in RESULTS.md. Starting SWE-bench Pro alone (5 tasks, sequential).
2026-11-19T11:03:27Z SWE (alone) 1/5 ansible: unresolved, agent timeout 9900 s after 11 model calls (vs 8 when shared); fix absent, egress blocked, only server reachable. element-web running.
2026-11-19T13:49:36Z SWE (alone) 2/5 element-web: unresolved, agent timeout after 11 model calls; fix 18c03daa absent (correct hash now), egress blocked, only server reachable. vuls running.
