# PIPELINE-NOTES: problems and fixes during local benchmark/evals (Qwen3.8-Flash-Next, QB2 M1)

Format: what broke / why / fix / should the bring-up pipeline have caught it.

1. **Final serving script self-terminates after 24 h.** `doc/optimized_vllm/run_server.sh` runs
   `exec timeout 86400 ...`. Evals on this port take far longer than 24 h (≈9.5 tok/s aggregate decode,
   ≈17 tok/s prefill). Fix: copy to `evals/server/run_server.sh` with `timeout 604800`; nothing else
   changed. Pipeline: yes - the hand-off serving command should not carry a 24 h kill switch (or should
   document it).
2. **Throughput reality vs eval scope (planning, not a bug).** Stage 10/11 measured prefill ≈17 tok/s
   (32K TTFT ≈1950 s, insensitive to token budget 9/17/33) and ≈9.5 tok/s aggregate decode with only 2
   resident slots (batch 2 gives no aggregate gain over batch 1). A 198-question thinking-mode GPQA run
   alone is tens of hours; SWE agents without prefix caching (recurrent-state exclusion) re-prefill their
   whole context each turn at ≈17 tok/s. Pipeline: stage 10's eval forecast (120 h for the default plan)
   already flagged this; the hand-off should state the per-eval wall-time explicitly.
3. **Served model id is the snapshot path.** The final serving command has no `--served-model-name`, so
   requests for `Qwen/Qwen3.8-Flash-Next` get 404. Fix (client side, no server restart): GPQA client, bench
   client and the SWE-bench Pro eval config use the snapshot path as model id (local commit `1b639753a` in
   `evals/tt-inference-server`). Pipeline: yes - the serving recipe should pass
   `--served-model-name Qwen/Qwen3.8-Flash-Next` so standard eval harnesses work unchanged.
4. **Device sampler check (OK).** Card thinking sampling (T=1, top_p=0.95, top_k=20, no penalties) is
   accepted by the adapter (device path supports top_k 1..40; penalties/logprobs/top_k>40 are rejected
   unless host sampling) and the route events show `sampling_mode:"device"` for it; no host fallback.

## 5. SWE-bench Pro agent budget vs device throughput (ansible task)
- What happened: the ansible trial hit Harbor's agent timeout (override_timeout_sec 9900 s, owner config) after
  only 8 mini-swe-agent model calls. Each turn re-prefills the growing context at ~17 tok/s and decodes at
  ~4 tok/s (device shared with GPQA), so one turn costs ~15-20 min. The verifier still ran on the partial diff
  (reward 0).
- Why: the owner's timeouts were sized for a faster deployment. On this QB2 recipe (2 resident slots, token
  budget 9, prefix caching off) a multi-turn agent cannot finish in 2.75 h.
- Fix: none applied. I kept the owner's exact timeouts as the CLAUDE.md asks, so the result is reported as
  "unresolved (agent timeout)". Not a harness bug.
- Should the pipeline have caught it: yes. Stage 11 should publish an "agentic feasibility" number, e.g.
  seconds per turn at 32K context, so eval budgets can be sized before a run. Enabling prefix caching for
  the recurrent model (currently excluded) is the biggest lever for agents.

## 6. My egress watcher checked the wrong fix commit for element-web and vuls (eval-side bug, fixed)
- What broke: `swepro/egress_watch.sh` derived the task from the container name with a regex on the full
  instance id, but Harbor lowercases and truncates container names (`instance_element-hq__element-web__xphjmiq`),
  so no match; the dataset glob then fell back to the first task dir and the "fix_absent" line for those two
  trials tested an unrelated ansible commit. Egress results (github/pypi/example.com blocked, server reachable)
  were unaffected.
- Fix: map by repo keyword; watcher restarted 22:17Z before the openlibrary container reached the 5-min mark.
  Fix-commit absence for element-web and vuls rests on the image verification of the identical task Dockerfiles
  (`verify_images_prebuild.txt`: fix absent, refs=0, remotes=0, HEAD=base) plus the build-time assertion in the task Dockerfile (build fails if the fix commit is reachable) and the leak scan; Harbor deletes its
  per-trial images, so they could not be re-checked after the fact.
- Pipeline should have caught: n/a (my script). Suggestion: Harbor/swebench_pro.py could log a fix-absence
  assertion per trial in trial.log so no external watcher is needed.

## 7. Running evals in parallel halved per-request speed (2 resident slots)
- What happened: GPQA (1 request) and SWE-bench Pro (1 agent) shared the server. With `--max-num-seqs 2`, two active
  requests split decode: ~9.7 tok/s for one user becomes ~4.3 tok/s each, and each agent turn's re-prefill (~17 tok/s)
  also stalled the GPQA decode. All three SWE trials hit the 2.75 h agent limit after 8-11 model calls.
- Decision (owner, 2026-11-18): run one workload at a time. The 10 GPQA answers already scored were kept (accuracy
  does not depend on load). The SWE trials from the shared run were discarded and will be rerun alone.
- Pipeline should have caught: partly. The CLAUDE.md lesson ("run SWE agents alongside GPQA") assumes spare slots.
  The stage-11 report should state the slot count and single-user vs 2-user tok/s, so eval plans know that
  concurrency does not add throughput on this recipe.
