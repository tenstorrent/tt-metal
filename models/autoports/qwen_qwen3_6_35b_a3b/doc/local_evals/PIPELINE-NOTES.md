# Pipeline notes: Qwen3.6-35B-A3B local evals on M2 (qb2-120-p04t09)

Every problem hit during the local benchmark/eval campaign, with cause, fix, and whether the bring-up pipeline
should have caught it.

## Setup observations
- HF token missing on M2 (`~/.cache/huggingface/token`): GPQA Diamond (gated) blocked until owner provides it.
  Pipeline: stage-0 setup could check for a token when gated evals are planned.
- `systemctl list-units 'actions.runner.*'` lists nothing on M2; the host runs rke2/kubelet, so any CI that resets
  chips would come from k8s pods, not a systemd runner. Check `journalctl` + pod list if the server dies unexplained.
- Server startup (weight load, 40 layers) takes ~20 min before traces/warmup.

## Sampler: card's presence_penalty forces host sampling (performance, not correctness)
- What: model-card thinking sampling is T=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5.
  `tt/serving_capabilities.py` declares `max_device_top_k: 64` (top_k=20 OK on device) but
  `supports_device_penalties: False`, so `model_runner._can_sample_on_device` returns False for any batch that
  contains a request with presence_penalty != 0 -> host-logits route for the whole batch.
- Measured (1 user, 300 tok ignore-EOS decode, 32-slot server): presence_penalty 0 -> 29.6 ms/token;
  presence_penalty 1.5 -> 51.0 ms/token (1.7x slower).
- Fix: none applied (would need device penalty support in the model sampler). GPQA still uses the card sampling
  (correctness first); the cost is reported.
- Pipeline should have caught: yes. Stage 10/11 measured "all-device sampling" with greedy/no-penalty workloads,
  but the model card's recommended sampling (which every eval uses) includes presence_penalty=1.5. A stage-10 check
  "decode TPOT under the card's recommended sampling params" would have exposed it.

## Decode step time cliff above 4 active users (performance; known stage-10 limitation, now quantified per concurrency)
- What: 128/128 sweep: c=1 TPOT 31.4 ms (solo trace); c=8 407 ms; c=16 412 ms. With max-num-seqs 32 the adapter
  captures traces model_solo0 / model_2 / model_4 / model_fixed(32); `tt_supported_decode_batch_sizes = None`
  pads every batch of 5..32 rows to the fixed 32-row trace, whose step costs ~410 ms. So 8 users get 2.45 tok/s
  each (19.6 tok/s aggregate), slower in aggregate than 1 user (31.9 tok/s).
- Fix: none in this campaign (model performance work, out of scope). Matters for evals: aggregate throughput only
  beats single-stream at ~c>=13; agent runs (1 stream) get the fast solo trace.
- Pipeline should have caught: it did record the 32-user floor miss (owner-accepted), but no stage reported the
  per-concurrency step-time curve or the c=8 aggregate < c=1 aggregate inversion; stage 11 should sweep
  concurrency 1/2/4/8 on the 32-slot server and flag non-monotonic aggregate throughput.

## Agentic evals are prefill-bound: 1 SWE-bench task ran 5h19m (stopped by owner 3 h rule)
- What: mini-swe-agent on django__django-16100 (one agent, nothing else on the server), 01:27-06:46Z:
  27 model calls in 19,110 s; ~839,600 prompt tokens prefilled vs ~7,400 generated (server stats lines summed).
  Average prompt ~31K tokens per call; late turns took ~11 min each (requests at 05:28, 05:39, 05:50, 06:11,
  06:23, 06:34). Effective prefill ~44 tok/s over the whole window (serial prefill ~22 ms/token, the same rate the
  sweep shows: 4K = 91 s, 32K = 729 s). Decode is only ~1% of the time.
- Why: no prefix caching (`supports_prefix_caching: False`), so every turn re-prefills the whole growing transcript;
  prefill is ~45 tok/s on this port (vs thousands of tok/s typical on GPU). Cost per call grows linearly and total
  cost quadratically with turns. A 30-50-turn SWE task needs 5-15 h.
- Fix (harness): per-task 3 h wall clock: `agent.wall_time_limit_seconds=10800` (mini-swe-agent native, checked
  before each model call) + `evals/swebench/run_one.py` wrapper that submits the container's working-tree diff on
  TimeExceeded (mini would otherwise submit an empty patch) + 3.5 h outer hard stop. TB2 already used the card's
  3 h agent timeout.
- My own mistake while stopping task 1: `pgrep -f "bash .*run_swe.sh"` also matched my control shell, and killing
  the runner closed the tmux session, whose SIGHUP killed mini before it saved its trajectory. The patch was
  captured from the container first, so scoring is unaffected; step/token counts come from server logs.
- Pipeline should have caught: yes. Stage 11 measured 32K prefill (742 s) but did not translate it into an agent
  cost; a "multi-turn agent transcript" probe (e.g. 20 turns growing to 30K tokens) would show hours per task.
  Prefix caching (or at least much faster prefill) is the lever that matters for agentic use.
