# DiffusionGemma 26B-A4B-it — SWE-Bench Verified (GPU reference baseline)

Frozen, reproducible baseline measured on the A100 reference vLLM server, intended to be
re-run byte-identically against the TT vLLM plugin server. Measured 2026-08-04 → 2026-08-06.

## Result

100-instance fixed subset of SWE-Bench Verified, graded by the official harness
(`swebench` 4.1.0, `princeton-nlp/SWE-bench_Verified`, split `test`).

| metric | value |
| --- | --- |
| **resolved** | **12 / 100 = 12.0%** |
| submitted a patch | 22 / 100 |
| **resolved given a patch was submitted** | **12 / 22 = 54.5%** |
| empty patch (agent killed before submitting) | 78 / 100 |
| harness errors | 0 |

Read the two rates together: the model solves **54.5%** of the tasks whose trajectory it
manages to finish, but only **22%** of trajectories finish. The 12.0% headline is dominated
by trajectory mortality, not by the ability to write a correct patch.

Resolved instances: `django-9296, django-11099, django-11163, django-12143, django-13670,
django-13821, django-16493, django-17087, matplotlib-20859, matplotlib-25332,
psf__requests-1142, pydata__xarray-2905`.

## Why 78% of trajectories die

Every one of the 78 failures is `RepeatedFormatError`: mini-swe-agent kills an instance after
`max_consecutive_format_errors: 3` consecutive turns that do not contain exactly one parsable
action. Over all 8398 assistant turns:

| | valid action | failed to emit an action |
| --- | --- | --- |
| turns | 7053 (84.0%) | 1345 (16.0%) |
| median prompt tokens | 32356 | 47309 |
| median output tokens | 169 | — |

The per-turn miss rate scales with context, and that is the whole mechanism:

| prompt tokens | turns | failed |
| --- | --- | --- |
| < 4000 | 331 | 3.0% |
| < 16000 | 1067 | 5.0% |
| < 32000 | 1831 | 10.3% |
| 32000–64000 | 4076 | 22.7% |
| ≥ 64000 | 557 | 25.0% |

By turn index the same curve: 4.5% before turn 5, 8.6% by turn 40, 26.9% after turn 80. A 16%
per-turn miss rate against a 3-consecutive-miss kill rule is fatal over the median 77-turn
trajectory — the dead instances were mostly working normally for 100+ turns first (first miss
typically at turn 2–28, death much later). No instance hit the 250-step limit, so `step_limit`
was never the binding constraint.

Hard degeneracy is real but is not the bulk of it: only 1.7% of turns have 8-gram repetition
> 0.3, though 75.7% of those turns fail. The failures show up as corrupted fence tags
(`mswea_bash_`, `msa_bash_command`, `mswea_bashcommand`), ```python / bare fences instead of
the required one, several draft commands in one turn, and — in the worst cases — word salad
(`the the the`, `mmmm......`) or `ls` output pasted back as if it were a command.

## What was ruled out

Recorded so these are not re-investigated:

- **A tolerant fence parser — REFUTED.** Accepting fuzzy tags / ```bash / bare fences with a
  last-block-wins rule was built and replayed offline over 540 recorded format-error turns: it
  recovered only 11.3%, and the recovered "commands" were degenerate text. It would have
  manufactured a plausible-looking fake run. Do not reintroduce it. See `dg_mini_model.py`.
- **`max_tokens` too low — NOT the issue.** `finish_reason=length` is 0.0% of turns; output is
  median 169 / max 8141 tokens. Lowering the cap from 16384 to 4096 would cut 1% of generated
  tokens and truncate 0.2% of valid turns.
- **Server wedged / KV thrash — NO.** KV-cache usage 40%, 0 preemptions, prefix caching on with
  a 43% hit rate, 3 client retries in the entire run.
- **The expensive tail was worthless.** At 63/100 the score was already 12 resolved / 21
  submitted; the final 37 instances (≈24 h of exclusive GPU, the longest-running survivors)
  added 1 submission and **0** resolutions. A prediction that the partial number would
  *understate* the final was wrong — 63-instance 19.0% overstated the final 12.0%. If this is
  re-run under time pressure, the tail is not worth waiting for.

## Cost and the throughput wall

~43 h wall clock for the agent phase at 10 workers, plus ~3 min grading. The binding limit is
generation speed at long context: **~4 tokens/s per stream (25.6 tok/s aggregate over 6
concurrent requests)** versus ~280 tok/s at short context on the same server. Raising worker
count does not help — the GPU is already at 100% with `--max-num-seqs 6`. Agentic evals pay
this cost in repeated long-context forwards, not in output length.

## Context requirement — blocks the TT comparison

Per-instance peak prompt tokens: median **44835**, p90 **64368**, max **110648**.

| context cap | instances whose peak fits |
| --- | --- |
| 16384 | 16 / 100 |
| 32768 | 35 / 100 |
| 65536 | 91 / 100 |
| 131072 | 100 / 100 |

QB2 currently caps at `max_context 16384`, which covers 16/100 instances — the TT side cannot
run this benchmark meaningfully until that cap rises. 65536 covers 91/100.

## Frozen baseline contract

Any comparison run must match all of this:

| knob | value |
| --- | --- |
| dataset | `princeton-nlp/SWE-bench_Verified`, split `test` |
| subset | the 100 ids in `swebench_verified_subset100.json` (`random.Random(0).sample(sorted(ids), 100)`) |
| agent | mini-swe-agent 2.4.6, config `swebench_backticks.yaml` |
| model class | `dg_mini_model.DGTextbasedModel` (reasoning→content transport fix only) |
| `agent.step_limit` | 250 (the mini-swe-agent published value) |
| `agent.max_consecutive_format_errors` | 3 (stock default, NOT relaxed) |
| `model.model_kwargs.max_tokens` | 16384 |
| sampling | server-side defaults (`--generation-config vllm`), thinking **ON** |
| server | `--max-model-len 131072 --max-num-seqs 6 --max-num-batched-tokens 4096 --enable-chunked-prefill --reasoning-parser gemma4` |
| grading | `swebench` 4.1.0 `run_evaluation`, `--cache_level env` |

The single harness deviation is `dg_mini_model.py`: the `gemma4` reasoning parser frequently
leaves `message.content` empty and puts the whole formatted action in `reasoning`, so content
falls back to the reasoning text. Without it every instance dies within a minute (8/8 in the
first smoke) and the failure is easily misread as a model capability failure.

## Reproduce

```bash
# agent phase (A100 reference server)
./run_swebench.sh swe_verified_100 10

# same harness against a TT vLLM plugin server
DG_BASE_URL=http://<tt-host>:8000/v1 ./run_swebench.sh swe_verified_100_tt 10

# grade
./eval_swebench.sh swe_verified_100 dg_strict_100 12

# turn-level format-adherence / degeneracy analysis
python analyze_turns.py swe_verified_100
```

`make_subset.py` regenerates the subset id list and the `--filter` regex from seed 0.
