---
name: eval-run-audit
description: "Audit a finished DiffusionGemma GPQA eval run end to end: prove it is not a false green, LLM-judge every response for whether it is meaningful and actually answered, compare paired against a baseline run, and measure output speed from block telemetry. Use whenever a tt-shield CI evals job or a local run_upfront_gpqa run finishes and its score needs to be believed, or when two runs need comparing."
---

# Eval-run audit

A finished run gives you one number. This turns it into a defensible one: whether the run really
happened, whether the score is inflated by the regex extractor, whether a difference against another
run is real, and what it cost in wall clock.

Everything here is offline analysis of files a finished run already produced. **No device, no server, no
`tt-smi`.** It cannot break a run and needs no hardware.

## Step 0 — find the run

**tt-shield CI job.** The samples file ships inside the workflow-logs artifact:

```bash
gh run view <run-id> --repo tenstorrent/tt-shield --json displayTitle,conclusion,createdAt,updatedAt
gh api repos/tenstorrent/tt-shield/actions/jobs/<job-id> --jq '{name,started_at,completed_at,conclusion}'
gh run download <run-id> --repo tenstorrent/tt-shield -n workflow_logs_evals_<model>_<device>_default
#   -> reports_output/evals/**/samples_*.jsonl     the 198 responses
#   -> reports_output/evals/**/results_*.json      the official score
#   -> docker_server/vllm_*.log                    the block telemetry
```

**Local run on the box.** `~/dg_runs/<name>/full/**/samples_*.jsonl` plus `<name>/server.log`. Analyse
the log **on the box** (`scp` the script over, not the 20 MB log back).

## Step 1 — prove it is not a false green

Do this before reading any score. A green tt-shield evals job has published a full-looking score after
the engine died on request 8 of 198, because lm_eval writes a results file regardless.

```bash
grep -c prefill_block0 <server.log>       # must be >= the question count
grep -c flexible-extract <samples.jsonl>  # questions actually written
```

- **Count == questions** → real.
- **Count slightly above** (200 for 198) → the smoke stage ran 2 questions against the same server.
  Expected; `block_speed.py --expect 198` drops them.
- **Count below** → the engine died mid-run. **The score is void.** Say so and stop; do not report it.

Also check `empty 0` in the response-length line printed by step 2, and that
`results_*.json` carries `exact_match,flexible-extract` (strict-match is near-zero by design and is not
the score).

## Step 2 — judge every response

```bash
export ANTHROPIC_API_KEY=...        # pip install anthropic
gate/llm_judge.py <samples.jsonl> --effort high --max-chars 44000 --out verdicts.jsonl
```

Per response it returns `meaningful`, `failure_mode`, `language`, `answered`, `selected_letter`. The
judge never sees gold; correctness is computed locally, so it cannot launder a non-answer into the right
letter. Set `--max-chars` above the longest response (check the `max` in its length line) so nothing is
elided. `--votes 3` majority-votes when a call is borderline; there is no temperature on Opus 5, so the
spread comes from sampling non-determinism.

**The number to look for is `regex credited a non-answer`.** That is lm_eval's `flexible-extract`
scoring a response that never committed to an answer — a real measured 16 of 197 on one run and 2 of 197
on another. The gap between the official score and judge-confirmed correctness is how inflated the
official number is (8.3 pp before a fix landed, 1.1 pp after).

~198 calls, about 20 s and roughly $7 at `--effort high`. Costs real money: say the estimate before
running it on someone's behalf.

## Step 3 — compare against a baseline, PAIRED

```bash
gate/compare_runs.py baseline_verdicts.jsonl candidate_verdicts.jsonl --labels "CI" "candidate"
```

**Never conclude from the aggregates.** 72.1% vs 69.5% reads as a 2.6 pp regression; paired it was 18
questions one way, 13 the other, two-sided p = 0.47 — noise. Binomial sigma at n=198 is already 3.3 pp,
so anything under ~6 pp needs this test before it means anything.

Joins on `doc["Record ID"]`. Not `doc_hash` (lm_eval reshuffles the choices per run, so it matches 18 of
198) and not letters (the same answer is a different letter in each run). Both runs must cover the same
question set — a partial run's answered set is a prefix, not a sample, so comparing it to a full run's
figure is wrong in an unknown direction.

The `BOTH` list at the end is the payoff: responses that failed in every run are question-specific and
directly reproducible, not flake. Five Record IDs have now failed across three separate runs.

## Step 4 — output speed

```bash
gate/block_speed.py <server.log> --expect 198
```

**Ignore vLLM's `Avg generation throughput` line.** DG commits a 256-token block at once, so vLLM reports
instantaneous spikes — 921, 972, 1100 tok/s in a log whose sustained rate was 44.7. This script uses
committed tokens over block latency.

Reads out aggregate tok/s, per-block p50/p90, decode-block latency, the denoise/commit split, denoise
steps per block with the halt rate, and TTFT. Cross-check the total block time against lm_eval's own
progress-bar wall clock (`grep -oE '[0-9]+/198 \[[0-9:]+<' full.log`) — they should agree to a few
percent, and agreement across those independent measures is what makes a speed claim credible.

Attribute a speedup honestly: compare `blocks`, `committed_tokens`, and response-char mean too. A run
that is 20% faster with the same block count, token count, and text length got faster **per block**; one
with fewer blocks produced less output, which is a different claim.

## Step 5 — report

Lead with whether the run is real, then the two accuracy numbers side by side (official and
judge-confirmed) with the laundered count between them, then the failure-mode table, then speed. State
the denominator whenever it is not the full question count.

## Traps that have actually cost time

- **The judge refuses one specific question** (`recTs7qzfJs6kfLUK`, category `bio`) on every run. Stable,
  not flake. It makes the denominator 197 — say so rather than quietly reporting /198.
- **GitHub truncates gist and API file content.** A file with `"truncated": true` comes back short (one
  came back 15 KB light, cutting the last question in half). Fetch `raw_url` instead.
- **`str.splitlines()` shreds real model output.** It splits on U+2028/U+2029/form-feed/NEL, which JSON
  does not escape, so one inside a response cuts its JSON line in half — a real 198-question file has
  four. Split on `"\n"`.
- **Response length is not a proxy for quality.** One run had identical mean chars to its baseline
  (11,069 vs 11,110) while a note predicted 27% shorter; and non-answers are concentrated in the
  *longest* responses, which are degenerate loops that ran to the length cap.
- **A `--limit` on the judge is not a sample.** It takes the first N, so its score is not comparable to a
  full run's.
- Validate any parse you write against something the source already states — per-question char counts,
  a published table, the official score. A parse that reproduces those is trustworthy; one that merely
  runs is not.
