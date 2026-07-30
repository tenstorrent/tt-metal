# The degeneracy guard: what it measures, and the terminal-padding fix

Status: current. The RNG root cause it was built against is fixed elsewhere
([device Gumbel restored](device_gumbel_restored.md)); the guard remains as defence in depth.
Owns: `tt/degeneracy.py` — what `top_frac`/`max_run` measure, why both are needed, the
healthy-vs-collapsed calibration, the termination-is-not-degeneration exemption (fixed 2026-07-28),
and the policy env var.
See also: [refuted list](../REFUTED.md) · [decision fidelity](README.md) · [plan](../../plan.md)

Over the 110-line target: the calibration numbers, the 07-28 defect measurement and the replay are
the reason this guard is trustworthy, and none of them is cut for length.

The `host` Gumbel mode this document originally recommended was **deleted 2026-07-28** — it drifts on
exactly the same prompts as `device`, repairs 0, and costs 1.40× per request. Every measurement below
stands as recorded; only the recommendation to use or fall back to `host` is void. `device` is the
only materialized Gumbel source and therefore the only mode valid under up-front capture. The commit
that landed the real fix (hiding the prefill pad keys) is credited two different ways in this tree —
see the [open contradictions](../REFUTED.md#open-contradictions-unexplained).

## What a collapse looks like

GPQA-Diamond doc_id=0 produced 3256 characters and no reasoning: a wall of `ní`, then `▁\` (id 621),
then `1` (id 236770). Ids 621 and 236770 are **the two most frequent tokens of the HEALTHY blocks of
that same LaTeX-heavy physics prompt**. So a collapsed canvas is not emitting noise — it is collapsing
onto the prompt's own most probable token, which is the signature of positions losing independence.

**Reproduction prerequisite:** those traces were produced in **thinking mode**, and `serving_smoke`
renders through `tokenize_prompt`, which could not emit the `<|think|>` turn until `--enable-thinking`
was wired in. With it, doc_id=0 replays at `prompt_len=157` — the exact length the server logged.

## The measure, and its calibration

`tt/degeneracy.py` reports **`top_frac`** (share taken by the most frequent id) and **`max_run`**
(longest consecutive repeat). **Both are needed:** the `\ \ \ \` 2-cycle has `max_run == 1`.

| on real traced blocks (12 blocks forced past the natural end) | distinct ids / 256 | top_frac | max_run |
|---|---|---|---|
| healthy | 54–106 | 0.06–0.08 | 1–2 |
| collapsed | 1–16 | 0.94–1.00 | 240–256 |

Over 192 committed canvases (the 10 worst GPQA docs replayed plus both 4-seed sweeps): healthy and
not stop-dominated n=136, max `top_frac` **0.1836**, max `max_run` **18**; degenerate n=1,
`top_frac` 0.8516, `max_run` 86. The defaults sit in that gap by an order of magnitude, and on that
replay nine of ten docs committed no degenerate canvas at all — doc0, the trace that emitted the
2000-character wall of `1` in serving, answers **C**, correctly.

**Termination is not degeneration.** Once the answer is complete the model fills the canvas with
`<eos>`, scoring `top_frac 1.0 / max_run 256` — the same numbers, the opposite meaning. The verdict
therefore takes the caller's stop-token set and never flags a canvas whose dominant id is a stop token.

**Placement is the point:** the check runs **after denoise and before `commit_fn`**. A degenerate
canvas that reaches the KV cache conditions every later block, making the state near-absorbing —
`P(nonhalt | prev nonhalt) = 85.7%` against an 8.2% base rate.

**Wiring defects only device runs exposed:** `serving.decode_block` was calling the commit path
**without** `stop_token_ids`, so the termination exclusion was inert on the path that matters and the
first `<eos>` canvas raised; and `stop_token_ids` can be a bare int from `eos_token_id`, so `set()`
on it raised `TypeError`.

**Design rule:** a refused block must not fail the whole request. The first attempt raised an
exception and lost the good text — "no degenerate output" must not mean "no output". The session is
marked finished and a zero-token terminal emission hands the caller every healthy block it already
received.

## 2026-07-28: the guard was rejecting normal completions

**Status: fixed — the verdict is now taken on the canvas's CONTENT region.** Three correct-looking
decisions composed into a wrong one: `is_degenerate()` exempts a stop-token-dominated canvas *only if
the caller passes `stop_token_ids`*; `tt/serving.py` passed the session's **stop policy** into that
argument; and `tt/generator_vllm.py:_make_session()` deliberately sets `stop_token_ids=[]` because
vLLM owns the stop decision. So on the vLLM path the exemption was **dead code** and the whole-canvas
rule ran unrestricted — and a block that ends at position 149 pads 107 positions with `<eos>`, i.e.
`top_frac 0.58` **and** `max_run 107`. **The terminal block of every answer shorter than the canvas is
structurally degenerate under that rule.**

Measured on tt-shield run 30285823000 (2026-07-27, 198 GPQA-Diamond requests): **130 of 198** requests
ended by the guard; **110 (85%)** had a stop token as dominant id, of which **108 were answer +
`<eos>` padding** (median 55 distinct ids in the discarded block); real tokens thrown away median
**107/block, 11135 total**; only 20 trips were on a content id (14 of them a full 256-wide wall); 9
requests returned an empty string. Score effect: **65.15 → 48.99** on `gpqa_diamond_cot_zeroshot`,
with responses reaching a final-answer statement down 122/198 → 37/198 and `\boxed` presence 119 → 43.

- **TRAP:** that 48.99 came mostly from the harness's fallback extractor picking the last `(X)` out of
  a truncated response — 66 of 97 correct answers, at 43% against a 25% random baseline. See the
  [three-denominator rule](README.md#gpqa-measurement-traps).

**The fix.** `block_degeneracy(tokens, stop_token_ids=...)` also reports the four statistics for the
**content region** (the canvas with its terminal stop-token run removed), and `is_degenerate()`
prefers those: answer + `<eos>` padding commits; an all-stop canvas is benign termination; a wall of
a content id still rejects; with no stop ids declared the whole-canvas behaviour is unchanged and the
serving layer logs why. The stop ids no longer come from the stop *policy* —
`serving._resolve_degeneracy_stop_ids()` takes an explicit set, else the session's policy if
non-empty, else **every special id the tokenizer knows** (a tail of `<eos>`/`<end_of_turn>`/`<pad>` is
padding under any tokenizer).

**Replay against the fix** — `gate/replay_degeneracy_verdicts.py <evals.log>` re-decides every canvas
from the `DG_DEGENERACY` telemetry. On the 07-27 run: 1842 canvases, **0 reconstruction mismatches**,
130 guard trips → **110 now allowed to commit, 20 still rejected, 0 newly rejected** among the 1712
healthy blocks. The 20 survivors are the shape the guard exists for: top ids 239054 (11×), 63405,
107, 167, 1340, 236743, 237808, 238408 at `top_frac` 0.55–1.00 over a full 256-token content region.

- **What the replay does NOT claim:** it proves the verdicts flip as intended on the real
  distribution; it does not predict the score. The 07-27 outputs differ from 07-24 from the first
  ~120 characters (retention default, Gumbel layout and a MoE HiFi2 revert all landed in that window),
  so only a re-run separates the guard's contribution from the model's.

## Not covered

Progressive degradation. The block immediately before a refused one was already repetitive
(`the the the ... ,,,1,1111`) at `top_frac` under 0.5. Catching that precursor needs a bound near
0.2–0.3 against a measured healthy maximum of 0.1836 — too thin a margin, so it is left uncaught
rather than traded for false positives on ordinary text. The residual case (doc7) sits in the
**over-generation** regime: it ran out of context before it ran out of reasoning, which is the
context-length bottleneck of
[gpqa_thinking3072](../optimize_perf/gpqa_thinking3072_sub40_20260723.md), a separate piece of work.

Halt telemetry cannot substitute for this guard — refuted, see the
[refuted list](../REFUTED.md#sampling-rng-and-decision-fidelity).

## Reproduce

```bash
# env: see plan.md — serving smoke that replays the doc-0 collapse regime
DG_TRACE_REGION_SIZE=12884901888 MESH_DEVICE=P150x4 \
python -m models.experimental.diffusion_gemma.demo.serving_smoke \
  --max-seq-len 4096 --num-blocks 12 --gumbel-mode device --upfront --reveal-pmax 4096 \
  --enable-thinking --disable-eos-stop --seed 0 --prompt "<gpqa doc 0>"

DG_DEGENERACY_POLICY=warn  ...   # logs DG_DEGENERACY start_pos=... top_frac=... max_run=...
DG_DEGENERACY_POLICY=stop  ...   # ends generation instead of committing a collapsed canvas
```

Current code state (`tt/degeneracy.py`): `POLICIES = ("off", "warn", "stop", "retry")`,
`DEFAULT_POLICY = "stop"`, `DEFAULT_TOP_FRAC = 0.5`, `DEFAULT_MAX_RUN = 64`, `DEFAULT_RETRIES = 2`.
Earlier revisions of this file claimed the default was `off` and separately `warn`, and quoted
`max_run >= 32` — all three are wrong; trust the code.

Unit coverage: `tests/test_degeneracy.py` pins the five real terminal-padding `(content_len,
top_frac)` tuples, the mixed-stop-id tail, both still-degenerate collapses and the no-stop-ids
fallback; `tests/test_serving_block_contract.py` pins that an empty stop policy still leaves the
guard a stop set — the exact composition that caused the 07-28 defect;
`tests/test_degeneracy_retry.py` covers the retry policy.
