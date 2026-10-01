# Measurement history

The recall and precision measurements behind the skill's design, in the order they were taken (2026-09-26). SKILL.md
keeps only what they mean for running an audit (*What the measurements mean*). This file is not loaded when the skill
runs; read it before changing the prompts, the classes or the packs.

## How the benchmark works, and why its numbers are not audit recall
`bench.py prepare` takes held-out real bugs from the repo's history and audits each one at the commit before its fix.
Each case is one batch holding only the files the fix touched: 36 of the 50 cases in the fresh holdout are a single
file, and none has more than 3. The hunter prompt also tells the hunter that it is auditing a benchmark tree. A real
audit hands a hunter up to 20 files and 300 lines (1,500-3,500 before the batch-size change below), and nobody knows whether any bug is in them. So the
benchmark measures recall when the broken file is already known. Recall in a real audit is probably lower, and
precision on normal-sized batches has not been measured.

Scoring is semantic (`bench.py judge-inputs`, then `judge-wave.js`, then `bench.py judged`): two blind judges per case,
who agreed on 809 of 810 findings. An earlier line-proximity scorer overstated some arms by up to 11 points and is not
used. A post-mortem found that 9 of the first 75 held-out "bugs" were not defects in the pre-fix code; they are excluded.

Run-to-run noise is about ±5 cases, so a difference smaller than that between two single runs means nothing.

## Round 1: first holdout, first version
Three arms, audited on 53 valid tt-metal and 13 valid tt-llk held-out bugs:

| Repo (valid held-out bugs) | Universal classes | + Tenstorrent classes | + mined pack |
|---|---|---|---|
| tt-metal (53) | 28% | 36% | 32% |
| tt-llk (13) | 38% | 38% | 38% (54% counting unconfirmed candidates) |

**Post-mortem of the pack arm's 45 misses** (37 tt-metal, 8 tt-llk, counted before the 9 invalid cases were excluded,
which is why 45 matches no cell above). The report is kept in the benchmark run's directory, not in this repo.
- Cross-file tracing not done (13), and the defect lived outside the batch (6): the largest group. The evidence was
  one hop away. Examples: a caller's `uint32_t` narrowed into a callee's `uint8_t` parameter that is packed into a
  4-bit field; `validate()` accepting layouts the program factory does not implement; a renamed keyword argument that
  callers still pass; unpack and math MOPs disagreeing on dvalid counts.
- Class missing from the lists (6), or domain knowledge missing (4): symbol visibility across shared libraries,
  preprocessor-macro collisions and `#elif` on undefined macros, namespace lookup shadowing, `.begin()` on a
  possibly-empty container, stall-resource versus stall-condition constants, register-literal consistency.
- Hunt variance (5): another arm found it. Attention (3): the site was read, but a low-salience defect went unnoticed.
- Verifier kills (2, both tt-llk): each finding was right but was held back because the library entry point had no
  in-tree caller; the callers live in another repo.
- Not statically visible (2): a compiled stack-frame overflow, and an undocumented NoC erratum.

**Fixes applied from it:** a mandatory one-hop contract trace, sibling comparison and mechanical sweeps in the hunter
prompt; the missing classes (`sibling-divergence`, `dead-store`, `empty-access`, `lossy-compare`, `validate-vs-impl`,
`symbol-resolution`, `preprocessor`, `linkage-visibility`, and `tt-` classes for dvalid balance, stall operands,
register literals, hardcoded arch constants, core-flavour maps, JIT defines and dispatch field width); a verifier rule
that library entry points are reachable by default; a stricter triage definition of "code bug"; a holdout validity
screen; deep-read verdicts overriding triage in the weights; and the second-pass step.

## Round 2: the same holdout, after the fixes (in-sample)

| Repo | Universal classes | + Tenstorrent classes | + mined pack |
|---|---|---|---|
| tt-metal (53) | 42% | 45% | 38% |

The universal arm went from 15 to 22 of 53 (8 gained, 1 lost; exact McNemar p ≈ 0.04). **This is in-sample:** the
classes and the trace rule came from this holdout's own misses, the gain is 7 cases against ±5 noise, and p ≈ 0.04 is
one of several comparisons. It is not evidence the fixes generalise; round 3 is the out-of-sample number.

- **The pack showed no measurable recall benefit in either round.** Before the fixes it scored 5 vs 3 against the
  baseline (p ≈ 0.7); after them, 1 vs 5 against the domain arm (p ≈ 0.2). A "significant pack win" first reported was
  a line-proximity artifact. A transcript diagnostic found no case where the pack misled a hunter; hunters spent about
  5-20% of their tool calls on it. The pack arm confirmed the most findings in the same files (145, against 128 and
  116), but whether those extra findings are real was not measured.
- Three post-fix arms together found 26 of 53 (49%), against 45% for the best single arm. All six runs together: 30.
- One defect per finding: a finding that bundles a real defect with a wrong claim gets refuted whole.

**Post-mortem of the best post-fix arm's 29 misses:** the hunter read the buggy lines in 27 of 29 but only skimmed
them; the required contract trace was done fully in 2 (partly in 20, not at all in 5). For 11 of 29, execution was the
cheapest reliable catch: a non-default build flavour (5), a sanitizer or static analyzer (4), an existing test on the
target arch (2). For 16, the analysts proposed a narrow prompt rule or class, which would overfit if added one by one.
Only 1 of 29 was not findable from source or execution. **Built from it** (never measured in isolation): the recorded
and audited contract-trace ledger, smaller priority-A batches, the "look hard / one defect per finding" hunter rules,
and the optional execution tier.

## Round 3: a fresh holdout (the only out-of-sample number)
`packs/tt-metal-holdout-v2.jsonl`: 50 screened bugs, none in the first holdout or among the deep-read fixes; 3 excluded
as contaminated. Static only; universal + Tenstorrent classes, **with the pack handed to hunters** (the default no
longer does this), the contract-trace ledger and its trace audit, and the post-mortem hunter rules.

- **Recall: 23 of 47 = 49%, exact 95% interval 34-64%**, on benchmark batches of 1-3 files (see the top of this file).
- **Precision:** 24 of a random 25 non-benchmark confirmations were judged real and reachable by two independent
  reviewers (96%, 95% interval about 80-99%), who agreed on all 25; 19 of the 25 were later fixed on main by other
  commits. Same 1-3 file batches, so precision on normal-sized batches is unmeasured.
- The trace audit overturned 24 of 223 re-checked "consistent" verdicts (11%).
- Cost: one 50-batch wave took 37M tokens and 881 agents (about 17.7 agents per batch).

## Lines per hunter: what sets depth on full-size code
The benchmark's 1-3 file cases cannot measure batch size, so it was measured on real code: 75 files of ttnn untilize
and argmax at a pinned commit, hunt only, the same tree and class lists in every arm. Four candidates that no
default-size hunter reported were verified 3-0 by the standard screen and deep verifiers.

| Arm | Lines per agent | Verified bugs found (of 4) | Hunter cost |
|---|---|---|---|
| Default batches | 1,600-2,600 | 0 | 1.28M |
| Hunter fills a unit x class-family grid | 1,600-2,600 | 0 | 4.59M |
| Engine hands each agent a unit list | 800 | 1 | 2.83M |
| Engine hands each agent a unit list | 300 | 4 | 5.57M |
| Default hunter, `--max-lines 300` (two runs) | about 300 | 4, then 4 | 6.30M, 6.34M (trace audits another 3.3M) |

Cost is price-weighted tokens (cache reads at 0.1, cache writes at 1.25, output at 5). Each arm ran once, except
the shipped one, which was rerun on the same batches: both runs found the same 4 bugs, plus the same 6 of the 7
that the default batches found, at the same cost.
- Depth follows lines per agent; listing the units to check does not add it. Both unit-list arms assigned every unit.
- A hunter-written coverage table is satisfied mechanically: hunters scripted it, filling "n/a" from per-kind defaults
  and deriving each citation's line from its quote, and a 945-cell table was too large to return. Coverage evidence
  has to come from the engine, never from the hunter.
- Whether a hunter handling every class family at once skips some families was measured once, with 68 bugs planted
  across 12 families in the same 300-line batches: hunters caught 58 of them (85%), but only 1 of 5 numerics plants.
  The plants also displaced the real findings (0 of the 10 real bugs were reported), so calibrate in a separate run
  and never plant inside an audit.

## Does verification earn its cost?
On round 3 verification confirmed 260 of 261 candidates: tiny batches that really held a bug produced almost no false
candidates. On noisier input it refutes a lot:

| Run | Input | Refuted by verification |
|---|---|---|
| Whole-repo tt-metal audit (July) | normal batches, up to 20 files | 262 of 920 (28%), with a refute-if-unsure verifier |
| Sibling sweep | history leads, the current three-valued verifier | 220 of 776 (28%), plus 20 uncertain |
| Round 3 benchmark | 1-3 files, a known bug in each | 1 of 261 |

## How to re-measure
After any change to the prompts, classes or packs: `bench.py prepare` into fresh run directories, then
`score --cases <holdout>`. Measure rules derived from a holdout's misses on a FRESH holdout (`select.py holdout --seed
<new>` plus the screen, excluding the old ones), never on the cases they came from. Do not edit a knowledge file while
a run is in flight: hunters read them at start time. For long runs, launch each arm with `engine/run_headless.py --run
<arm> --bench-cases <holdout>` in tmux (a plain `claude -p` exits after about 10 minutes and kills its workflow).
