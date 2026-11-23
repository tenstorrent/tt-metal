# Draft: regression rule in the prompt (meeting action item)

*Status: DRAFT. Nothing has been added to the harness or the prompt. Only `test_eltwise_binary -k add`.*

## 1. What was asked
From the meeting notes (shared 2026-11-20): if the agent's code is worse than the current (original) code, the agent should look for a different way to use HAL and try a few times. If it is still worse, it should report it.

No prompt contains such a rule today (`grep -i regress` over `hal_sweep/harness_run5rules/prompt-*.md` finds nothing). Run 8b is therefore a plain repeat of 8a.

## 2. Decisions needed before adding it

### 2.1 What "worse" means
| Option | Measure | For | Against |
|---|---|---|---|
| **A. Cycles** | LLK perf test `perf_eltwise_binary.py -k Elwadd` (16 variants), L1→L1 cycles per tile, asserts off | real speed; repeatable: original 32.80 on all 4 boards | 73–83 s per measurement incl. reset; the perf test does not check results; only 4 of 16 variants are in the functional set |
| B. Code size (`.text`) | build without a card + count | fast, no card | smaller code is not faster code (see below) |
| C. TTI / TT | elfray / `tt_tti_counts.py` | — | the agent has no such tool in its workspace; TT is a secondary measure not fully trusted (meeting 2026-10-28) |

**Proposal: A (cycles) as the regression measure; B for information only.**
- Literature: a static instruction count is not a speed measure. KernelCraft reports 94.5% fewer instructions for only 2.9% more speed (`literatura-paket/1_specifikacija_benchmark_i_metrike.md:131`).
- Our data: all 6 HAL runs (7a–7e, 8a) are slower than the original in cycles. 7a–7e are 2.82–3.83× slower and 8a is 1.32× slower (`perf_ciklusi/reports_en/report_perf_cycles_2026-11-21.md`, table 5.1). The asserts-off code-size comparison will be added when the running rebuild finishes.

### 2.2 Other decisions
1. **Threshold.** How much slower counts as a regression? For 8a the largest spread over 3 repeated measurements was 0.1% (TILE_LOOP, `perf_ciklusi/rezultati/out/compare_8a_na.md`), so a threshold of a few percent keeps noise from triggering the rule. Proposal: 2%.
2. **Who measures.** The rule needs the agent to react, so the agent must measure during the run. A postcheck measurement comes too late.
3. **How the agent learns the original's number.**
   - (i) The harness measures the original before the agent starts, like the baseline, and writes the number into the prompt. Proposed: one number, no code visible.
   - (ii) The agent runs the original perf test itself. The original code is read-denied, but running a test is not reading.
4. **Number of attempts.** "A few"; proposal: 3.
5. **Budget.** Attempts mean more iterations, so more time and cost. Decide whether TIMEOUT 4 h and MAX_BUDGET $60 stay.
6. **Correctness first.** Every attempt must pass the full test and the 4-step reset / no-reset sequence, as now.

## 3. Draft prompt text
```
## Performance compared to the original
Correctness comes first: every version you keep must pass the full test and the 4-step
sequence above.
After your implementation passes, measure its speed with:
    {{PERF_COMMAND}}
It runs the HAL perf test (about 80 s) and prints the median cycles per tile
(L1-to-L1 tile loop, Elwadd variants). The perf test does not check results; only the
functional test does.
The original LLKs measure {{ORIG_CYCLES}} cycles per tile on this card.
If your implementation is more than {{THRESHOLD}}% slower, look for a different way to
express the slow part with HAL, pass the full test again and measure again.
Try at most {{N_ATTEMPTS}} different approaches.
Keep the fastest version that passes the full test. If it is still slower than the
original, say so in the final report: every approach you tried, its cycles per tile,
and what you think makes it slower (with the source for each claim).
```

## 4. What the harness would need if the rule is adopted
1. A HAL version of the perf test in the workspace, as in `perf_ciklusi/skripte/setup_ws.sh`: a perf `.cpp` with `#include "../hal_llk/..."` and a `.py` pointing to it.
2. A wrapper command that runs perf with `TT_LLK_DISABLE_ASSERTS=1` after a reset and prints only the median cycles per tile.
3. A measurement of the original before the agent starts, with the number written into the prompt (`{{ORIG_CYCLES}}`).
4. Permission for that command in `hal-settings-B.json`.
5. `metrics.py` counting perf runs separately, not as functional-test iterations.

## 5. Open for the user / mentor
- Measure (A / B / C) and threshold.
- Whether the agent may know the original's cycle count.
- Number of attempts and budget.
- Whether the rule goes only into the HAL arm or also into the RAW arm of set (b). For a fair comparison it should go into both.
