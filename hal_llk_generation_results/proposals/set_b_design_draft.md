# Draft: set (b) "spec only", HAL and RAW in parallel (meeting action item)

*Status: DRAFT for discussion. Nothing has been launched or changed. Only `test_eltwise_binary -k add`.*

## 1. What was asked
From the meeting notes (shared 2026-11-20), mentor 2 asks for two **separate** test sets:
- **(a) "HAL performance":** take an existing implementation and convert it to HAL. This is our mode A, never run on the frozen configuration.
- **(b) "Spec only":**
  1. the existing kernel calls + a **short description of each API**, written **before** the kernel is deleted;
  2. then the kernel is **deleted**;
  3. HAL and RAW are generated **in parallel**.

This draft covers only (b). The RAW runs 7a_raw–7e_raw were deleted on 2026-11-21 as wrongly generated, so RAW starts from zero in set (b).

## 2. What stays and what is new
**Now** (mode B, runs 1–8b):
- `make_spec.py` makes the spec. It keeps comments, `#include`s, namespaces, enum/struct declarations and function signatures with their doc comments. It replaces function bodies with `;` and drops macros with raw instruction encodings (`hal_sweep/harness_run5rules/make_spec.py`, lines 2–6).
- The original LLKs stay in the workspace but are read-denied. The harness uses them for the baseline and for the dirty-state check.

**New in set (b):**
| | Now | Set (b) |
|---|---|---|
| API description | existing doc comments only | **a new short description per function** |
| original kernel in the agent's workspace | present, denied | **deleted** |
| HAL and RAW | run at different times | **at the same time**, same input |

## 3. Proposed steps
1. **API list:** every LLK function that `eltwise_binary_test.cpp` calls for add.
2. **API descriptions** (the key decision). Who writes them:
   - (A) a separate agent reads the original LLK and writes 2–4 sentences per function, then a human reviews them;
   - (B) the team writes them by hand;
   - (C) existing doc comments only, which does not meet the request.

   Content rule: **what** a function does (inputs, outputs, the state it leaves for later calls), **not how** (no instruction names, registers or constants). Otherwise the description would help one arm more than the other.

   **Proposal:** (A) + review. The file is frozen (md5) before the first run, and the **same file** goes to HAL and to RAW.
3. **Deleting the kernel:** in the agent's workspace, the files of the LLK functions the test uses are replaced by the spec and descriptions instead of staying read-denied. The baseline and dirty-state checks run in a **separate reference workspace** that the agent cannot see.
   - It needs checking which files are deleted.
   - It needs checking that the HAL and RAW tests still compile without them. In the HAL arm, HAL stays but is read-denied, as now.
4. **Parallel launch:**
   - HAL and RAW of the same repeat start at the same time on two boards, with a random board per arm;
   - same model, same claude version (2.1.287, pinned), same budget and same prompt template (`prompt-base.md` + `api-hal.md` / `api-raw.md`);
   - with 4 boards, 2 pairs can run at once.
   - Lessons from the earlier parallel launch: stagger starts ≥ 20 s, then check that every workspace denies all the others.
5. **Repeats:** the spec asks for ≥ 5 per arm (`literatura-paket/1_specifikacija_benchmark_i_metrike.md:80`). For add that is 5 pairs = 10 runs. One HAL run cost $21.24–33.56 (7a–7e and 8a, `logs/SUMMARY.txt`). There is no valid RAW number. If RAW costs about the same, 10 runs are about $210–340.
6. **RAW reference material:** the exact list of headers and ISA docs is a team decision (`1_specifikacija_benchmark_i_metrike.md:219`).
7. **Measurement, the same for both arms:** success (independent check + 4 steps), cost, iterations and failure types (`metrics.py` v2), `.text` / TTI / TT **asserts off**, cycles (perf test).

## 4. Open questions
1. Who writes the API descriptions and what they may contain.
2. Which files are deleted and how that is checked.
3. RAW reference material.
4. Number of repeats and budget.
5. Whether the regression rule also goes into set (b), in both arms.
6. On bh-30 there is a folder `results/set2_hal_vs_raw` that is not from my notes and that I have not opened. Is it already the start of set (b)?
