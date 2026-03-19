# Matmul N150 protocol — acceptance policy (Milestone 1)

This policy describes **criteria for judging** matmul model-traced runs on the **smoke / train / holdout** protocol. It is **documentation only** in Milestone 1: **no automated enforcement** in CI or local scripts unless you opt in later.

## Goals

- **Regression prevention** on a **diverse** set of traced shapes/sources, not a single hand-tuned case.
- **Separation of concerns**: **train** visibility for iteration; **holdout** for decisions that should not be overfit during tuning.
- **Stability**: correctness and absence of new hangs/timeouts before trusting performance numbers.

## Required conditions (when evaluating a change)

1. **Correctness** — No increase in assertion/PCC failures vs the baseline for protocol vectors (same `input_hash` set and manifest version). Treat unexpected **fail** statuses on previously passing hashes as regressions.

2. **Hangs / timeouts** — **Zero new** `fail_crash_hang` / timeout outcomes relative to baseline for the same manifest. (Infrastructure flakes should be triaged; the protocol is meant to make hangs visible.)

3. **Holdout p95 e2e (primary perf bar)** — For vectors in **holdout**, **p95** of `e2e_perf_ms` (with the same `--perf` / runner settings as baseline) must be **no worse than baseline** within an agreed tolerance, or **improved**. Suggested starting tolerance for exploratory work: **5%** relative increase allowed before calling it a regression (tune with more data).

4. **Train vs holdout** — **Train** metrics may improve during tuning; **holdout** is the bar for claiming general improvement. Reporting **train** and **holdout** side by side is mandatory when claiming perf wins.

5. **Memory (optional)** — If `--measure-memory` is enabled, track p50 **peak_l1_memory_per_core_bytes** (and aggregates if present). Policy: **no sustained regression** vs baseline on holdout without justification (e.g. deliberate accuracy/footprint trade documented elsewhere).

## Out of scope for Milestone 1

- Hard-failing CI on the above rules.
- Replacing the full model-traced sweep matrix with only this protocol (the protocol is an **additional** lens).

## Revision

Bump **`protocol_version`** in the manifest when split logic or partition parameters change so historical comparisons remain interpretable.
