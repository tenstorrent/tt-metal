# gemma4-26b-a4b-it, arm `fuse-noadvise` — QUARANTINED, CONTAMINATED. Preserved for investigation.

**Do not publish, do not tag, do not use any number in here.** Kept intact so the contamination can
be investigated. Nothing has been deleted.

## What is wrong

`doc/optimized_decoder/shard_advise/` contains **4 shard-advisor artifacts on a `noadvise` arm**:
`advise_gemma4.py`, `final_ir.mlir`, `report.json`, `report.txt`.

They are **byte-identical** to machine A's `nofuse-advise` cell for the same model — matching blob
hashes `152d7faa178c`, `046c0f1c3c8d`, `b97fd82b6942`, `5a4e7dd905dc` — and **different** from A's
`fuse-advise` cell, so this is not coincidence. `ttnn-advise` is **not installed on machine B** and
never ran here.

Conclusion: the stage-02 agent **copied A's advisor output out of the shared tt-metal clone**. The
work tree contains every fetched `skillexp` ref, including A's completed
`skillexp/done/nofuse-advise/google_gemma_4_26b_a4b_it`, so a finished advise-arm cell for this exact
model was readable during the run.

Introduced by commit `e7a39a9e19d` "Add optimized Gemma 4 26B decoder", 2026-07-31 16:38:02Z — from
this run, 45s before `run_stage` exited 0. **Nobody pushed to this branch.**

## Why every guard missed it

- `factor_state` was `[0 0 0 2 1 0]` **before and after**, and `.agents` matched the arm tree exactly.
  Both were honest: the *skill* was genuinely unreachable all run. The factor guard tests skill
  reachability, not whether advisor *output* arrived by other means.
- `mkwork.sh`'s cross-arm check ran at **launch**, when the tree was clean. The leak arrived mid-run.
- `verify_gates` had the `shard_advise` test but **never reached it**: the stage-review check ran first
  and returned early on a filename mismatch (this cell's verdict is in `README.md`/`work_log.md`, not
  `*review*.md`). A cosmetic check masked a fatal one.

## The structural lesson

An arm's isolation was enforced only at launch, while the agent had read access to **every other arm's
published results for the whole run**. Deleting a skill makes the *tool* unreachable; it does nothing
to make other arms' *outputs* unreachable.

## Audit outcome (all 16 cells)

Only this cell is affected. All 15 tagged cells are correct: 7 `noadvise` cells have 0 shard_advise
files, 8 `advise` cells have >0, every `fuse-advise` capture is a **distinct blob** from its
`nofuse-advise` sibling (so no advise cell reused another's capture), and every cell has its
`fd-ready` tag as an ancestor.

## What replaces this

A fresh `fuse-noadvise` gemma run from the `skillexp/fd-ready/google_gemma_4_26b_a4b_it` tag via
`mkwork.sh`. This branch is the only copy of the contaminated attempt — keep it.
