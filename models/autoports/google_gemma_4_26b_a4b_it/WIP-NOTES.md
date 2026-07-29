# WIP — gemma4-26b-a4b-it, arm nofuse-noadvise, PARKED (void measurement)

**Do not treat any number in here as a result.** This branch exists so hours of real work and the
evidence of *why* it is void survive (MONITORING.md §6b).

## Provenance

- base: `b9e6c242a34` (pinned `mvasiljevic/qb2/skillexp/base`)
- started from tag: `skillexp/fd-ready/google_gemma_4_26b_a4b_it` → `fd/google_gemma_4_26b_a4b_it` @ `00cade41989`
- work branch was `skillexp-work-gemma` @ `ed9dff651d7` (fd tip + arm `.agents` applied)
- arm: `nofuse-noadvise` @ `51b17c3da34`, `.agents` tree `33ada6f46a4` (verified identical to arm)

## Why this is void — two stacked problems

**1. Wrong `DECODE_BATCH`.** The first attempt ran 08:54Z–11:18Z at `DECODE_BATCH=32`. Machine A's
P26 then changed policy: `DECODE_BATCH` is per-model, **32 dense / 1 MoE**, and gemma4 is MoE. At
batch 32 the routed-expert cost swamps any factor effect (north-mini: 20.5 ms/layer vs 0.245 for
dense candidates), so a b32 number for an MoE model is not comparable to the other arms. That run
was killed deliberately.

**2. §6c isolation violation — the fatal one.** The b32 run's output was **untracked**, and
`git reset --hard` does not remove untracked files. So when the batch-1 run was relaunched at
11:18:39Z into the same tree, `doc/optimized_decoder/` already contained b32 artifacts:

- 16 files with mtimes in the b32 window (08:54–11:18)
- 47 files created after the b1 launch

The b1 run was therefore measuring **already-optimized → more optimized**, not **FD → optimized**.
Per §6c that is not comparable to any cell, and nothing downstream can detect it — the artifacts
look completely normal. Killed at ~11:59Z.

## What is proven good (do not re-derive)

- The functional decoder for this model is complete and gated: full 262144 context, `clean-pass`,
  trace-replay PCC 0.999861, traced decode sliding 2.991 / full 3.177 ms at b1.
- The arm tree construction works: `.agents` stayed byte-identical to the arm across a full
  2h21m stage on north-mini (no factor drift).

## To resume correctly

1. Fresh tree with **no** `models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/`,
   no `tt/optimized_decoder.py`, no `tests/test_optimized_decoder.py`.
2. `DECODE_BATCH=1`.
3. Run `scripts/skillexp_preflight.sh` first — it refuses precisely this violation.
4. Work branch should be `merge(fd tip, arm)` so **both** invariants hold: fd tip an ancestor
   (removes the B16 rebase incentive) *and* the arm an ancestor (satisfies the shared monitor's
   `merge-base --is-ancestor` check, which currently false-alarms on B's fd-tip-only construction).

## Evidence

- `/home/mvasiljevic/skillexp-logs/p2-nofuse-noadvise-google_gemma_4_26b_a4b_it/` (b32 attempt)
- `/home/mvasiljevic/skillexp-logs/p2-nofuse-noadvise-google_gemma_4_26b_a4b_it-b1/` (b1 attempt)
- consoles: `p2-nofuse-gemma.console.log`, `p2-nofuse-gemma-b1.console.log`
