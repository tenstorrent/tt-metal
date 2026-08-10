# advchal-v3 — run log: every problem hit, and its fix

Kept live during the run, in the shape of v2's `PROBLEMS.md`: one entry per problem, what it cost, and what
was done. Appended as things happen, not reconstructed afterwards.

**Stage frozen at `advchal-v3/stage-frozen` = `4ea2fb1fb7d`.** From here the `.agents` tree must not move.
Every cell's `.agents` is asserted byte-identical to it by `confirm-cell.sh`, and the driver's `ARM_REF` check
turns any edit into a contamination signal.

## Protocol for each cell

1. driver builds a fresh per-cell work branch from the frozen skill tree + the incumbent SHA;
2. cell runs, watch reports `OK`/`BUSY-QUIET`/`WEDGE`/`RUNNER-GONE` per tick;
3. gate runs, driver publishes and tags on pass;
4. `confirm-cell.sh <short> <md> <v2-commit>` re-derives isolation: `.agents` non-drift, author dates,
   reflog, blob identity vs the v2 run, all 156 parked refs still unnamed, provenance fields, exclusivity;
5. result and any problem appended here, and the branch pushed.

## R0 — the shakedown, and why it does not count as a run

`nmFN` ran 13:32–14:17 against the **pre-audit** stage (`9f918ab4428`). `rc=0`, gate PASSED, published.
Result: shipped `topk110`, −37.3 µs/model inside a ±50.1 µs band — **while holding `norm16` at 0.543590,
measured four times and never oracle-checked**, with the two slower rungs both oracle-passing on real weights.

Kept as `parked/SHAKEDOWN-advchal-v3-nmFN` / tag `advchal-v3/shakedown/nmFN`. Its `done` tag was deleted so
the cell re-runs against the frozen stage. It stays in the record because it is the measurement that found
four design defects, and because its incumbent reproduced v2's to four decimals (`0.1727` vs `0.1727`), which
is what makes every comparison in this run mean anything.

---
