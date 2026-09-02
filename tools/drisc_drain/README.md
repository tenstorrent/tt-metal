# drisc_drain — DRISC PCIe hang investigation

`FINDINGS.md` is the running record. Read the newest section first; earlier conclusions are
superseded in place and several are explicitly retracted.

## Harness

Two scripts, both driven over ssh from a Mac against an IRD box.

| script | purpose |
|---|---|
| `drisc_hang_harness.sh` | single-condition runs, classified |
| `drisc_hang_compare.sh` | armed-vs-unarmed, randomized order, per-arm breakdown |

```sh
TAG=v1 DELAY=125 N=40 ARMED=0 STOP_ON_WEDGE=1 ./drisc_hang_harness.sh
TAG=cmp DELAY=125 N=40 ./drisc_hang_compare.sh          # N = runs PER ARM
```

Env: `TT_HOST` `TT_PORT` `TT_REMOTE` `OUT_DIR`. Writes `summary.txt` and `runs.csv`
(`k,delay,armed,rc,dur_s,card,class`).

## Why it scores four axes

Every one of these corresponds to an error that produced a wrong finding earlier in this
investigation. Do not remove them.

- **CARD STATE is authoritative, never exit code.** "Hang" pooled two distinct failure modes —
  a genuine PCIe endpoint wedge (`Unknown|63`, all-ones config space) and a teardown
  `wait_until_cores_done` that never completes on a perfectly healthy card.
- **DURATION.** A 9x slowdown (45 s vs the 2.1 s baseline) hid inside ~280 runs that all exited 0.
- **MASKED.** `TT_METAL_OPERATION_TIMEOUT_SECONDS` also bounds the teardown core-wait; the
  exception is caught and the run exits 0. Such a run is *not* clean — unarmed it would hang.
  Scoring these as clean is what produced, and then destroyed, the §N+18 "periodic read prevents
  the wedge" claim.
- **rc**, last and least.

## Method rules learned the hard way

- **Randomize arm order, never alternate.** The wedge straddles run boundaries: the failure
  surfaces in the run *after* the one that seeded it. Under strict alternation that always lands
  on the same arm, which is a bias, not noise — more runs make the wrong answer look stronger.
  This voided a `num_hw_cqs` comparison and, earlier, a Tensix-vs-DRISC one.
- **Watchdog the local ssh client.** A remote process can exit while the local client hangs,
  stalling a whole block silently. Hence `ServerAliveInterval`, `timeout -k`, and a local kill.
- **bash 3.2 on macOS has no associative arrays.** Tally from filenames, not in-script counters —
  string subscripts silently collapse to index 0 and every arm reports the same number.
- Recovery: try `tt-smi -r` first (seconds, and it does clear a wedged card), fall back to a host
  reboot only if the link stays `Unknown`.

## Reclassifying after the fact

`drisc_reclassify.py <run_dir>` re-derives every run's class from its log and writes
`runs_reclassified.csv`. Use it on any block collected before a classification rule was fixed —
the raw logs hold everything, so no block ever needs re-running for a scoring bug.

The discriminator is **how far the log got**, not the exit code and not a log signature:

| ending | card | class |
|---|---|---|
| `Cluster destructor completed` | — | CLEAN |
| stops at the profiler teardown block | healthy | TEARDOWN |
| stops earlier | `Unknown\|63` | WEDGE |
| reaches the end, but logged a caught timeout | healthy | MASKED |

The signature-only rule was wrong because `waiting for physical cores to finish` is emitted **only
when the timeout is armed**. Unarmed runs hang at the identical place in silence, so every unarmed
teardown hang was landing in OTHER.

## Egress amplifier as a candidate fast repro

`REPEAT=<n>` sets `TT_METAL_STREAMING_PROFILER_SHIP_REPEAT`, re-shipping each staged frame n times so egress
stops being bounded by producer rate. The payload becomes duplicate frames, so it is a STRESS tool,
never a capture (`NO_DECODE` is already on).

```sh
TAG=amp8 DELAY=500 N=60 ARMED=0 REPEAT=8 ./drisc_hang_harness.sh
```

**Calibrate before trusting it as a faster repro.** The historical basis is a single cell -- repeat=8,
delay 500, 1 hang in 12 runs. That is ~8%, but against the ~2.5% base rate measured at delay 125/150/500
it is p ~ 0.3, i.e. consistent with no difference at all. And the 84-run monitored churn used a full
SHIP_REPEAT ladder {1,2,4,6,8,12,16} and produced zero wedges. Measure the rate with the classifying
harness before spending a campaign on it, and check that what it produces is a WEDGE
(card `Unknown|63`) rather than a TEARDOWN -- a faster repro of the wrong failure mode is worse than
no repro, because it looks like progress.
