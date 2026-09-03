# METHODOLOGY.md — measuring compiler-vs-expert speedups on a 32-chip galaxy

How the galaxy-kit turns a 32-chip Blackhole node into an A/B measurement
instrument, and why each design choice is the right one for the question we
are actually asking. This is the reviewer's companion: `README.md` gives the
three commands, `EXABOX.md` gives the route and the etiquette, and this file
explains what the numbers mean and why they can be trusted.

The one-line claim we are testing: *the compiler's kernels are faster than the
expert-written ones, and that is a real property of the compiler, not an
artifact of the machine we happened to measure on.* Everything below exists to
make that claim falsifiable.

---

## 1. The question decides the instrument

There are two completely different questions you can ask with a profiler, and
they need different tools:

- **"Where did the time go?"** — a pipeline question. You want to see the whole
  program: host dispatch, data movement, kernel launch, the gaps between ops.
  The right tool is a **tracing profiler** (tracy): it timestamps events across
  the host and device and draws you a timeline. Wall-clock time is the honest
  unit, because dispatch and DMA are part of the story.

- **"Is kernel A faster than kernel B?"** — an A/B question. You want one
  number, computed the same way for both arms, with everything that is *not*
  the kernel held constant. Tracing is the wrong tool here: its timeline is
  dominated by host scheduling and transfer jitter that have nothing to do with
  the compiled math, and its own instrumentation perturbs the very thing you are
  timing.

This kit answers the second question, so it does **not** use tracy and does
**not** use wall time. It uses the **on-device cycle counter, scoped to the
kernel** — the same `KERNEL_*_E2E` marker the canonical board uses. The device
counts its own clock cycles between the marker's start and end, on the chip,
with no host in the loop. That is the correct instrument for an A/B comparison
for three reasons:

1. **It measures the thing under test and nothing else.** The marker brackets
   the compiled kernel body. Host launch overhead, PCIe transfer, and Python
   dispatch all sit outside it and cannot leak in.
2. **It is deterministic.** A cycle count off a free-running on-device counter,
   for a fixed kernel on a fixed chip, is the same integer every time (see §5) —
   there is no clock-domain conversion, no timestamp quantization, no scheduler
   noise to average away.
3. **It is the board's own unit.** Booking the compiler's wins against the
   experts already uses this marker. Replicating in the same unit means we are
   reproducing the actual recorded numbers, not a wall-time proxy that would
   need its own separate argument.

tracy remains the right tool for its own job — profiling the pipeline to find
where a workload spends its time. It is simply not an A/B stopwatch, and this
methodology is an A/B stopwatch.

---

## 2. The unit of work is `(op, chip)`, both arms in one session

The atom of measurement is a single operation raced on a single chip. For that
atom the kit runs **both arms — the compiler's kernel (`sem`) and the expert's
hand-written kernel (`hand`) — back-to-back on the same physical chip, in the
same worker session.** Neither arm ever runs on a different chip from its
partner.

This is the load-bearing decision. Chips in a galaxy are not identical: small
differences in binning, temperature, and voltage mean chip 07 may run a few
percent hot or cold relative to chip 22. If the two arms were measured on
different chips — or even on the same chip at different times under different
thermal conditions — that chip-to-chip difference would contaminate the ratio.
By pairing both arms on one chip in one session, **every per-chip effect divides
out of the sem/hand ratio exactly.** What survives is the compiler's causal
contribution, which is what we are booking.

The consequence, stated plainly: **the galaxy's raw cycle counts are not the
record.** The chip is a comparator, not a canonical clock. The statistic we
report and replicate is the **same-chip ratio** of compiled to expert; the
absolute cycle numbers are scaffolding for computing it. (See §7.)

---

## 3. Work-stealing across 32 chips

The node has 32 chips; the campaign has a few hundred `(op, chip)` cells. The
launcher starts **one worker process per chip** (`lib/galaxy_launch.sh` under a
single `srun --overlap` step), and the workers pull from **one shared queue** of
work items (`lib/seed.py`, `lib/worker.py`). A worker finishes a cell, writes
its result, and takes the next item off the queue.

Work-stealing rather than a static partition, because sessions are not all the
same length (a corr gate plus five perf reps on a short kernel is seconds; a
heavier op is longer) and chips are not all the same speed. A static
chip→op assignment would leave fast chips idle waiting for the slowest; the
queue keeps all 32 busy until the work is gone. One rule constrains the steal:
**a worker never takes two copies of the same op** — the `-k N` distinct-chip
requirement (§6) is enforced at the queue, so each op lands on N *different*
chips rather than N reps on whichever chip grabbed it first.

Sessions are solo on their chip (one worker owns one chip at a time), which
makes attribution trivial: a cycle count belongs unambiguously to one op on one
chip with nothing else contending for that chip's cores.

---

## 4. Correctness gate first — golden before fast

A speed number for a kernel that computes the wrong answer is worse than
useless. So for every `(op, chip, arm)` the kit runs the **correctness (corr)
node first**, against the golden reference, and **only if it passes** does that
arm proceed to its timed perf reps. A failing gate writes `<arm>-CORR-FAIL.txt`
and that arm contributes no perf data — the failure is recorded, never silently
dropped and never quietly replaced by a fast-but-wrong number.

This ordering is what lets the ratio mean "faster *at the same answer*." A pair
only exists when both arms passed golden on that chip, so every booked ratio is
a comparison of two kernels that are both correct.

---

## 5. Five reps, expected cycle-identical

Each arm is timed **5 times** (solo perf sessions, `-r 5`). Because §1's
instrument is a deterministic on-device cycle count, the reps are *expected to
be bit-identical* — not "averaged to reduce noise," but checked for the absence
of noise. The ledger records the per-arm **spread** (max − min across reps); a
spread of 0 means the chip returned the same integer five times.

The reps are therefore a **falsification test, not an error bar.** If a cell's
reps disagreed by a meaningful amount, that would signal contention, a thermal
excursion, or a measurement bug — and it would show up in the spread column
rather than being smoothed over. In practice the spreads are 0 or a handful of
cycles (§8), which is the evidence that the instrument is as deterministic as §1
claims.

---

## 6. Chip id recorded per cell; distinct chips per op

Every result row carries the **chip id and the node hostname** it was measured
on. This is not bookkeeping ceremony — it is what makes the same-chip pairing
auditable after the fact (you can confirm both arms of any pair share a chip)
and what lets a single op be spread across **N distinct chips** (`-k N`, default
8) to show the ratio is stable across the machine, not a quirk of one chip.

A subtlety the chip id has to survive: **the galaxy's device index is not its
reset id** (see §7, wall 1). The kit records the `TT_VISIBLE_DEVICES` mask value
consistently as the chip id, and — critically — uses the *same* mask for both
arms of a pair on the same worker. So whatever physical chip a mask opens, both
arms open the same one, and the pairing is immune to the index transposition.
The transposition only bites code that tries to *reset* by that index, which is
exactly what §7 forbids.

---

## 7. The four walls of running 32-wide (and how each is handled)

A galaxy is a shared, 32-chip machine with one reset domain. Four things will
corrupt a naive campaign; the kit closes each one. (`EXABOX.md` §7 is the
operations bible for these; summarized here for the measurement argument.)

1. **Chip-id transposition for chips ≥ 16.** `TT_VISIBLE_DEVICES=k` opens PCIe
   id `k` for `k < 16` but `k XOR 8` for `k ≥ 16` — tt-smi sorts chips by PCI
   bus while the device mask walks the root complexes, transposing the two
   high groups (verified on silicon: mask 22 ↔ chip 30, 16 ↔ 24). The trap is a
   per-worker recovery reset: `tt-smi -r $CHIP` for `k ≥ 16` resets the *swap
   partner*, so workers reset each other mid-measurement. **Handled by:** never
   assuming `CHIP == reset-id`, recording the mask consistently, and using the
   same mask for both arms so pairing is transposition-proof (§6).

2. **One shared reset domain — no per-worker resets.** A Blackhole `tt-smi -r`
   cascades across the whole tray; a per-worker reset would knock over the other
   31 workers. **Handled by:** exactly **one upfront `tt-smi -r`** for the whole
   node, marker-guarded (`.reset-done-<host>`), before any worker starts.
   Workers never touch tt-smi. The one retry on a transient failure is
   kill-tree only — it never resets.

3. **No tracy.** For the A/B reasons in §1, and because a tracing profiler
   perturbs the timed region and serializes poorly across 32 concurrent chips.
   The measurement is the on-device kernel-scoped counter, nothing else.

4. **The Slurm soft process limit.** `srun` steps inherit a soft `nproc` of 512;
   32 workers each spawning pytest/compile subprocesses blow past it. **Handled
   by:** the launcher raises `ulimit -u` to the hard limit before forking
   workers.

Two etiquette walls from `EXABOX.md` bound the whole thing: the kit only ever
does `srun --overlap --jobid <held-id>` (it never `salloc`s, `scancel`s, or
releases a hold it did not create), and it stages small archives through the
Mac relay and cleans up. Compilation happens entirely at home on the quietbox;
the galaxy runs **execute-only** consumer sessions (`pytest --compile-consumer`)
of pre-compiled ELFs, so no compiler ever ships to or runs on the cluster.

---

## 8. Empirical validation — this methodology, on a real run

The kit's first full campaign (pin-55 toolchain, a held 32-chip Blackhole
galaxy, 2026-09-03) is itself the evidence that the design above is sound. Every
number here is derived from the run's own ledgers
(`REPLICATION-LEDGER.tsv`, `REPLICATION-PAIRS.tsv`, `REPLICATION-VERDICTS.tsv`
in `~/sfpi-uplift/laneLK-evidence-20260903/`):

- **14,028 device sessions across all 32 chips, 100% passed** their correctness
  gate; zero device resets consumed, zero retries, zero parse anomalies. The
  corr-first ordering (§4) held for every timed reading.
- **The instrument is deterministic (§1, §5).** 930 of the 1,169 same-chip pairs
  are **cycle-identical across all 5 repeats on both arms**; the largest
  rep-to-rep spread anywhere in the campaign is **14 cycles**, on kernels of
  10³–10⁵ cycles — under 0.1%. The reps found no noise to average, exactly as a
  kernel-scoped on-device counter predicts.
- **The same-chip ratio is stable across the machine (§2, §6).** For **145 of
  146** operations the speed-vs-expert ratio is **identical to 0.00% across all
  8 chips** it was measured on; the single exception moves by 0.01%. The
  chip-to-chip differences that §2 divides out are, empirically, divided out.
- **The replicated numbers land on the home board.** The paper's headline cells
  reproduce their booked values on foreign silicon — e.g. the licensed
  trigonometry win at −1.51, erfinv at −42.19, silu at −19.30, addcmul at
  −10.87, exp at −0.86 — with nine of ten headline cells matching the home board
  exactly.
- **Class agreement is 108 of 135 raced rows, and every one of the 27
  differences is attributable, not a measurement disagreement:** 21 are rows
  whose home-board cell was booked with an extra compiler knob switched on and
  were re-measured here at plain flags (the one knob leg that *was* run at its
  booked flags reproduced exactly); 6 are stale plain cells the compiler
  genuinely moved past since they were booked, four of them in the compiler's
  favor. None is a case of the two machines disagreeing about the same kernel.

The headline reading: **the speedups reproduce on an independent 32-chip machine
with essentially zero measurement noise** — which is the strongest possible
statement the methodology could hope to earn.

---

## 9. What the galaxy is not

Stated once more, because it is the easiest thing to get wrong: **galaxy cycle
counts are not promoted to the canonical record.** The p150 board remains the
booking authority. The galaxy is a *replication* instrument — it re-runs the
whole set on a different machine to confirm the compiler's advantage is real —
and the only quantity it certifies is the **same-chip ratio of compiled to
expert.** A class move measured here against a stale or knob-booked home cell is
a re-book *candidate* for the canonical board, never a booking made on the
galaxy.

---

## See also

- **`README.md`** — the three-command runbook (`stage.sh` / `run_bench.sh` /
  `collect.sh`), prerequisites, and the honesty rules baked into each stage.
- **`EXABOX.md`** — the route to the cluster (the Mac relay, both hops), the
  Slurm-hold-vs-ssh split, the hard etiquette rules, and the §7 walls in
  operational detail.
- **`lib/`** — `gen_spec.py` (spec generation), `worker.py` (the per-chip
  work-stealing worker: corr gate, same-chip pairs, reps), `seed.py` (the queue),
  `galaxy_launch.sh` (the node launcher with the reset marker and ulimit raise),
  `ledger.py` (result → the three TSVs).
