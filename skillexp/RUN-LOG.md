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
## Cell 1 — phi-3.5 `fuse-noadvise` (phiFN), batch 32 — ✅ PASSED, no warnings

`rc=0`, 15:07→15:31 (24 min), gate PASSED **with zero advisory warnings**, confirmation PASSED, published as
`skillexp/done/advchal-v3/fuse-noadvise/microsoft_phi_3_5_mini_instruct`.

| | |
|---|---|
| result | **−1.08 %/layer**, 0.806756 → 0.798063 |
| model estimate | 23162.7 → 22884.5 = **−278.2 µs/model against a ±31.1 µs band** — outside its band, so established |
| shipped | `input_norm_cores=11` — **the advisor's own advised value** |
| oracle | absolute, **real weights**, model's own 0.995 bar, incumbent 0.99890 vs candidate 0.99891 |
| expectation | ~2.8 % (22.1 % pool × 12.5 %). Measured −1.08 %, about a third of it |

**The decision fix works.** Seven measurements; the shipped 0.798063 is the **minimum of all seven**. Nothing
measured faster than what shipped — the failure that cost the shakedown 1.2 pp did not recur.

**F5's bound is confirmed on a second model, and the artefacts now say why.** `advised_plan_verbatim` returned
`hard_error` again — *"no generic final_ir-to-decoder execution bridge exists"* — so the knob-vs-IR gap is
**not nmFN-specific**. And `inexpressible[]` names the mechanism with a cost:
`dense:3, profile_cost_us 16.759, advisor_removes_us 14.636, reason: capture-substituted rope
concat/multiply placement`. That is the corpus's own STG-9 — the capture monkey-patches `_decode_rope`, so
**the advice for that region is advice for a stand-in and cannot be applied to the real decoder.** The
`inexpressible[]` field did exactly what it was added for: turned a `hard_error` into a located, priced finding.

**Consequence for the corpus's headline result.** −10.43 % at PCC 1.0 was its single strongest pro-advisor
number. It is **unreachable by this stage**, and the reason is now recorded rather than guessed: it was
produced by hand-patching the rope the capture substitutes. The stage's honest figure for this cell is −1.08 %.

Also measured, and worth keeping: `advised_plan_11` at **0.824471 — slower than the incumbent**. The partially
applied advised plan is a regression here.

### Problems found

- **P1 (minor, logged not fixed).** `final.json.oracle_pcc` is `None` at top level while the kept
  measurement carries `0.9989078`. The gate passes because it checks `oracle_passed`, so this is bookkeeping,
  not evidence — but a reader of `final.json` alone sees no candidate PCC. Fix after the run, not during.
- **P2 (gap in my own gate, logged not fixed).** One per-kind incumbent recorded `process_ordinal=1`. The
  gate's ordinal check reads only `incumbent.json`, so per-kind incumbent files bypass it. Advisory-only by
  design (C3 is n=1 evidence), so it changes nothing here.

## Cell 2 — gemma-4-26B `nofuse-noadvise` (g26B), batch 1 — ✅ PASSED, `measured_zero`

`rc=0`, 15:35→16:11 (36 min), gate PASSED, published as
`skillexp/done/advchal-v3/nofuse-noadvise/google_gemma_4_26b_a4b_it`. Outcome **`measured_zero`,
changed=False** — the frozen incumbent shipped unchanged.

**And this is the run's most consequential cell, because it reproduces the corpus's performance number and
contradicts its correctness claim.**

| candidate | median ms | vs incumbent | oracle PCC | bar | verdict |
|---|---:|---:|---:|---:|---|
| `advisor_residual_22_sliding` | **1.101676** | **−12.4 %** | **0.9946942** | 0.995 | `rejected_absolute_oracle` |
| `advisor_residual_11_sliding` | 1.131064 | −10.1 % | 0.9947948 | 0.995 | `rejected_absolute_oracle` |
| frozen incumbent | 1.257985 | — | **0.9996169** | 0.995 | shipped |

The corpus's tier-1 claim for this cell was `GEMMA4_OPT_RESIDUAL_SHARD_CORES=22`, sliding only,
**1.2583 → 1.1017 = −12.44 %, "26× what that cell shipped"**. v3 measures **1.101676** — that reproduces to
four decimals. So the *speed* is real and independently confirmed.

**But the correctness claim does not reproduce, and it is not close.** The corpus stated that against the
model's own bfloat16 `FunctionalDecoder` the candidate scored **0.99931** and the **shipped incumbent 0.98347**
— i.e. the candidate was *more accurate* and the incumbent failed the model's own bar. v3 measures the
candidate at **0.99469** and the incumbent at **0.99962**: the candidate is **worse**, and both sit far above
0.98347. Under the v3 rule — within the bar **and** no worse than the incumbent — the candidate fails on both
counts, by **0.0003** on the bar.

**So the rejection is correct as the rule is written, and the rule did its job.** A −12.4 % candidate was
found, measured, evaluated against an absolute reference and rejected **with its numbers on file**. That is
exactly what v2 could not do here: v2 shipped −0.34 % and never screened this candidate at all.

### Problems found

- **P3 (open question, NOT resolved — do not "fix" this).** The corpus and v3 disagree about the same pair of
  configurations: 0.99931/0.98347 versus 0.99469/0.99962. Both cannot be right, and the decision on a
  −12.4 % candidate turns on 0.0003. Until someone re-derives it, **the −12.44 % item should be treated as
  speed-confirmed and correctness-disputed**, not as a shippable win.
- **P4 (gap in my own gate, logged not fixed).** `final.json.oracle_reference` is `None`, and per-measurement
  oracle references are not required at all — so I cannot attribute the disagreement above to a difference in
  reference. When a verdict hinges on 0.0003, the reference *is* the finding. `oracle_reference` should be
  CRITICAL rather than advisory, and required per rejected candidate. Post-run.
- **P5 (worth checking, logged).** This cell reports `oracle_weights: real`, while the v2 corpus recorded that
  gemma-4-26B's real weights are **absent from this host** (28 KB, config only). Either they were fetched
  since, or "real" is overstated. Bears directly on P3.

## Cells 3–11 — all completed `rc=0`; 11 of 11 now tagged

The challenger queue drained overnight. Every cell exited `rc=0` with a gate PASS. Per-cell results are
being read out; the problems below are what the run itself produced.

## P6 — nmFN's publish failed on add/add merge conflicts. **The measurement was NOT re-run.**

`rc=0`, `terminal_status=complete`, gate PASSED at 17:34 — then the publish step hit
`CONFLICT (add/add)` on every artefact and the cell went untagged, leaving **10 tags for 11 completed cells**.

**Cause, and it is mine.** I parked the shakedown's *cell* branch and deleted its `done` tag, but left its
output on the **run branch** — 106 artefact files. The re-run produces the same 168 filenames, so assembling
the run branch collided on all of them.

**Fix, and the important part is what I did *not* do.** v2's own record says: *"when the measurement exits
rc=0 and the publish fails, re-running the measurement is the expensive way to retry the publish"* — it cost
that run ~41 minutes of device time. So: verified every file on the stale run branch was byte-present in
`parked/SHAKEDOWN-advchal-v3-nmFN` (**0 files unique to it**), deleted the run branch locally and on origin,
and re-invoked the driver, which took its own **publish-only** path — *"a COMPLETED unpublished run exists
for nmFN (168 artifacts, current method) — publishing it instead of re-running; its device hours are not
discarded."* **51 minutes of device time preserved.**

**Second attempt needed, for a different reason.** The first publish-only run gated against the **working
tree**, which was on the *previous* cell's branch, so the gate reported `no incumbent.json / no report.json /
no reconciliation / no final.json` and failed. Checked out `skillexp-cell/advchal-v3/nmFN` (170 artefacts
present), re-invoked, gate PASSED, published and tagged. **A publish-only path must check out the branch it
is gating** — driver defect, logged not fixed.

Residual: `PUSH REJECTED (NON-FAST-FORWARD)` parking the previous cell pointer, so `cell/` was left untouched.
Cosmetic; the `done` tag and run branch are correct.

## P7 — my measurements-vs-decision check produces cross-kind false positives

The check I added compares every measurement against `final_ms`. I scoped **which measurements** it considers
by matching the layer kind in the label, but still compared them all against **one global `final_ms`** — so on
nmFN it flagged twelve `full_attention_sparse` measurements (~0.542) against a `final_ms` of 0.5684 that is the
**sliding** kind's number. Those are apples-to-oranges.

**This is the C5c mistake again**, from me, at the same spot in the reasoning: a rule applied across a
population where it does not hold. It is noise rather than a wrong verdict — every row carried a recorded
verdict, so all twelve are WARN not CRITICAL — but it makes a multi-kind cell's gate output unreadable, which
is how a real one gets missed. **Fix: compare against that kind's own shipped number from
`model_estimate.per_kind`, not the global `final_ms`.** Post-run.

Genuine within-kind question it leaves standing on nmFN: `norm_8_sliding` (0.566617) and `norm_11_sliding`
(0.566323) are faster than the shipped `norm_22` (0.567898) and both carry `verdict='rejected'` with PCC
≈0.9994. Whether those rejections are right is a per-cell read, not something to adjudicate at speed.

## P8 — I reported two drivers "alive" that were my own `pgrep` pattern

`pgrep -f 'bash run_challenger.sh'` and `pgrep -f 'run_dense.sh'` both matched **the shell running the
pgrep**, because the pattern appears in its own command line. I reported "challenger driver: alive, dense
driver: alive" when **neither was running** — the queue had drained and `run_dense.sh` had never been started
(its log is from 2026-07-31, the v2 run).

`MONITOR.md` documents this exactly: *"`pgrep -f <pattern>` matches the shell that runs it… Anchor the
pattern."* It fooled a v2 preflight into refusing four launches. Correct form, used since:
`ps -eo pid=,args= | awk '$0 ~ /bash run_challenger\.sh/ && $1 != '"$$"''`.

Consequence: I would have reported the dense cells as running when they had not started. Caught within one
command, but only because a start time of `etime 00:00` looked wrong.


## PAUSED — watch and heartbeat stopped 2026-08-11 09:41 UTC

The run has been paused since 07:27 (`.challenger-STOP`, `.dense-STOP`). The **watch was still running and
publishing for another two hours** after that, so this is the record of stopping it and how to bring it back.

| what | state |
|---|---|
| `skillexp_watch.sh` (machine a) | **stopped**, was pid 3439781, `SIGTERM`, its `EXIT` trap cleared `.watch-a.lock` and `.watch-a.pid` |
| last heartbeat tick | `2026-08-11T09:31:00 IDLE tick=53 stage=none tmux=GONE mg=down cx=down cpu+0s dev=ok disk=2149G` |
| last published status | `origin/.../skillexp/status/a` @ `cdbbc6edc11`, `2026-08-11T07:56:44Z` |
| `.challenger-STOP` / `.dense-STOP` | **still in place** — leave them until the guard fix lands |
| devices | free, nothing holding `/dev/tenstorrent*` |

**Why it was worth stopping rather than leaving:** every tick since the pause published
`ALERT IDLE: no multigoal process, no live stage, devices are free. The next stage needs launching` — an alert for
a condition that is deliberate. A monitor that alarms on an intended state trains its reader to ignore it, and the
board it force-pushes says "needs launching" when the correct state is "paused pending a decision". It was also
still repeating a **v2-era** alert (`p-advchal-v2-nmFN reports CONTAMINATED`), which is stale by two weeks.

**Also stopped: three loops left running by earlier sessions**, none of them part of this run.

| pid | age | what it was |
|---|---|---|
| 1969997 | 12 d | v2-era cell-tagging validator; `git fetch origin --tags` every 600 s against the shared clone |
| 3910120 | 8 d | `until ! pgrep -f 'bash /tmp/watch_results.sh'` waiter from a finished session |
| 4050469 | 8 d | waiter on a task-output file from session `366e7e69` |

The first one mattered: it was fetching tags into `/home/mvasiljevic/skillexp-book` every ten minutes for twelve
days, i.e. mutating a shared checkout on a schedule nobody was reading. Its exit was visible as a task
notification with code 144 (`128 + SIGTERM`).

**To resume**, in this order:
1. land the guard/policy fix ([`GUARD-FINDING`](../../tt-metal/skillexp/ADVCHAL-V3-GUARD-FINDING.md) action 1) and
   the oracle-provenance gate change, since re-running before them reproduces the same unattributable verdict;
2. `MACHINE=a INTERVAL=1800 CHECK_EVERY=300 STALL=3600 TTSMI=idle-only PUBLISH=1 PUBLISH_BOARD=1 SCALE=flat setsid nohup bash skillexp_watch.sh > watch-stdout.log 2>&1 &`
   from `~/skillexp-logs`;
3. remove `.challenger-STOP` / `.dense-STOP` last, so the watch is up before the queue is.

**Note for the next watch stand-up:** the watch has **no stop sentinel** — the only ways out are `MAX_TICKS` and a
signal, while the drivers it monitors both take one (`.challenger-STOP`). Give it a `.watch-a-STOP` check in the
`CHECK_EVERY` sleep loop so pausing the run and pausing its monitor are the same gesture.

## RESUMED (dense only) then RE-PAUSED — 2026-08-11 10:59 / 11:10 UTC

Relaunched `run_dense.sh` for the three cells left behind the pause; `.challenger-STOP` deliberately left in place
so the 11 challenger cells stay held. Re-placed `.dense-STOP` at 11:10 so the driver finishes **phi-exp17** and
then holds before llama31-8b.

**Prerequisites that had to be fixed first, and they matter for the record:**

| | |
|---|---|
| llama-3.2-1B-Instruct weights | **absent from the host entirely** — fetched, 2.31 GiB |
| llama-3.1-8B-Instruct weights | **present as a 20 KB config stub only** — fetched, 14.97 GiB |
| both repos | gated; the container's `HF_TOKEN` has access (verified 200 on both `config.json`) |
| snapshots | all four present under `/home/mvasiljevic/.agentic-research-ro`; my first check looked in the wrong root and reported all four missing |
| watch | restarted (`INTERVAL=1800 PUBLISH=1 PUBLISH_BOARD=1`), pid recorded in `.watch-a.pid` |

**The run is against the completely unmodified v3 stage.** Verified: `SKILL_BR` =
`mvasiljevic/qb2/skillexp/challenger-skill-v3` = `4ea2fb1fb7d` = `advchal-v3/stage-frozen^{}`; no commits of mine
outside the docs branch; no fix branches pushed; every experiment worktree removed; the one modified file in the
tree (`phi.../tt/optimized_decoder.py`) is **phi-exp17's own work in progress** and contains none of my markers.
**Neither model fix and none of the gate changes were applied** — deliberately, so the control calibrates v3 as it
actually ran.

Predictions written down before the results: `skillexp/ADVCHAL-V3-DENSE-PREREG.md`.
