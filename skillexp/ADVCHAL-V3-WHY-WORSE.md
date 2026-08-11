# Why v3 was worse than v2 — the accounting, and the answer I had not reached

Asked whether I had fully understood and documented *all* the differences. **No.** This file is the audit of my own
coverage, three hypotheses of mine that the evidence kills, and the synthesis that replaces them.

## 1. How much of the loss is actually explained

11,303 µs of improvement was lost across seven cells (v2's delta minus v3's, per cell):

| cell | µs lost | status |
|---|---:|---|
| gemma-4-26B `-onA` | **5,907** | ✅ **fully explained, fix measured** (−5,260 recoverable) |
| **north-mini `fuse-noadvise`** | **2,200** | ❌ **NOT explained** — I dismissed it as *"v2's cell was `CONTAMINATED`, the number was never checkable"* and **never verified that** |
| phi-3.5 `nofuse-noadvise` | **1,285** | ✅ **fully explained, fix measured** (−1,285 recoverable) |
| phi-3.5 `fuse-noadvise` | 989 | ⚠ partial — "rope inexpressible"; **never tested whether v2's rope ports**, which is exactly what fixed phiB |
| qwen3.6 `fuse-noadvise` | 434 | ❌ never dug — "nothing measured faster", its 97 %-of-model kind still unmeasured |
| phi-3.5 `-onA` | 340 | ⚠ partial — clause-2 artefacts identified in the artefacts, never re-measured |
| gemma-4-26B `nofuse-noadvise` | 148 | ⚠ partial — P3 contradiction unresolved |

| | µs | share |
|---|---:|---:|
| **fully explained with a measured fix** | **7,192** | **64 %** |
| not explained or partial | **4,111** | **36 %** |

Offsetting gains: +2,701 from three coverage cells (attributed to the tracer handlers, **asserted, never isolated**)
and +195 where v3 beat v2 on `g26FN` (**never explained why**).

**So a third of the regression is unexplained, and the largest single hole is the one I waved away.**

## 2. Three hypotheses of mine, killed

I assumed the regression came from my own rebuild. Checked, and it did not — at least not in the ways I assumed.

| hypothesis | test | verdict |
|---|---|---|
| *"I cut the prompt 4049 → 2885 chars and removed guidance that mattered"* | **v2's prompt is 2541 chars; v3's is 2889** | ❌ **dead.** I was trimming my own inflated draft, not v2's guidance. v3's prompt is **348 chars longer** |
| *"my `inexpressible[]` design licensed agents to substitute a nearby geometry, which is what broke phiB"* | the sentence *"Only after that fails is a compatible geometry of your own the…"* is **v3 line 137 and v2 line 66 — identical** | ❌ **dead.** The substitution licence **predates v3**. v2's agent had the same licence and wrote a correct implementation |
| *"v3's guidance was weaker"* | `.agents` diff: **+1,388 / −149 lines**, SKILL.md +453, reconcile.py +619, gate +249 | ❌ **dead.** v3's guidance is strictly larger and its gate strictly stricter |

So: **same prompt, same substitution licence, more guidance, stricter gate — and worse model code.** That rules out
the explanation I found most comfortable (that I broke it) and forces a harder one.

## 3. The selection effect in my own audit, and what removing it shows

I then reached for *"v3's agents simply wrote worse code — 3 defects in 11 cells against 0 in 15."* **That count is
worthless**, because I audited v3's model code line by line and used v2's as the reference. "v2 has no known
defects" largely means **I did not look.**

And when I *did* look, I found one — documented, then not counted:

> **v2's gemma guard is `x.shape[-2] > TILE_SIZE → skip`, so it shards prefill only when prefill fits in one tile
> row. Measured: at seq 64 v2's `phase=both` returns the decode-only number to sixteen digits.** v2's configuration
> is phase-consistent **only at prefill ≤ 32 — exactly its oracle's fixture.** At production prefill lengths it is
> the same mismatched configuration that cost v3 the cell.

**So both versions ship the same class of defect.** v2's is invisible at its oracle's fixture; v3's is visible at
the same fixture because v3's guard is phase-gated rather than shape-gated.

## 4. The answer

> **v3 is not worse at writing code or worse at judging. v3's defect was *detected* and v2's was not — and v3 had
> no way to attribute what it detected, so it discarded the win instead of fixing the bug.**

On gemma-4-26B `-onA`:

| | v2 | v3 |
|---|---|---|
| the model code | phase-inconsistent above 32 prefill rows | phase-inconsistent at all prefill lengths |
| its oracle at seq 32 | **0.9996293 — passes**, because the guard happens to fire there | **0.9943717 — fails**, correctly |
| what the stage concluded | *the placement is good* → **shipped −5,919 µs** | *the placement is bad* → **vetoed, 0 µs** |
| what was actually true | a latent cross-phase bug, shipped | **the same bug, caught** |

**v3 detected a real defect and was penalised 5,919 µs for detecting it**, because the stage could only express the
finding as "this placement fails correctness". A detection with no attribution path is indistinguishable from a
rejection — and it is *worse than not detecting*, because it also destroys the win.

That is why the fix list is what it is. `drop_index=1` recovers −5,260 µs **while keeping the correctness v3 was
right about**; v2's number was never safe, it was unmeasured. **The corpus's −15,177 µs v2 baseline is therefore
partly a measurement of what v2 failed to check**, which is the single most important caveat on
[`RESULTS`](ADVCHAL-V3-RESULTS.md) and it was not stated there until now.

## 5. What this supersedes

[`CORE-ISSUE`](ADVCHAL-V3-CORE-ISSUE.md) said the harness's scope determined what the agents wrote and what the gate
could see. That still holds for **what the agents wrote**. But its account of *why v3 lost* was incomplete: it
framed v3's phase-gated guard as the defect and v2's shape-gated guard as *"the coin-flip v2 got right"*. **v2 did
not get it right — v2's guard is defective too, and only its fixture hid it.** The difference between the versions
is **detection**, not correctness.

## 6. What is still owed, in priority order

1. **north-mini `fuse-noadvise`, 2,200 µs — the largest unexplained gap.** Verify or refute the `CONTAMINATED`
   dismissal from v2's artefacts and driver log, then measure v3's cell against it. **This is the next thing I run.**
2. **Try porting v2's rope into phi `fuse-noadvise`** (989 µs). The identical move recovered phiB entirely; I never
   tested it on the cell where I claimed inexpressibility.
3. **qwen `fuse-noadvise`** (434 µs) — never examined.
4. **Isolate the +2,701 coverage gain** to the tracer handlers instead of asserting it. If some of it is not
   coverage, v3's total is overstated.
5. **Explain `g26FN`'s +195 µs.** An unexplained *win* is as much a gap as an unexplained loss, and it is the one
   cell where v3's approach beat v2's outright.
6. **Audit v2's shipped knobs with the intensity I applied to v3's.** One pass found a defect in v2's gemma guard;
   the count of "v2 defects" is currently a measure of my attention, not of v2.
