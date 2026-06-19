# Step E — Style Bake-off (`mcast_pipe`)

> **STATUS: PLAN (beat 1) — awaiting user OK before ANY device run.**
> Results sections are stubs until beat 2.

The bake-off resolves the four style forks the API draft (Step ★) left open, by measurement, not
argument: **coverage screen first (cheap, correctness/hang), perf+L1 only on survivors.**

> **Round-5 re-entry (2026-06-19) — E re-confirm NO-OP, no device.** feedback.txt item 1 (template
> `McastRect` on the NoC id, precompute corners in the ctor) touches NONE of the four style forks
> (F1 fence / F2 staging / F3 loopback / F4 linking). It changes where a constexpr-foldable corner
> swap is computed (ctor vs per-call), not which mitigation a fork takes — no new variant, no new
> topology/dtype cell. All coverage maps and ns/L1 numbers below stand verbatim. Items 2 & 3 are
> pure renames (Step F). No re-measure, no re-decide.

---

## E.1 — Triage: which forks are style (→ bake-off) vs use-case (→ knob)

| Fork | Style or forced? | To bake-off? |
|---|---|---|
| **F1 fence: flush vs write-barrier** (data path) | STYLE — VC-4 FIFO (INV4) makes both correct; caller doesn't force it | **YES** |
| F1 atomic-barrier | FORCED — required *after an atomic inc* (counter path); not a choice | no (mechanism) |
| **F3: INCLUDE_SRC loopback vs EXCLUDE_SRC+local-copy** (sender ∈ rect, needs own data) | STYLE — both deliver the sender its copy | **YES** |
| F3: EXCLUDE (sender ∉ rect) / degenerate→unicast | FORCED — by geometry / rect size | no (internal dispatch) |
| **F4: linked-pair+flush vs unlinked+barrier-between** | **DECIDED by user: linked where possible** | coverage-only (confirm linked lands bit-exact + no deadlock; unlinked = fallback where predicate forbids linking) |
| **F2: flag+reset vs counter+wait_min** (multi-round staging) | borderline — counter is *coverage*-immune to stale-retrigger; perf is the open question | **YES** (coverage may pre-decide) |
| KNOB `pre_handshake` | USE-CASE (dest reused vs fresh slot) | no — settled on paper |

**Interactions (bake the cross-product, not each axis alone):**
- F2=counter ⇒ F1=atomic-barrier (forced) — so the F2 bake carries its own fence.
- F4=linked ⇒ flush (no barrier allowed) — so F4's linked arm pins F1=flush.
- ⇒ The independent axes to vary are **F1(flush|write-barrier)**, **F3(loopback|exclude+copy)**,
  **F4(linked|unlinked)**, **F2(flag|counter)**. F1×F4 share the flush cell; bake F4 as a 2-cell
  pair and F1 as a 2-cell pair on the *unlinked data* path.

---

## E.2 — Matched micro-kernels (raw object-API, identical except one axis)

All under `tests/ttnn/unit_tests/kernel_lib/kernels/bakeoff_*`, written on the **object API**
(`Noc`, `Semaphore<>`, `MulticastEndpoint`). Throwaway-but-kept → become the raw baseline for
`build-helper`. **Base micro-op:** one sender core mcasts `size` bytes to an R×C receiver rectangle,
`N` iterations, with an R→S pre-handshake; receivers verify bit-exact payload and signal back.

| Pair | Kernel A | Kernel B | Sole difference |
|---|---|---|---|
| **F1** | `bakeoff_f1_flush_{sender,receiver}.cpp` | `bakeoff_f1_barrier_{sender,receiver}.cpp` | post-mcast fence: `async_writes_flushed()` vs `async_write_barrier()` (data path, unlinked) |
| **F3** | `bakeoff_f3_loopback_{sender,receiver}.cpp` | `bakeoff_f3_excludecopy_{sender,receiver}.cpp` | sender-in-rect self-fill: `INCLUDE_SRC` vs `EXCLUDE_SRC` + local memcpy |
| **F4** | `bakeoff_f4_linked_{sender,receiver}.cpp` | `bakeoff_f4_unlinked_{sender,receiver}.cpp` | data;flag issued `linked=true`+flush vs `linked=false`+barrier-between |
| **F2** | `bakeoff_f2_flag_{sender,receiver}.cpp` | `bakeoff_f2_counter_{sender,receiver}.cpp` | staging: `set_multicast`+`wait`+reset vs `inc_multicast`+`wait_min` (+atomic-barrier) |

(8 sender + 8 receiver kernels; the receiver differs only where the handshake style demands, e.g.
F2 receiver `wait` vs `wait_min`.)

## Harness — DECIDED: Python pytest (`ttnn.generic_op` + `ProgramDescriptor`)
**Chosen at the checkpoint.** Feasibility confirmed against `tests/ttnn/unit_tests/operations/debug/test_generic_op.py`:
`ttnn.ProgramDescriptor(kernels=[...], semaphores=[...], cbs=[...])` + `ttnn.generic_op(io_tensors, pd)`
exposes `KernelDescriptor` (file-path source, per-kernel `core_ranges`, CT + runtime args,
Reader/Writer config), `CBDescriptor`, and a semaphore list — enough to place the sender kernel on
the sender core, receiver kernels across the R×C rectangle, wire the handshake semaphores + payload
CBs, and verify the output tensor bit-exact. Lives at `tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py`,
run with `scripts/run_safe_pytest.sh --dev`. No per-kernel rebuild. (A stale `.pyc` of this name from
unrelated prior work will be overwritten fresh.)

### (rejected) C++ host alternative
The program-setup (core grid, rectangle, CBs, semaphores, runtime args, payload verify) is
API-agnostic. Two options:
- **(Recommended) C++ host harness** modeled on `tt_metal/programming_examples/contributed/multicast/multicast.cpp`
  (already does intra-chip rectangle-mcast+handshake setup + verify). One parametrized host that
  loads the chosen bakeoff kernel pair, sizes the rect/iters/payload from argv, verifies bit-exact,
  wraps the send block in `DeviceZoneScoped` for ns. **Cost:** needs `./build_metal.sh` once; then
  kernel edits are free.
- **(Alt) Python pytest** via the ttnn low-level program API at
  `tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py`, run with `scripts/run_safe_pytest.sh --dev`.
  No rebuild per kernel; but raw-kernel-on-arbitrary-rect program construction from python is more
  fiddly. (A stale `.pyc` of this name exists from unrelated prior work — will be overwritten fresh.)

Either way: **Pass 1 runs under `--dev`** so a hang is a captured triage result, not a crash.

---

## E.3 — The matrix (staged: coverage → perf)

### Coverage axes (Pass 1 — every variant, correctness/hang only, under `--dev`)
| Axis | Cells |
|---|---|
| rectangle shape | `1×2` (min), `1×8` (row), `8×8` (full), `2×4` (block) |
| sender placement | out-of-rect · in-rect skip-self · in-rect loopback · **self-only (degenerate `num_dests==1`)** |
| iterations | `N=1` · `N=8` (exposes stale-flag re-trigger) |
| payload size | `1 tile` (< burst) · `> NOC_MAX_BURST_SIZE` (forces chunking) |

**Pass-1 pass = data landed bit-exact on every receiver AND no hang.** Record pass/hang/mismatch per
(variant × cell). Any fork where one variant fails cells the other passes is **decided here, no perf**.

### Perf pass (Pass 2 — survivors only, only on cells both pass)
| Metric | How |
|---|---|
| NoC latency (ns) | `DeviceZoneScoped` around the send block (or tracy `-r`) |
| L1 footprint (bytes) | sem slots + CB depth per variant |

---

## E.4 — Decision rules (recorded here at beat 2)
Per fork: DOMINANT (one variant wins coverage+perf → single path) · TRADEOFF+recognizable predicate
(→ internal dual/tri-path, only if perf gap ≥10% **and** predicate is `constexpr`) · TRADEOFF+
non-recognizable (→ safe-global or documented-precondition knob) · NOISE-TIE (→ qualitative).
Dual-path cap ~2/helper. Micro-bench perf gaps are **provisional** (confirm against the real op in
build-helper); coverage verdicts are final.

## Hypotheses going in (to be confirmed/refuted by data — NOT the decision)
- **F1:** flush dominates — INV4 (data-before-flag, same VC) + the receiver's flag-wait already prove
  arrival, so the barrier round-trip is expected pure overhead. *Risk:* a BH erratum could make
  barrier load-bearing → matrix includes the arch we run on; flag if barrier is ever *required*.
- **F3:** loopback (INCLUDE_SRC) beats EXCLUDE+local-copy when sender needs its own copy (one NoC op
  vs op+memcpy), but **hangs at `num_dests==1`** (degenerate) — so tri-path with the degenerate guard
  is the likely outcome, predicate `constexpr` (rect size, sender∈rect).
- **F4:** linked+flush beats unlinked+barrier (skips a NoC round-trip and a VC re-acquire); predicate
  = `linked` (constexpr) → dual-path candidate.
- **F2:** counter is coverage-safer at `N=8` (no stale-retrigger window); if perf is within noise,
  keep both as the `STAGING` use-case knob rather than forcing one.

---

## RESULTS — Pass 1 coverage map

**Harness validated** (`test_smoke_flush`, BH p150a): object-API sender mcast → 2 receivers →
bit-exact DRAM readback PASS. Virtualization hardcode logical→virtual `(+1,+2)` confirmed correct
(data landed at the right cores). JIT-compiles the object-API kernels cleanly.

**Pass-1 matrix: 72/72 cells PASS** (6 variants × rects {1×2,1×8,4×2} × N {1,8} × payload {1,4 tiles},
sender out-of-rect). f1_flush, f1_barrier, f2_flag, f2_counter, f4_linked, f4_unlinked all land
bit-exact across every cell with no hang.

**Coverage-decided correctness facts (final, not provisional):**
- **F2 counter requires an atomic barrier**, not a write flush. `Semaphore::inc_multicast` issues a
  NON-POSTED multicast atomic that expects `num_dests` ACKs; fencing it with `async_writes_flushed`
  (writes only) **hangs** (observed at 1×2, N=1 and N=8). Fixed by `noc.async_atomic_barrier()`.
  ⇒ F1's atomic-barrier value is **forced on the counter path** (confirms the Step-D ternary-F1 amendment).
- **F3 same-core sender+receiver (R6) hangs** (dispatch/program-construction exception) — confirms
  the rotating-sender/role-flip hybrid is out of scope for one Pipe object (Step ★). INCLUDE_SRC
  loopback itself is proven in production (census C1/C2), not indicted.
- **F4 linked passes all coverage** — no deadlock in any cell.

| Fork | variant A | variant B | coverage |
|---|---|---|---|
| F1 | flush | barrier | both PASS all 24 cells each |
| F2 | flag | counter (+atomic barrier) | both PASS all 24 cells each |
| F4 | linked | unlinked | both PASS all cells |

## RESULTS — Pass 2 perf (amplifying cell: 1×8 rect, N=8, payload 4 tiles; sender=NCRISC)

| variant | sender NCRISC (ns) | overall kernel (ns) | vs baseline |
|---|---|---|---|
| **f4_linked** | **3585** | **4770** | fastest |
| f2_flag | 5505 | 6960 | — |
| f1_flush | 5576 | 7039 | F1/F4 baseline |
| f1_barrier | 7616 | 8580 | +37% vs flush |
| f2_counter | 7719 | 8668 | +40% vs flag |

(L1 footprint: all variants identical — same CB depth + 2 sem slots; flag vs counter differ by 0 bytes.
F4 footprint identical. So L1 is not a tiebreaker.)

### F3 perf — sender-in-rect self-fill: INCLUDE_SRC loopback vs EXCLUDE_SRC + local self-copy
Run separately (clean setup: sender ∈ rect runs only the sender kernel + writes its own shard;
other column cores are plain receivers — avoids the R6 same-core hang). 8-core column, N=8.
Both arms PASS coverage (sender's own shard bit-exact) at payloads {1,4,16}. Perf (sender NCRISC ns):

| payload | INCLUDE_SRC | EXCLUDE_SRC + local self-copy | winner |
|---|---|---|---|
| 1 tile (2 KB)   | 2359  | 2335  | tie (~1%, noise) |
| 16 tiles (32 KB)| 7939  | 9985  | **INCLUDE +26%** |
| 64 tiles (128 KB)| 27630 | 38889 | **INCLUDE +41%** |

The EXCLUDE arm pays a *separate serial NoC self-read* of the whole payload after the mcast;
INCLUDE_SRC folds the self-write into the broadcast tree for free, so the gap grows with payload.

## OUTCOMES per fork

- **F1 — flush DOMINANT (single path).** flush 5576 ns vs barrier 7616 ns = **−27% sender / −18% kernel**,
  >10% threshold, full coverage. The receiver's flag-wait + same-VC ordering (INV4) already proves
  arrival, so the barrier round-trip is pure overhead — hypothesis confirmed. **Bake in flush.**
  Atomic-barrier remains forced on the counter path only (mechanism, not a choice).
- **F2 — flag DOMINANT on perf (default flag; counter stays a use-case knob).** flag 5505 ns vs counter
  7719 ns = **−29%**, and counter carries extra complexity (atomic barrier, ACK accounting). flag is the
  baked-in default. Counter is **not** a perf dual-path; it is exposed as `STAGING::Counter` ONLY for ops
  whose protocol genuinely needs a monotone, reset-free counter (e.g. layernorm phase-2 streaming, census C3).
- **F4 — linked-pair DOMINANT (user-decided + confirmed).** linked 3585 ns = fastest, **−36% vs unlinked
  flush**, full coverage. Matches the user directive "linked where possible." Bake linked-pair+flush as the
  default; fall back to unlinked+barrier-between only where the call site forbids linking (predicate =
  `linked` constexpr — the F4 dual-path), e.g. a barrier is structurally required between data and flag.
- **F3 — tri-path; the sender-in-rect arm is INCLUDE_SRC, decided by MEASUREMENT (not argument).**
  Three branches on a constexpr predicate (sender∈rect?, rect==1?):
  - **sender ∉ rect (or src==dst):** EXCLUDE_SRC — forced (no self to fill); proven by all coverage.
  - **sender ∈ rect, needs own copy:** **INCLUDE_SRC loopback DOMINANT** — both arms correct, but INCLUDE
    is **+26% (32 KB) to +41% (128 KB)** faster than EXCLUDE_SRC+local-self-copy, tie only at 1 tile. The
    "EXCLUDE + unicast/local-copy on the sender" alternative is **empirically rejected** — strictly ≥ INCLUDE's
    cost across the measured range. (Corrects the earlier hand-waved "not a perf winner".)
  - **degenerate (`num_dests==1`/self-only):** internal unicast/local-copy guard (documented hang otherwise).
  Same-core role-flip (R6) stays out of scope.

### Dual-path budget check
Single-path defaults for F1 (flush), F2 (flag). **Two** internal dispatch points, both constexpr-clean:
**F4** (linked vs unlinked, predicate = `linked`) and **F3** (loopback mode, predicate = sender∈rect/rect
size). Within the ~2-per-helper cap. F1/F2 are NOT dual-paths (flush/flag dominate globally; counter is a
use-case knob). Perf gaps are micro-bench (provisional per E.4) but F1/F2/F4 all clear ≥10% comfortably;
F3 is correctness/geometry-forced (not provisional).
