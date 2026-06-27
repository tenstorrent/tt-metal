# mcast `Pipe` — Round 2: drop `MCAST`, add active-core count

**Branch:** `sjovic/mcast-helpers-round2`
**Trigger:** review feedback after the tune-helper + apply-helper round —
*"I don't want mcast mode to be exposed; you're just missing the number of active
cores as a parameter — with that we should be able to infer the mcast modes."*

This is a **feedback round** on an already-materialized + already-rolled-out helper.
The job is to apply one API change coherently across every artifact the skill chain
produced, and to leave a round log so the *next* round is cheap.

---

## Decision record (locked with the user)

| # | Decision | Choice |
|---|---|---|
| D1 | How does the new count surface, given data-mcast destinations can diverge from the handshake ACK count (the conv-1D weights hang: `active_cores-1 ≠ num_cores-1`)? | **Two counts.** `McastRect` becomes pure geometry (its area = data-mcast destination population); the Pipe takes a *separate* `num_active_cores` for the handshake. |
| D2 | How is EXCLUDE_SRC vs INCLUDE_SRC (loopback) chosen now that the knob is gone? | **Runtime inference.** The Pipe compares its own NoC coords (`my_x`/`my_y` in `noc_`'s index space) against the rect at send time. No caller input. |

### Why this is the right shape
- The three modes are **fully determined by geometry**, so the knob was always redundant:
  - sender **outside** rect ⇒ `EXCLUDE_SRC` (matmul in0: sender col 0, receivers cols 1…N)
  - sender **inside** rect ⇒ `INCLUDE_SRC` loopback (sharded gn/ln: all cores incl. sender)
  - `num_active_cores == 1` ⇒ degenerate self-only → local copy
- `my_x`/`my_y` come from the per-NoC `NOC_ID_LOGICAL` register
  (`noc_nonblocking_api.h:695`), i.e. the core's coords in the **same NoC-index space
  the rect is expressed in**. The existing `local_copy_` already uses
  `my_x[noc_.get_noc_id()]`, so the comparison is sound **provided it is done in the
  Pipe's `noc_` index space** (implementation rule, IR1 below).
- The **second count is not optional polish** — it is the fix for the conv-1D sender
  hang from the round-1 rollout. The data mcast covers the whole rectangle (geometry,
  = `McastRect` area) but only the *active* cores ACK the handshake. Today's single
  `num_dests` cannot express that, which is part of why senders migrated worse than
  receivers. Two counts unblocks the senders.

---

## New API (target)

```cpp
// PURE GEOMETRY now — no num_dests field. Area = data-mcast destination population.
struct McastRect {
    uint32_t x0{}, y0{}, x1{}, y1{};
    static constexpr McastRect single_core(uint32_t x, uint32_t y) { return {x, y, x, y}; }
};

template <
    Staging STAGING = Staging::Flag,   // use-case knob (unchanged)
    bool    PRE_HANDSHAKE = true,       // use-case knob (unchanged)
    bool    LINK = true>                // internal F4 (unchanged)
class Pipe {
public:
    // num_active_cores = how many cores participate in the handshake (the ACK count).
    Pipe(const Noc& noc, const McastRect& dest, uint32_t num_active_cores,
         Semaphore<> data_ready, Semaphore<> consumed);
    // ... send/receive/send_signal/receive_signal unchanged in signature ...
};
```

What the Pipe now derives **internally** (was caller-facing):
- **loopback mode** — runtime `sender_in_rect = inside(my_x[noc], my_y[noc], dest)`.
- **data-mcast destination count** — from `dest` geometry (rect area). This is the
  `num_dests` argument to `noc_async_write_multicast[_loopback_src]`.
- **handshake ACK count** — `num_active_cores` (the `consumed_.wait(...)` value and the
  degenerate `==1` collapse trigger).

**Removed:** the `MCAST` template parameter (first param today), and the `num_dests`
field on `McastRect`. `Noc::McastMode` is no longer named at any call site.

### Implementation rules
- **IR1** — do the sender-in-rect comparison in `noc_.get_noc_id()`'s coord space (same
  space the rect uses). Never mix NoC-0/NoC-1 coords.
- **IR2** — degenerate collapse keys off `num_active_cores == 1`, not rect area
  (a 1×1 rect that *is* the sender is the self-only case).
- **IR3** — when `sender_in_rect` and rect-area == active-count, the loopback `num_dests`
  still counts self (INV8); keep the data-mcast count = rect area, ACK count =
  `num_active_cores`, and let them differ where the op needs it (conv-1D).

---

## Phased plan

### Phase 0 — pin the contract (no code)
- Create `helper_design/mcast_pipe/changelog.md`: a round log
  (Round 1 = original tune+apply; Round 2 = this change). Each entry: trigger, decisions,
  artifacts touched, verification result. This is the cheap-iteration mechanism for
  the feedback still to come.
- This plan file is the Round-2 design record the changelog points at.

### Phase 1 — helper (`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`)
- `McastRect` → pure geometry; drop `num_dests`; fix `single_core`.
- Constructor gains `uint32_t num_active_cores`; store it.
- Delete the `MCAST` template param; add a private `sender_in_rect_()` (IR1).
- `send_data_`: pick `noc_async_write_multicast_loopback_src` vs `noc_async_write_multicast`
  on the runtime flag; data-mcast count = rect area; loopback degenerate guard on
  `num_active_cores == 1` (IR2).
- `send()` PRE_HANDSHAKE wait + degenerate short-circuit key off `num_active_cores`.
- `raise_flag_` / `set_multicast`: destination count = rect area (geometry), mode = runtime flag.
- Rewrite the header contract block: the F3 tri-path is now *inferred*, not a knob;
  document the two counts and IR1–IR3.
- No device cost to edit — kernels JIT-build.

### Phase 2 — unit test + bake-off (the gate)
- `tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py`: stop sweeping `MCAST`/`num_dests`;
  instead pass geometry + `num_active_cores` and **assert the Pipe infers** EXCLUDE
  (sender out of rect), INCLUDE (sender in rect), and degenerate (active==1). Keep the
  divergent count case (rect area > active) as a regression for the conv-1D hang.
- `kernels/bakeoff_mcast_*.cpp`: drop the `LOOPBACK_INCLUDE` define-switch; the kernel
  now lets the Pipe decide. (Keep `test_mcast_pipe_bakeoff.py` raw baseline as-is — it
  measures primitives, not the Pipe API.)
- Run on device: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py`
  — first one parametrization to clear compile, then the full suite. **This is the gate
  before touching any op.**

### Phase 3 — migrate the 13 call sites
Same discipline as round 1: tier-ordered (receivers + canonical senders first), each
guarded by its mapped test from `migration/test_map.json`, **atomic git commit per
kernel**, `--mode=halt` (stop at first failure, leave the diff to debug).

Per call site the edit is mechanical:
- drop the `Pipe<Noc::McastMode::…>` template arg → `Pipe<>` (or just the use-case knobs);
- `McastRect{…, num_dests}` → `McastRect{…}` + pass the active-core count as the new ctor arg.

Senders to re-verify with care (they carry the two-count divergence):
matmul in0/in1 senders (`in0_mcast_num_dests` vs `in0_mcast_num_cores`),
conv weights sender, sharded-LN sender. The receivers are count-independent and trivial.

### Phase 4 — reconcile the record
- Docs: `proposed_helpers.md`, `api_feasibility.md`, `style_bakeoff.md` (F3 section),
  `hazards_catalog.md` (INV8 count rule), `primitive_contracts.md` — change every place
  that frames MCAST as a caller knob to "inferred from geometry + active count."
- `migration/report.md` + `tiers.md`: append a Round-2 section (what changed, re-verified set).
- Memory: update `project_mcast_pipe_rollout` (the API-limit bullet about single
  `num_dests` is now resolved) and `project_mcast_pipe_tunehelper` (final API).
- Append the Round-2 entry to `changelog.md` with the device result.

---

## Risks / watch items
- **R-A (coord space, IR1):** the central correctness bet. De-risked by code read
  (`NOC_ID_LOGICAL`, existing `local_copy_` usage) but *proven* only by the Phase-2
  device run with a sender both in- and out-of-rect.
- **R-B (perf):** runtime branch per `send()` replaces a constexpr `if constexpr`. Cold
  path, one comparison vs a NoC transfer — expected negligible. Confirm against a
  bake-off perf number if any op regresses.
- **R-C (already-deferred set unchanged):** R6 role-flip, R4 streaming, CCL, interleaved
  group_norm stay out of scope — this change does not revive them.
- **R-D (Pipe v2 multi-rect):** still single-rect. gn/welford multi-rect senders remain a
  separate limitation; not addressed here. Flag if the next feedback round wants it.
