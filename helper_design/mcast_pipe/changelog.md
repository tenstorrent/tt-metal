# mcast `Pipe` — change log

Round-by-round record of the helper's API and rollout. Each round: trigger →
decisions → artifacts touched → device verification. Diff API states here when a new
feedback round lands.

---

## Round 1 — tune-helper + apply-helper (2026-06-04 / 06-05)

- **Trigger:** standalone tune-helper run for the recurring NoC-multicast + semaphore
  handshake block; then apply-helper rollout.
- **API delivered:** `Pipe<MCAST, STAGING, PRE_HANDSHAKE, LINK>` + `McastRect{x0,y0,x1,y1,num_dests}`.
  Caller picks `MCAST` (EXCLUDE_SRC/INCLUDE_SRC) and a matching `num_dests`.
- **Bake-off winners baked in:** flush fence (−27%), level flag (−29%), linked pair (−36%),
  INCLUDE_SRC loopback for sender-in-rect (+26–41%).
- **Rollout:** 13 kernels migrated (atomic commits, mapped tests green), 7 reverted under
  `--mode=run-all`. 310 lines of open-coded mcast removed.
- **Artifacts:** `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`,
  `tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py` (45/45),
  `helper_design/mcast_pipe/*`, `migration/*`.
- **Known limits flagged:** single-rect `num_dests` couldn't express
  data-mcast-population ≠ handshake-ACK-count (conv-1D weights sender HUNG); senders
  migrated worse than receivers; R6 role-flip / R4 streaming / CCL deferred.

---

## Round 2 — drop `MCAST`, add active-core count (in progress)

- **Trigger:** review feedback — *"don't expose mcast mode; you're missing the number of
  active cores as a parameter — infer the modes from that."*
- **Decisions** (see `round2_active_cores_plan.md`):
  - D1: **two counts** — `McastRect` = pure geometry (area = data-mcast destination
    population); new ctor arg `num_active_cores` = handshake ACK count. Fixes the round-1
    population≠ACK divergence (conv-1D).
  - D2: **runtime mode inference** — Pipe compares its own NoC coords to the rect; deletes
    the `MCAST` template param and the `num_dests` field.
- **API after:** `Pipe<STAGING, PRE_HANDSHAKE, LINK>` + `McastRect{x0,y0,x1,y1}` +
  `Pipe(noc, dest, num_active_cores, data_ready, consumed)`.
- **Artifacts touched:** helper, unit test/bake-off kernels, 13 call sites, design docs,
  `migration/` report, memory.
- **Verification — Phase 2 unit gate (Blackhole p150a):** `test_mcast_pipe.py` **45/45 PASS**.
  The mode knob is gone, so the same suite now doubles as the runtime-inference gate and
  proves IR1 (coord-space comparison): EXCLUDE inferred (sender out-of-rect: coverage/smoke/
  pre_handshake), INCLUDE_SRC loopback inferred (sender in-rect: f3_loopback), degenerate
  self-only collapsed to local copy (num_active_cores==1: f3_degenerate). No hangs.
- **Status:** helper + unit gate done (committed). 13 call sites (Phase 3) pending review.

### Round 2 — inference refinement (Phase 3 finding)

Phase 3 migrating matmul exposed a flaw in D2 as first implemented. The first rule was
`loopback iff sender_in_box`, with `num_dsts = rect.area()`. That hung matmul 1d: the in0
sender sits at the **top-left corner of its own broadcast box** but uses EXCLUDE_SRC — it
already holds the in0 block as its mcast source and must not self-overwrite. "Sender in box"
alone cannot tell EXCLUDE-in-box (matmul) from INCLUDE (conv-WS round-robin).

**Refined rule (still no mode knob, still inferred):**
- `num_active_cores` is the **recipient count** = the NoC `num_dsts` for data+flag = the ACK
  count. (Confirmed against the proven raw matmul: it mcast to `in0_mcast_num_cores` with the
  comment *"num_dests must not include source"*; round-1 used that same value for the wait and
  passed → recipients == acks in the tested configs.)
- **loopback (INCLUDE_SRC) iff `sender_in_box && num_active_cores == rect.area()`** — the sender
  is in the box AND counted as a recipient. matmul: 15 recipients ≠ 16-core box → EXCLUDE ✓;
  conv-WS: readers == box → INCLUDE ✓.
- `rect.area()` is used ONLY for that test, never as a transfer count.

Gap the unit suite missed (sender-in-box + EXCLUDE) is now covered by the matmul mapped test;
a synthetic unit case is a follow-up.

### Round 2 — Phase 3 migration progress (per family, mapped-test gated, --mode=halt)
- **matmul (4 kernels):** in0 sender/receiver, in1 sender/receiver. 1d + 2d PASS. ✓
- **conv (3 migrated + 1 reverted):** 1d-receiver (HS) PASS; 2d sender+receiver (BS) PASS;
  **width-sharded activation sender REVERTED to raw** — un-inferable partial-box self-gather
  (see `loopback_inference_limitation.md`), raw test PASS. ✓
- **groupnorm (2 kernels):** reduce-receiver (legacy) + welford-receiver PASS. ✓
- **topk (1 kernel):** reader_final_topk send_signal PASS. ✓
- **layernorm (1 kernel):** reader_mcast_sender_unary_sharded_ln send_signal PASS. ✓

**Phase 3 result: 11 kernels migrated to the new API (all mapped tests green), 1 (conv-WS)
intentionally kept raw + documented.** All on BH p150a, single parametrization each.
