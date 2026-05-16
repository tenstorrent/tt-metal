<!-- SUMMARY: Opus audit of FIX AD (TCP-style symmetric ETH handshake) commits 749979f96f4 + aaa3b13a1e2
     KEYWORDS: opus-audit, fix-ad, symmetric-handshake, prepare_handshake_state, fabric_symmetric_handshake, deadlock, #42429
     SOURCE: Code review of edm_handshake.hpp, fabric_router_eth_handshake.hpp, fabric_erisc_router.cpp, erisc_datamover_builder.cpp, catchall_catch22_analysis_20260516.md, strategy_report_20260516_2137.txt
     SCOPE: Correctness, commit message accuracy, dead code, open gaps, next steps
     USE WHEN: Reviewing FIX AD correctness, planning follow-on fixes, PR review prep -->

# Opus Audit: FIX AD — TCP-style Symmetric ETH Handshake

**Date**: 2026-05-16 23:xx UTC
**Commits**: `749979f96f4` (FIX AD) + `aaa3b13a1e2` (research notes) + `cb91a6c8a37` (unused variable fix)
**Branch**: `nsexton/0-racecondition-hunt`
**Auditor**: Task 2 Opus Audit agent

---

## VERDICT: CORRECT — APPROVE WITH NOTES

FIX AD is architecturally sound and eliminates the STARTED-STARTED deadlock class by construction. The three commits are clean. The primary issue is one commit message inaccuracy (low severity). Two open gaps are not introduced by FIX AD — they preexist — but are worth tracking.

---

## 1. Correctness Assessment

### 1.1 Core Protocol (PASS)

`prepare_handshake_state()` is called at line 3260 in `fabric_erisc_router.cpp`, immediately after `*edm_status_ptr = STARTED` (line 3251). Object Setup begins at line 3268. The handshake loop calls `fabric_symmetric_handshake()` at line 3590.

Timeline on each side:
```
STARTED write (3251)
  → prepare_handshake_state() [local_value=0, scratch=MAGIC] (3260)
  → ~318 lines of Object Setup (channel alloc, interface construction) (3268–3584)
  → wait_for_other_local_erisc() (3582)
  → fabric_symmetric_handshake() — sends MAGIC + polls MAGIC (3590)
```

For the old erase race to resurface, the peer must reach `fabric_symmetric_handshake()` and send to our `local_value` BEFORE we execute `prepare_handshake_state()`. This would require the peer to complete its entire ~318-line Object Setup in less time than our single `prepare_handshake_state()` function call. This is practically impossible under any realistic firmware execution skew.

**VERDICT: Race window is closed by sufficient timing margin.**

### 1.2 Symmetric Protocol (PASS)

Both sides now run the identical `fabric_symmetric_handshake()` loop: send MAGIC AND poll for MAGIC every iteration. No sender/receiver role assignment. No compile-time flag driving different code paths. The STARTED-STARTED deadlock (where both sides wait for the other to send first) cannot occur.

The post-loop final send (FIX HS1/HS2) is retained as belt-and-suspenders. It's harmless and handles the degenerate case where one side exits the loop before the other enters it.

**VERDICT: Protocol is correct and deadlock-free.**

### 1.3 IS_HANDSHAKE_SENDER=0 (PASS)

`erisc_datamover_builder.cpp` now sends `IS_HANDSHAKE_SENDER=0`. The `is_handshake_sender` constexpr in `fabric_erisc_router_ct_args.hpp` remains for compile compat but is never read by `fabric_erisc_router.cpp`. Confirmed by grep: `is_handshake_sender` does not appear in the router. Dead but harmless.

**VERDICT: Correct. The dead constant can be cleaned up in a separate PR.**

### 1.4 Legacy Compat (PASS)

`sender_side_handshake()` and `receiver_side_handshake()` are kept as thin wrappers over `init_handshake_info()` + `symmetric_handshake()`. Non-fabric callers of the deprecated API get symmetric semantics automatically. DEPRECATED comments are present. No callers appear to be relying on the old asymmetric behavior for correctness.

**VERDICT: Backward compat is safe.**

### 1.5 Diagnostic Fields (PASS, minor gap)

`diag_send_count` is now incremented in `fabric_symmetric_handshake()` but was missing from the old `fabric_sender_side_handshake()`. This is a net improvement. The `diag_local_val_at_init` field now records the value before the unconditional zero (without the FIX HX conditional), which is still useful for diagnosing unexpected non-zero values at init time.

---

## 2. Issues Found

### 2.1 COMMIT MESSAGE INACCURACY (LOW SEVERITY)

The commit message (and `fix_ad_implementation_notes.md`) states:

> `prepare_handshake_state()` — called during Object Setup (before edm_status=STARTED)

**This is incorrect.** The code places `prepare_handshake_state()` AFTER `*edm_status_ptr = STARTED` (line 3251) and BEFORE the Object Setup comment block (line 3268). The correct description is:

> Called immediately after `edm_status = STARTED` is written, before Object Setup begins.

The FIX AD correctness argument does not depend on the call being before STARTED — it depends on it being before the handshake loop. The safety margin comes from Object Setup duration, not the STARTED gate. The misdescription is non-critical but should be corrected in `fix_ad_implementation_notes.md` to avoid confusion for future readers.

**Action**: Update `fix_ad_implementation_notes.md` to say "after `edm_status = STARTED`, before Object Setup" not "before `edm_status = STARTED`".

### 2.2 TIMING DEPENDENCY IS NOT HARD GUARANTEE (LOW SEVERITY, PRE-EXISTING)

The safety of `prepare_handshake_state()` running before the peer's `symmetric_handshake()` depends on Object Setup taking sufficient time — it is not enforced by an explicit ordering barrier. If Object Setup were ever significantly shortened (e.g., compile-time channel counts reduced to 1, loop elided), the window could theoretically reopen. This was noted in `catchall_catch22_analysis_20260516.md` as Catch-22 #3.

This is pre-existing and not introduced by FIX AD. The current timing margin is large (~318 lines including loops). A future hardening option would be an L1 flag written by the peer after prepare, with a poll before `symmetric_handshake()`.

**Action**: Document as known gap. No code change required now.

### 2.3 DOMINANT OPEN FAILURE: NON-MMIO RELAY CATCH-22 (PRE-EXISTING)

Per `strategy_report_20260516_2137.txt`, CI run 25973430393 failed for a different reason: all 4 non-MMIO devices (4,5,6,7) had dirty relay channels from the prior session. FIX NX timed out on all 4 (5s each), excluded all non-MMIO devices from the fabric, and the ring sync master had no peer → 120s timeout → test FAILED.

**This is not caused by FIX AD.** The handshake deadlock class is confirmed CLOSED. The open issue is the relay bootstrap paradox (Catch-22 #2).

Recommended next steps from the strategy report:
1. **FIX DW** — add 50ms delay before FIX DU heartbeat poll (prevents stale 0ms read, same pattern as FIX DS)
2. **FIX DX** — fast ring sync skip when all non-MMIO relay channels confirmed broken (saves 120s timeout)
3. **OPTION B** (medium-term) — ETH-DMA based non-MMIO reset via MMIO fabric firmware

---

## 3. Code Quality

### 3.1 edm_handshake.hpp
- `prepare_handshake_state()` is clearly named and well-commented
- `symmetric_handshake()` comment accurately describes the LLDP-style protocol
- FIX HX removal justification is correct and documented inline
- Legacy function comments include DEPRECATED tags and usage guidance

### 3.2 fabric_router_eth_handshake.hpp
- `fabric_symmetric_handshake()` adds `diag_send_count` tracking (improvement over prior omission)
- Termination signal handling (non-Wormhole post-loop guard) is preserved correctly
- Legacy wrappers are thin and correct

### 3.3 fabric_erisc_router.cpp
- `prepare_handshake_state()` callsite is well-guarded by `if constexpr (enable_ethernet_handshake)`
- The replacement of the `if constexpr (is_handshake_sender)` branch with a naked `{}` block is clean
- Callsite comment explains the two-phase protocol (prepare at Object Setup, handshake later)

### 3.4 erisc_datamover_builder.cpp
- `IS_HANDSHAKE_SENDER = 0` comment is clear about the reason
- `is_handshake_master` retained for `IS_LOCAL_HANDSHAKE_MASTER` — correct
- The unused variable warning caught by CI was a minor oversight, fixed in cb91a6c8a37

---

## 4. Failure Class Table (Updated)

```
Class                            Status      Fix
──────────────────────────────── ─────────── ─────────────────────────────
STARTED-STARTED deadlock         CLOSED ✅   FIX AD (symmetric handshake)
MMIO base-UMD stale firmware     CLOSED ✅   FIX S9 + FIX DU
FIX XZ stale heartbeat           CLOSED ✅   FIX DS (50ms pre-poll delay)
Non-MMIO relay catch-22          OPEN ❌    FIX DX (fast skip) + OPTION B
FIX DU stale read (0ms)          OPEN ❌    FIX DW (50ms pre-poll delay)
FIX DV consuming code            OPEN ⚠️    Wire quiesce_relay_transitioned_
Ring sync 120s timeout           OPEN ⚠️    FIX DX closes the symptom
prepare_handshake_state timing   Safe ✅     (timing margin, not hard barrier)
```

---

## 5. Recommended Actions (Priority Order)

1. **Fix commit message / research notes** (cosmetic, low effort):
   In `fix_ad_implementation_notes.md`, change "before edm_status=STARTED" to
   "immediately after edm_status=STARTED, before Object Setup begins".

2. **Implement FIX DW** (50ms delay before FIX DU poll):
   Follows FIX DS precedent. Prevents false "0ms heartbeat" positives.

3. **Implement FIX DX** (fast ring sync skip when non-MMIO relay all broken):
   Saves 120s per CI run when relay catch-22 is active. Test still fails but fast.

4. **Plan OPTION B** (ETH-DMA non-MMIO reset via MMIO fabric firmware):
   True fix for the dominant open failure class. Medium-term.

5. **Clean up dead code** (separate PR, optional):
   Remove `is_handshake_sender` constexpr from `fabric_erisc_router_ct_args.hpp`
   and the `IS_HANDSHAKE_SENDER` arg from `erisc_datamover_builder.cpp`.

---

## 6. CI Run Note

CI run 25972426760 (FIX AD commit) failed on build (unused variable, fixed in cb91a6c8a37).
CI run 25973430393 (cb91a6c8a37) built cleanly, test failed due to non-MMIO relay catch-22.
The handshake deadlock did NOT occur in run 25973430393. FIX AD is confirmed correct.

