<!-- SUMMARY: Implementation notes for FIX AD — TCP-style symmetric ETH handshake (Fix A+D) refactor
     KEYWORDS: fix-ad, symmetric-handshake, prepare_handshake_state, fabric_symmetric_handshake, deadlock, #42429
     SOURCE: Implementation session 2026-05-16, based on tcp_vs_fabric_init_comparison.md
     SCOPE: What was changed, why, and where — commit 749979f96f4
     USE WHEN: Reviewing the FIX AD commit, debugging handshake regressions, planning further protocol changes -->

# FIX AD Implementation Notes (2026-05-16)

## Commit: `749979f96f4` on `nsexton/0-racecondition-hunt`

## What Changed

### 1. `edm_handshake.hpp`
- **`init_handshake_info()`** renamed to **`prepare_handshake_state()`** with key change: unconditionally zeros `local_value` (no FIX HX guard needed — runs during Object Setup, long before any peer can send).
- `init_handshake_info()` kept as a thin wrapper around `prepare_handshake_state()` for backward compat.
- **New `symmetric_handshake()`**: Both sides send MAGIC AND poll. Takes only `handshake_register_address` and timeout — no mesh/device args (those were only needed for init, which is done earlier).
- Old `sender_side_handshake()` and `receiver_side_handshake()` kept but now delegate to `init_handshake_info()` + `symmetric_handshake()`. Marked DEPRECATED.

### 2. `fabric_router_eth_handshake.hpp`
- **New `fabric_symmetric_handshake()`**: Same symmetric pattern with termination signal support. Takes only `handshake_register_address`, `termination_signal_ptr`, and timeout.
- Old `fabric_sender_side_handshake()` and `fabric_receiver_side_handshake()` kept but delegate to `init_handshake_info()` + `fabric_symmetric_handshake()`. Marked DEPRECATED.
- Added `diag_send_count` tracking (was missing in old fabric_sender; only in base sender).

### 3. `fabric_erisc_router.cpp`
- **Object Setup (after `edm_status = STARTED`)**: Added `prepare_handshake_state()` call, guarded by `enable_ethernet_handshake`.
- **Handshake section (~line 3584)**: Replaced `if constexpr (is_handshake_sender)` branch with single `fabric_symmetric_handshake()` call.

### 4. `erisc_datamover_builder.cpp`
- `IS_HANDSHAKE_SENDER` now hardcoded to `0` (was `is_handshake_master`). The `is_handshake_master` variable is retained because `IS_LOCAL_HANDSHAKE_MASTER` still uses it for the local ERISC sync phase.

### 5. `fabric_erisc_router_ct_args.hpp`
- `is_handshake_sender` constexpr kept for compile compat, comment added noting it's deprecated and always 0.

## Why This Is Correct

1. **Fix A eliminates the root cause**: `local_value = 0` happens during Object Setup, hundreds of microseconds before the handshake loop. No peer can have sent MAGIC yet (they haven't reached STARTED). The FIX HX guard (check for MAGIC before zeroing) is no longer needed.

2. **Fix D eliminates the timing dependency**: Both sides send and poll, so even if they start at different times, whichever starts first will keep sending MAGIC until the other arrives and sees it. No sender/receiver asymmetry to get wrong.

3. **Post-loop final send (FIX HS1/HS2) kept as defense-in-depth**: Not required for correctness with Fix A+D, but harmless and provides extra safety margin.

4. **Host-side launch ordering (FIX AE/AF) kept**: No longer required for correctness, but doesn't hurt. Can be simplified later.

## CI Run
- Workflow: 119782334 (T3K unit tests)
- Run ID: 25972426760
- Inputs: `{"t3000-unit": "racecondition-hunt"}`
