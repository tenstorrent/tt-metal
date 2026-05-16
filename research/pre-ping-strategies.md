<!--
SUMMARY: Five alternative strategies for the FIX CZ pre-ping handshake rendezvous regression
KEYWORDS: fabric, pre-ping, handshake, ERISC, ETH DMA, TXQ, racecondition-hunt, FIX CZ, FIX DA, FIX DB
SOURCE: BrAIn analysis session 2026-05-15, thread 1778776874.773369
SCOPE: Pre-ping rendezvous strategies for nsexton/0-racecondition-hunt branch
USE WHEN: Debugging fabric ERISC startup deadlock; revisiting pre-ping implementation
-->

# Pre-Ping Handshake Rendezvous Strategies

Background: FIX CZ introduced an ETH DMA pre-ping (non-MMIO → MMIO) to prevent the
simultaneous-handshake deadlock. The DMA approach has proven fragile due to
`init_handshake_info`'s defensive TXQ flush aborting the in-flight packet.

## Strategy 1 — Flip the pre-ping direction
**MMIO → non-MMIO instead of non-MMIO → MMIO.**

MMIO sends the pre-ping to non-MMIO. Non-MMIO spins on its `preping_addr` before
entering the handshake. MMIO has no `init_handshake_info` flush hazard on the send
side — it's already at PPWT and won't enter handshake until after the send completes.

- Pro: minimal code change; same DMA mechanism, reversed roles
- Con: MMIO send happens before MMIO knows non-MMIO is ready to receive;
  if non-MMIO hasn't loaded firmware yet the packet is lost

## Strategy 2 — Replace ETH DMA with a host-mediated signal  ← ACTIVE
**Non-MMIO ERISC writes nonce to local L1; host relays it to MMIO L1.**

Protocol:
1. Non-MMIO ERISC: write nonce to `preping_addr` in own L1 (plain store, no TXQ)
2. Non-MMIO ERISC: set `edm_status = HANDSHAKE_READY`, proceed to handshake
3. Host (FabricFirmwareInitializer): polls non-MMIO `preping_addr` via UMD relay until nonce appears
4. Host: writes nonce to MMIO chip's `preping_addr` via PCIe
5. MMIO ERISC: polls `preping_addr` in own L1, sees nonce, proceeds to handshake

- Pro: eliminates TXQ entirely; host already polls per-device status; no race with `init_handshake_info`
- Con: adds host to the startup critical path; relay polling adds latency;
  host must know (preping_addr, nonce) for each link — requires new CT arg plumbing to host

## Strategy 3 — Non-MMIO polls MMIO edm_status instead of pre-ping
**Non-MMIO ERISC reads MMIO L1 via ETH relay to gate handshake entry.**

Non-MMIO waits until MMIO's `edm_status == HANDSHAKE_READY` before entering the
handshake loop. Non-MMIO can read MMIO L1 via `internal_::eth_read_remote_reg()`.
No DMA, no TXQ.

- Pro: no DMA; no host involvement; symmetric/clean
- Con: `eth_read_remote_reg` is also TXQ-based (uses ETH DMA read); same hazard
  may apply. Need to verify.

## Strategy 4 — Flush TXQ before pre-ping send, not after
**Pre-flush guarantees TXQ is idle going into `eth_send_packet`.**

Current flow: send → TXQ busy → enter handshake → `init_handshake_info` flushes → abort.
New flow: flush → send → enter handshake → `init_handshake_info` flush is no-op (TXQ idle by the time DMA completes).

This is close to FIX DA, but the flush is moved BEFORE the send. After the send, we still
need to wait for DMA completion (same as FIX DA) OR trust that `init_handshake_info`'s
flush fires late enough. Does not fundamentally eliminate the race — just changes timing.

- Pro: one-line change; no new protocol
- Con: still relies on timing; `init_handshake_info` flush fires if DMA takes
  longer than expected; not a structural fix

## Strategy 5 — Remove pre-ping; stagger via edm_status poll
**Eliminate pre-ping entirely; use `edm_status` polling to serialize handshake entry.**

Non-MMIO ERISC spins reading MMIO's `edm_status_address` via ETH relay until it
sees `HANDSHAKE_READY`, then enters its own handshake. This guarantees MMIO enters
first and is listening before non-MMIO initiates.

- Pro: removes all DMA from the pre-ping path; clean protocol
- Con: ERISC ETH relay reads use a different TXQ (receive-side); need to confirm
  they don't conflict with `init_handshake_info`'s flush. Also adds round-trip
  relay latency per poll iteration.

---

## Status

| # | Name | Status |
|---|------|--------|
| 1 | Flip direction | Queued |
| 2 | Host-mediated relay | **ACTIVE** |
| 3 | Non-MMIO polls MMIO status | Queued |
| 4 | Pre-flush before send | Queued |
| 5 | Remove pre-ping, stagger via edm_status | Queued |
