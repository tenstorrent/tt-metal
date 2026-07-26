# TT-RDMA on Blackhole — Bring-up → Production Plan

Status: living plan (2026-07-24). The methodical path from what is validated on the bench today to a
full production solution. **Principle: no shortcuts, nothing left behind.** Every capability advances
through the same gate discipline — (1) works on silicon, (2) has an automated regression test, (3) error
paths handled, (4) reproducible from a clean state with zero manual steps, (5) documented. A checkbox is
only ticked when all five hold. "It worked once on the bench" is bring-up, not done.

Companion specs: `tt-rdma-tx-ring-spec.md` (TX), `tt-rdma-rx-dispatch-spec.md` (RX), `tt-rdma-gateway-sender.md`
(DOCA TX leg), `bf3-gateway-design.md` / `tt-rdma-dpa-gateway-spec.md` (gateway), `tt-rdma-bh-bf3-impl-plan.md`
(phase map). The base FW (`bh-erisc`) is untouched except where a phase explicitly calls for a reflash.

---

## Honest current state (what "validated" really means)

| Area | Bench-validated | NOT yet production |
|---|---|---|
| TX ring | 200G/rail, 397G aggregate, byte-exact | no automated regression; `pace` is a manual throttle, not accept-ahead; unpaced wedge uncharacterized |
| RX dispatch + MR WRITE | byte-exact to Tensix via NoC; 15.8 Gbps/rail | WRITE-only; no SEND/READ/ACK; no CRC check; MR is one hard-coded slot; overload degrades (resync tax) |
| Streaming ring | 128KB BUF_WRAP, resync-on-lap | resync tuning ad-hoc; no PFC so lossy above ceiling |
| Jumbo | 4080B both ways | MTU config is manual + lost on every reboot |
| DOCA sender | ~143 Gbps HW-TX, repo-reproducible | sends a FIXED frame — not a real gateway (no RoCE translation) |
| Link / MTU / DPU | works after manual recovery | not persistent; every reboot needs hand re-training + re-MTU + tmfifo IP |
| Tests | manual bench runs | **no CI, no automated regression for ANY of the above** |

The gap to production is mostly **hardening, tests, reliability, completeness, and the real gateway** —
not new datapath proofs.

---

## Phase 0 — Consolidate & make bring-up repeatable (foundation; no new features)

Rationale: everything below rests on a clean, reproducible, tested baseline. Do this first or every later
phase inherits manual bench fragility.

- **0.1 Persistent bench bring-up. ✅ DONE** — `tt_metal/tt_rdma/bh0/bringup.sh` (idempotent): `mst start`,
  force-200G both rails, tmfifo IP, MTU 9000 (host PF + DPU p0/p1/pf0hpf/pf1hpf, re-applying host after
  the uplink flap), then a VERIFY gate = MTU + DPU-reachable + **BH sees both external rails**
  (authoritative link-up proof; mlxlink State is informational). `--verify-only` checks without changing.
  Passes green on the current rig. TODO: wrap in a systemd unit / boot hook so it runs automatically on
  boot (currently run-on-demand).
- **0.2 Automated regression harness. ✅ core DONE** — `tt_metal/tt_rdma/bh0/regression.sh`: asserted
  PASS/FAIL + exit code for the core invariants — T1 golden wire-header vectors, T2 TX egress (frames on
  the wire), T3 RX inbound WRITE byte-exact (dispatch + MR + noc_async_write to Tensix, landing == "TTWR"),
  T4 RX streaming lossless (BUF_WRAP, bad==0). Uses 256B host-sender frames so correctness gates don't
  depend on jumbo/DPU state. Runs green (5/5) on the rig. TODO: add TX-aggregate (2-rail), jumbo-both-ways,
  and RX-ceiling perf assertions (perf tests, separate from correctness); fold into a `make test` target.
- **0.3 Close known rough edges.** `risc_touch` hang (remove or guard), TX `pace`→real accept-ahead
  (characterize the unpaced wedge, replace the spin-pace), RX resync-on-lap tuned + counted, all the
  "uncommitted diagnostic" tooling either committed clean or removed.
- **0.4 CI wiring.** The regression harness runs on a labeled bench runner (nightly/on-PR where HW is
  available); a HW-less subset (golden vectors, header self-tests, builds) runs on every PR.

Exit gate: a documented, one-command path from cold rig to all current claims re-verified automatically.

## Phase 1 — Complete the RX protocol (correctness & completeness)

No opcode left behind; every path tested + error-handled.

- **1.1 CRC-32C validation** on every inbound header (drop + count on mismatch); golden-vector tests.
- **1.2 SEND / SEND_IMM** → host RxWqeRing (NoC→PCIe push to a hugepage ring), completion to the host;
  test byte-exact delivery + CQE.
- **1.3 READ_REQ / READ_RESP** — target-side READ handler (NoC read from MR → RESP frame via TXQ),
  initiator correlation; round-trip byte-exact test.
- **1.4 ACK (0x40)** reception + cumulative-ACK accounting (pairs with Phase 2 reliability).
- **1.5 WRITE_IMM / imm_data** completions.
- **1.6 MR table lifecycle** — CONTROL-opcode register/deregister, rkey `(slot<<24)|rand|gen` generation
  + rotation, 64-slot management, and **access-control enforcement tests** (rkey_miss / rkey_access /
  rkey_bounds / rkey_wrap each provably dropped + counted). Security-relevant — no shortcut.

Exit gate: all 8 v1 opcodes exercised end-to-end with automated byte-exact + error-path tests.

## Phase 2 — Reliability & flow control (production robustness)

- **2.1 PFC-lossless (BH.6)** — Rianta PFC bring-up (`eth_init.cpp:538` TODO). **The one base-FW change
  + reflash** (human reboot per CLAUDE.md). Gate: pause counters non-zero under overload, 0 buffer-discard,
  RX lossless at ≤ ceiling with the 143G sender.
- **2.2 Software reliability (Phase R)** for the external/raw path — cumulative-ACK + retransmit;
  selective-ACK (v1.1) for lossy fabrics. Auto-on for one-sided WRITE/READ.
- **2.3 Overload & fault matrix** — ring overflow, lapping, link flap, partner timeout, malformed bursts;
  each has a defined, tested, graceful behavior (never a wedge, never silent corruption).
- **2.4 Long soak** — 1h+ sustained both directions; assert no counter drift, no admit-rate decay, no
  leak, link stable. This is a hard gate for "production," not optional.

## Phase 3 — Performance to target (close the measured gaps)

- **3.1 RX BW** — coalesce `noc_async_write` for contiguous same-MR frames, cut per-poll overhead
  (stats/pace out of the hot loop), and **multi-rail RX aggregate** (both external rails, like TX's 397G).
  Target: push past ~16 Gbps/rail toward the parse ceiling; re-measure against the 143G DOCA sender.
- **3.2 TX** — replace `pace` with real accept-ahead depth; characterize + remove the unpaced wedge; confirm
  sustained line rate without the safety throttle.
- **3.3 Jumbo end-to-end persisted** (folds into 0.1); confirm line-rate both directions at 4080B.
- **3.4 On-chip MR targets** — DRAM / remote-chip NoC addresses in the MR (not just Tensix L1), the
  "line-rate, host-bypassed" case; measure vs the host-hugepage PCIe-bound case.

## Phase 4 — The real gateway (RoCE interop — the value proposition)

The current DOCA sender emits a fixed frame; the gateway does real per-packet translation so unmodified
apps work. This is the impl-plan G.x MVP and the reason the project exists.

- **4.1 WAN RoCE QP termination** — `doca_rdma` RC QP on the BF3 "wan" port; RDMA-CM connection setup.
- **4.2 Per-packet translation** — BTH/RETH parse → 32B TT header (opcode/rkey/remote_offset/seq); the
  `tt_v1_codec` sharing the frozen `tt_rdma_wire.h` golden vectors with the chip (M-3 loopback contract).
- **4.3 MR / QP / PSN↔seq tables** + `bh_mr_agent` (host-side MR registration bridge; BF3 has no PCIe to
  the chip).
- **4.4 Interop gate** — `ib_write_bw` / `ib_read_bw` from a stock ConnectX → gateway → bytes land in a
  BH MR, byte-exact, ACKs working. Then MPI/UCX hello-world.
- **4.5 Translation tiering** — T1 Arm (correctness) → T2 zero-copy header-only → **T3 DPA** (line-rate,
  in-datapath) per `tt-rdma-dpa-gateway-spec.md`. Each tier measured.

## Phase 5 — Full production (SDK, scale, ops, sign-off)

- **5.1 Host SDK** — `TtRdmaEndpoint` (`register_mr`/`post_send_*`/`poll_cq`) retargeted to BH, productized,
  with API tests; folds in the RxWqeRing + completion model.
- **5.2 Scale & HA** — multi-rail matrix across the 14 ETH SS, gateway rail-matrix, single-BF3-SPOF failover
  (drain + RDMA-CM reconnect; rkey is rail-agnostic).
- **5.3 Coexistence** — the RDMA active-eth kernel running concurrently with stock tt-fabric on TT↔TT rails
  in the same mesh (impl-plan §11.2 gate); confirm no reserved-set collision.
- **5.4 Security review** — rkey/access/bounds/PD enforcement audited; no path writes an unvalidated NoC
  address; the raw-external trust boundary documented.
- **5.5 Observability** — per-rail counters/telemetry (admit, drop reasons, BW, PFC pause, retx), exported.
- **5.6 Docs & onboarding** — every spec reconciled to shipped reality (kill the stale WH `fw-arch-rx.md`
  and `blackhole-port.md` TCAM premise), runbooks, and a clean onboarding path.
- **5.7 Production sign-off** — the full regression + soak + interop suite green on a cold rig, reviewed.

---

## Working discipline (applies to every phase)

1. **Test before tick.** Each item lands with its automated regression test in the same change.
2. **Reproducible from cold.** If it needs a manual bench step, it isn't done — script it (Phase 0.1).
3. **Error paths are features.** Every drop/failure has a defined behavior + a counter + a test.
4. **Commit small, push, keep the branch green.** Each item is a reviewable change on
   `aperezvicente/tt-rdma-bh-bf3`.
5. **No stale docs.** Findings update the specs; wrong premises are corrected, not left to mislead.
6. **Honest ledger.** This file's "current state" table is updated as items move from bench-validated to
   production-done; nothing silently drops off.
