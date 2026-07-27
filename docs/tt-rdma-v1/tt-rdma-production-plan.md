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
  T4 RX streaming lossless (BUF_WRAP, bad==0), T5 CRC drop, T6 SEND-ring, T7 READ round-trip, T8 ACK
  cumulative, T9 WRITE_IMM completion. Uses small host-sender frames so correctness gates don't depend on
  jumbo/DPU state. Runs green (**9/9**) on the rig.
- **0.2b Per-phase PERF gate. ✅ DONE** — `tt_metal/tt_rdma/bh0/perf.sh` (companion to regression.sh):
  measures **latency** (READ round-trip p50/p99, host-only) + **bandwidth** (RX WRITE processing rate,
  rate-matched DPU sender) and flags a **regression vs a recorded baseline** (`perf_baseline.txt`; lat
  +25%, bw −15% tolerances). **Baseline: p50 ≈ 51 µs, bw ≈ 12.5 Gbps (crc-on).** Standing rule: **every
  phase change runs `regression.sh` (correctness) AND `perf.sh` (BW+latency non-regression)** before the
  item is ticked. NB perf.sh never SIGTERMs a kernel — an abrupt kill wedges the eth core (needs
  `tt-smi -r` + re-force BF3 200G); it sizes hold_s + waits for the clean stop-flag shutdown.
- **0.3 Close known rough edges. ✅ mostly DONE.**
  - **`risc_touch` removed** from the TX ring kernel + host — it was a diagnostic red herring that hangs
    the TXQ (busy-waits a stuck `CMD_ONGOING`); a footgun that must not ship. TX egress re-verified after
    removal (no regression). The useful TXQ counter snapshots are kept.
  - **TX `pace` = the required raw-mode rate-limit, not a hack.** Raw `START_RAW` has no deep accept-ahead
    FIFO; over-arming sticks `CMD_ONGOING` and wedges the TXQ (needs a board reset — §11). `pace` throttles
    to the sustainable arm rate; the safe operating point (pace ≥ 100, line rate with MAX_PKT split) is
    validated. Deliberately NOT re-triggering the wedge to "characterize" it — it's understood and
    documented; a deeper HW fix (why `CMD_ONGOING` sticks) is a future investigation, not a bench reset now.
  - **RX resync-on-lap** counts lap events (`n_bad`) and degrades gracefully — done in the RX-ceiling work.
  - Diagnostic tooling is committed clean (counter snapshots) or removed (`risc_touch`).
- **0.4 CI wiring. ✅ content DONE.**
  - **HW-less PR gate:** `tt_metal/tt_rdma/bh0/ci_golden_test.cpp` (+ `ci_hwless.sh`) — the wire-header
    golden-vector oracle (SEND/WRITE_IMM/ACK byte-exact + canonical CRC-32C), plain `g++`, no HW/tt-metal/
    DOCA. Catches wire-format / CRC / struct-packing drift before it ships to the chip kernel *or* the DOCA
    gateway codec. Runs green (4/4) in ~1s anywhere.
  - **On-silicon suite:** `bringup.sh` → `regression.sh` is the one-command cold-rig gate.
  - TODO (infra, with the CI team): wire `ci_hwless.sh` into the per-PR GitHub Actions job, and
    `bringup.sh && regression.sh` into a labeled self-hosted bench runner (nightly / on-label).

Exit gate: a documented, one-command path from cold rig to all current claims re-verified automatically —
`bringup.sh` (restore rig) + `regression.sh` (5/5 datapath invariants) + `ci_hwless.sh` (wire oracle). **Met**
except the GitHub-Actions job wiring (infra task).

## Phase 1 — Complete the RX protocol (correctness & completeness)

No opcode left behind; every path tested + error-handled.

- **1.1 CRC-32 validation** on every inbound header (drop + count on mismatch); golden-vector tests.
  **Done (SW).** Header `header_cksum` is **CRC-32 (poly 0x04C11DB7, reflected 0xEDB88320)** — the BH
  ETH-CTRL `ROCE_ICRC` hardware polynomial (was Castagnoli/CRC-32C; switched pre-freeze so the RX check
  can offload to that inline engine). Kernel drops + counts mismatches (`crc_err`); validated on silicon
  both paths (good→0, corrupt→all dropped). Golden vectors regenerated (`crc32("123456789")=0xCBF43926`),
  HW-less CI + regression T5 green. **Follow-up:** wire the `ROCE_ICRC` engine (regs @ `0xFFB98100`,
  `RX_CHECK_EN` + `RX_CALCULATED`/`RX_RECEIVED`) to remove the CRC from the RISC hot path — needs an
  on-silicon bit-order/init calibration pass (see tt-rdma-rx-dispatch-spec §9).
- **1.2 SEND / SEND_IMM** → host RxWqeRing (NoC→PCIe push to a hugepage ring), completion to the host;
  test byte-exact delivery + CQE.
  - **1.2a Done (on-core).** SEND/SEND_IMM publish an RxWqeRing slot (host-sdk §3: 32B header + payload,
    `OWNED_BY_HOST` last) to a NoC-addressable core (Tensix L1) + bump a producer index (the completion).
    Byte-exact on silicon, regression **T6** (prod_idx=20, 8/8 slots `TTWR`-exact). Kernel gotcha fixed:
    `noc_async_write` source must be RDMA-L1, not the RISC stack (stage header/prod_idx in TX_BUF0 scratch).
  - **1.2b Pending.** Swap the ring target to a host hugepage (NoC→PCIe), map it + resolve its NoC address
    on the host SDK side, and raise the host-visible CQE.
- **1.3 READ_REQ / READ_RESP** — target-side READ handler (NoC read from MR → RESP frame via TXQ),
  initiator correlation; round-trip byte-exact test.
  - **1.3a Done (target side).** READ_REQ → MR lookup (REMOTE_READ + bounds) → `noc_async_read` →
    READ_RESP (tag+seq echoed, valid CRC) → `tt_rdma_send_raw` to the initiator. Byte-exact on the wire,
    regression **T7** (5/5 READ_RESP frames, op 0x21, 'READ' pattern). RX kernel now bidirectional.
    Fixed a runt-padding framing bug: header-only opcodes (READ_REQ/ACK) are padded to 48 B
    (`TT_RDMA_HDR_ONLY_BYTES`) so the MAC never runt-pads them (which desyncs header-only framing).
  - **1.3b Pending.** Initiator side: BH issues READ_REQ and correlates the RESP by tag (needs a
    read-correlation table, `TT_RDMA_READ_CORR`).
- **1.4 ACK (0x40)** reception + cumulative-ACK accounting (pairs with Phase 2 reliability). **Done.**
  Inbound ACK carries the peer's cumulative `ack_seq` in the seq field (header-only frame); the RX kernel
  tracks the highest via wraparound-safe signed compare (the "acked-up-to" watermark the TX/initiator side
  reads to free retransmit buffers). Validated on silicon (regression **T8**): 40 fresh ACKs → watermark
  40; 20 stale ACKs (seq ≤ 40) → watermark held at 40 (advance-on-newer + ignore-stale). Publishes
  `ack` count + `ack_seq` watermark.
- **1.5 WRITE_IMM / imm_data** completions. **Done.** A WRITE_IMM (0x11, `version_flags.IMM`) lands the
  payload at the MR (like WRITE) **and** raises a receiver completion: a length-0 RxWqeRing slot carrying
  `imm_data`, `mr_table_idx` = the target MR slot (host-sdk §3). The SEND publish was refactored into a
  shared `rxwqe_publish` helper (SEND passes payload; WRITE_IMM passes length 0 + imm). Validated on
  silicon (regression **T9**): payload byte-exact at the MR + completion slots with `op=0x11, len=0,
  imm=0xC0DE1257, owned`. Sender sets the IMM flag + imm for `_IMM` opcodes.
- **1.6 MR table lifecycle** — CONTROL-opcode register/deregister, rkey `(slot<<24)|rand|gen` generation
  + rotation, 64-slot management, and **access-control enforcement tests** (rkey_miss / rkey_access /
  rkey_bounds / rkey_wrap each provably dropped + counted). Security-relevant — no shortcut.
  - **1.6a Access-control enforcement. Done.** WRITE validation refactored into separately-counted drop
    classes — `rkey_miss` (slot OOR / rkey incl. generation mismatch), `rkey_access` (no REMOTE_WRITE),
    `rkey_bounds` (roff+len > mr_len); each an unauthorized WRITE provably **not landed** (stats[14..16]).
    Silicon-validated, regression **T10** (15 each → miss/access/bounds=15, write_ok=0; valid path T3 still
    lands write_ok=40). Both gates green (regression 12/12, perf OK).
  - **1.6b Pending.** CONTROL-opcode (0xF0) MR register/deregister over the wire + rkey generation/rotation
    (reuse detection) + 64-slot management tests. Also the MR-registration plumbing 3.1b + 4F.1 extend.

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

- **3.1 RX BW — line-rate via the Tensix drainer pool.** *(Rewritten 2026-07-26 after the RX line-rate
  investigation — see `tt-rdma-rx-linerate-research.md`. The old "make the single eth RISC faster" target
  was the wrong approach: one ~1 GHz RISC cannot hit 6M frames/s. The answer is to fan the per-frame work
  out to a Tensix worker pool, with the eth RISC as control plane.)* **Feasibility proven on silicon** —
  three green experiments: ingress MAC→L1 **198 Gbps drop-free** (exp 1), eth L1 sustains **200 G write +
  619 G read concurrent** drop-free (exp 2), Tensix pool processes (read+parse+`rkey`→MR+validate) at
  **~99 Gbps/worker linear → ~2–3 workers per 200 G link** (exp 3). Tools: `bh1_ingest_probe`,
  `bh1_l1_bw_test`, `bh1_rx_worker_test`. Build steps (research doc §7):
  - **3.1a** Port the Phase-1 per-frame body (parse / `rkey`→MR / validate / land) into the worker kernel
    (`bh_rdma_rx_worker.cpp` is the prototype). **Done (prototype).**
  - **3.1b** Shared MR table (RISC1 writes on registration, workers read) + atomic multi-consumer ring
    claim (NoC-atomic head) + framing (recommend fixed-size TT frames from the gateway → zero framing
    overhead; else RISC1 posts `(offset,len)` descriptors).
  - **3.1c** Remote-dest landing: worker `noc_read`(eth ring)→`noc_write`(arbitrary MR) (~2× workers).
  - **3.1d** Worker-posted completions → host RxWqeRing (extends Phase 1.2a).
  - **3.1e** RISC1 as control plane only (MR mgmt, exceptions, ACK, READ_RESP) — off the per-frame path.
  - **3.1f** Fold in PFC-lossless (Phase 2.1); acceptance: sustained **200 Gbps/link** vs the DOCA sender,
    drop=0, byte-exact landing. Multi-rail then aggregates N links for more BW / redundancy.
  - The Phase-1 single-RISC1 dispatch stays as the correctness reference + control-plane + regression oracle.
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

### 4F — ext-TT-RDMA ↔ TT-fabric bridge (inbound cross-chip landing)

Makes the node a **mesh** endpoint: an inbound op lands on an MR on *any* TT chip behind the gateway, not
just the edge chip. Closes the "tight bridge" gap (`bidirectional-mesh-gap.md`, `impl-plan §11.1`). Full
design + caveats: **`tt-rdma-fabric-bridge-design.md`**. Depends on 1.6 (MR lifecycle) + 3.1 (Tensix
drainer pool). Key insight: TT-fabric already routes + lands cross-chip with completions, and the drainer
workers are Tensix cores = native fabric clients — so the bridge is MR entries carrying a **fabric-global
address `{mesh,chip,noc_addr}`** + a one-branch fabric write (`udm::fabric_fast_write_any_len`) in the drainer.

- **4F.1** Fabric-global MR entry (`is_local` fast path) + cluster MR-registration service (interior-chip MR → edge `rkey` table).
- **4F.2** Drainer fabric-write branch; validate byte-exact on a 2-chip mesh (edge + interior) via the DOCA sender.
- **4F.3** Completion: fabric-barrier-gated CQE + deferred ACK reflecting the *interior* landing.
- **4F.4** **Measure cross-chip inbound BW** (interior MR, 1..k hops) — TT-fabric is RISC store-and-forward,
  so remote-MR is fabric-BW-bound (local-MR stays the measured 200 G). Establishes the remote-MR tier.
- **4F.5** Reliability (fabric-hop failure + retransmit), MR-across-chips consistency, and the coexistence
  gate (custom external-rail RX kernel + stock TT-fabric EDM on the same chip's different eth cores).

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

1. **Test before tick.** Each item lands with its automated regression test in the same change, and runs
   **both** gates: `regression.sh` (correctness, byte-exact) **and** `perf.sh` (BW + latency vs baseline).
   A correctness pass with a perf regression is not done.
2. **Reproducible from cold.** If it needs a manual bench step, it isn't done — script it (Phase 0.1).
3. **Error paths are features.** Every drop/failure has a defined behavior + a counter + a test.
4. **Commit small, push, keep the branch green.** Each item is a reviewable change on
   `aperezvicente/tt-rdma-bh-bf3`.
5. **No stale docs.** Findings update the specs; wrong premises are corrected, not left to mislead.
6. **Honest ledger.** This file's "current state" table is updated as items move from bench-validated to
   production-done; nothing silently drops off.
