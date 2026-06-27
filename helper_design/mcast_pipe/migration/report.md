# mcast_pipe rollout — final report (re-entry @ v8, 2026-06-20)

## Run header
- **Helper version (CURRENT):** `MCAST_PIPE_API_VERSION 8` (Round 10 — the **D2 count split**). Unit
  test `test_mcast_pipe.py` green **50/50** at intake. Helper **NOT modified** this run (only read).
- **Entry mode:** re-entry on a v7→v8 bump. **Migration mode:** `run-all` (autonomous, no halt).
  **Machine:** single-chip **Wormhole b0**.
- **Scope (per user request):** (1) remigrate every already-migrated kernel to v8 (the staleness
  sweep); (2) upgrade the conv-WS `activation_reader_width_sharded.cpp` from its partial
  (SenderPipe-only + raw receiver) form to a **full dual-pipe** (SenderPipe + ReceiverPipe).
- **Headline:** **19 / 19 migrated@v8, 0 failures, 0 quarantined.** Every stale kernel remigrated;
  the conv-WS dual-pipe upgrade — the highest-risk item, quarantined once at v7 — **passed on device**.

## The v7→v8 caller-facing move (what made everything stale)
`SenderPipe` **dropped its 3rd template param** `NUM_ACTIVE_RECEIVER_CORES`. Fan-out is now derived
from the rect's `area()`; a divergent consumer-ack count moves to a runtime ctor arg
`consumer_ack_count` (default `ACK_EQUALS_FANOUT` = the EXCLUDE fan-out). `ReceiverPipe` **unchanged**.
- **SenderPipe sites** = caller-facing-broken → rewritten (dense: pure arg deletion; divergent: delete
  arg + pass `consumer_ack_count`).
- **ReceiverPipe-only sites** = code unchanged → re-verified (their test runs alongside the sender) +
  version bumped 7→8.

## Rollout state @ v8 (from ledger.json) — 69 entries
| status | count |
|---|---|
| **migrated (current @ v8)** | **19** |
| pending | **0** |
| quarantined | **0** |
| deferred | **50** (8 original design-deferred + 19 design-gap + 23 coverage-gap) |

**0 pending and 0 stale → the migratable-on-this-chip fleet is fully current at v8.** Every remaining
kernel is deferred for a documented reason (helper design gap or no single-chip coverage).

## Per-tier results
| tier | scope | kernels | migrated | failed | quarantined |
|---|---|---|---|---|---|
| 0a | matmul (sender remigrate ×3 + receiver re-verify ×2) | 5 | 5 | 0 | 0 |
| 0b | layernorm sharded (sender ×3 + receiver ×3) | 6 | 6 | 0 | 0 |
| 0c | groupnorm v2 (receiver-only ×2, no code edit) | 2 | 2 | 0 | 0 |
| 0d | topk (sender ×1 + receiver ×1) | 2 | 2 | 0 | 0 |
| 0e | conv weights (sender ×1 + receiver ×2) | 3 | 3 | 0 | 0 |
| 1 | **conv-WS dual-pipe upgrade** (net-new) | 1 | 1 | 0 | 0 |
| **total** | | **19** | **19** | **0** | **0** |

## Per-kernel detail
### Tier 0a — matmul
| kernel | face | action | commit | test result |
|---|---|---|---|---|
| reader_bmm_tile_layout_in0_sender_padding.cpp | sender | **divergent** — drop arg + `consumer_ack_count=in0_mcast_num_dests` (num_cores-1) | 2dc2b7e5617 | 1D 48/0 |
| reader_bmm_tile_layout_in1_sender_writer_padding.cpp | sender | dense — pure deletion | d5ca9f7dbe1 | 2D 56/0 |
| reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp | sender+recv | dense — pure deletion (+ dead-var cleanup); recv arm unchanged | 5a9d07277df | 2D 56/0 |
| reader_bmm_tile_layout_in0_receiver.cpp | receiver | no code edit; re-verify + bump | (kept) | 1D 48/0 |
| reader_bmm_tile_layout_in1_receiver_writer_padding.cpp | receiver | no code edit; re-verify + bump | (kept) | 2D 56/0 |

> **Notable:** `in0_sender_padding` was NOT dense — under `uneven_width` the mcast rect covers more
> cores than actually ack, so the v8-default `ACK_EQUALS_FANOUT` over-waits → HANG. The explicit
> `consumer_ack_count` (exactly the D2 mechanism Round 10 added) was required. This is the first
> production confirmation of the divergent-ack arm outside the unit test.

### Tier 0b — layernorm (all dense, pure deletions; PRE_HANDSHAKE=false)
| kernel | face | commit | test |
|---|---|---|---|
| reader_mcast_sender_unary_sharded_ln.cpp | sender | 2b662e1ba1f | single_stage 64/0 |
| reader_mcast_receiver_unary_sharded_ln.cpp | receiver | (kept) | single_stage 64/0 |
| reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp | sender (send_signal) | 4c7f3b919f2 | pre_allgather 32/0 |
| reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp | receiver | (kept) | pre_allgather 32/0 |
| reader_mcast_sender_unary_sharded_ln_post_allgather.cpp | sender (loopback INCLUDE) | 65d3debd959 | post_allgather 64/0 |
| reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp | receiver | (kept) | post_allgather 64/0 |

### Tier 0c — groupnorm v2 (ReceiverPipe-only, no code edit; senders are deferred design-gaps)
| kernel | commit | test |
|---|---|---|
| reader_mcast_receiver_unary_sharded_gn_v2.cpp | (kept) | gn_v2_8x4_grid 6/6 |
| welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp | (kept) | gn_v2_8x4_grid (welford) |

### Tier 0d — topk
| kernel | face | commit | test |
|---|---|---|---|
| reader_final_topk.cpp | sender (send_signal, PRE_HANDSHAKE=false) — pure deletion | f8879d8c5ce | topk W=8192: 80 pass / 80 xfail / 0 fail |
| writer_local_topk.cpp | receiver | (kept) | topk W=8192 |

### Tier 0e — conv weights
| kernel | face | commit | test |
|---|---|---|---|
| writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | sender — dense pure deletion | 0d31bb2d615 | BLOCK_SHARDED 48/0, PCC 0.9999992 |
| writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp | receiver | (kept) | BLOCK_SHARDED |
| reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp | receiver | (kept) | HEIGHT_SHARDED 256×256 act_block_h=32 |

### Tier 1 — conv-WS dual-pipe upgrade (THE user request) ✅
| kernel | before | after | commit | test |
|---|---|---|---|---|
| activation_reader_width_sharded.cpp | SenderPipe-only (PRE_HANDSHAKE=false) + **raw receiver** (`act_mcast_receiver_sem.wait(VALID)`) + **raw fan-in counter** (`act_mcast_sender_sem.wait_min(num_mcast_cores-1)`) | **SenderPipe** (PRE_HANDSHAKE=true, `consumer_ack_count = num_mcast_cores-1`) **+ ReceiverPipe** (`.receive(sx,sy)`) | 4063ade12d0 (+ ledger 239a53c42fa) | WIDTH_SHARDED matrix **48 passed / 16 RM+bf8 skips / 0 fail**, smoke PCC 0.9999565, no hang |

**Why it works now (it was quarantined at v7):**
- The data-mcast fan-out (`num_reader_cores` = rect `area()`) **diverges** from the consumer-ack count
  (`num_mcast_cores - 1`). v7 had one count for both → the receiver/handshake stayed raw. **v8's
  `consumer_ack_count` decouples them** — the exact D2 gap Round 10 closed.
- The `ReceiverPipe` clear-AFTER-wait (H11) discipline that made this self-mcast hang at v7 is defused
  by Round 9's **per-send VALID re-assert (M12b)** in `send()` (same fix that lifted block_sharded).
- Sem wiring confirmed against the host factory: CTA13 = `act_mcast_receiver_sem` (data-ready flag),
  CTA12 = `act_mcast_sender_sem` (consumer-ready fan-in), both host-created on `all_cores` init 0.

## Commit hygiene
Per-kernel atomic commits + paired ledger write-backs throughout (revert-clean, bisectable). All
**local** — nothing pushed / rebased / reset. The **helper header was never touched** this run (correct;
design fixes are tune-dm-helper's job). Per-kernel verbose logs in `migration/log/`.

## Coverage notes / risks
- **conv-WS coverage upgraded:** its ledger `validation_set` was empty + `coverage_confidence=unknown`
  at intake. Now filled (`test_conv_features` WIDTH_SHARDED matrix, device-JIT-verified) with
  `coverage_confidence=high`. The single highest-risk item of the run, now well-tested.
- All other migrations land on already-green, device-verified call sites; no new coverage gaps.

## Out of scope (unchanged): 50 deferred kernels
Stay deferred per the prior run + user request — 8 original design-deferred, 19 design-gap
(D1/D3–D9), 23 coverage-gap (multi-device / no single-chip test). Not touched this run.

## Hand-off / observation for the next tune-dm-helper round
The v8 `consumer_ack_count` mechanism that unblocked conv-WS also directly addresses the other **D2
divergence deferrals** — most concretely the conv-1D weights **sender**
`reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` (deferred at v7 on
the split mcast-dest vs consumer-ack count, A/B-confirmed on device) and the dram-sharded matmul
sender. These are now candidates to **un-defer in a future divergent-tier apply-dm-helper run** (out of
this run's user-requested scope). D1 (runtime num_dests for gn_v2 senders), D3–D9 remain genuine gaps.
