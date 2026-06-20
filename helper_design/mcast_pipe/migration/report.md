# mcast_pipe rollout — re-entry report @ v7 (2026-06-20)

## Run header
- **Helper version (CURRENT):** `MCAST_PIPE_API_VERSION 7`. Unit test `test_mcast_pipe.py` 39/39 PASS (green baseline).
- **Entry mode:** re-entry. **Migration mode:** `run-all`. **Scope:** Tier 0 only (remigrate stale) — the 48 `pending` + 8 `deferred` were untouched per the request.
- **Machine:** single-chip **Wormhole** (this run). NOTE: the prior migration + ledger were last verified on **Blackhole p150a** — the two hangs below surfaced only now, on WH.
- **Remigration breakdown:** 13 stale kernels owed (all recorded at v4). **11 remigrated v4→v7 (device-verified). 2 quarantined.** 0 net-new (Tier-0-only run).

## Rollout state @ v7 (from ledger.json)
| status | count |
|---|---|
| migrated (current @ v7) | 11 |
| quarantined | 2 |
| pending | 48 |
| deferred | 8 |
- **0 stale remain** among the formerly-migrated set: every migrated entry is now at v7 (or quarantined). The migrated fleet is current at v7.

## Per-kernel results
| kernel | group | role | status | validation | commit | Δlines |
|---|---|---|---|---|---|---|
| reader_mcast_receiver_unary_sharded_gn_v2 | groupnorm | receiver | migrated v7 | PASS | 720f0ede3ac | 0 |
| welford_reader_mcast_receiver_unary_sharded_gn_v2 | groupnorm | receiver | migrated v7 | PASS | 7e8e0a90bcb | 0 |
| reader_final_topk | topk | sender (control) | migrated v7 | PASS | 15c9fa95984 | 0 |
| reader_writer_tiled_out_1d_mcast_receiver_conv_weights | conv HS | receiver | migrated v7 | PASS | d93798c8e07 | 0 |
| writer_tiled_out_2d_mcast_sender_conv_weights | conv BS | sender | migrated v7 | PASS | 65d233ef348 | 0 |
| writer_tiled_out_2d_mcast_receiver_conv_weights | conv BS | receiver | migrated v7 | PASS | b37b1853d20 | 0 |
| **activation_reader_width_sharded** | conv WS | sender (PRE_HANDSHAKE=false, round-robin self-mcast) | **QUARANTINED** | **HANG** (raw PASS) | 9b5d9b5823f (raw revert) | reverted |
| reader_bmm_tile_layout_in0_sender_padding | matmul | sender | migrated v7 | PASS | 3d31e412f9d | 1 |
| reader_bmm_tile_layout_in0_receiver | matmul | receiver | migrated v7 | PASS | 85cc48b1373 | ~4 |
| reader_bmm_tile_layout_in1_sender_writer_padding | matmul | hybrid | migrated v7 | PASS | 3a90bb89286 | 1 |
| reader_bmm_tile_layout_in1_receiver_writer_padding | matmul | receiver | migrated v7 | PASS | 592e9216927 | ~4 |
| **reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded** | matmul | role-flip (refactor) | **QUARANTINED** | **HANG** (raw PASS) | 49c5bd8f96f (raw revert) | reverted |
| reader_mcast_sender_unary_sharded_ln | layernorm | sender (two-phase) | migrated v7 | PASS | 00fa7199641 | 4 |

Tree is green at HEAD: re-ran both quarantined kernels' validation post-run (matmul 2D in0-sharded PASS, conv WS PASS) — confirms the quarantined kernels run raw and coexist with their migrated siblings.

## Headline finding — the 2 quarantined kernels (feed back to tune-dm-helper)
Both are the **`refactor`-tagged INCLUDE_SRC / loopback / round-robin self-mcast** kernels. In each, the
v7 translation was **API-correct** (verified against changelog R8 D1 signatures), compiled, but **HANGS**
on device, while the **raw pre-helper version PASSES the same test** (decisive A/B bisect each time).

Same root-cause class: the helper `send()`'s **runtime loopback-mode inference** + **degenerate
local-copy path** + its **single `NUM_ACTIVE_RECEIVER_CORES` recipient count** do not reproduce these
kernels' rotating-sender / self-mcast data+flag ordering and *split* recipient-count semantics. Concretely
for conv-WS: the pipe broadcasts the data-ready flag to `num_reader_cores-1`, but the kernel's raw fan-in
counter waits `num_mcast_cores-1 = max(num_input,num_output)-1`; when these differ some receivers never
see VALID → hang. This is a **helper-contract gap for the rotating/self-mcast topology**, not a Tier-0
remigration error → it belongs in `tune-dm-helper` (the loopback/rotating-role contract), and matches the
STAR-only capability note already in `proposed_helpers.md` (CHAIN/rotating = GAP).

These two map to the audit's `refactor` tag — they were the riskiest of the 13, as predicted.

## Coverage gaps
None. All 13 had high-confidence device-verified validation (block_sharded device-verified during this
re-entry's Phase 1). No kernel was migrated without a passing gate.

## Discrepancies worth noting
1. **Ledger said uniform v4; the tree was a MIX of intermediate APIs.** The 5 matmul kernels were truly at
   the dead `Pipe<>` (v4); topk/conv/ln senders were at an *old SenderPipe arg order* (count-first, no
   NOC_ID, un-templated `McastRect{}`, `send_signal(VALID)`); the GN/conv receivers were at a *pre-v7
   2-arg ReceiverPipe* with the consumed-sem drifted into v7's `PRE_HANDSHAKE` slot. So prior partial
   remigrations had happened in the tree but were never recorded — the ledger's blanket `migrated_api_version=4`
   under-described reality. All were brought to v7 regardless.
2. **conv-WS commit `9b5d9b5823f` is mislabeled** "remigrate ... to v7" but actually contains the *raw
   revert* (the subagent amended raw content over the migration commit). Functionally correct (raw,
   quarantined); only the message is wrong. Left as-is (cosmetic; rewriting history needs a rebase).

## Next steps
- **tune-dm-helper:** close the rotating/self-mcast loopback gap (split recipient-count + deterministic
  loopback ordering), then re-enter `apply-dm-helper` to lift the 2 quarantined kernels off raw. Until
  then they stay raw alongside their `pending` siblings.
- The 48 `pending` backlog is unchanged and out of scope for this run.
