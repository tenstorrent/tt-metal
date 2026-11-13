# M5: sparse MTP attention (GLM53_SPEC_MTP_SPARSE)

## Problem

The MTP layer (layer 45) attends densely over every position, from a private replicated latent cache. This adds
~1.6 ms per step per 32 K positions to the speculative step.

| Single-user sampled K 3 (2026-11-13) | ISL 127 | ISL 32 K | ISL ~1 M |
|---|---|---|---|
| spec step | 25.2 ms | 28.4 ms | 79.0 ms |
| speedup | 2.45x | 2.35x | **0.73x** |

The served profiles work around this by capping speculation: `GLM53_SPEC_MAX_CTX` is 262144 for the single-user
profile and 32768 for `glm53_flash_mtp_mu`. AgentX traffic has mean ISL 129 K and P95 378 K, so the multi-user cap
leaves most of it unspeculated.

## Reference semantics

Layer 45 ships a full DSA indexer: `wq_b`, `wk`, `weights_proj`, `k_norm`, `index_kpool_compress_{ape,gate}`.
The config sets `index_topk` 2048, `index_kpool` 4 and `index_share_for_mtp_iteration = True`. In the reference,
MTP draft 1 runs the indexer and selects; drafts 2..K reuse that selection
(`models/demos/deepseek_v3_d_p/tt/mtp_prefill/tt_mtp.py`).

## Design (v1, knob off by default)

`GLM53_SPEC_MTP_SPARSE=1` builds layer 45 as a sequence-sharded sparse_dsa layer, the same way a main one is built.

| Piece | Implementation |
|---|---|
| Attention | `MtpLayer(sparse=True)` builds `TPMlaShard(dsa=DsaIndexer(45))` with `TPDsaLayer._build_mla`'s knobs (indexer K blocks and rings). It shares the first main sparse_dsa layer's fabric phases and stream sync tensors. |
| Latent | A per-chip pool with the main layers' local block count, addressed by the main layers' page tables (`pt_v`, owner chip `g % C`). |
| DSA state | `DsaIndexer.stack_state`: a per-slot pooled-key store and ring, with the first main layer's slots, seats and local page tokens. |
| Catch-up | A one-slot verify-mode launch (`MLA_DSA_ONE_SLOT`; `GROUP_ROWS` for multi-user) at the verify's rows, positions and `pt_v`. |
| Chained rows | Plain-mode launches: row 0 is slot 0 and the other rows are inactive. |
| Ring protocol | `spec_ctl.cpp` `SPEC_MODE 4` (RING), described below. |
| Prefill | `TPPrefill._mtp_fill` writes the MTP latent rows to their owner chips (`tiles_to(ci.lat_idx)`). The MTP indexer's pooled keys and ring go through `DsaIndexer(45, st).keys(hn, ci)`, the same path as a main layer's. |

### Ring protocol

The catch-up writes rows in the verify layout. For pool q, rows go to `4 (q - q0) + p % 4` with `q0 = c / 4`.

1. **RING phase 1**, after the catch-up: rows `4 grp ..` move to the canonical rows 0..3, with
   `grp = (c + n + 1) / 4 - c / 4` (ACCEPT's rule). At K >= 2, rows 0..3 are also saved to rows 8..11.
2. **Chained rows** run in plain mode. A chained row that enters the next pool rewrites rows 0..3.
3. **RING phase 2**, after the chain: rows 8..11 are restored to rows 0..3. The next catch-up, whose base is
   `c + n + 1`, then finds its pool's committed rows.

`tests/test_spec_mtp_sparse_ring.py` models this protocol:

- I1: the canonical rows hold the committed rows.
- I2: a committed close reads committed rows only.

It also shows the protocol breaks without the save/restore at K 3.

### v1 scope (generator.spec_mtp_sparse_refusals)

- Supported:
  - single user, any K <= 3;
  - multi-user at K 1 (G 8 K 1), where there are no chained rows.
- Requirements:
  - the shard;
  - the per-slot store (not `GLM53_DSA_PAGED_POOLS`);
  - no dense window;
  - the in-trace MTP fill (`GLM53_SPEC_MTP_PREFILL` 1).
- Refused: multi-user K >= 2. Its chained launch row u addresses DSA slot u, but the user's state lives in slot
  u (K + 1). Fixing this is v2: a row-to-slot stride define in `ix_dm1` / `att_dsa_shard_dm0` / the select readers,
  or chained rows laid out at u (K + 1).
- Deviation from the reference: the chained drafts run the MTP indexer themselves instead of reusing draft 1's
  selection. This affects draft quality only; the verify keeps the output exact.

## DRAM per chip (8x4, TP 32)

| | B 16 x 1 M | B 4 x 1 M |
|---|---|---|
| MTP latent pool, sharded | ~512 MiB (8 K local blocks x 64 KiB) | ~128 MiB |
| MTP pooled-key store | ~32 MiB (16 slots x 2 MiB) | ~8 MiB |
| Ring | 0.25 MiB | 0.06 MiB |
| MTP indexer weights, decode + prefill copies | ~30 MB | ~30 MB |
| Dense MTP cache freed | 128 MiB (G 4 x 32 K cap) | 1 GiB uncapped / 256 MiB at the 256 K cap |

## L1

No new L1 buffers are expected:

- The MTP `MkMla(dsa)` programs have the main sparse_dsa programs' shapes.
- Their persistent buffers are seeded as row views of layer 43's (`_seed_persist`).
- The stream sync tensors are shared.
- RING is one core with a 1 KB CB.

## Validation

`_logs/mesh/chain_m5_1.sh` runs on `run_mk_m5.sh` (`wt_m5`, shadow root `tt-metal-m5`).

## Served profile changes for the cap lift (TIS)

- **Single-user `glm53_flash_mtp`:** add `GLM53_SPEC_MTP_SPARSE: "1"` and drop `GLM53_SPEC_MAX_CTX: "262144"`.
- **`glm53_flash_mtp_mu`:** v1 serves G 8 K 1 sparse: `GLM53_SPEC_K 1`, `GLM53_SPEC_MULTI_USERS 8`,
  `GLM53_SPEC_MTP_SPARSE 1`, and no `GLM53_SPEC_MAX_CTX`. G 4 K 3 stays dense and capped until v2.
