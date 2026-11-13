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
- Multi-user K >= 2 is v2 (below); v1 refused it.
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

## v2: multi-user at any K, the alternate step, the partial-entry seed

### Chained rows on strided slots (mk_mla rev 74)

A multi-user chained launch has row u = user u (`spec_widths` ch: the smallest width >= G), but user u's DSA state
lives on its group base, generator slot u (K + 1). v2 adds `MLA_DSA_SLOT_STRIDE = S` for plain rows:

| Piece | Change |
|---|---|
| `ix_dm1.cpp` | plain row t writes the ring / pooled keys of slot t S (`#error` with ONE_SLOT or the paged pools) |
| `att_dsa_shard_dm0.cpp`, `att_dsa_dm0.cpp` | plain row t scores the pooled keys of slot t S |
| `mk_mla.py` | `dsa_state["slot_stride"]` -> the define on the index and reader programs; `dsa_slot_addrs` bound-checks the last strided slot (`stride_rows`: the live rows) |
| `spec_mtp.py` | `pt_vc`: the chained rows' LOCAL page table (row u = user u's local row; `set_page_tables` writes it); `_chain` launches with `slot_stride = K + 1`, `stride_rows = G` |
| `generator.spec_mtp_sparse_refusals` | multi-user K >= 2 allowed (only G (K + 1) > 16 refused) |

Page tables and positions stay per launch row (`pt_vc`, `mpos_ch` row u); only the DSA state is strided. The ring
protocol is unchanged: SPEC_MODE 4 already addresses user u's ring at slot u (K + 1).

### The alternate step (GLM53_SPEC_MULTI_ALT = "G:K")

The served 4 x 3 step declined every step at 5..8 live users (smtp3a 2026-11-13, c8: 244 declines, 24.7 tok/s/u plain
at width 16). `GLM53_SPEC_MULTI_ALT=8:1` builds a second `SpecMTP` (8 users x 1 draft) on the same build with `share=`
the primary: one MTP layer, one sparse latent pool / DSA state (both per generator slot), one MkEh. `SpecSwitch` makes
one step current; every attribute not defined on it is the current step's.

| Rule | Where |
|---|---|
| The alternate's group stride divides the primary's (2 | 4): the primary's bases 0 / 4 / 8 / 12 are alternate bases, no request ever moves | `generator.spec_multi_alt` |
| New requests take the primary's free bases, then the alternate's others (0, 4, 8, 12, 2, 6, 10, 14) | `_admit`, `spec_all_slots` |
| The step serving the live users: the first whose bases hold all of them (partial groups) | `_spec_multi_pick` / `spec_pick_step` |
| A step switch leaves the running speculation first, then `spec_select` | `_propose_multi`, `_spec_seed` |
| The plugin's block is always 1 + effective_k (3) wide; the alternate offers `num_valid` 1 and verifies columns 0..1 (columns 2..3 padded -1) | `_fused_rows`, `_spec_ids` |
| k_by_rows ((4, 3), (8, 1), (16, 1)) already caps 5..8 live rows at 1 draft | `spec_multi_k_by_rows` |

Requires `GLM53_SPEC_MTP_SPARSE=1` (the dense private MTP caches are per group index, not per slot).

### The partial-entry seed

`_spec_seed` entered every group base (`spec_enter` without slots): with partial groups and fewer live users than G it
raised "the entering bases [0, 4, 8, 12] must be exactly the live slots" (smtp3a: 20 of 52 entries unseeded). It now
enters the picked step's live subset.

### Narrow widths with multi-user speculation

A plain step's width is the smallest captured width covering slots 0..max live slot. Users on group bases
0 / 4 / 8 / 12 therefore decode plain steps at width 16 as soon as one sits on slot 8 or 12; narrow width 4 serves
only a lone user on slot 0 (native width 8: users on 0 / 4). Nothing in the step breaks the narrow widths (the chained
launch runs at ch = 4 when width 4 is built: the MTP layer's t = 4 programs on the narrow phases). The lever for plain
steps is fewer of them: partial groups + the alternate step keep speculating as users come and go (smtp3a: 512 fused
vs 46 fallback steps at c4).

### Memory

| | Per chip |
|---|---|
| `pt_vc` (v2) | 32 rows x bps_local int32 (64 KiB at 1 M) |
| The alternate step | its own verify / MTP-catch-up programs, heads, records, `pt_v`, the 16-row flat MTP residual (1 MiB); no second MTP layer or state |
| L1 | no new buffer expected: the alternate's persistent buffers are row views (`_seed_persist`); measured by the chain's GLM53_L1_REPORT rows |
